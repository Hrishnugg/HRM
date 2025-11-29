"""
Multi-GPU Training for HRM-Augmented A*.

Optimized for multi-GPU training on Modal with B200/H100 GPUs.
Called as subprocess per Modal docs: https://modal.com/docs/guide/gpu#multi-gpu-training

Usage:
    python -m training.train_multi_gpu --algorithm PPO --timesteps 10000000
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU training."""
    
    # Algorithm
    algorithm: str = "PPO"
    
    # Training
    timesteps: int = 10_000_000
    n_envs: int = 64
    batch_size: int = 2048
    n_steps: int = 8192
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    
    # Environment
    map_size: int = 32
    num_obstacles: int = 8
    max_episode_steps: int = 500
    use_curriculum: bool = True
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "hrm-astar-multi-gpu"
    wandb_run_name: Optional[str] = None
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    save_freq: int = 100_000
    eval_freq: int = 50_000
    
    # Device
    device: str = "cuda"


def setup_multi_gpu():
    """Setup for multi-GPU training."""
    num_gpus = torch.cuda.device_count()
    
    if num_gpus == 0:
        print("⚠️  No GPUs detected, falling back to CPU")
        return "cpu", 1
    
    print(f"🚀 Multi-GPU Setup: {num_gpus} GPUs available")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {props.name}")
        print(f"      Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"      Compute: {props.major}.{props.minor}")
    
    # Set CUDA environment for optimal multi-GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    # Enable TF32 for Ampere+ GPUs (faster training)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable cudnn benchmark for consistent input sizes
    torch.backends.cudnn.benchmark = True
    
    return "cuda", num_gpus


def train_distributed(
    algorithm: str = "PPO",
    timesteps: int = 10_000_000,
    n_envs: int = 64,
    batch_size: int = 2048,
    n_steps: int = 8192,
    use_curriculum: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "hrm-astar",
    wandb_run_name: Optional[str] = None,
    log_dir: str = "logs",
    checkpoint_dir: str = "checkpoints",
    map_size: int = 32,
    num_obstacles: int = 8,
    **kwargs,
):
    """
    Main training function for multi-GPU setups.
    
    Optimizes for high GPU utilization with:
    - Large batch sizes
    - Many parallel environments
    - Efficient data loading
    """
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import (
        EvalCallback,
        CheckpointCallback,
        CallbackList,
    )
    from stable_baselines3.common.monitor import Monitor
    from gymnasium.wrappers import TimeLimit
    
    from envs.dynamic_pathfinding_env import DynamicPathfindingEnv, EnvConfig
    from envs.prediction.hrm_predictor import HRMObstaclePredictor
    from training.sb3_wrapper import (
        HRMFeatureExtractor,
        PredictorRewardWrapper,
        CurriculumEnvWrapper,
        create_curriculum_stages,
    )
    
    # Setup
    device, num_gpus = setup_multi_gpu()
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("HRM-Augmented A* Multi-GPU Training")
    print("=" * 60)
    print(f"Algorithm: {algorithm}")
    print(f"Timesteps: {timesteps:,}")
    print(f"GPUs: {num_gpus}")
    print(f"Environments: {n_envs}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per update: {n_steps}")
    print(f"Map size: {map_size}x{map_size}")
    print(f"Obstacles: {num_obstacles}")
    print(f"Curriculum: {use_curriculum}")
    print()
    
    # Initialize W&B
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    "algorithm": algorithm,
                    "timesteps": timesteps,
                    "n_envs": n_envs,
                    "batch_size": batch_size,
                    "n_steps": n_steps,
                    "num_gpus": num_gpus,
                    "map_size": map_size,
                    "num_obstacles": num_obstacles,
                },
            )
        except Exception as e:
            print(f"⚠️  W&B init failed: {e}")
            use_wandb = False
    
    # Create HRM predictor
    print("Creating HRM predictor...")
    predictor = HRMObstaclePredictor(
        hidden_size=128,
        history_length=10,
        prediction_horizon=5,
        num_obstacles=num_obstacles,
        device=device,
    )
    
    # Environment factory
    def make_env(rank: int, is_eval: bool = False):
        def _init():
            env_config = EnvConfig(
                map_size=map_size,
                num_obstacles=num_obstacles,
                max_steps=500,
            )
            env = DynamicPathfindingEnv(env_config)
            env = TimeLimit(env, max_episode_steps=500)
            
            if not is_eval:
                env = PredictorRewardWrapper(env, predictor=predictor)
                
                if use_curriculum:
                    stages, thresholds = create_curriculum_stages(
                        base_obstacles=3,
                        max_obstacles=num_obstacles,
                        num_stages=5,
                    )
                    env = CurriculumEnvWrapper(env, stages, thresholds)
            
            env = Monitor(env)
            return env
        
        return _init
    
    # Create vectorized environments
    print(f"Creating {n_envs} parallel environments...")
    
    env = SubprocVecEnv(
        [make_env(i) for i in range(n_envs)],
        start_method="spawn",  # Better for CUDA
    )
    env = VecMonitor(env)
    
    # Evaluation environment
    eval_env = SubprocVecEnv(
        [make_env(0, is_eval=True) for _ in range(4)],
        start_method="spawn",
    )
    eval_env = VecMonitor(eval_env)
    
    # Policy kwargs
    policy_kwargs = {
        "features_extractor_class": HRMFeatureExtractor,
        "features_extractor_kwargs": {
            "features_dim": 256,
            "hrm_predictor": predictor,
            "use_predictor_encoder": True,
        },
        "net_arch": dict(pi=[512, 512, 256], vf=[512, 512, 256]),
        "activation_fn": nn.ReLU,
    }
    
    # Create model
    print(f"Creating {algorithm} model...")
    
    if algorithm.upper() == "PPO":
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=log_dir,
            device=device,
            policy_kwargs=policy_kwargs,
        )
    elif algorithm.upper() == "SAC":
        model = SAC(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            batch_size=batch_size,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto",
            verbose=1,
            tensorboard_log=log_dir,
            device=device,
            policy_kwargs=policy_kwargs,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Print model info
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Callbacks
    callbacks = []
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=log_dir,
        eval_freq=max(50_000 // n_envs, 1000),
        n_eval_episodes=20,
        deterministic=True,
    )
    callbacks.append(eval_callback)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 5000),
        save_path=checkpoint_dir,
        name_prefix="hrm_astar",
    )
    callbacks.append(checkpoint_callback)
    
    if use_wandb:
        try:
            from wandb.integration.sb3 import WandbCallback
            callbacks.append(WandbCallback(
                model_save_path=checkpoint_dir,
                verbose=2,
            ))
        except Exception:
            pass
    
    # Train
    print("\n🚀 Starting training...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=CallbackList(callbacks),
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted")
    
    elapsed = time.time() - start_time
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_model")
    model.save(final_path)
    print(f"\n💾 Model saved to: {final_path}")
    
    # Save predictor
    predictor_path = os.path.join(checkpoint_dir, "predictor.pt")
    predictor.save(predictor_path)
    print(f"💾 Predictor saved to: {predictor_path}")
    
    # Stats
    steps_per_second = timesteps / elapsed
    print(f"\n✅ Training complete!")
    print(f"   Time: {elapsed/3600:.2f} hours ({elapsed/60:.1f} min)")
    print(f"   Steps/sec: {steps_per_second:,.0f}")
    print(f"   Steps/sec/GPU: {steps_per_second/num_gpus:,.0f}")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass
    
    return {
        "elapsed_time": elapsed,
        "steps_per_second": steps_per_second,
        "checkpoint_dir": checkpoint_dir,
    }


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Multi-GPU training for HRM-augmented A*"
    )
    
    # Algorithm
    parser.add_argument("--algorithm", "-a", default="PPO", choices=["PPO", "SAC"])
    
    # Training params
    parser.add_argument("--timesteps", "-t", type=int, default=10_000_000)
    parser.add_argument("--n-envs", "-e", type=int, default=64)
    parser.add_argument("--batch-size", "-b", type=int, default=2048)
    parser.add_argument("--n-steps", type=int, default=8192)
    
    # Environment
    parser.add_argument("--map-size", type=int, default=32)
    parser.add_argument("--num-obstacles", type=int, default=8)
    parser.add_argument("--use-curriculum", action="store_true", default=True)
    parser.add_argument("--no-curriculum", dest="use_curriculum", action="store_false")
    
    # Logging
    parser.add_argument("--use-wandb", action="store_true", default=False)
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    parser.add_argument("--wandb-project", default="hrm-astar-multi-gpu")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    
    args = parser.parse_args()
    
    train_distributed(
        algorithm=args.algorithm,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        map_size=args.map_size,
        num_obstacles=args.num_obstacles,
        use_curriculum=args.use_curriculum,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()

