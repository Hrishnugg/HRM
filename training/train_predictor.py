"""
Training script for HRM-augmented pathfinding.

Uses Stable-Baselines3 with PPO/SAC for training the agent
with curriculum learning and HRM-based feature extraction.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import json

import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from envs.dynamic_pathfinding_env import DynamicPathfindingEnv, EnvConfig
from envs.prediction.hrm_predictor import HRMObstaclePredictor
from training.sb3_wrapper import (
    HRMFeatureExtractor,
    PredictorRewardWrapper,
    CurriculumEnvWrapper,
    CurriculumCallback,
    create_curriculum_stages,
)


@dataclass
class TrainConfig:
    """Training configuration."""
    
    # Environment
    map_size: int = 32
    num_obstacles: int = 5
    obstacle_speed_range: tuple = (0.5, 2.0)
    max_episode_steps: int = 500
    
    # Algorithm
    algorithm: str = "PPO"  # "PPO" or "SAC"
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    n_steps: int = 2048  # For PPO
    batch_size: int = 64
    n_epochs: int = 10  # For PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95  # For PPO
    clip_range: float = 0.2  # For PPO
    ent_coef: float = 0.01  # Entropy coefficient
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: List[Dict] = field(default_factory=list)
    
    # Feature extractor
    features_dim: int = 256
    use_hrm_encoder: bool = True
    
    # Reward shaping
    prediction_reward_weight: float = 0.1
    safety_reward_weight: float = 0.2
    
    # Training
    n_envs: int = 8  # Number of parallel environments
    eval_freq: int = 10000
    n_eval_episodes: int = 20
    save_freq: int = 50000
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "hrm-astar-rl"
    wandb_run_name: Optional[str] = None
    log_dir: str = "logs/hrm_astar"
    checkpoint_dir: str = "checkpoints/hrm_astar"
    
    # Device
    device: str = "auto"
    
    # Seed
    seed: int = 42


def make_env_fn(
    config: TrainConfig,
    rank: int,
    predictor: Optional[HRMObstaclePredictor] = None,
    is_eval: bool = False,
):
    """
    Factory function for creating environments.
    
    Args:
        config: Training configuration
        rank: Environment rank (for seeding)
        predictor: HRM predictor for reward shaping
        is_eval: Whether this is an evaluation environment
    """
    def _init():
        env_config = EnvConfig(
            map_size=config.map_size,
            num_obstacles=config.num_obstacles,
            obstacle_speed_range=config.obstacle_speed_range,
            max_steps=config.max_episode_steps,
        )
        
        env = DynamicPathfindingEnv(env_config)
        
        # Add time limit
        env = TimeLimit(env, max_episode_steps=config.max_episode_steps)
        
        # Add reward shaping (only for training)
        if not is_eval and predictor is not None:
            env = PredictorRewardWrapper(
                env,
                predictor=predictor,
                prediction_weight=config.prediction_reward_weight,
                safety_weight=config.safety_reward_weight,
            )
        
        # Add curriculum (only for training)
        if not is_eval and config.use_curriculum:
            stages, thresholds = create_curriculum_stages()
            env = CurriculumEnvWrapper(env, stages, thresholds)
        
        # Wrap with monitor
        env = Monitor(env)
        
        # Seed
        env.reset(seed=config.seed + rank)
        
        return env
    
    return _init


def create_model(
    config: TrainConfig,
    env: gym.Env,
    predictor: Optional[HRMObstaclePredictor] = None,
) -> Any:
    """
    Create the RL model.
    
    Args:
        config: Training configuration
        env: Training environment
        predictor: HRM predictor for feature extraction
        
    Returns:
        SB3 model
    """
    # Policy kwargs with custom feature extractor
    policy_kwargs = {
        "features_extractor_class": HRMFeatureExtractor,
        "features_extractor_kwargs": {
            "features_dim": config.features_dim,
            "hrm_predictor": predictor,
            "use_predictor_encoder": config.use_hrm_encoder,
        },
        "net_arch": dict(pi=[256, 256], vf=[256, 256]),
    }
    
    if config.algorithm.upper() == "PPO":
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            verbose=1,
            tensorboard_log=config.log_dir,
            device=config.device,
            policy_kwargs=policy_kwargs,
            seed=config.seed,
        )
    elif config.algorithm.upper() == "SAC":
        model = SAC(
            "MultiInputPolicy",
            env,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            gamma=config.gamma,
            ent_coef="auto",
            verbose=1,
            tensorboard_log=config.log_dir,
            device=config.device,
            policy_kwargs=policy_kwargs,
            seed=config.seed,
        )
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")
    
    return model


def train_predictor(config: Optional[TrainConfig] = None):
    """
    Main training function.
    
    Args:
        config: Training configuration (uses defaults if None)
    """
    config = config or TrainConfig()
    
    # Create directories
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    print("=" * 60)
    print("HRM-Augmented A* RL Training")
    print("=" * 60)
    print(f"Algorithm: {config.algorithm}")
    print(f"Total timesteps: {config.total_timesteps:,}")
    print(f"Parallel environments: {config.n_envs}")
    print(f"Map size: {config.map_size}x{config.map_size}")
    print(f"Obstacles: {config.num_obstacles}")
    print(f"Curriculum learning: {config.use_curriculum}")
    print(f"Device: {config.device}")
    print()
    
    # Initialize W&B
    if config.use_wandb and WANDB_AVAILABLE:
        run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=vars(config),
            sync_tensorboard=True,
        )
        print(f"W&B run: {run.url}")
    
    # Create HRM predictor
    print("Creating HRM predictor...")
    predictor = HRMObstaclePredictor(
        hidden_size=128,
        history_length=10,
        prediction_horizon=5,
        num_obstacles=config.num_obstacles,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Create environments
    print("Creating training environments...")
    
    if config.n_envs > 1:
        env = SubprocVecEnv([
            make_env_fn(config, i, predictor, is_eval=False)
            for i in range(config.n_envs)
        ])
    else:
        env = DummyVecEnv([
            make_env_fn(config, 0, predictor, is_eval=False)
        ])
    
    env = VecMonitor(env)
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([
        make_env_fn(config, 0, None, is_eval=True)
    ])
    eval_env = VecMonitor(eval_env)
    
    # Create model
    print("Creating model...")
    model = create_model(config, env, predictor)
    
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create callbacks
    callbacks = []
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config.checkpoint_dir,
        log_path=config.log_dir,
        eval_freq=config.eval_freq // config.n_envs,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True,
    )
    callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq // config.n_envs,
        save_path=config.checkpoint_dir,
        name_prefix="hrm_astar",
    )
    callbacks.append(checkpoint_callback)
    
    # W&B callback
    if config.use_wandb and WANDB_AVAILABLE:
        wandb_callback = WandbCallback(
            model_save_path=config.checkpoint_dir,
            verbose=2,
        )
        callbacks.append(wandb_callback)
    
    # Curriculum callback
    if config.use_curriculum:
        # Get curriculum wrapper from first env
        base_env = env.envs[0]
        while hasattr(base_env, 'env'):
            if isinstance(base_env, CurriculumEnvWrapper):
                curriculum_callback = CurriculumCallback(base_env)
                callbacks.append(curriculum_callback)
                break
            base_env = base_env.env
    
    # Train
    print("\nStarting training...")
    print("=" * 60)
    
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Save final model
    final_path = os.path.join(config.checkpoint_dir, "final_model")
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")
    
    # Save predictor
    predictor_path = os.path.join(config.checkpoint_dir, "predictor.pt")
    predictor.save(predictor_path)
    print(f"Predictor saved to: {predictor_path}")
    
    # Save config
    config_path = os.path.join(config.checkpoint_dir, "config.json")
    with open(config_path, 'w') as f:
        # Convert config to dict, handling non-serializable types
        config_dict = {}
        for key, value in vars(config).items():
            if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                config_dict[key] = value
            else:
                config_dict[key] = str(value)
        json.dump(config_dict, f, indent=2)
    print(f"Config saved to: {config_path}")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    print("\nTraining complete!")
    
    return model, predictor


def load_trained_model(
    checkpoint_dir: str,
    device: str = "auto",
) -> tuple:
    """
    Load a trained model and predictor.
    
    Args:
        checkpoint_dir: Directory containing saved model
        device: Device to load model on
        
    Returns:
        (model, predictor, config) tuple
    """
    # Load config
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    config = TrainConfig(**config_dict)
    
    # Load predictor
    predictor = HRMObstaclePredictor(
        hidden_size=128,
        history_length=10,
        prediction_horizon=5,
        num_obstacles=config.num_obstacles,
        device=device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    
    predictor_path = os.path.join(checkpoint_dir, "predictor.pt")
    if os.path.exists(predictor_path):
        predictor.load(predictor_path)
    
    # Load model
    model_path = os.path.join(checkpoint_dir, "final_model.zip")
    
    # Create dummy env for model loading
    env_config = EnvConfig(
        map_size=config.map_size,
        num_obstacles=config.num_obstacles,
    )
    dummy_env = DynamicPathfindingEnv(env_config)
    
    if config.algorithm.upper() == "PPO":
        model = PPO.load(model_path, env=dummy_env, device=device)
    else:
        model = SAC.load(model_path, env=dummy_env, device=device)
    
    return model, predictor, config


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train HRM-augmented A* agent")
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "SAC"])
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--map-size", type=int, default=32)
    parser.add_argument("--obstacles", type=int, default=5)
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    config = TrainConfig(
        algorithm=args.algorithm,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        map_size=args.map_size,
        num_obstacles=args.obstacles,
        use_curriculum=not args.no_curriculum,
        use_wandb=args.wandb,
        seed=args.seed,
    )
    
    train_predictor(config)

