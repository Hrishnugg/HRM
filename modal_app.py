"""
Modal App for HRM-Augmented A* RL Training.

Deploy training to Modal's serverless GPU infrastructure for fast,
scalable training without local hardware requirements.

Usage:
    # Deploy and run training
    modal run modal_app.py::train
    
    # Run with custom config
    modal run modal_app.py::train --timesteps 2000000 --n-envs 16
    
    # BLAZING FAST: 4x H100 GPUs
    modal run modal_app.py::train_4gpu --timesteps 10000000
    
    # Download trained model
    modal run modal_app.py::download_checkpoint --run-id <run_id>
    
    # Evaluate a trained model
    modal run modal_app.py::evaluate --checkpoint-path /vol/checkpoints/<run_id>/final_model.zip

See https://modal.com/docs for Modal documentation.
"""

import modal
from pathlib import Path

# =============================================================================
# Modal App Configuration
# =============================================================================

# Create the Modal app
app = modal.App("hrm-astar-rl")

# Persistent volume for checkpoints and data
volume = modal.Volume.from_name("hrm-astar-volume", create_if_missing=True)
VOLUME_PATH = "/vol"
CHECKPOINT_DIR = f"{VOLUME_PATH}/checkpoints"
LOG_DIR = f"{VOLUME_PATH}/logs"
DATA_DIR = f"{VOLUME_PATH}/data"

# GPU configurations (per Modal docs: https://modal.com/docs/guide/gpu)
# Supported: T4, L4, A10, A100, A100-40GB, A100-80GB, L40S, H100, H200, B200
# Multi-GPU: up to 8 GPUs for B200, H200, H100, A100, L4, T4, L40S; up to 4 for A10
GPU_CONFIGS = {
    "T4": "T4",
    "L4": "L4",
    "A10": "A10",
    "L40S": "L40S",
    "A100": "A100",
    "A100-40GB": "A100-40GB",
    "A100-80GB": "A100-80GB",
    "H100": "H100",
    "H200": "H200",
    "B200": "B200",  # Blackwell architecture (closest to GB200)
    # Multi-GPU configs
    "2xA100": "A100:2",
    "4xA100": "A100:4",
    "8xA100": "A100:8",
    "2xH100": "H100:2",
    "4xH100": "H100:4",
    "8xH100": "H100:8",
    "2xH200": "H200:2",
    "4xH200": "H200:4",
    "8xH200": "H200:8",
    "2xB200": "B200:2",
    "4xB200": "B200:4",  # 4x Blackwell - BLAZING FAST
    "8xB200": "B200:8",  # 8x Blackwell - MAXIMUM POWER
}

# CUDA version for Blackwell (B200/GB200) support
# Blackwell requires CUDA 12.4+ and PyTorch 2.5+
CUDA_VERSION = "12.8.1"
CUDA_FLAVOR = "devel"  # Full toolkit for compilation
CUDA_OS = "ubuntu24.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

# Container image with CUDA toolkit for Blackwell/GB200 support
# Using official NVIDIA CUDA image per: https://modal.com/docs/guide/cuda
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .entrypoint([])  # Remove verbose logging
    .apt_install("git", "ffmpeg", "openssh-client", "libopenmpi-dev")
    .pip_install(
        # Core ML - PyTorch 2.5+ for Blackwell support
        "torch>=2.5.0",
        "numpy>=1.26.0",
        "einops>=0.8.0",
        "pydantic>=2.0.0",
        
        # RL
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.3.0",
        "shimmy>=1.0.0",
        "sb3-contrib>=2.3.0",
        
        # Visualization
        "matplotlib>=3.8.0",
        "tqdm>=4.66.0",
        
        # Logging
        "wandb>=0.16.0",
        "tensorboard>=2.15.0",
        
        # Config
        "omegaconf>=2.3.0",
        "pyyaml>=6.0",
        
        # HTTP
        "requests>=2.31.0",
        
        # Multi-GPU / Distributed
        "accelerate>=0.27.0",
    )
    .env({
        "NCCL_DEBUG": "WARN",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)

# Simpler image for non-Blackwell GPUs (faster cold start)
image_standard = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch>=2.2.0",
        "numpy>=1.24.0",
        "einops>=0.7.0",
        "pydantic>=2.0.0",
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.2.0",
        "shimmy>=1.0.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "wandb>=0.15.0",
        "tensorboard>=2.14.0",
        "omegaconf>=2.3.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
    )
)

# Add local code to the image (Modal's new API replaces Mount)
# Exclude .git, checkpoints, logs, __pycache__ etc.
image = image.add_local_dir(
    Path(__file__).parent,
    remote_path="/app",
    copy=True,  # Copy files into image layer for faster startup
    ignore=[
        ".git",
        ".git/**",
        "__pycache__",
        "**/__pycache__",
        "*.pyc",
        "checkpoints",
        "checkpoints/**",
        "logs",
        "logs/**",
        ".venv",
        ".venv/**",
        "*.egg-info",
        ".pytest_cache",
    ],
)


# =============================================================================
# Training Function
# =============================================================================

@app.function(
    image=image,
    gpu="A100",  # Use A100 for fast training (alternatives: "T4", "A10G", "H100")
    timeout=60 * 60 * 12,  # 12 hour timeout
    volumes={VOLUME_PATH: volume},
    # secrets=[modal.Secret.from_name("wandb-secret")] if you set up W&B
)
def train(
    algorithm: str = "PPO",
    timesteps: int = 1_000_000,
    n_envs: int = 16,  # More envs for faster data collection
    map_size: int = 32,
    num_obstacles: int = 5,
    use_curriculum: bool = True,
    use_wandb: bool = False,
    seed: int = 42,
    run_name: str = None,
):
    """
    Train the HRM-augmented A* agent on Modal GPUs.
    
    Args:
        algorithm: "PPO" or "SAC"
        timesteps: Total training timesteps
        n_envs: Number of parallel environments
        map_size: Grid map size
        num_obstacles: Number of dynamic obstacles
        use_curriculum: Enable curriculum learning
        use_wandb: Enable W&B logging
        seed: Random seed
        run_name: Custom run name (auto-generated if None)
    """
    import os
    import sys
    import time
    from datetime import datetime
    
    # Add app to path
    sys.path.insert(0, "/app")
    
    # Generate run ID
    run_id = run_name or f"modal_{algorithm.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Set up directories
    checkpoint_dir = f"{CHECKPOINT_DIR}/{run_id}"
    log_dir = f"{LOG_DIR}/{run_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print("=" * 60)
    print("HRM-Augmented A* Training on Modal")
    print("=" * 60)
    print(f"Run ID: {run_id}")
    print(f"Algorithm: {algorithm}")
    print(f"Timesteps: {timesteps:,}")
    print(f"Parallel envs: {n_envs}")
    print(f"GPU: {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'N/A')}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print()
    
    # Import training module
    from training.train_predictor import TrainConfig, train_predictor
    
    # Create config
    config = TrainConfig(
        algorithm=algorithm,
        total_timesteps=timesteps,
        n_envs=n_envs,
        map_size=map_size,
        num_obstacles=num_obstacles,
        use_curriculum=use_curriculum,
        use_wandb=use_wandb,
        wandb_project="hrm-astar-modal",
        wandb_run_name=run_id,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
        device="cuda",
        # Optimized settings for GPU training
        batch_size=256,  # Larger batch for GPU
        n_steps=4096,  # More steps per update
    )
    
    # Train
    start_time = time.time()
    model, predictor = train_predictor(config)
    elapsed = time.time() - start_time
    
    print(f"\nTraining completed in {elapsed/3600:.2f} hours")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    
    # Commit volume changes
    volume.commit()
    
    return {
        "run_id": run_id,
        "checkpoint_dir": checkpoint_dir,
        "elapsed_time": elapsed,
        "timesteps": timesteps,
    }


# =============================================================================
# Fast Training with Multiple GPUs
# =============================================================================

@app.function(
    image=image,
    gpu="A100:2",  # 2x A100 GPUs
    timeout=60 * 60 * 24,  # 24 hour timeout
    volumes={VOLUME_PATH: volume},
    # secrets=[modal.Secret.from_name("wandb-secret")] if needed
)
def train_multi_gpu(
    algorithm: str = "PPO",
    timesteps: int = 5_000_000,
    n_envs: int = 32,
    use_wandb: bool = True,
    run_name: str = None,
):
    """
    Train with 2x A100 GPUs.
    
    Uses larger batch sizes and more parallel environments.
    """
    import os
    import sys
    from datetime import datetime
    
    sys.path.insert(0, "/app")
    
    run_id = run_name or f"modal_2xA100_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = f"{CHECKPOINT_DIR}/{run_id}"
    log_dir = f"{LOG_DIR}/{run_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    from training.train_predictor import TrainConfig, train_predictor
    
    config = TrainConfig(
        algorithm=algorithm,
        total_timesteps=timesteps,
        n_envs=n_envs,
        use_curriculum=True,
        use_wandb=use_wandb,
        wandb_project="hrm-astar-modal",
        wandb_run_name=run_id,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        device="cuda",
        batch_size=512,
        n_steps=8192,
    )
    
    model, predictor = train_predictor(config)
    volume.commit()
    
    return {"run_id": run_id, "checkpoint_dir": checkpoint_dir}


# =============================================================================
# BLAZING FAST: 4x B200 (Blackwell/GB200) Training
# =============================================================================

@app.function(
    image=image,  # Full CUDA toolkit image for Blackwell
    gpu="B200:4",  # 4x B200 Blackwell GPUs (GB200-class)
    memory=256 * 1024,  # 256GB system RAM
    cpu=32,  # More CPU cores for env parallelization
    timeout=60 * 60 * 48,  # 48 hour timeout
    volumes={VOLUME_PATH: volume},
    # secrets=[modal.Secret.from_name("wandb-secret")] if needed
)
def train_4gpu(
    algorithm: str = "PPO",
    timesteps: int = 10_000_000,
    n_envs: int = 128,  # 32 envs per GPU
    batch_size: int = 4096,  # Massive batches for Blackwell
    n_steps: int = 16384,  # Large rollout buffer
    use_wandb: bool = True,
    run_name: str = None,
    use_curriculum: bool = True,
    map_size: int = 32,
    num_obstacles: int = 8,
):
    """
    🚀 BLAZING FAST: 4x B200 Blackwell GPUs (GB200-class)
    
    Optimized for NVIDIA Blackwell architecture:
    - 128 parallel environments (32 per GPU)
    - 4096 batch size (leverages Blackwell's massive memory bandwidth)
    - 16384 steps per update
    - CUDA 12.8 + PyTorch 2.5 for Blackwell optimization
    
    B200 specs: 192GB HBM3e, 8TB/s bandwidth
    
    Estimated: 10M timesteps in ~15-20 minutes
    """
    import os
    import sys
    import subprocess
    import time
    import torch
    from datetime import datetime
    
    sys.path.insert(0, "/app")
    
    # Check GPU availability
    num_gpus = torch.cuda.device_count()
    print(f"🚀 BLACKWELL MODE: {num_gpus}x B200 GPUs detected")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
    
    run_id = run_name or f"modal_4xB200_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = f"{CHECKPOINT_DIR}/{run_id}"
    log_dir = f"{LOG_DIR}/{run_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Scale with GPU count
    scaled_n_envs = n_envs * num_gpus // 4
    scaled_batch = batch_size * num_gpus // 4
    
    print(f"\n📊 Training Configuration:")
    print(f"   Timesteps: {timesteps:,}")
    print(f"   Environments: {scaled_n_envs}")
    print(f"   Batch size: {scaled_batch}")
    print(f"   Steps per update: {n_steps}")
    print(f"   Estimated updates: {timesteps // (scaled_n_envs * n_steps):,}")
    
    start_time = time.time()
    
    # Per Modal docs: run as subprocess for multi-GPU frameworks
    # https://modal.com/docs/guide/gpu#multi-gpu-training
    subprocess.run(
        [
            sys.executable, "-m", "training.train_multi_gpu",
            "--algorithm", algorithm,
            "--timesteps", str(timesteps),
            "--n-envs", str(scaled_n_envs),
            "--batch-size", str(scaled_batch),
            "--n-steps", str(n_steps),
            "--checkpoint-dir", checkpoint_dir,
            "--log-dir", log_dir,
            "--map-size", str(map_size),
            "--num-obstacles", str(num_obstacles),
            "--use-curriculum" if use_curriculum else "--no-curriculum",
            "--use-wandb" if use_wandb else "--no-wandb",
            "--wandb-project", "hrm-astar-blackwell",
            "--wandb-run-name", run_id,
        ],
        check=True,
        cwd="/app",
    )
    
    elapsed = time.time() - start_time
    steps_per_second = timesteps / elapsed
    
    print(f"\n✅ Training complete!")
    print(f"   Total time: {elapsed/3600:.2f} hours ({elapsed/60:.1f} min)")
    print(f"   Steps/second: {steps_per_second:,.0f}")
    print(f"   Checkpoints: {checkpoint_dir}")
    
    volume.commit()
    
    return {
        "run_id": run_id,
        "checkpoint_dir": checkpoint_dir,
        "elapsed_time": elapsed,
        "steps_per_second": steps_per_second,
        "num_gpus": num_gpus,
        "gpu_type": "B200",
    }


@app.function(
    image=image,
    gpu="B200:8",  # 8x B200 Blackwell - ABSOLUTE MAXIMUM
    memory=512 * 1024,  # 512GB system RAM
    cpu=64,
    timeout=60 * 60 * 72,  # 72 hour timeout
    volumes={VOLUME_PATH: volume},
    # secrets=[modal.Secret.from_name("wandb-secret")] if needed
)
def train_8gpu(
    algorithm: str = "PPO",
    timesteps: int = 50_000_000,
    n_envs: int = 256,  # 32 envs per GPU
    batch_size: int = 8192,
    n_steps: int = 32768,
    use_wandb: bool = True,
    run_name: str = None,
):
    """
    🔥 ABSOLUTE MAXIMUM: 8x B200 Blackwell GPUs
    
    For massive training runs (50M+ timesteps).
    Total: 1.5TB+ GPU memory, 64TB/s aggregate bandwidth
    
    Estimated: 50M timesteps in ~30-45 minutes
    """
    import os
    import sys
    import subprocess
    import time
    import torch
    from datetime import datetime
    
    sys.path.insert(0, "/app")
    
    num_gpus = torch.cuda.device_count()
    print(f"🔥 MAXIMUM POWER: {num_gpus}x B200 Blackwell GPUs")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
    
    run_id = run_name or f"modal_8xB200_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = f"{CHECKPOINT_DIR}/{run_id}"
    log_dir = f"{LOG_DIR}/{run_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    start_time = time.time()
    
    # Run as subprocess for multi-GPU
    subprocess.run(
        [
            sys.executable, "-m", "training.train_multi_gpu",
            "--algorithm", algorithm,
            "--timesteps", str(timesteps),
            "--n-envs", str(n_envs),
            "--batch-size", str(batch_size),
            "--n-steps", str(n_steps),
            "--checkpoint-dir", checkpoint_dir,
            "--log-dir", log_dir,
            "--use-curriculum",
            "--use-wandb" if use_wandb else "--no-wandb",
            "--wandb-project", "hrm-astar-8xB200",
            "--wandb-run-name", run_id,
        ],
        check=True,
        cwd="/app",
    )
    
    elapsed = time.time() - start_time
    volume.commit()
    
    return {
        "run_id": run_id,
        "elapsed_time": elapsed,
        "steps_per_second": timesteps / elapsed,
        "gpu_type": "B200",
        "num_gpus": num_gpus,
    }


# =============================================================================
# H100/H200 Alternatives (if B200 unavailable)
# =============================================================================

@app.function(
    image=image,
    gpu="H100:4",  # Fallback to 4x H100
    memory=128 * 1024,
    cpu=32,
    timeout=60 * 60 * 48,
    volumes={VOLUME_PATH: volume},
    # secrets=[modal.Secret.from_name("wandb-secret")] if needed
)
def train_4x_h100(
    algorithm: str = "PPO",
    timesteps: int = 10_000_000,
    n_envs: int = 64,
    batch_size: int = 2048,
    n_steps: int = 16384,
    use_wandb: bool = True,
    run_name: str = None,
):
    """
    4x H100 GPUs - Great alternative if B200 unavailable.
    
    May auto-upgrade to H200 per Modal docs.
    """
    import os
    import sys
    import subprocess
    import time
    from datetime import datetime
    
    sys.path.insert(0, "/app")
    
    run_id = run_name or f"modal_4xH100_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = f"{CHECKPOINT_DIR}/{run_id}"
    log_dir = f"{LOG_DIR}/{run_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    start_time = time.time()
    
    subprocess.run(
        [
            sys.executable, "-m", "training.train_multi_gpu",
            "--algorithm", algorithm,
            "--timesteps", str(timesteps),
            "--n-envs", str(n_envs),
            "--batch-size", str(batch_size),
            "--n-steps", str(n_steps),
            "--checkpoint-dir", checkpoint_dir,
            "--log-dir", log_dir,
            "--use-curriculum",
            "--use-wandb" if use_wandb else "--no-wandb",
            "--wandb-project", "hrm-astar-h100",
            "--wandb-run-name", run_id,
        ],
        check=True,
        cwd="/app",
    )
    
    elapsed = time.time() - start_time
    volume.commit()
    
    return {"run_id": run_id, "elapsed_time": elapsed, "gpu_type": "H100"}


# =============================================================================
# Evaluation Function
# =============================================================================

@app.function(
    image=image,
    gpu="T4",  # Smaller GPU for eval
    timeout=60 * 30,  # 30 min timeout
    volumes={VOLUME_PATH: volume},
)
def evaluate(
    checkpoint_path: str,
    num_episodes: int = 100,
    render_video: bool = False,
):
    """
    Evaluate a trained model.
    
    Args:
        checkpoint_path: Path to model checkpoint (in volume)
        num_episodes: Number of evaluation episodes
        render_video: Whether to render evaluation video
    """
    import os
    import sys
    
    sys.path.insert(0, "/app")
    
    from training.train_predictor import load_trained_model
    from evaluation.metrics import evaluate_model
    from envs.dynamic_pathfinding_env import DynamicPathfindingEnv, EnvConfig
    
    print(f"Loading model from: {checkpoint_path}")
    
    # Construct full path
    if not checkpoint_path.startswith("/"):
        checkpoint_path = f"{CHECKPOINT_DIR}/{checkpoint_path}"
    
    checkpoint_dir = str(Path(checkpoint_path).parent)
    
    # Load model
    model, predictor, config = load_trained_model(checkpoint_dir)
    
    # Create eval environment
    env_config = EnvConfig(
        map_size=config.map_size,
        num_obstacles=config.num_obstacles,
        render_mode="rgb_array" if render_video else None,
    )
    env = DynamicPathfindingEnv(env_config)
    
    # Simple agent wrapper
    class ModelAgent:
        def __init__(self, model):
            self.model = model
        
        def act(self, obs):
            action, _ = self.model.predict(obs, deterministic=True)
            return action
    
    agent = ModelAgent(model)
    
    # Evaluate
    print(f"\nEvaluating over {num_episodes} episodes...")
    results = evaluate_model(
        env=env,
        agent=agent,
        predictor=predictor,
        num_episodes=num_episodes,
        verbose=True,
    )
    
    return results.to_dict()


# =============================================================================
# Checkpoint Management
# =============================================================================

@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
)
def list_checkpoints():
    """List all available checkpoints in the volume."""
    import os
    
    checkpoints = []
    
    if os.path.exists(CHECKPOINT_DIR):
        for run_id in os.listdir(CHECKPOINT_DIR):
            run_dir = os.path.join(CHECKPOINT_DIR, run_id)
            if os.path.isdir(run_dir):
                files = os.listdir(run_dir)
                has_final = "final_model.zip" in files
                checkpoints.append({
                    "run_id": run_id,
                    "path": run_dir,
                    "has_final_model": has_final,
                    "files": files,
                })
    
    return checkpoints


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
)
def download_checkpoint(run_id: str):
    """
    Download a checkpoint from the volume.
    
    Returns the checkpoint files as bytes.
    """
    import os
    import zipfile
    import io
    
    run_dir = f"{CHECKPOINT_DIR}/{run_id}"
    
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Checkpoint not found: {run_id}")
    
    # Create zip archive
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(run_dir):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, CHECKPOINT_DIR)
                zf.write(filepath, arcname)
    
    buffer.seek(0)
    return buffer.getvalue()


# =============================================================================
# Hyperparameter Sweep
# =============================================================================

@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 6,  # 6 hours per run
    volumes={VOLUME_PATH: volume},
    # secrets=[modal.Secret.from_name("wandb-secret")] if needed
)
def train_sweep_run(
    config: dict,
    sweep_id: str,
):
    """Single run in a hyperparameter sweep."""
    import os
    import sys
    
    sys.path.insert(0, "/app")
    
    from training.train_predictor import TrainConfig, train_predictor
    
    run_id = f"sweep_{sweep_id}_{config.get('seed', 0)}"
    checkpoint_dir = f"{CHECKPOINT_DIR}/{run_id}"
    log_dir = f"{LOG_DIR}/{run_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    train_config = TrainConfig(
        **config,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        device="cuda",
    )
    
    model, predictor = train_predictor(train_config)
    volume.commit()
    
    return {"run_id": run_id, "config": config}


@app.local_entrypoint()
def sweep(
    n_runs: int = 5,
    base_timesteps: int = 500_000,
):
    """
    Run a hyperparameter sweep.
    
    Launches multiple training runs with different hyperparameters.
    """
    import itertools
    
    # Define sweep space
    learning_rates = [1e-4, 3e-4, 1e-3]
    batch_sizes = [64, 128, 256]
    n_envs_options = [8, 16]
    
    configs = []
    for i, (lr, bs, ne) in enumerate(itertools.product(
        learning_rates, batch_sizes, n_envs_options
    )):
        if len(configs) >= n_runs:
            break
        configs.append({
            "learning_rate": lr,
            "batch_size": bs,
            "n_envs": ne,
            "total_timesteps": base_timesteps,
            "seed": 42 + i,
            "use_wandb": True,
            "wandb_project": "hrm-astar-sweep",
        })
    
    print(f"Launching {len(configs)} sweep runs...")
    
    # Launch runs in parallel
    results = list(train_sweep_run.starmap([
        (cfg, f"sweep_{i}") for i, cfg in enumerate(configs)
    ]))
    
    print("\nSweep complete!")
    for r in results:
        print(f"  - {r['run_id']}: lr={r['config']['learning_rate']}, bs={r['config']['batch_size']}")
    
    return results


# =============================================================================
# Local CLI Entrypoints
# =============================================================================

@app.local_entrypoint()
def main(
    action: str = "train",
    algorithm: str = "PPO",
    timesteps: int = 1_000_000,
    n_envs: int = 16,
    map_size: int = 32,
    num_obstacles: int = 5,
    use_curriculum: bool = True,
    use_wandb: bool = False,
    run_name: str = None,
    checkpoint_path: str = None,
    num_episodes: int = 100,
):
    """
    Main entrypoint for Modal app.
    
    Actions:
        train: Train a new model
        train_fast: Train with 2x A100 GPUs
        evaluate: Evaluate a trained model
        list: List available checkpoints
        download: Download a checkpoint
    """
    if action == "train":
        result = train.remote(
            algorithm=algorithm,
            timesteps=timesteps,
            n_envs=n_envs,
            map_size=map_size,
            num_obstacles=num_obstacles,
            use_curriculum=use_curriculum,
            use_wandb=use_wandb,
            run_name=run_name,
        )
        print(f"\nTraining result: {result}")
        
    elif action == "train_fast":
        result = train_multi_gpu.remote(
            algorithm=algorithm,
            timesteps=timesteps,
            n_envs=n_envs * 2,
            use_wandb=use_wandb,
            run_name=run_name,
        )
        print(f"\nTraining result: {result}")
        
    elif action == "evaluate":
        if not checkpoint_path:
            print("Error: --checkpoint-path required for evaluate action")
            return
        result = evaluate.remote(
            checkpoint_path=checkpoint_path,
            num_episodes=num_episodes,
        )
        print(f"\nEvaluation result: {result}")
        
    elif action == "list":
        checkpoints = list_checkpoints.remote()
        print("\nAvailable checkpoints:")
        for cp in checkpoints:
            print(f"  - {cp['run_id']}")
            print(f"    Path: {cp['path']}")
            print(f"    Has final model: {cp['has_final_model']}")
            
    elif action == "download":
        if not run_name:
            print("Error: --run-name required for download action")
            return
        data = download_checkpoint.remote(run_name)
        output_path = f"{run_name}_checkpoint.zip"
        with open(output_path, "wb") as f:
            f.write(data)
        print(f"Downloaded checkpoint to: {output_path}")
        
    else:
        print(f"Unknown action: {action}")
        print("Available actions: train, train_fast, evaluate, list, download")

