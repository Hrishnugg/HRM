#!/usr/bin/env python3
"""
Convenience script for running HRM-Augmented A* training on Modal.

This script provides an easy CLI for common Modal operations.

Setup:
    1. Install Modal: pip install modal
    2. Authenticate: modal token new
    3. (Optional) Set up W&B secret: modal secret create wandb-secret WANDB_API_KEY=<your_key>

Usage Examples:
    # Quick training run (1M steps, single A100)
    python modal_run.py train
    
    # Long training with more timesteps
    python modal_run.py train --timesteps 5000000
    
    # Fast training with 2x A100 GPUs
    python modal_run.py train-fast --timesteps 10000000
    
    # Train with W&B logging
    python modal_run.py train --wandb
    
    # List all checkpoints
    python modal_run.py list
    
    # Evaluate a checkpoint
    python modal_run.py evaluate <run_id>
    
    # Download checkpoint locally
    python modal_run.py download <run_id>
    
    # Run hyperparameter sweep
    python modal_run.py sweep --n-runs 10
"""

import subprocess
import sys
import argparse


def run_modal_command(cmd: list):
    """Run a modal command and stream output."""
    full_cmd = ["modal"] + cmd
    print(f"Running: {' '.join(full_cmd)}")
    print("-" * 60)
    
    process = subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding='utf-8',
        errors='replace',  # Replace undecodable chars instead of crashing
    )
    
    for line in process.stdout:
        print(line, end="")
    
    process.wait()
    return process.returncode


def cmd_train(args):
    """Run training on Modal."""
    cmd = [
        "run", "modal_app.py::main",
        "--action", "train",
        "--algorithm", args.algorithm,
        "--timesteps", str(args.timesteps),
        "--n-envs", str(args.n_envs),
        "--map-size", str(args.map_size),
        "--num-obstacles", str(args.num_obstacles),
    ]
    
    if args.use_curriculum:
        cmd.extend(["--use-curriculum", "True"])
    else:
        cmd.extend(["--use-curriculum", "False"])
    
    if args.wandb:
        cmd.extend(["--use-wandb", "True"])
    
    if args.name:
        cmd.extend(["--run-name", args.name])
    
    return run_modal_command(cmd)


def cmd_train_fast(args):
    """Run fast training with 2x A100 GPUs."""
    cmd = [
        "run", "modal_app.py::train_multi_gpu",
        "--algorithm", args.algorithm,
        "--timesteps", str(args.timesteps),
        "--n-envs", str(args.n_envs),
    ]
    
    if args.wandb:
        cmd.extend(["--use-wandb", "True"])
    
    if args.name:
        cmd.extend(["--run-name", args.name])
    
    return run_modal_command(cmd)


def cmd_train_4gpu(args):
    """Run BLAZING FAST training with 4x B200 Blackwell GPUs."""
    cmd = [
        "run", "modal_app.py::train_4gpu",
        "--algorithm", args.algorithm,
        "--timesteps", str(args.timesteps),
        "--n-envs", str(args.n_envs),
        "--batch-size", str(args.batch_size),
    ]
    
    if args.wandb:
        cmd.extend(["--use-wandb", "True"])
    
    if args.name:
        cmd.extend(["--run-name", args.name])
    
    if not args.use_curriculum:
        cmd.extend(["--use-curriculum", "False"])
    
    return run_modal_command(cmd)


def cmd_train_8gpu(args):
    """Run MAXIMUM POWER training with 8x B200 Blackwell GPUs."""
    cmd = [
        "run", "modal_app.py::train_8gpu",
        "--algorithm", args.algorithm,
        "--timesteps", str(args.timesteps),
        "--n-envs", str(args.n_envs),
        "--batch-size", str(args.batch_size),
    ]
    
    if args.wandb:
        cmd.extend(["--use-wandb", "True"])
    
    if args.name:
        cmd.extend(["--run-name", args.name])
    
    return run_modal_command(cmd)


def cmd_evaluate(args):
    """Evaluate a trained model."""
    cmd = [
        "run", "modal_app.py::main",
        "--action", "evaluate",
        "--checkpoint-path", args.run_id,
        "--num-episodes", str(args.episodes),
    ]
    
    return run_modal_command(cmd)


def cmd_list(args):
    """List available checkpoints."""
    cmd = ["run", "modal_app.py::main", "--action", "list"]
    return run_modal_command(cmd)


def cmd_download(args):
    """Download a checkpoint."""
    cmd = [
        "run", "modal_app.py::main",
        "--action", "download",
        "--run-name", args.run_id,
    ]
    return run_modal_command(cmd)


def cmd_sweep(args):
    """Run hyperparameter sweep."""
    cmd = [
        "run", "modal_app.py::sweep",
        "--n-runs", str(args.n_runs),
        "--base-timesteps", str(args.timesteps),
    ]
    return run_modal_command(cmd)


def cmd_deploy(args):
    """Deploy the app for scheduled/triggered runs."""
    cmd = ["deploy", "modal_app.py"]
    return run_modal_command(cmd)


def cmd_serve(args):
    """Serve the app for interactive development."""
    cmd = ["serve", "modal_app.py"]
    return run_modal_command(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Run HRM-Augmented A* training on Modal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument("--algorithm", "-a", default="PPO", choices=["PPO", "SAC"])
    train_parser.add_argument("--timesteps", "-t", type=int, default=1_000_000)
    train_parser.add_argument("--n-envs", "-e", type=int, default=16)
    train_parser.add_argument("--map-size", type=int, default=32)
    train_parser.add_argument("--num-obstacles", type=int, default=5)
    train_parser.add_argument("--no-curriculum", dest="use_curriculum", action="store_false")
    train_parser.add_argument("--wandb", "-w", action="store_true", help="Enable W&B logging")
    train_parser.add_argument("--name", "-n", type=str, help="Custom run name")
    train_parser.set_defaults(func=cmd_train)
    
    # Train fast command (2x A100)
    fast_parser = subparsers.add_parser("train-fast", help="Train with 2x A100 GPUs")
    fast_parser.add_argument("--algorithm", "-a", default="PPO", choices=["PPO", "SAC"])
    fast_parser.add_argument("--timesteps", "-t", type=int, default=5_000_000)
    fast_parser.add_argument("--n-envs", "-e", type=int, default=32)
    fast_parser.add_argument("--wandb", "-w", action="store_true")
    fast_parser.add_argument("--name", "-n", type=str)
    fast_parser.set_defaults(func=cmd_train_fast)
    
    # 4x B200 Blackwell (GB200-class) - BLAZING FAST
    b200_4_parser = subparsers.add_parser(
        "train-4gpu",
        help="🚀 BLAZING: 4x B200 Blackwell GPUs (GB200-class)"
    )
    b200_4_parser.add_argument("--algorithm", "-a", default="PPO", choices=["PPO", "SAC"])
    b200_4_parser.add_argument("--timesteps", "-t", type=int, default=10_000_000)
    b200_4_parser.add_argument("--n-envs", "-e", type=int, default=128)
    b200_4_parser.add_argument("--batch-size", "-b", type=int, default=4096)
    b200_4_parser.add_argument("--wandb", "-w", action="store_true")
    b200_4_parser.add_argument("--name", "-n", type=str)
    b200_4_parser.add_argument("--no-curriculum", dest="use_curriculum", action="store_false")
    b200_4_parser.set_defaults(func=cmd_train_4gpu, use_curriculum=True)
    
    # 8x B200 Blackwell - MAXIMUM POWER
    b200_8_parser = subparsers.add_parser(
        "train-8gpu",
        help="🔥 MAXIMUM: 8x B200 Blackwell GPUs"
    )
    b200_8_parser.add_argument("--algorithm", "-a", default="PPO", choices=["PPO", "SAC"])
    b200_8_parser.add_argument("--timesteps", "-t", type=int, default=50_000_000)
    b200_8_parser.add_argument("--n-envs", "-e", type=int, default=256)
    b200_8_parser.add_argument("--batch-size", "-b", type=int, default=8192)
    b200_8_parser.add_argument("--wandb", "-w", action="store_true")
    b200_8_parser.add_argument("--name", "-n", type=str)
    b200_8_parser.set_defaults(func=cmd_train_8gpu)
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("run_id", type=str, help="Run ID or checkpoint path")
    eval_parser.add_argument("--episodes", "-e", type=int, default=100)
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available checkpoints")
    list_parser.set_defaults(func=cmd_list)
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download a checkpoint")
    download_parser.add_argument("run_id", type=str, help="Run ID to download")
    download_parser.set_defaults(func=cmd_download)
    
    # Sweep command
    sweep_parser = subparsers.add_parser("sweep", help="Run hyperparameter sweep")
    sweep_parser.add_argument("--n-runs", "-n", type=int, default=5)
    sweep_parser.add_argument("--timesteps", "-t", type=int, default=500_000)
    sweep_parser.set_defaults(func=cmd_sweep)
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy app to Modal")
    deploy_parser.set_defaults(func=cmd_deploy)
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Serve app for development")
    serve_parser.set_defaults(func=cmd_serve)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

