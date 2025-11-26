"""
Optimized training script for HRM-ACT-v1 on Maze puzzles.

Features:
- Large batch sizes for full GPU utilization
- Multi-worker data loading for CPU efficiency
- W&B integration for experiment tracking
- Live training visualizations
- Gradient accumulation for even larger effective batches
"""

import os
import sys
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
from tqdm import tqdm
import wandb

# Add parent directory to path to import original utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset.common import PuzzleDatasetMetadata

# Add src to path for HRM-v2 imports
sys.path.insert(0, str(Path(__file__).parent / "src"))
from hrm.models import HRMACTv1, CastedSparseEmbeddingSignSGD_Distributed


IGNORE_LABEL_ID = -100


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100):
    """Stablemax cross entropy loss (numerically stable alternative to softmax)."""
    def s(x, epsilon=1e-30):
        return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)
    
    def log_stablemax(x, dim=-1):
        s_x = s(x)
        return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))
    
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
    valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(
        logprobs, 
        index=transformed_labels.to(torch.long).unsqueeze(-1), 
        dim=-1
    ).squeeze(-1)
    
    return -torch.where(valid_mask, prediction_logprobs, 0)


class OptimizedPuzzleDataset(IterableDataset):
    """Optimized puzzle dataset with multi-worker support."""
    
    def __init__(self, dataset_path: str, split: str, batch_size: int, epochs: int = 1, seed: int = 42):
        super().__init__()
        self.dataset_path = dataset_path
        self.split = split
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        
        # Load metadata
        with open(os.path.join(dataset_path, split, "dataset.json"), "r") as f:
            self.metadata = PuzzleDatasetMetadata(**json.load(f))
        
        # Get data paths (will be loaded per-worker)
        self.split_dir = os.path.join(dataset_path, split)
        self.data_files = {
            "inputs": os.path.join(self.split_dir, "all__inputs.npy"),
            "labels": os.path.join(self.split_dir, "all__labels.npy"),
            "puzzle_ids": os.path.join(self.split_dir, "all__puzzle_identifiers.npy"),
            "puzzle_indices": os.path.join(self.split_dir, "all__puzzle_indices.npy"),
        }
        
        # Load metadata only (not full data yet)
        self.num_examples = None
        print(f"Initialized {split} dataset:")
        print(f"  Dataset path: {dataset_path}")
        print(f"  Vocab size: {self.metadata.vocab_size}")
    
    def _load_data(self, worker_id: int):
        """Load data in worker process."""
        self.inputs = np.load(self.data_files["inputs"], mmap_mode="r")
        self.labels = np.load(self.data_files["labels"], mmap_mode="r")
        self.puzzle_ids = np.load(self.data_files["puzzle_ids"])
        self.num_examples = len(self.inputs)
    
    def __iter__(self):
        # Load data in worker
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        self._load_data(worker_id)
        
        for epoch in range(self.epochs):
            # Shuffle with per-epoch seed
            rng = np.random.default_rng(self.seed + epoch)
            indices = rng.permutation(self.num_examples)
            
            # Shard across workers
            if worker_info:
                indices = indices[worker_id::worker_info.num_workers]
            
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                # Get batch data
                batch = {
                    "inputs": torch.from_numpy(self.inputs[batch_indices].astype(np.int32)),
                    "labels": torch.from_numpy(self.labels[batch_indices].astype(np.int32)),
                    "puzzle_identifiers": torch.from_numpy(self.puzzle_ids[batch_indices].astype(np.int32)),
                }
                
                # Handle ignore labels
                if self.metadata.ignore_label_id is not None:
                    batch["labels"] = torch.where(
                        batch["labels"] == self.metadata.ignore_label_id,
                        IGNORE_LABEL_ID,
                        batch["labels"]
                    )
                
                yield batch


@dataclass
class TrainConfig:
    """Optimized training configuration for RTX 5090."""
    # Data
    data_path: str = "../data/maze-30x30-hard-1k"
    
    # Model - Maze 30x30
    batch_size: int = 32  # Optimized for 30x30 mazes (900 seq_len is large!)
    seq_len: int = 900  # 30x30 maze
    vocab_size: int = 6  # Maze charset: "# SGo" + padding/special tokens
    hidden_size: int = 512
    num_heads: int = 8
    expansion: float = 4.0
    H_cycles: int = 2
    L_cycles: int = 2
    H_layers: int = 4
    L_layers: int = 4
    halt_max_steps: int = 16
    halt_exploration_prob: float = 0.1
    puzzle_emb_ndim: int = 0  # Disabled for maze (only 1 puzzle type)
    
    # Training
    epochs: int = 100
    lr: float = 1e-4
    puzzle_emb_lr: float = 1e-4
    weight_decay: float = 1.0
    puzzle_emb_weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 500
    grad_clip: float = 1.0
    
    # Data loading (multi-worker)
    num_workers: int = 8  # 8 workers for 32 virtual cores
    prefetch_factor: int = 4  # Prefetch 4 batches per worker
    
    # Evaluation
    eval_every: int = 200  # Evaluate every 200 steps
    eval_batches: int = 50  # Number of eval batches
    
    # W&B
    use_wandb: bool = False  # Disabled for now
    wandb_project: str = "hrm-v2-maze"
    wandb_run_name: Optional[str] = None
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpoint
    checkpoint_dir: str = "checkpoints/maze"
    save_every: int = 1000  # Save checkpoint every 1000 steps


def cosine_schedule(step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float = 0.1) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def compute_metrics(carry, outputs, labels):
    """Compute training metrics."""
    mask = labels != IGNORE_LABEL_ID
    loss_counts = mask.sum(-1)
    
    preds = torch.argmax(outputs["logits"], dim=-1)
    is_correct = mask & (preds == labels)
    seq_is_correct = is_correct.sum(-1) == loss_counts
    
    valid_metrics = carry.halted & (loss_counts > 0)
    
    metrics = {
        "count": valid_metrics.sum().item(),
        "accuracy": torch.where(valid_metrics, 
                                (is_correct.float().sum(-1) / loss_counts.clamp_min(1)), 
                                torch.tensor(0.0, device=is_correct.device)).sum().item(),
        "exact_accuracy": (valid_metrics & seq_is_correct).sum().item(),
        "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum().item(),
        "steps": torch.where(valid_metrics, carry.steps.float(), torch.tensor(0.0, device=carry.steps.device)).sum().item(),
    }
    
    return metrics


def train_step(model, carry, batch, device):
    """Single training step."""
    # Move batch to device
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # Forward pass
    carry, outputs = model(carry, batch)
    
    # Compute loss
    lm_loss = stablemax_cross_entropy(
        outputs["logits"], 
        carry.current_data["labels"],
        ignore_index=IGNORE_LABEL_ID
    )
    
    # Q-learning loss (if training)
    total_loss = lm_loss.mean()
    q_loss_val = 0.0
    
    if model.training and "target_q_continue" in outputs:
        q_loss = F.binary_cross_entropy_with_logits(
            outputs["q_continue_logits"],
            outputs["target_q_continue"]
        )
        total_loss = total_loss + 0.1 * q_loss
        q_loss_val = q_loss.item()
    
    # Compute metrics
    metrics = compute_metrics(carry, outputs, carry.current_data["labels"])
    metrics["lm_loss"] = lm_loss.mean().item()
    metrics["q_loss"] = q_loss_val
    
    return carry, total_loss, metrics


def evaluate(model, dataloader, device, max_batches: Optional[int] = None):
    """Evaluate model on dataset."""
    model.eval()
    
    all_metrics = {
        "count": 0,
        "accuracy": 0.0,
        "exact_accuracy": 0,
        "q_halt_accuracy": 0,
        "steps": 0.0,
        "lm_loss": 0.0,
    }
    
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            
            # Move batch to device and initialize carry
            batch = {k: v.to(device) for k, v in batch.items()}
            carry = model.initial_carry(batch)
            
            # Run multiple steps until all sequences halt
            for _ in range(model.config.halt_max_steps):
                carry, _, metrics = train_step(model, carry, batch, device)
                
                # Accumulate metrics
                for k in all_metrics:
                    all_metrics[k] += metrics.get(k, 0)
                
                # Check if all halted
                if carry.halted.all():
                    break
            
            num_batches += 1
    
    # Average metrics
    if all_metrics["count"] > 0:
        all_metrics["accuracy"] /= all_metrics["count"]
        all_metrics["exact_accuracy"] /= all_metrics["count"]
        all_metrics["q_halt_accuracy"] /= all_metrics["count"]
        all_metrics["steps"] /= all_metrics["count"]
    
    if num_batches > 0:
        all_metrics["lm_loss"] /= num_batches
    
    return all_metrics


def main():
    """Main training loop."""
    config = TrainConfig()
    
    # Initialize W&B
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or f"maze-bs{config.batch_size}-lr{config.lr}",
            config=vars(config),
        )
    
    print("=" * 60)
    print("HRM-v2 Optimized Training: Maze 30x30")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size} (optimized for RTX 5090)")
    print(f"Data workers: {config.num_workers}")
    print(f"Learning rate: {config.lr}")
    print(f"Epochs: {config.epochs}")
    print(f"W&B tracking: {config.use_wandb}")
    print()
    
    # Create model
    model_config = {
        "batch_size": config.batch_size,
        "seq_len": config.seq_len,
        "vocab_size": config.vocab_size,
        "num_puzzle_identifiers": 1,  # Maze dataset has only 1 puzzle type
        "hidden_size": config.hidden_size,
        "num_heads": config.num_heads,
        "expansion": config.expansion,
        "H_cycles": config.H_cycles,
        "L_cycles": config.L_cycles,
        "H_layers": config.H_layers,
        "L_layers": config.L_layers,
        "halt_max_steps": config.halt_max_steps,
        "halt_exploration_prob": config.halt_exploration_prob,
        "puzzle_emb_ndim": config.puzzle_emb_ndim,
        "pos_encodings": "rope",
        "forward_dtype": "bfloat16",
    }
    
    print("Creating model...")
    model = HRMACTv1(model_config).to(config.device)
    model.train()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Create datasets with multi-worker support
    print("Setting up data loaders...")
    train_dataset = OptimizedPuzzleDataset(
        config.data_path, "train", config.batch_size, epochs=config.epochs
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,  # Dataset already returns batches
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=True if config.num_workers > 0 else False,
        pin_memory=True,
    )
    
    eval_dataset = OptimizedPuzzleDataset(
        config.data_path, "test", config.batch_size, epochs=1
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=None,
        num_workers=2,  # Fewer workers for eval
        pin_memory=True,
    )
    
    print(f"✅ Train loader: {config.num_workers} workers, prefetch={config.prefetch_factor}")
    print(f"✅ Eval loader: 2 workers")
    print()
    
    # Setup optimizers
    main_params = [
        p for n, p in model.named_parameters() 
        if not n.startswith("inner.puzzle_emb")
    ]
    
    optimizer = torch.optim.AdamW(
        main_params,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay
    )
    
    # Puzzle embedding optimizer (SignSGD)
    puzzle_emb_optimizer = None
    if hasattr(model.inner, "puzzle_emb") and config.puzzle_emb_ndim > 0:
        puzzle_emb_params = [
            model.inner.puzzle_emb.weights,
            model.inner.puzzle_emb.local_weights,
            model.inner.puzzle_emb.local_ids,
        ]
        puzzle_emb_optimizer = CastedSparseEmbeddingSignSGD_Distributed(
            puzzle_emb_params,
            world_size=1,
            lr=config.puzzle_emb_lr,
            weight_decay=config.puzzle_emb_weight_decay
        )
        print("✅ Sparse embedding optimizer enabled (SignSGD)")
    
    # Calculate total steps
    # Note: Approximate since dataset is iterable
    total_steps = 10000  # Estimate for LR schedule
    
    # Training loop
    print("Starting training...")
    print()
    
    global_step = 0
    running_metrics = {
        "lm_loss": 0.0,
        "q_loss": 0.0,
        "accuracy": 0.0,
        "exact_accuracy": 0.0,
        "steps_taken": 0.0,
    }
    running_count = 0
    
    try:
        pbar = tqdm(train_loader, desc="Training", dynamic_ncols=True)
        
        for batch_idx, batch in enumerate(pbar):
            # Learning rate schedule
            lr_mult = cosine_schedule(global_step, total_steps, config.warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr * lr_mult
            
            # Initialize carry for new batch (ensure on correct device)
            batch = {k: v.to(config.device) for k, v in batch.items()}
            carry = model.initial_carry(batch)
            
            # Training step
            optimizer.zero_grad()
            if puzzle_emb_optimizer:
                puzzle_emb_optimizer.zero_grad()
            
            # Run hierarchical reasoning until all sequences halt
            step_losses = []
            for step_idx in range(model.config.halt_max_steps):
                carry, loss, metrics = train_step(model, carry, batch, config.device)
                step_losses.append(loss.item())
                
                # Accumulate metrics
                if metrics["count"] > 0:
                    for k in running_metrics:
                        running_metrics[k] += metrics.get(k, 0) * metrics["count"]
                    running_count += metrics["count"]
                
                # Check if all sequences halted
                if carry.halted.all():
                    break
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            # Optimizer step
            optimizer.step()
            if puzzle_emb_optimizer:
                puzzle_emb_optimizer.step()
            
            global_step += 1
            
            # Update progress bar
            if running_count > 0:
                pbar.set_postfix({
                    "loss": f"{running_metrics['lm_loss'] / running_count:.3f}",
                    "acc": f"{running_metrics['accuracy'] / running_count:.3f}",
                    "exact": f"{running_metrics['exact_accuracy'] / running_count:.3f}",
                })
            
            # Evaluation and logging
            if (batch_idx + 1) % config.eval_every == 0:
                # Compute training metrics
                train_metrics = {}
                if running_count > 0:
                    for k, v in running_metrics.items():
                        train_metrics[f"train/{k}"] = v / running_count
                    running_metrics = {k: 0.0 for k in running_metrics}
                    running_count = 0
                
                # Run evaluation
                print("\n" + "=" * 60)
                print(f"Evaluation at step {global_step}")
                print("=" * 60)
                
                eval_metrics_raw = evaluate(model, eval_loader, config.device, max_batches=config.eval_batches)
                eval_metrics = {f"eval/{k}": v for k, v in eval_metrics_raw.items()}
                
                # Print metrics
                print("Training metrics:")
                for k, v in train_metrics.items():
                    print(f"  {k}: {v:.4f}")
                
                print("\nEvaluation metrics:")
                for k, v in eval_metrics.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")
                print()
                
                # Log to W&B
                if config.use_wandb:
                    wandb.log({
                        **train_metrics,
                        **eval_metrics,
                        "step": global_step,
                        "lr": optimizer.param_groups[0]["lr"],
                        "puzzle_emb_lr": config.puzzle_emb_lr if puzzle_emb_optimizer else 0,
                    }, step=global_step)
                
                model.train()
            
            # Save checkpoint
            if (global_step % config.save_every) == 0:
                checkpoint_dir = Path(config.checkpoint_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                checkpoint_path = checkpoint_dir / f"checkpoint_step_{global_step}.pt"
                torch.save({
                    "model_config": model_config,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": global_step,
                }, checkpoint_path)
                
                print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        print("\n✅ Training complete!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    
    finally:
        # Save final checkpoint
        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / "checkpoint_final.pt"
        torch.save({
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": global_step,
        }, checkpoint_path)
        
        print(f"\n💾 Final checkpoint saved: {checkpoint_path}")
        
        if config.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()

