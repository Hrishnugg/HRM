"""
Training script for HRM-ACT-v1 on Sudoku puzzles.

This script trains the HRM-v2 model on the Sudoku-Extreme dataset.
"""

import os
import sys
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
from tqdm import tqdm

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


class SimplePuzzleDataset(IterableDataset):
    """Simplified puzzle dataset for single-GPU training."""
    
    def __init__(self, dataset_path: str, split: str, batch_size: int, epochs: int = 1):
        super().__init__()
        self.dataset_path = dataset_path
        self.split = split
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Load metadata
        with open(os.path.join(dataset_path, split, "dataset.json"), "r") as f:
            self.metadata = PuzzleDatasetMetadata(**json.load(f))
        
        # Load data
        split_dir = os.path.join(dataset_path, split)
        self.inputs = np.load(os.path.join(split_dir, "all__inputs.npy"), mmap_mode="r")
        self.labels = np.load(os.path.join(split_dir, "all__labels.npy"), mmap_mode="r")
        self.puzzle_ids = np.load(os.path.join(split_dir, "all__puzzle_identifiers.npy"))
        self.puzzle_indices = np.load(os.path.join(split_dir, "all__puzzle_indices.npy"))
        
        print(f"Loaded {split} dataset:")
        print(f"  Total examples: {len(self.inputs)}")
        print(f"  Num puzzles: {self.puzzle_indices.shape[0] - 1}")
        print(f"  Vocab size: {self.metadata.vocab_size}")
    
    def __iter__(self):
        for epoch in range(self.epochs):
            # Shuffle examples
            indices = np.random.permutation(len(self.inputs))
            
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
    """Training configuration."""
    # Data
    data_path: str = "../data/sudoku-extreme-1k-aug-1000"
    
    # Model
    batch_size: int = 16
    seq_len: int = 81  # Sudoku 9x9
    vocab_size: int = 11  # 0-10 (0=blank, 1-9=digits, 10=padding/ignore)
    hidden_size: int = 512
    num_heads: int = 8
    expansion: float = 4.0
    H_cycles: int = 2
    L_cycles: int = 2
    H_layers: int = 4
    L_layers: int = 4
    halt_max_steps: int = 16
    halt_exploration_prob: float = 0.1
    puzzle_emb_ndim: int = 512  # Same as hidden_size
    
    # Training
    epochs: int = 100
    lr: float = 1e-4
    puzzle_emb_lr: float = 1e-2
    weight_decay: float = 0.1
    puzzle_emb_weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 100
    
    # Evaluation
    eval_every: int = 10
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpoint
    checkpoint_dir: str = "checkpoints"


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
    
    if model.training and "target_q_continue" in outputs:
        q_loss = F.binary_cross_entropy_with_logits(
            outputs["q_continue_logits"],
            outputs["target_q_continue"]
        )
        total_loss = total_loss + 0.1 * q_loss
    
    # Compute metrics
    metrics = compute_metrics(carry, outputs, carry.current_data["labels"])
    metrics["lm_loss"] = lm_loss.mean().item()
    
    return carry, total_loss, metrics


def evaluate(model, dataset, device, max_steps: Optional[int] = None):
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
        for batch_idx, batch in enumerate(dataset):
            if max_steps and batch_idx >= max_steps:
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
    
    print("=" * 60)
    print("HRM-v2 Training: Sudoku-Extreme")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.lr}")
    print(f"Epochs: {config.epochs}")
    print()
    
    # Create model
    model_config = {
        "batch_size": config.batch_size,
        "seq_len": config.seq_len,
        "vocab_size": config.vocab_size,
        "num_puzzle_identifiers": 1000,  # Sudoku dataset has 1000 puzzles
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
    print(f"Total parameters: {total_params:,}")
    print()
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = SimplePuzzleDataset(
        config.data_path, "train", config.batch_size, epochs=config.epochs
    )
    eval_dataset = SimplePuzzleDataset(
        config.data_path, "test", config.batch_size, epochs=1
    )
    
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
    
    # Puzzle embedding optimizer (using SignSGD for sparse updates)
    puzzle_emb_optimizer = None
    if hasattr(model.inner, "puzzle_emb") and config.puzzle_emb_ndim > 0:
        # The SignSGD optimizer needs: weights (buffer), local_weights (buffer with grad), local_ids (buffer)
        puzzle_emb_params = [
            model.inner.puzzle_emb.weights,
            model.inner.puzzle_emb.local_weights,
            model.inner.puzzle_emb.local_ids,
        ]
        puzzle_emb_optimizer = CastedSparseEmbeddingSignSGD_Distributed(
            puzzle_emb_params,
            world_size=1,  # Single GPU
            lr=config.puzzle_emb_lr,
            weight_decay=config.puzzle_emb_weight_decay
        )
    
    # Training loop
    print("Starting training...")
    print()
    
    global_step = 0
    running_metrics = {k: 0.0 for k in ["lm_loss", "accuracy", "exact_accuracy"]}
    running_count = 0
    
    # Calculate total steps (approximate)
    total_steps = (len(train_dataset.inputs) // config.batch_size) * config.epochs
    
    try:
        for batch_idx, batch in enumerate(tqdm(train_dataset, desc="Training")):
            # Learning rate schedule
            lr_mult = cosine_schedule(global_step, total_steps, config.warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr * lr_mult
            
            # Initialize carry for new batch (ensure on correct device)
            batch_device = {k: v.to(config.device) for k, v in batch.items()}
            carry = model.initial_carry(batch_device)
            
            # Training step
            optimizer.zero_grad()
            if puzzle_emb_optimizer:
                puzzle_emb_optimizer.zero_grad()
            
            # Run hierarchical reasoning until all sequences halt
            for _ in range(model.config.halt_max_steps):
                carry, loss, metrics = train_step(model, carry, batch, config.device)
                
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
            
            # Optimizer step
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if puzzle_emb_optimizer:
                puzzle_emb_optimizer.step()
            
            global_step += 1
            
            # Evaluation
            if (batch_idx + 1) % config.eval_every == 0:
                # Print training metrics
                if running_count > 0:
                    print(f"\n[Step {global_step}] Training metrics:")
                    for k, v in running_metrics.items():
                        print(f"  {k}: {v / running_count:.4f}")
                    running_metrics = {k: 0.0 for k in running_metrics}
                    running_count = 0
                
                # Run evaluation
                print("Running evaluation...")
                eval_metrics = evaluate(model, eval_dataset, config.device, max_steps=20)
                print("Evaluation metrics:")
                for k, v in eval_metrics.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")
                print()
                
                model.train()
        
        print("\nTraining complete!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Save final checkpoint
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_path = checkpoint_dir / "hrm_sudoku_final.pt"
    torch.save({
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
        "step": global_step,
    }, checkpoint_path)
    
    print(f"\nCheckpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()

