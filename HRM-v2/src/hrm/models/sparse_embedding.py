"""
Sparse embedding with custom optimizer for puzzle identifiers.

Ported from original HRM implementation.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Union

from ..utils.init import trunc_normal_init_


class CastedSparseEmbedding(nn.Module):
    """
    Sparse embedding layer for puzzle identifiers with local gradient accumulation.
    
    This embedding layer maintains global weights but accumulates gradients locally
    during training, which is more efficient for sparse lookups.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        batch_size: int,
        init_std: float = 0.0,
        cast_to: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            num_embeddings: Number of unique embeddings (puzzle identifiers)
            embedding_dim: Dimension of each embedding
            batch_size: Batch size (for local weight buffer)
            init_std: Initialization std (0.0 for zero init)
            cast_to: Target dtype for forward pass
        """
        super().__init__()
        self.cast_to = cast_to

        # Global weights (persistent)
        # Truncated normal init (or zero if init_std=0)
        self.register_buffer(
            "weights",
            trunc_normal_init_(
                torch.empty((num_embeddings, embedding_dim)),
                std=init_std
            ),
            persistent=True
        )

        # Local weights and IDs (non-persistent, for gradient accumulation)
        # Local embeddings with gradient enabled - NOT a buffer, just an attribute with requires_grad
        # This needs to be a leaf tensor for the optimizer to work
        self.local_weights = torch.zeros(batch_size, embedding_dim, requires_grad=True)
        
        # Local embedding IDs - can be a buffer since it doesn't need gradients
        self.register_buffer(
            "local_ids",
            torch.zeros(batch_size, dtype=torch.int32),
            persistent=False
        )
    
    def _apply(self, fn):
        """Override _apply to handle local_weights device movement."""
        super()._apply(fn)
        # Move local_weights to the same device/dtype as other tensors
        # Must detach and set requires_grad to keep it as a leaf tensor
        self.local_weights = fn(self.local_weights).detach().requires_grad_(True)
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Embedding indices of shape (batch_size,)
            
        Returns:
            Embeddings of shape (batch_size, embedding_dim)
        """
        if not self.training:
            # Evaluation mode: direct lookup, no gradient
            return self.weights[inputs].to(self.cast_to)
            
        # Training mode: copy to local weights for gradient accumulation
        # Handle variable batch sizes (last batch might be smaller)
        actual_batch_size = inputs.shape[0]
        with torch.no_grad():
            self.local_weights[:actual_batch_size].copy_(self.weights[inputs])
            self.local_ids[:actual_batch_size].copy_(inputs)

        return self.local_weights[:actual_batch_size].to(self.cast_to)


class CastedSparseEmbeddingSignSGD_Distributed(Optimizer):
    """
    SignSGD optimizer for sparse embeddings with distributed training support.
    
    Uses sign of gradient (SignSGD) which is similar to Adam for very sparse
    gradients, with decoupled weight decay.
    """
    
    def __init__(
        self,
        params: ParamsT,
        world_size: int,
        lr: Union[float, torch.Tensor] = 1e-3,
        weight_decay: float = 1e-2,
    ):
        """
        Args:
            params: Parameters to optimize (should be CastedSparseEmbedding params)
            world_size: Number of processes in distributed training
            lr: Learning rate
            weight_decay: Weight decay coefficient
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            world_size=world_size
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore
        """Perform a single optimization step."""
        for group in self.param_groups:
            # Find the sparse embedding components
            local_weights_grad = None
            local_ids = None
            weights = None
            
            assert len(group["params"]) == 3, "Expected 3 params: weights, local_weights, local_ids"
            
            for p in group["params"]:
                if p.requires_grad:
                    local_weights_grad = p.grad
                elif p.ndim == 1:
                    local_ids = p
                elif p.ndim == 2:
                    weights = p
                else:
                    assert False, f"Unexpected parameter shape: {p.shape}"
                
            assert local_weights_grad is not None, "No gradient found"
            assert local_ids is not None, "No local_ids found"
            assert weights is not None, "No weights found"
        
            # Find actual batch size (handle variable batch sizes)
            # Only process non-zero IDs (actual used embeddings)
            actual_batch_size = (local_ids != 0).sum().item() if (local_ids != 0).any() else local_ids.shape[0]
            # Better: find first occurrence where gradient is non-zero
            if local_weights_grad is not None:
                # Use gradient sparsity to determine actual batch size
                has_grad = (local_weights_grad.abs().sum(dim=1) > 0)
                actual_batch_size = has_grad.sum().item()
                if actual_batch_size == 0:
                    actual_batch_size = local_ids.shape[0]
            
            # Apply SignSGD with distributed all-gather (only on actual batch)
            _sparse_emb_signsgd_dist(
                local_weights_grad[:actual_batch_size],
                local_ids[:actual_batch_size],
                weights,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                world_size=group["world_size"]
            )


def _sparse_emb_signsgd_dist(
    local_weights_grad: torch.Tensor,
    local_ids: torch.Tensor,
    weights: torch.Tensor,
    lr: float,
    weight_decay: float,
    world_size: int
) -> None:
    """
    Apply SignSGD update to sparse embeddings with distributed aggregation.
    
    Args:
        local_weights_grad: Local gradients (batch_size, embedding_dim)
        local_ids: Local embedding IDs (batch_size,)
        weights: Global embedding weights (num_embeddings, embedding_dim)
        lr: Learning rate
        weight_decay: Weight decay coefficient
        world_size: Number of distributed processes
    """
    N, D = local_weights_grad.shape
    
    # All-gather gradients and IDs from all processes
    all_weights_grad = local_weights_grad
    all_ids = local_ids

    if world_size > 1:
        all_weights_grad = torch.empty(
            (world_size * N, D),
            dtype=local_weights_grad.dtype,
            device=local_weights_grad.device
        )
        all_ids = torch.empty(
            world_size * N,
            dtype=local_ids.dtype,
            device=local_ids.device
        )
    
        dist.all_gather_into_tensor(all_weights_grad, local_weights_grad)
        dist.all_gather_into_tensor(all_ids, local_ids)

    # Get unique IDs and aggregate gradients
    grad_ids, inv = all_ids.unique(return_inverse=True)

    # Accumulate gradients for each unique ID
    grad = torch.zeros(
        (grad_ids.shape[0], D),
        dtype=all_weights_grad.dtype,
        device=all_weights_grad.device
    )
    grad.scatter_add_(0, inv.unsqueeze(-1).expand(-1, D), all_weights_grad)

    # SignSGD with decoupled weight decay
    # p = p * (1 - lr * wd) - lr * sign(grad)
    p = weights[grad_ids]
    p.mul_(1.0 - lr * weight_decay).add_(torch.sign(grad), alpha=-lr)

    # Write updated embeddings back
    weights[grad_ids] = p


__all__ = [
    "CastedSparseEmbedding",
    "CastedSparseEmbeddingSignSGD_Distributed",
]

