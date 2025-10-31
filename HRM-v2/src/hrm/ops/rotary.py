"""
Rotary Position Embeddings (RoPE).

Ported from original HRM implementation and updated for modern PyTorch.
"""

import torch
import torch.nn as nn
from typing import Tuple


CosSin = Tuple[torch.Tensor, torch.Tensor]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dims of the input.
    
    Args:
        x: Input tensor
        
    Returns:
        Rotated tensor
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.
    
    Args:
        q: Query tensor of shape (bs, seq_len, num_heads, head_dim)
        k: Key tensor of shape (bs, seq_len, num_heads, head_dim)
        cos: Cosine tensor of shape (seq_len, head_dim)
        sin: Sine tensor of shape (seq_len, head_dim)
        
    Returns:
        Tuple of (rotated query, rotated key)
    """
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding module.
    
    Generates precomputed cosine and sine values for RoPE.
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: torch.device = None,
    ):
        """
        Args:
            dim: Dimension of the embeddings (typically head_dim)
            max_position_embeddings: Maximum sequence length
            base: Base for the frequency computation
            device: Device to place tensors on
        """
        super().__init__()

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Register as buffers (non-trainable)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self) -> CosSin:
        """
        Returns:
            Tuple of (cos, sin) tensors
        """
        return self.cos_cached, self.sin_cached


__all__ = [
    "CosSin",
    "rotate_half",
    "apply_rotary_pos_emb",
    "RotaryEmbedding",
]

