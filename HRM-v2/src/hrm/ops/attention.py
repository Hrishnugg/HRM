"""
Attention operations with SDPA default and optional FlashAttention 4 acceleration.

This module provides a unified interface for attention computation that automatically
falls back to PyTorch's SDPA when FlashAttention is not available or not suitable.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    Scaled Dot Product Attention using PyTorch's native implementation.
    
    Args:
        q: Query tensor of shape (batch, num_heads, seqlen, head_dim) or (batch, seqlen, num_heads, head_dim)
        k: Key tensor of shape (batch, num_heads, seqlen, head_dim) or (batch, seqlen, num_heads, head_dim)
        v: Value tensor of shape (batch, num_heads, seqlen, head_dim) or (batch, seqlen, num_heads, head_dim)
        attn_mask: Optional attention mask
        is_causal: Whether to apply causal masking
        dropout_p: Dropout probability
        
    Returns:
        Attention output of same shape as q
    """
    # PyTorch SDPA expects (batch, num_heads, seqlen, head_dim)
    # If input is (batch, seqlen, num_heads, head_dim), transpose
    needs_transpose = q.dim() == 4 and q.size(1) > q.size(2)
    
    if needs_transpose:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
    
    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal
    )
    
    if needs_transpose:
        out = out.transpose(1, 2)
    
    return out


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    FlashAttention implementation (if available).
    
    Args:
        q: Query tensor of shape (batch, seqlen, num_heads, head_dim)
        k: Key tensor of shape (batch, seqlen, num_heads, head_dim)
        v: Value tensor of shape (batch, seqlen, num_heads, head_dim)
        is_causal: Whether to apply causal masking
        dropout_p: Dropout probability
        
    Returns:
        Attention output of shape (batch, seqlen, num_heads, head_dim)
    """
    try:
        from flash_attn import flash_attn_func
        
        # FlashAttention expects (batch, seqlen, num_heads, head_dim)
        # and requires float16 or bfloat16
        if q.dtype not in [torch.float16, torch.bfloat16]:
            raise ValueError(f"FlashAttention requires float16 or bfloat16, got {q.dtype}")
        
        return flash_attn_func(q, k, v, dropout_p=dropout_p, causal=is_causal)
        
    except ImportError:
        raise ImportError("FlashAttention is not installed. Install with: pip install flash-attn")


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    use_flash: bool = True,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    Unified attention interface that automatically selects the best implementation.
    
    This function attempts to use FlashAttention when available and suitable,
    falling back to PyTorch SDPA otherwise.
    
    Args:
        q: Query tensor of shape (batch, seqlen, num_heads, head_dim)
        k: Key tensor of shape (batch, seqlen, num_heads, head_dim)
        v: Value tensor of shape (batch, seqlen, num_heads, head_dim)
        use_flash: Whether to attempt using FlashAttention (default: True)
        attn_mask: Optional attention mask (only used with SDPA)
        is_causal: Whether to apply causal masking
        dropout_p: Dropout probability
        
    Returns:
        Attention output of shape (batch, seqlen, num_heads, head_dim)
        
    Note:
        - FlashAttention requires float16 or bfloat16 dtypes
        - FlashAttention expects (batch, seqlen, num_heads, head_dim) layout
        - SDPA can handle (batch, num_heads, seqlen, head_dim) layout automatically
    """
    if use_flash:
        # Check if FlashAttention is suitable
        is_suitable_dtype = q.dtype in [torch.float16, torch.bfloat16]
        is_cuda = q.is_cuda
        
        if is_suitable_dtype and is_cuda:
            try:
                return flash_attention(q, k, v, is_causal=is_causal, dropout_p=dropout_p)
            except (ImportError, Exception):
                # Fall through to SDPA
                pass
    
    # Fall back to SDPA
    return sdpa(q, k, v, attn_mask=attn_mask, is_causal=is_causal, dropout_p=dropout_p)


__all__ = ["attention", "sdpa", "flash_attention"]

