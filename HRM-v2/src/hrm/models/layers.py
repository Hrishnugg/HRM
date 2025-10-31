"""
Core layers for HRM models.

Ported and modernized from original HRM implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..ops.attention import attention
from ..ops.rotary import CosSin, apply_rotary_pos_emb
from ..ops.norm import rms_norm
from ..utils.init import trunc_normal_init_


def _find_multiple(a: int, b: int) -> int:
    """Find the smallest multiple of b that is >= a."""
    return (-(a // -b)) * b


class CastedLinear(nn.Module):
    """
    Linear layer with automatic dtype casting and truncated normal initialization.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Whether to include bias
        """
        super().__init__()
        
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(
                torch.empty((out_features, in_features)),
                std=1.0 / (in_features ** 0.5)
            )
        )
        
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features,)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Input tensor
            
        Returns:
            Output tensor with automatic dtype casting
        """
        weight = self.weight.to(input.dtype)
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return F.linear(input, weight, bias=bias)


class CastedEmbedding(nn.Module):
    """
    Embedding layer with automatic dtype casting and custom initialization.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        init_std: float = 0.02,
        cast_to: Optional[torch.dtype] = None,
    ):
        """
        Args:
            num_embeddings: Vocabulary size
            embedding_dim: Embedding dimension
            init_std: Standard deviation for truncated normal initialization
            cast_to: Target dtype (if None, inferred from input)
        """
        super().__init__()
        self.cast_to = cast_to

        # Truncated normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(
                torch.empty((num_embeddings, embedding_dim)),
                std=init_std
            )
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Input indices
            
        Returns:
            Embedded tensor
        """
        target_dtype = self.cast_to if self.cast_to is not None else torch.get_default_dtype()
        return F.embedding(input, self.embedding_weight.to(target_dtype))


class AttentionWithRoPE(nn.Module):
    """
    Multi-head attention with optional RoPE and FlashAttention support.
    
    This is the HRM-style attention layer that integrates with the unified
    attention wrapper.
    """
    
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_key_value_heads: Optional[int] = None,
        causal: bool = False,
        use_flash: bool = True,
    ):
        """
        Args:
            hidden_size: Hidden dimension
            head_dim: Dimension per attention head
            num_heads: Number of query heads
            num_key_value_heads: Number of key/value heads (for GQA)
            causal: Whether to use causal masking
            use_flash: Whether to attempt using FlashAttention
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads or num_heads
        self.causal = causal
        self.use_flash = use_flash

        # QKV projection
        self.qkv_proj = CastedLinear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=False
        )
        
        # Output projection
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: Optional[CosSin] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            cos_sin: Optional RoPE (cos, sin) tensors
            
        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(hidden_states)

        # Split heads
        qkv = qkv.view(
            batch_size, seq_len,
            self.num_heads + 2 * self.num_key_value_heads,
            self.head_dim
        )
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # Apply RoPE if provided
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Attention
        attn_output = attention(
            query, key, value,
            use_flash=self.use_flash,
            is_causal=self.causal,
        )

        # Reshape and project output
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    """
    SwiGLU activation function (Swish-Gated Linear Unit).
    
    Combines SiLU (Swish) activation with gating mechanism.
    """
    
    def __init__(self, hidden_size: int, expansion: float = 2.6667):
        """
        Args:
            hidden_size: Hidden dimension
            expansion: Expansion factor for intermediate dimension
        """
        super().__init__()
        
        # Round to nearest multiple of 256 for efficiency
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class HRMTransformerBlock(nn.Module):
    """
    HRM-style transformer block with post-normalization.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expansion: float = 2.6667,
        rms_norm_eps: float = 1e-5,
        causal: bool = False,
        use_flash: bool = True,
    ):
        """
        Args:
            hidden_size: Hidden dimension
            num_heads: Number of attention heads
            expansion: FFN expansion factor
            rms_norm_eps: RMS norm epsilon
            causal: Whether attention is causal
            use_flash: Whether to use FlashAttention
        """
        super().__init__()

        head_dim = hidden_size // num_heads
        
        self.self_attn = AttentionWithRoPE(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=causal,
            use_flash=use_flash,
        )
        
        self.mlp = SwiGLU(
            hidden_size=hidden_size,
            expansion=expansion,
        )
        
        self.norm_eps = rms_norm_eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: Optional[CosSin] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Input tensor
            cos_sin: Optional RoPE (cos, sin) tensors
            
        Returns:
            Output tensor
        """
        # Post-normalization (as in original HRM)
        # Self attention
        hidden_states = rms_norm(
            hidden_states + self.self_attn(hidden_states, cos_sin=cos_sin),
            variance_epsilon=self.norm_eps
        )
        
        # Feed-forward
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.norm_eps
        )
        
        return hidden_states


__all__ = [
    "CastedLinear",
    "CastedEmbedding",
    "AttentionWithRoPE",
    "SwiGLU",
    "HRMTransformerBlock",
]

