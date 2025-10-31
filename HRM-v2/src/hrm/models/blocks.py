"""
Basic transformer blocks using the attention wrapper.

These blocks demonstrate how to use the unified attention interface
in actual model architectures.
"""

import torch
import torch.nn as nn
from typing import Optional

from ..ops.attention import attention


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module using the unified attention interface.
    
    This is a minimal implementation that demonstrates integration with
    the SDPA/FlashAttention wrapper.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        use_flash: bool = True,
    ):
        """
        Args:
            embed_dim: Total embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in projections
            use_flash: Whether to attempt using FlashAttention
        """
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.use_flash = use_flash
        
        # QKV projection
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seqlen, embed_dim)
            attn_mask: Optional attention mask
            is_causal: Whether to use causal masking
            
        Returns:
            Output tensor of shape (batch, seqlen, embed_dim)
        """
        batch_size, seqlen, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seqlen, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        # Apply attention
        out = attention(
            q, k, v,
            use_flash=self.use_flash,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        
        # Reshape and project output
        out = out.reshape(batch_size, seqlen, self.embed_dim)
        out = self.out_proj(out)
        
        return out


class FeedForward(nn.Module):
    """
    Simple feed-forward network with GELU activation.
    """
    
    def __init__(
        self,
        embed_dim: int,
        ff_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        """
        Args:
            embed_dim: Input/output dimension
            ff_dim: Hidden dimension (default: 4 * embed_dim)
            dropout: Dropout probability
            bias: Whether to use bias
        """
        super().__init__()
        
        if ff_dim is None:
            ff_dim = 4 * embed_dim
        
        self.fc1 = nn.Linear(embed_dim, ff_dim, bias=bias)
        self.fc2 = nn.Linear(ff_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seqlen, embed_dim)
            
        Returns:
            Output tensor of shape (batch, seqlen, embed_dim)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Standard transformer block with pre-normalization.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_flash: bool = True,
    ):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
            dropout: Dropout probability
            use_flash: Whether to use FlashAttention
        """
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_flash=use_flash,
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            dropout=dropout,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seqlen, embed_dim)
            attn_mask: Optional attention mask
            is_causal: Whether to use causal masking
            
        Returns:
            Output tensor of shape (batch, seqlen, embed_dim)
        """
        # Pre-norm attention
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)
        
        # Pre-norm feed-forward
        x = x + self.ff(self.norm2(x))
        
        return x


class MinimalTransformer(nn.Module):
    """
    Minimal transformer model for testing and demonstration.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: Optional[int] = None,
        max_seqlen: int = 512,
        dropout: float = 0.1,
        use_flash: bool = True,
    ):
        """
        Args:
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Feed-forward hidden dimension
            max_seqlen: Maximum sequence length
            dropout: Dropout probability
            use_flash: Whether to use FlashAttention
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_seqlen = max_seqlen
        
        # Token and position embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seqlen, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                use_flash=use_flash,
            )
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Tie weights
        self.head.weight = self.token_embed.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Input token IDs of shape (batch, seqlen)
            is_causal: Whether to use causal masking
            
        Returns:
            Logits of shape (batch, seqlen, vocab_size)
        """
        batch_size, seqlen = input_ids.shape
        
        # Embeddings
        token_embeds = self.token_embed(input_ids)
        positions = torch.arange(seqlen, device=input_ids.device).unsqueeze(0)
        pos_embeds = self.pos_embed(positions)
        
        x = self.dropout(token_embeds + pos_embeds)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, is_causal=is_causal)
        
        # Output projection
        x = self.norm(x)
        logits = self.head(x)
        
        return logits


__all__ = [
    "MultiHeadAttention",
    "FeedForward",
    "TransformerBlock",
    "MinimalTransformer",
]

