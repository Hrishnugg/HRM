"""Model architectures"""

from .blocks import (
    MultiHeadAttention,
    FeedForward,
    TransformerBlock,
    MinimalTransformer,
)
from .layers import (
    CastedLinear,
    CastedEmbedding,
    AttentionWithRoPE,
    SwiGLU,
    HRMTransformerBlock,
)
from .sparse_embedding import (
    CastedSparseEmbedding,
    CastedSparseEmbeddingSignSGD_Distributed,
)
from .hrm_act_v1 import (
    HRMACTv1,
    HRMACTv1Config,
    HRMACTv1Carry,
    HRMACTv1InnerCarry,
)

__all__ = [
    # Standard blocks
    "MultiHeadAttention",
    "FeedForward",
    "TransformerBlock",
    "MinimalTransformer",
    # HRM layers
    "CastedLinear",
    "CastedEmbedding",
    "AttentionWithRoPE",
    "SwiGLU",
    "HRMTransformerBlock",
    # Sparse embeddings
    "CastedSparseEmbedding",
    "CastedSparseEmbeddingSignSGD_Distributed",
    # HRM-ACT-v1
    "HRMACTv1",
    "HRMACTv1Config",
    "HRMACTv1Carry",
    "HRMACTv1InnerCarry",
]
