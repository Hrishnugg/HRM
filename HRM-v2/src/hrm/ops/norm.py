"""
Normalization operations.

Ported from original HRM implementation.
"""

import torch
import torch.nn as nn


def rms_norm(
    hidden_states: torch.Tensor,
    variance_epsilon: float = 1e-5,
) -> torch.Tensor:
    """
    RMS (Root Mean Square) Normalization.
    
    Args:
        hidden_states: Input tensor
        variance_epsilon: Small constant for numerical stability
        
    Returns:
        Normalized tensor
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    
    return hidden_states.to(input_dtype)


class RMSNorm(nn.Module):
    """
    RMS Normalization module.
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        """
        Args:
            hidden_size: Hidden dimension size
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Input tensor
            
        Returns:
            Normalized tensor
        """
        normalized = rms_norm(hidden_states, self.eps)
        return normalized * self.weight


__all__ = ["rms_norm", "RMSNorm"]

