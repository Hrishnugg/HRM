"""
Weight initialization utilities.

Ported from original HRM implementation.
"""

import math
import torch


def trunc_normal_init_(
    tensor: torch.Tensor,
    std: float = 1.0,
    lower: float = -2.0,
    upper: float = 2.0,
) -> torch.Tensor:
    """
    Truncated normal initialization (JAX-style).
    
    This is a mathematically correct truncated normal initialization,
    unlike PyTorch's nn.init.trunc_normal_ which doesn't preserve the
    actual standard deviation.
    
    Based on JAX implementation:
    https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L807-L848
    https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L162-L199
    
    Args:
        tensor: Tensor to initialize
        std: Standard deviation
        lower: Lower truncation bound (in units of std)
        upper: Upper truncation bound (in units of std)
        
    Returns:
        Initialized tensor (in-place)
    """
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * lower ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor


__all__ = ["trunc_normal_init_"]

