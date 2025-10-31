"""Environment and device utilities."""

import torch
from typing import Tuple


def get_device() -> torch.device:
    """
    Get the best available device.
    
    Returns:
        torch.device: CUDA device if available, else CPU
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    """
    Get the recommended dtype for the given device.
    
    Args:
        device: Target device
        
    Returns:
        torch.dtype: bfloat16 for CUDA, float32 for CPU
    """
    if device.type == "cuda":
        # Check if bfloat16 is supported
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    return torch.float32


def get_device_info() -> dict:
    """
    Get detailed device information.
    
    Returns:
        dict: Device information including name, compute capability, memory, etc.
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if info["cuda_available"]:
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        
        devices = []
        for i in range(info["device_count"]):
            props = torch.cuda.get_device_properties(i)
            devices.append({
                "index": i,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": props.total_memory / 1024**3,
                "multi_processor_count": props.multi_processor_count,
            })
        info["devices"] = devices
    
    return info


def check_flash_attention() -> Tuple[bool, str]:
    """
    Check if FlashAttention is available and get version.
    
    Returns:
        Tuple[bool, str]: (is_available, version_or_error)
    """
    try:
        import flash_attn
        return True, flash_attn.__version__
    except ImportError:
        return False, "Not installed"
    except Exception as e:
        return False, f"Error: {str(e)}"


def print_env_info():
    """Print detailed environment information."""
    print("=" * 60)
    print("Environment Information")
    print("=" * 60)
    
    # PyTorch
    print(f"PyTorch version: {torch.__version__}")
    
    # Device info
    info = get_device_info()
    print(f"CUDA available: {info['cuda_available']}")
    
    if info["cuda_available"]:
        print(f"CUDA version: {info['cuda_version']}")
        print(f"cuDNN version: {info['cudnn_version']}")
        print(f"Device count: {info['device_count']}")
        
        for dev in info["devices"]:
            print(f"\nGPU {dev['index']}: {dev['name']}")
            print(f"  Compute capability: {dev['compute_capability']}")
            print(f"  Total memory: {dev['total_memory_gb']:.2f} GB")
            print(f"  Multi-processors: {dev['multi_processor_count']}")
    
    # FlashAttention
    fa_available, fa_version = check_flash_attention()
    print(f"\nFlashAttention: {'✓' if fa_available else '✗'} {fa_version}")
    
    # Recommended dtype
    device = get_device()
    dtype = get_dtype(device)
    print(f"\nRecommended device: {device}")
    print(f"Recommended dtype: {dtype}")
    
    print("=" * 60)


__all__ = [
    "get_device",
    "get_dtype",
    "get_device_info",
    "check_flash_attention",
    "print_env_info",
]

