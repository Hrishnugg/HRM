# HRM-v2 Quick Start Guide

## Prerequisites

- **OS**: Native Linux installation
- **GPU**: NVIDIA RTX 5090 (Blackwell, sm_100)
- **CUDA**: 12.8+ toolkit installed
- **Python**: 3.12+
- **Tools**: `uv` (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`)

## Installation (One Command)

```bash
cd HRM-v2
bash scripts/setup_uv.sh
```

This will:
1. Create a Python 3.12 virtual environment
2. Install PyTorch 2.8.x with CUDA 12.8 support
3. Install all project dependencies
4. Build FlashAttention 4 from source (targeting sm_100)

**Note**: The FlashAttention build can take 10-15 minutes.

## Verify Installation

```bash
source .venv/bin/activate
python scripts/verify_gpu.py
```

Expected output:
```
✓ PyTorch version: 2.8.x
✓ CUDA available: True
✓ CUDA version: 12.8
✓ GPU 0: NVIDIA GeForce RTX 5090
  ✓ Blackwell architecture detected (sm_100)
✓ BFloat16 supported
✓ Float16 supported
✓ PyTorch SDPA works
✓ FlashAttention installed: 4.x.x
✓ FlashAttention forward pass successful
```

## Quick Test

```bash
# Run smoke test
bash scripts/smoke_train.sh

# Run unit tests
pytest tests/
```

## Basic Usage

### 1. Simple Attention

```python
import torch
from hrm.ops.attention import attention

# Create sample tensors (Blackwell prefers bfloat16)
device = torch.device("cuda")
q = torch.randn(2, 128, 8, 64, dtype=torch.bfloat16, device=device)
k = torch.randn(2, 128, 8, 64, dtype=torch.bfloat16, device=device)
v = torch.randn(2, 128, 8, 64, dtype=torch.bfloat16, device=device)

# Automatically uses FlashAttention 4 if available, else SDPA
out = attention(q, k, v, use_flash=True, is_causal=False)
```

### 2. Minimal Transformer

```python
from hrm.models import MinimalTransformer

model = MinimalTransformer(
    vocab_size=10000,
    embed_dim=512,
    num_heads=8,
    num_layers=6,
).to("cuda").to(torch.bfloat16)

# Forward pass
input_ids = torch.randint(0, 10000, (2, 64), device="cuda")
logits = model(input_ids)  # Shape: (2, 64, 10000)
```

### 3. HRM-Style Layers

```python
from hrm.models.layers import HRMTransformerBlock
from hrm.ops.rotary import RotaryEmbedding

# Create RoPE
rope = RotaryEmbedding(dim=64, max_position_embeddings=512)

# Create HRM block
block = HRMTransformerBlock(
    hidden_size=512,
    num_heads=8,
    use_flash=True,
).to("cuda").to(torch.bfloat16)

# Forward
x = torch.randn(2, 128, 512, dtype=torch.bfloat16, device="cuda")
cos_sin = rope()
out = block(x, cos_sin=cos_sin)
```

## Project Structure

```
HRM-v2/
├── src/hrm/              # Main package
│   ├── ops/              # Low-level operations (attention, RoPE, norms)
│   ├── models/           # Model architectures (blocks, layers)
│   ├── train/            # Training utilities (placeholder)
│   └── utils/            # Helper functions (env, init)
├── tests/                # Unit tests
├── configs/              # Configuration files
├── scripts/              # Setup and verification scripts
├── pyproject.toml        # Python packaging
├── README.md             # Project overview
├── PORTING_GUIDE.md      # Migration from HRM-v1
└── QUICKSTART.md         # This file
```

## Key Features

### ✅ Unified Attention Interface
- FlashAttention 4 (sm_100 optimized) when available
- Automatic fallback to PyTorch SDPA
- Single API: `attention(q, k, v, use_flash=True)`

### ✅ Modern Architecture
- Python 3.12 with type hints
- Clean module organization
- Comprehensive testing

### ✅ Performance Optimized
- CUDA 12.8+ kernels
- BFloat16 training support
- Blackwell (sm_100) targeted

### ✅ Developer Friendly
- One-command setup
- Detailed error messages
- Extensive documentation

## Common Issues

### FlashAttention build fails
```bash
# Ensure CUDA toolkit is installed
nvcc --version  # Should show 12.8+

# Set environment variables explicitly
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="10.0"
```

### GPU not detected
```bash
# Check driver
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Import errors
```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Reinstall in editable mode
uv pip install -e .
```

## Next Steps

1. **Explore the code**: Browse `src/hrm/` to see the architecture
2. **Run tests**: `pytest tests/ -v` for detailed test output
3. **Port your model**: Follow `PORTING_GUIDE.md` to migrate HRM-v1 components
4. **Add datasets**: Port dataset builders from original HRM
5. **Train**: Implement training loops using the new infrastructure

## Resources

- PyTorch 2.8 Docs: https://pytorch.org/docs/stable/
- FlashAttention 4: https://github.com/Dao-AILab/flash-attention
- CUDA 12.8: https://developer.nvidia.com/cuda-toolkit

## Getting Help

If you encounter issues:
1. Check `scripts/verify_gpu.py` output
2. Review `PORTING_GUIDE.md` for architecture details
3. Run `pytest tests/ -v` to identify failing components
4. Ensure all prerequisites are met (CUDA 12.8+, Python 3.12+)

Happy coding! 🚀

