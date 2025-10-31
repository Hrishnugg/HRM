# HRM-v2 Porting Guide

This document describes how the original HRM codebase has been modernized and restructured in HRM-v2.

## Architecture Changes

### Attention System

**Original (HRM-v1):**
- Direct import from `flash_attn` package
- No fallback mechanism
- Hardcoded FlashAttention 2/3 compatibility checks

**New (HRM-v2):**
- Unified attention wrapper in `src/hrm/ops/attention.py`
- Automatic fallback: FlashAttention 4 → PyTorch SDPA
- Clean API: `attention(q, k, v, use_flash=True, is_causal=False)`
- Compatible with both sm_90 (H100) and sm_100 (Blackwell/RTX 5090)

### Module Organization

```
Original:                   HRM-v2:
models/                     src/hrm/
├── common.py              ├── ops/           # Low-level operations
├── layers.py              │   ├── attention.py
├── losses.py              │   ├── rotary.py
├── sparse_embedding.py    │   └── norm.py
└── hrm/                   ├── models/        # Model architectures
    └── hrm_act_v1.py      │   ├── blocks.py
                           │   └── layers.py  # HRM-specific layers
                           ├── train/         # Training utilities
                           └── utils/         # Helpers
                               ├── env.py
                               └── init.py
```

## Ported Components

### ✅ Core Operations

| Component | Original | HRM-v2 Location | Status |
|-----------|----------|-----------------|--------|
| RoPE | `models/layers.py` | `src/hrm/ops/rotary.py` | ✅ Complete |
| RMS Norm | `models/layers.py` | `src/hrm/ops/norm.py` | ✅ Complete |
| Attention | `models/layers.py` | `src/hrm/ops/attention.py` | ✅ Modernized |
| Truncated Normal Init | `models/common.py` | `src/hrm/utils/init.py` | ✅ Complete |

### ✅ Layer Primitives

| Component | Original | HRM-v2 Location | Status |
|-----------|----------|-----------------|--------|
| CastedLinear | `models/layers.py` | `src/hrm/models/layers.py` | ✅ Complete |
| CastedEmbedding | `models/layers.py` | `src/hrm/models/layers.py` | ✅ Complete |
| SwiGLU | `models/layers.py` | `src/hrm/models/layers.py` | ✅ Complete |
| AttentionWithRoPE | `models/layers.py` | `src/hrm/models/layers.py` | ✅ Modernized |
| HRMTransformerBlock | `models/layers.py` | `src/hrm/models/layers.py` | ✅ Complete |

### 📋 To Be Ported (As Needed)

| Component | Original | Status | Priority |
|-----------|----------|--------|----------|
| HRM-ACT-v1 Model | `models/hrm/hrm_act_v1.py` | Pending | Medium |
| Sparse Embedding | `models/sparse_embedding.py` | Pending | Low |
| Custom Losses | `models/losses.py` | Pending | Medium |
| Dataset Builders | `dataset/` | Pending | High |

## Key Improvements

### 1. Modern Python Packaging
- `pyproject.toml` with proper dependencies
- Editable install: `uv pip install -e .`
- Clean namespace: `from hrm.ops import attention`

### 2. FlashAttention 4 Support
- Targets Blackwell (sm_100) architecture
- Pinned to commit: `5c1627a7a1cda9c32cb9b937a053564e663f81bc`
- Build flags: `TORCH_CUDA_ARCH_LIST=10.0`

### 3. Environment Setup
- One-command setup: `bash scripts/setup_uv.sh`
- GPU verification: `python scripts/verify_gpu.py`
- Smoke tests: `bash scripts/smoke_train.sh`

### 4. Testing Infrastructure
- Comprehensive unit tests in `tests/`
- Pytest-based: `pytest tests/`
- GPU/CPU compatibility checks

### 5. Type Safety & Documentation
- Type hints throughout
- Docstrings for all public APIs
- Clear separation of concerns

## Migration Path

If you want to use the full HRM-ACT-v1 model:

1. **Port the main model** (when needed):
   ```python
   # Create src/hrm/models/hrm_act_v1.py
   # Adapt HierarchicalReasoningModel_ACTV1 to use new layers
   ```

2. **Port sparse embeddings** (if using puzzle embeddings):
   ```python
   # Create src/hrm/models/sparse_embedding.py
   # Update imports to use new base classes
   ```

3. **Port datasets**:
   ```python
   # Adapt dataset/ builders to work with new structure
   # Update paths and imports
   ```

## Dependencies

### Required
- Python 3.12+
- PyTorch 2.8+ (CUDA 12.8)
- NumPy, einops, PyYAML, tqdm, rich

### Optional
- FlashAttention 4 (Linux only; built from source)
- pytest, ruff (development)

## Build Instructions

### Standard Installation
```bash
cd HRM-v2
bash scripts/setup_uv.sh
```

### Manual FlashAttention 4 Build
```bash
export TORCH_CUDA_ARCH_LIST="10.0"  # Blackwell
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
uv pip install -v --no-binary=:all: --no-build-isolation \
  "git+https://github.com/Dao-AILab/flash-attention@5c1627a7a1cda9c32cb9b937a053564e663f81bc"
```

## Verification

After installation:
```bash
source .venv/bin/activate
python scripts/verify_gpu.py  # Check GPU and FlashAttention
pytest tests/                 # Run all tests
bash scripts/smoke_train.sh   # Quick end-to-end test
```

## Next Steps

The current HRM-v2 provides a clean foundation with:
- ✅ Modern attention infrastructure
- ✅ Core HRM layers (RoPE, RMS norm, SwiGLU, etc.)
- ✅ Unified FlashAttention 4 / SDPA interface
- ✅ Testing and verification tools

**To complete the port**, you can now:
1. Add the full HRM-ACT-v1 model when needed
2. Port dataset builders for your specific tasks
3. Implement training loops and evaluation scripts
4. Add task-specific components as required

The architecture is designed to be **extensible and minimal** - port only what you need!

