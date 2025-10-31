# HRM-v2 Installation Checklist

Use this checklist to ensure your environment is properly set up for HRM-v2.

## ☑️ Pre-Installation

- [ ] **Linux OS**: Native Linux installation (not WSL)
- [ ] **GPU**: NVIDIA RTX 5090 (Blackwell) installed
- [ ] **Driver**: NVIDIA driver 550+ installed (`nvidia-smi` works)
- [ ] **CUDA Toolkit**: CUDA 12.8+ installed (`nvcc --version` shows 12.8+)
- [ ] **Python**: Python 3.12+ available (`python3 --version`)
- [ ] **uv**: Package manager installed (`uv --version`)
  - If not: `curl -LsSf https://astral.sh/uv/install.sh | sh`

## ☑️ Installation

```bash
cd HRM-v2
bash scripts/setup_uv.sh
```

Expected timeline:
- Virtual environment creation: ~10 seconds
- PyTorch installation: ~2 minutes
- Project dependencies: ~30 seconds
- FlashAttention 4 build: ~10-15 minutes

Progress indicators:
- [ ] Virtual environment created (`.venv/` folder exists)
- [ ] PyTorch installed (no errors during pip install)
- [ ] Project installed in editable mode
- [ ] FlashAttention building (shows compilation progress)
- [ ] FlashAttention installed (build completes successfully)

## ☑️ Verification

```bash
source .venv/bin/activate
python scripts/verify_gpu.py
```

Expected checks:
- [ ] ✓ PyTorch version: 2.8.x or higher
- [ ] ✓ CUDA available: True
- [ ] ✓ CUDA version: 12.8 or higher
- [ ] ✓ cuDNN version: 90000+ (9.x)
- [ ] ✓ GPU 0: NVIDIA GeForce RTX 5090
- [ ] ✓ Compute capability: 10.0 (sm_100)
- [ ] ✓ Blackwell architecture detected
- [ ] ✓ BFloat16 supported
- [ ] ✓ Float16 supported
- [ ] ✓ PyTorch SDPA works
- [ ] ✓ FlashAttention installed: 4.x.x
- [ ] ✓ FlashAttention forward pass successful

## ☑️ Testing

```bash
# Run smoke test
bash scripts/smoke_train.sh
```

Expected output:
- [ ] Testing attention wrapper...
- [ ] Using device: cuda
- [ ] Using dtype: torch.bfloat16
- [ ] Input shape: torch.Size([2, 128, 8, 64])
- [ ] ✓ SDPA output shape: torch.Size([2, 128, 8, 64])
- [ ] ✓ FlashAttention output shape: torch.Size([2, 128, 8, 64])
- [ ] ✓ Smoke test passed!

```bash
# Run unit tests
pytest tests/ -v
```

Expected results:
- [ ] All tests in `test_attention.py` pass
- [ ] All tests in `test_models.py` pass
- [ ] No failures or errors
- [ ] Total tests: 20+

## ☑️ Quick Functionality Test

Create a test file `test_basic.py`:
```python
import torch
from hrm.ops.attention import attention
from hrm.models import MinimalTransformer

# Test 1: Attention
print("Test 1: Attention")
device = torch.device("cuda")
q = torch.randn(2, 128, 8, 64, dtype=torch.bfloat16, device=device)
k = torch.randn(2, 128, 8, 64, dtype=torch.bfloat16, device=device)
v = torch.randn(2, 128, 8, 64, dtype=torch.bfloat16, device=device)
out = attention(q, k, v, use_flash=True)
print(f"✓ Output shape: {out.shape}")

# Test 2: Model
print("\nTest 2: Minimal Transformer")
model = MinimalTransformer(
    vocab_size=1000,
    embed_dim=256,
    num_heads=8,
    num_layers=4,
).to(device).to(torch.bfloat16)
input_ids = torch.randint(0, 1000, (2, 64), device=device)
logits = model(input_ids)
print(f"✓ Logits shape: {logits.shape}")

print("\n✅ All basic tests passed!")
```

Run it:
```bash
python test_basic.py
```

Expected output:
- [ ] Test 1: Attention ✓
- [ ] Test 2: Minimal Transformer ✓
- [ ] ✅ All basic tests passed!

## ☑️ Environment Info

Capture your environment for reference:
```bash
python -c "
from hrm.utils import print_env_info
print_env_info()
" > environment_info.txt
```

Review `environment_info.txt` and confirm:
- [ ] PyTorch 2.8+
- [ ] CUDA 12.8+
- [ ] RTX 5090 detected
- [ ] sm_100 architecture
- [ ] FlashAttention 4 available

## 🚨 Troubleshooting

### Issue: CUDA not available
```bash
# Check driver
nvidia-smi

# Check PyTorch build
python -c "import torch; print(torch.version.cuda)"

# Reinstall with correct CUDA version
PIP_INDEX_URL=https://download.pytorch.org/whl/cu128 \
  uv pip install --force-reinstall torch torchvision
```

### Issue: FlashAttention build fails
```bash
# Check CUDA toolkit
nvcc --version  # Must be 12.8+

# Set environment variables
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="10.0"
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)

# Retry build
uv pip install -v --no-binary=:all: --no-build-isolation \
  "git+https://github.com/Dao-AILab/flash-attention@5c1627a7a1cda9c32cb9b937a053564e663f81bc"
```

### Issue: Import errors
```bash
# Ensure venv is activated
source .venv/bin/activate

# Reinstall package
uv pip install -e .

# Check installation
python -c "import hrm; print(hrm.__version__)"
```

### Issue: Tests fail
```bash
# Run with verbose output
pytest tests/ -v -s

# Run specific test
pytest tests/test_attention.py::TestSDPA::test_sdpa_basic -v

# Check CUDA memory
nvidia-smi
```

## ✅ Completion

Once all items are checked:
- [ ] ✅ Environment is fully set up
- [ ] ✅ All verification tests pass
- [ ] ✅ Basic functionality works
- [ ] ✅ Ready for development

**Next steps**: See `QUICKSTART.md` for usage examples and `PORTING_GUIDE.md` for migrating additional components.

---

**Estimated total time**: 20-30 minutes (mostly FlashAttention build)  
**Disk space required**: ~5 GB (PyTorch + FlashAttention + dependencies)  
**Memory required**: 16 GB+ RAM recommended for building FlashAttention

