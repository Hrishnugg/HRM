#!/bin/bash
set -euo pipefail

echo "=== HRM-v2 Environment Setup ==="
echo "Target: CUDA 12.8+, PyTorch 2.8+, FlashAttention 4 (Blackwell sm_100)"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not installed. Install it with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo "WARNING: nvcc not found. Make sure CUDA 12.8+ toolkit is installed."
    echo "CUDA_HOME will default to /usr/local/cuda"
fi

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
echo "Using CUDA_HOME: $CUDA_HOME"

# Create virtual environment
echo ""
echo "Step 1: Creating Python 3.12 virtual environment..."
uv venv -p 3.12 .venv

# Activate environment
echo "Step 2: Activating environment..."
source .venv/bin/activate

# Install PyTorch 2.8.x with CUDA 12.8
echo ""
echo "Step 3: Installing PyTorch 2.8.x with CUDA 12.8..."
uv pip install --upgrade pip setuptools wheel

# PyTorch 2.8+ with cu128
PIP_INDEX_URL=https://download.pytorch.org/whl/cu128 \
  uv pip install "torch>=2.8.0,<2.9.0" torchvision --extra-index-url https://pypi.org/simple

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install project dependencies
echo ""
echo "Step 4: Installing project dependencies..."
uv pip install -e .

# Install dev dependencies
echo ""
echo "Step 5: Installing dev dependencies..."
uv pip install -e ".[dev]"

# Build FlashAttention 4 from source (pinned commit for sm_100 support)
echo ""
echo "Step 6: Building FlashAttention 4 from source (this may take several minutes)..."
echo "Commit: 5c1627a7a1cda9c32cb9b937a053564e663f81bc"

# Set build flags for Blackwell (sm_100)
export TORCH_CUDA_ARCH_LIST="10.0"
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
export MAX_JOBS=$(nproc)

# Install build dependencies
uv pip install ninja cmake pybind11

# Build FlashAttention 4
uv pip install -v --no-binary=:all: --no-build-isolation \
  "git+https://github.com/Dao-AILab/flash-attention@5c1627a7a1cda9c32cb9b937a053564e663f81bc"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To verify the installation, run:"
echo "  python scripts/verify_gpu.py"
echo ""

