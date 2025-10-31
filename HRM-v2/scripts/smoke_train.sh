#!/bin/bash
set -euo pipefail

echo "=== HRM-v2 Smoke Test ==="
echo "Running minimal training to validate GPU and attention kernels"
echo ""

# Check if virtual environment is activated
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Run smoke training
python -c "
import torch
import torch.nn as nn
from hrm.ops.attention import attention

print('Testing attention wrapper...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

if device.type == 'cpu':
    print('WARNING: No CUDA available, running on CPU')
    dtype = torch.float32
else:
    dtype = torch.bfloat16
    print(f'Using dtype: {dtype}')

# Test with small tensors
batch_size, seqlen, num_heads, head_dim = 2, 128, 8, 64

q = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
k = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
v = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)

print(f'Input shape: {q.shape}')

# Test SDPA path
print('Testing SDPA path...')
out_sdpa = attention(q, k, v, use_flash=False, is_causal=False)
print(f'✓ SDPA output shape: {out_sdpa.shape}')

# Test FlashAttention path (if available)
print('Testing FlashAttention path...')
try:
    out_flash = attention(q, k, v, use_flash=True, is_causal=False)
    print(f'✓ FlashAttention output shape: {out_flash.shape}')
except Exception as e:
    print(f'⚠ FlashAttention not available: {e}')
    print('  (This is OK, model will use SDPA)')

print('')
print('✓ Smoke test passed!')
"

echo ""
echo "=== Smoke Test Complete ==="

