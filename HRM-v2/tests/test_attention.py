"""Tests for attention operations."""

import pytest
import torch
from hrm.ops.attention import attention, sdpa, flash_attention


class TestSDPA:
    """Test PyTorch Scaled Dot Product Attention."""
    
    @pytest.fixture
    def device(self):
        """Get test device (CUDA if available, else CPU)."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def dtype(self, device):
        """Get test dtype (bfloat16 for CUDA, float32 for CPU)."""
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    
    def test_sdpa_basic(self, device, dtype):
        """Test basic SDPA functionality."""
        batch_size, num_heads, seqlen, head_dim = 2, 8, 64, 64
        
        q = torch.randn(batch_size, num_heads, seqlen, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, num_heads, seqlen, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, num_heads, seqlen, head_dim, dtype=dtype, device=device)
        
        out = sdpa(q, k, v)
        
        assert out.shape == q.shape
        assert out.dtype == dtype
        assert out.device == device
    
    def test_sdpa_causal(self, device, dtype):
        """Test SDPA with causal masking."""
        batch_size, num_heads, seqlen, head_dim = 2, 8, 64, 64
        
        q = torch.randn(batch_size, num_heads, seqlen, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, num_heads, seqlen, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, num_heads, seqlen, head_dim, dtype=dtype, device=device)
        
        out = sdpa(q, k, v, is_causal=True)
        
        assert out.shape == q.shape
        assert out.dtype == dtype
        assert out.device == device
    
    def test_sdpa_alternate_layout(self, device, dtype):
        """Test SDPA with (batch, seqlen, num_heads, head_dim) layout."""
        batch_size, seqlen, num_heads, head_dim = 2, 64, 8, 64
        
        q = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
        
        out = sdpa(q, k, v)
        
        assert out.shape == q.shape
        assert out.dtype == dtype
        assert out.device == device


class TestFlashAttention:
    """Test FlashAttention implementation."""
    
    @pytest.fixture
    def device(self):
        """Get test device (CUDA required for FlashAttention)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda")
    
    @pytest.fixture
    def dtype(self):
        """Get test dtype (FlashAttention requires float16/bfloat16)."""
        return torch.bfloat16
    
    def test_flash_attention_available(self):
        """Test if FlashAttention is available."""
        try:
            import flash_attn
            assert hasattr(flash_attn, 'flash_attn_func')
        except ImportError:
            pytest.skip("FlashAttention not installed")
    
    def test_flash_attention_basic(self, device, dtype):
        """Test basic FlashAttention functionality."""
        try:
            import flash_attn
        except ImportError:
            pytest.skip("FlashAttention not installed")
        
        batch_size, seqlen, num_heads, head_dim = 2, 64, 8, 64
        
        q = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
        
        out = flash_attention(q, k, v)
        
        assert out.shape == q.shape
        assert out.dtype == dtype
        assert out.device == device
    
    def test_flash_attention_causal(self, device, dtype):
        """Test FlashAttention with causal masking."""
        try:
            import flash_attn
        except ImportError:
            pytest.skip("FlashAttention not installed")
        
        batch_size, seqlen, num_heads, head_dim = 2, 64, 8, 64
        
        q = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
        
        out = flash_attention(q, k, v, is_causal=True)
        
        assert out.shape == q.shape
        assert out.dtype == dtype
        assert out.device == device
    
    def test_flash_attention_invalid_dtype(self, device):
        """Test that FlashAttention raises error for invalid dtypes."""
        try:
            import flash_attn
        except ImportError:
            pytest.skip("FlashAttention not installed")
        
        batch_size, seqlen, num_heads, head_dim = 2, 64, 8, 64
        
        q = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=torch.float32, device=device)
        k = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=torch.float32, device=device)
        v = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=torch.float32, device=device)
        
        with pytest.raises(ValueError, match="float16 or bfloat16"):
            flash_attention(q, k, v)


class TestUnifiedAttention:
    """Test unified attention interface."""
    
    @pytest.fixture
    def device(self):
        """Get test device (CUDA if available, else CPU)."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def dtype(self, device):
        """Get test dtype (bfloat16 for CUDA, float32 for CPU)."""
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    
    def test_attention_sdpa_path(self, device, dtype):
        """Test attention with SDPA path explicitly."""
        batch_size, seqlen, num_heads, head_dim = 2, 64, 8, 64
        
        q = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
        
        out = attention(q, k, v, use_flash=False)
        
        assert out.shape == q.shape
        assert out.dtype == dtype
        assert out.device == device
    
    def test_attention_flash_path(self, device, dtype):
        """Test attention with FlashAttention path (if available)."""
        if device.type != "cuda":
            pytest.skip("FlashAttention requires CUDA")
        
        batch_size, seqlen, num_heads, head_dim = 2, 64, 8, 64
        
        q = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
        
        # This should not raise even if FlashAttention is not available
        # (should fall back to SDPA)
        out = attention(q, k, v, use_flash=True)
        
        assert out.shape == q.shape
        assert out.dtype == dtype
        assert out.device == device
    
    def test_attention_causal(self, device, dtype):
        """Test attention with causal masking."""
        batch_size, seqlen, num_heads, head_dim = 2, 64, 8, 64
        
        q = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
        
        out = attention(q, k, v, is_causal=True, use_flash=False)
        
        assert out.shape == q.shape
        assert out.dtype == dtype
        assert out.device == device
    
    def test_attention_different_seqlens(self, device, dtype):
        """Test attention with various sequence lengths."""
        batch_size, num_heads, head_dim = 2, 8, 64
        
        for seqlen in [16, 32, 64, 128, 256]:
            q = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
            k = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
            v = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
            
            out = attention(q, k, v, use_flash=False)
            
            assert out.shape == q.shape
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_attention_dtypes(self, device):
        """Test attention with different dtypes on CUDA."""
        batch_size, seqlen, num_heads, head_dim = 2, 64, 8, 64
        
        for dtype in [torch.float16, torch.bfloat16]:
            q = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
            k = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
            v = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
            
            out = attention(q, k, v, use_flash=False)
            
            assert out.shape == q.shape
            assert out.dtype == dtype

