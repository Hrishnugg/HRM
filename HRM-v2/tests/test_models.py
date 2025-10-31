"""Tests for model architectures."""

import pytest
import torch
from hrm.models import MultiHeadAttention, FeedForward, TransformerBlock, MinimalTransformer


class TestMultiHeadAttention:
    """Test MultiHeadAttention module."""
    
    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def dtype(self, device):
        """Get test dtype."""
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    
    def test_mha_forward(self, device, dtype):
        """Test MultiHeadAttention forward pass."""
        batch_size, seqlen, embed_dim = 2, 64, 256
        num_heads = 8
        
        mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads).to(device).to(dtype)
        x = torch.randn(batch_size, seqlen, embed_dim, dtype=dtype, device=device)
        
        out = mha(x)
        
        assert out.shape == (batch_size, seqlen, embed_dim)
        assert out.dtype == dtype
        assert out.device == device
    
    def test_mha_causal(self, device, dtype):
        """Test MultiHeadAttention with causal masking."""
        batch_size, seqlen, embed_dim = 2, 64, 256
        num_heads = 8
        
        mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads).to(device).to(dtype)
        x = torch.randn(batch_size, seqlen, embed_dim, dtype=dtype, device=device)
        
        out = mha(x, is_causal=True)
        
        assert out.shape == (batch_size, seqlen, embed_dim)


class TestFeedForward:
    """Test FeedForward module."""
    
    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def dtype(self, device):
        """Get test dtype."""
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    
    def test_ff_forward(self, device, dtype):
        """Test FeedForward forward pass."""
        batch_size, seqlen, embed_dim = 2, 64, 256
        
        ff = FeedForward(embed_dim=embed_dim).to(device).to(dtype)
        x = torch.randn(batch_size, seqlen, embed_dim, dtype=dtype, device=device)
        
        out = ff(x)
        
        assert out.shape == (batch_size, seqlen, embed_dim)
        assert out.dtype == dtype
        assert out.device == device
    
    def test_ff_custom_ff_dim(self, device, dtype):
        """Test FeedForward with custom hidden dimension."""
        batch_size, seqlen, embed_dim = 2, 64, 256
        ff_dim = 512
        
        ff = FeedForward(embed_dim=embed_dim, ff_dim=ff_dim).to(device).to(dtype)
        x = torch.randn(batch_size, seqlen, embed_dim, dtype=dtype, device=device)
        
        out = ff(x)
        
        assert out.shape == (batch_size, seqlen, embed_dim)


class TestTransformerBlock:
    """Test TransformerBlock module."""
    
    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def dtype(self, device):
        """Get test dtype."""
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    
    def test_block_forward(self, device, dtype):
        """Test TransformerBlock forward pass."""
        batch_size, seqlen, embed_dim = 2, 64, 256
        num_heads = 8
        
        block = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads).to(device).to(dtype)
        x = torch.randn(batch_size, seqlen, embed_dim, dtype=dtype, device=device)
        
        out = block(x)
        
        assert out.shape == (batch_size, seqlen, embed_dim)
        assert out.dtype == dtype
        assert out.device == device
    
    def test_block_causal(self, device, dtype):
        """Test TransformerBlock with causal masking."""
        batch_size, seqlen, embed_dim = 2, 64, 256
        num_heads = 8
        
        block = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads).to(device).to(dtype)
        x = torch.randn(batch_size, seqlen, embed_dim, dtype=dtype, device=device)
        
        out = block(x, is_causal=True)
        
        assert out.shape == (batch_size, seqlen, embed_dim)


class TestMinimalTransformer:
    """Test MinimalTransformer model."""
    
    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def dtype(self, device):
        """Get test dtype."""
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    
    def test_model_forward(self, device, dtype):
        """Test MinimalTransformer forward pass."""
        vocab_size = 1000
        batch_size, seqlen = 2, 64
        embed_dim = 256
        num_heads = 8
        num_layers = 4
        
        model = MinimalTransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        ).to(device).to(dtype)
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seqlen), device=device)
        
        logits = model(input_ids)
        
        assert logits.shape == (batch_size, seqlen, vocab_size)
        assert logits.device == device
    
    def test_model_different_seqlens(self, device, dtype):
        """Test MinimalTransformer with various sequence lengths."""
        vocab_size = 1000
        batch_size = 2
        embed_dim = 256
        num_heads = 8
        num_layers = 2
        
        model = MinimalTransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        ).to(device).to(dtype)
        
        for seqlen in [16, 32, 64, 128]:
            input_ids = torch.randint(0, vocab_size, (batch_size, seqlen), device=device)
            logits = model(input_ids)
            assert logits.shape == (batch_size, seqlen, vocab_size)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_inference_mode(self, device):
        """Test MinimalTransformer in inference mode."""
        vocab_size = 1000
        batch_size, seqlen = 2, 64
        embed_dim = 256
        num_heads = 8
        num_layers = 2
        
        model = MinimalTransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        ).to(device).to(torch.bfloat16)
        
        model.eval()
        
        with torch.no_grad():
            input_ids = torch.randint(0, vocab_size, (batch_size, seqlen), device=device)
            logits = model(input_ids)
            
            assert logits.shape == (batch_size, seqlen, vocab_size)

