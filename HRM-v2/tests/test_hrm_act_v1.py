"""Tests for HRM-ACT-v1 model."""

import pytest
import torch
from hrm.models import (
    HRMACTv1,
    HRMACTv1Config,
    CastedSparseEmbedding,
)


class TestCastedSparseEmbedding:
    """Test sparse embedding layer."""
    
    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def dtype(self, device):
        """Get test dtype."""
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    
    def test_sparse_embedding_forward(self, device, dtype):
        """Test sparse embedding forward pass."""
        num_embeddings = 100
        embedding_dim = 256
        batch_size = 4
        
        emb = CastedSparseEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            init_std=0.02,
            cast_to=dtype
        ).to(device)
        
        # Test evaluation mode
        emb.eval()
        indices = torch.randint(0, num_embeddings, (batch_size,), device=device)
        out = emb(indices)
        
        assert out.shape == (batch_size, embedding_dim)
        assert out.dtype == dtype
        assert out.device == device
    
    def test_sparse_embedding_training(self, device, dtype):
        """Test sparse embedding in training mode."""
        num_embeddings = 100
        embedding_dim = 256
        batch_size = 4
        
        emb = CastedSparseEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            init_std=0.02,
            cast_to=dtype
        ).to(device)
        
        # Test training mode
        emb.train()
        indices = torch.randint(0, num_embeddings, (batch_size,), device=device)
        out = emb(indices)
        
        assert out.shape == (batch_size, embedding_dim)
        assert out.requires_grad  # Local weights have gradient
        assert out.dtype == dtype


class TestHRMACTv1Config:
    """Test HRM-ACT-v1 configuration."""
    
    def test_config_creation(self):
        """Test creating configuration."""
        config_dict = {
            "batch_size": 8,
            "seq_len": 128,
            "num_puzzle_identifiers": 100,
            "vocab_size": 1000,
            "H_cycles": 3,
            "L_cycles": 2,
            "H_layers": 2,
            "L_layers": 2,
            "hidden_size": 256,
            "num_heads": 8,
            "pos_encodings": "rope",
            "halt_max_steps": 5,
            "halt_exploration_prob": 0.1,
        }
        
        config = HRMACTv1Config(**config_dict)
        
        assert config.batch_size == 8
        assert config.seq_len == 128
        assert config.hidden_size == 256
        assert config.H_cycles == 3
        assert config.L_cycles == 2


class TestHRMACTv1Model:
    """Test HRM-ACT-v1 model."""
    
    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def dtype(self, device):
        """Get test dtype."""
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    
    @pytest.fixture
    def config_dict(self):
        """Get test configuration."""
        return {
            "batch_size": 4,
            "seq_len": 64,
            "puzzle_emb_ndim": 0,  # Disable puzzle embeddings for simplicity
            "num_puzzle_identifiers": 10,
            "vocab_size": 1000,
            "H_cycles": 2,  # Reduced for testing
            "L_cycles": 1,
            "H_layers": 2,
            "L_layers": 2,
            "hidden_size": 256,
            "num_heads": 8,
            "pos_encodings": "rope",
            "halt_max_steps": 3,
            "halt_exploration_prob": 0.1,
            "forward_dtype": "bfloat16",
        }
    
    def test_model_creation(self, config_dict, device, dtype):
        """Test creating HRM-ACT-v1 model."""
        model = HRMACTv1(config_dict).to(device)
        
        # Check model was created
        assert model.inner is not None
        assert model.config.hidden_size == 256
    
    def test_initial_carry(self, config_dict, device, dtype):
        """Test creating initial carry state."""
        model = HRMACTv1(config_dict).to(device)
        
        batch = {
            "inputs": torch.randint(0, 1000, (4, 64), device=device),
            "puzzle_identifiers": torch.randint(0, 10, (4,), device=device),
        }
        
        carry = model.initial_carry(batch)
        
        assert carry.steps.shape == (4,)
        assert carry.halted.shape == (4,)
        assert carry.halted.all()  # All start halted
        assert carry.steps.sum() == 0  # All start at 0 steps
    
    def test_forward_pass(self, config_dict, device, dtype):
        """Test forward pass through HRM-ACT-v1."""
        if device.type == "cpu":
            config_dict["forward_dtype"] = "float32"
        
        model = HRMACTv1(config_dict).to(device)
        
        batch = {
            "inputs": torch.randint(0, 1000, (4, 64), device=device),
            "puzzle_identifiers": torch.randint(0, 10, (4,), device=device),
        }
        
        carry = model.initial_carry(batch)
        new_carry, outputs = model(carry, batch)
        
        # Check outputs
        assert "logits" in outputs
        assert "q_halt_logits" in outputs
        assert "q_continue_logits" in outputs
        
        # Check shapes
        assert outputs["logits"].shape == (4, 64, 1000)  # (batch, seq_len, vocab_size)
        assert outputs["q_halt_logits"].shape == (4,)
        assert outputs["q_continue_logits"].shape == (4,)
        
        # Check carry updated
        assert new_carry.steps.sum() > 0  # Some steps taken
    
    def test_multi_step_forward(self, config_dict, device, dtype):
        """Test multiple forward passes (simulation)."""
        if device.type == "cpu":
            config_dict["forward_dtype"] = "float32"
        
        model = HRMACTv1(config_dict).to(device)
        model.eval()  # Evaluation mode for consistent behavior
        
        batch = {
            "inputs": torch.randint(0, 1000, (4, 64), device=device),
            "puzzle_identifiers": torch.randint(0, 10, (4,), device=device),
        }
        
        carry = model.initial_carry(batch)
        
        # Run until all sequences halt
        max_iters = 10
        for i in range(max_iters):
            carry, outputs = model(carry, batch)
            
            if carry.halted.all():
                print(f"All sequences halted at iteration {i+1}")
                break
        
        # Check all sequences eventually halted
        assert carry.halted.all()
        assert (carry.steps <= config_dict["halt_max_steps"]).all()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_training_mode(self, config_dict, device):
        """Test forward pass in training mode."""
        model = HRMACTv1(config_dict).to(device)
        model.train()
        
        batch = {
            "inputs": torch.randint(0, 1000, (4, 64), device=device),
            "puzzle_identifiers": torch.randint(0, 10, (4,), device=device),
        }
        
        carry = model.initial_carry(batch)
        new_carry, outputs = model(carry, batch)
        
        # In training mode, should have target Q values
        if config_dict["halt_max_steps"] > 1:
            assert "target_q_continue" in outputs
            assert outputs["target_q_continue"].shape == (4,)
    
    def test_with_puzzle_embeddings(self, config_dict, device, dtype):
        """Test model with puzzle embeddings enabled."""
        if device.type == "cpu":
            config_dict["forward_dtype"] = "float32"
        
        # Enable puzzle embeddings
        config_dict["puzzle_emb_ndim"] = 128
        
        model = HRMACTv1(config_dict).to(device)
        
        batch = {
            "inputs": torch.randint(0, 1000, (4, 64), device=device),
            "puzzle_identifiers": torch.randint(0, 10, (4,), device=device),
        }
        
        carry = model.initial_carry(batch)
        new_carry, outputs = model(carry, batch)
        
        # Should still work with puzzle embeddings
        assert outputs["logits"].shape == (4, 64, 1000)
    
    def test_learned_positions(self, config_dict, device, dtype):
        """Test model with learned position embeddings."""
        if device.type == "cpu":
            config_dict["forward_dtype"] = "float32"
        
        config_dict["pos_encodings"] = "learned"
        
        model = HRMACTv1(config_dict).to(device)
        
        batch = {
            "inputs": torch.randint(0, 1000, (4, 64), device=device),
            "puzzle_identifiers": torch.randint(0, 10, (4,), device=device),
        }
        
        carry = model.initial_carry(batch)
        new_carry, outputs = model(carry, batch)
        
        assert outputs["logits"].shape == (4, 64, 1000)


class TestHRMACTv1Integration:
    """Integration tests for complete HRM-ACT-v1 workflow."""
    
    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gradient_flow(self, device):
        """Test that gradients flow correctly."""
        config_dict = {
            "batch_size": 2,
            "seq_len": 32,
            "puzzle_emb_ndim": 0,
            "num_puzzle_identifiers": 10,
            "vocab_size": 500,
            "H_cycles": 1,
            "L_cycles": 1,
            "H_layers": 1,
            "L_layers": 1,
            "hidden_size": 128,
            "num_heads": 4,
            "pos_encodings": "rope",
            "halt_max_steps": 2,
            "halt_exploration_prob": 0.0,
            "forward_dtype": "bfloat16",
        }
        
        model = HRMACTv1(config_dict).to(device)
        model.train()
        
        batch = {
            "inputs": torch.randint(0, 500, (2, 32), device=device),
            "puzzle_identifiers": torch.randint(0, 10, (2,), device=device),
        }
        
        carry = model.initial_carry(batch)
        new_carry, outputs = model(carry, batch)
        
        # Compute simple loss
        logits = outputs["logits"]
        targets = torch.randint(0, 500, (2, 32), device=device)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 500),
            targets.reshape(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "No gradients found in model"

