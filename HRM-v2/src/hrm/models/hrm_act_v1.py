"""
HRM-ACT-v1: Hierarchical Reasoning Model with Adaptive Computation Time.

This is a complete port of the original HRM-ACT-v1 model, updated to use
the modern infrastructure (FlashAttention 4, unified attention API, etc.).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from pydantic import BaseModel

from .layers import CastedLinear, CastedEmbedding, HRMTransformerBlock
from .sparse_embedding import CastedSparseEmbedding
from ..ops.rotary import RotaryEmbedding, CosSin
from ..utils.init import trunc_normal_init_


@dataclass
class HRMACTv1InnerCarry:
    """Inner state for HRM-ACT-v1 (hierarchical levels)."""
    z_H: torch.Tensor  # High-level state
    z_L: torch.Tensor  # Low-level state


@dataclass
class HRMACTv1Carry:
    """Complete carry state for HRM-ACT-v1 including ACT (Adaptive Computation Time) state."""
    inner_carry: HRMACTv1InnerCarry
    steps: torch.Tensor  # Number of steps taken
    halted: torch.Tensor  # Whether sequence has halted
    current_data: Dict[str, torch.Tensor]  # Current input data


class HRMACTv1Config(BaseModel):
    """Configuration for HRM-ACT-v1 model."""
    # Data dimensions
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0  # 0 to disable puzzle embeddings
    num_puzzle_identifiers: int
    vocab_size: int

    # Hierarchical cycles
    H_cycles: int  # High-level reasoning cycles
    L_cycles: int  # Low-level reasoning cycles per high cycle

    # Layer configuration
    H_layers: int  # Number of high-level transformer layers
    L_layers: int  # Number of low-level transformer layers

    # Transformer architecture
    hidden_size: int
    expansion: float = 4.0  # FFN expansion factor (matches original HRM)
    num_heads: int
    pos_encodings: str = "rope"  # "rope" or "learned"

    # Normalization and RoPE
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Adaptive Computation Time (ACT) - Halting Q-learning
    halt_max_steps: int
    halt_exploration_prob: float

    # Precision
    forward_dtype: str = "bfloat16"


class HRMACTv1ReasoningModule(nn.Module):
    """
    Reasoning module consisting of multiple transformer blocks.
    
    Applies input injection (addition) before processing through layers.
    """
    
    def __init__(self, layers: List[HRMTransformerBlock]):
        """
        Args:
            layers: List of transformer blocks
        """
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Current state
            input_injection: Input to inject (add)
            **kwargs: Additional arguments (e.g., cos_sin for RoPE)
            
        Returns:
            Updated hidden states
        """
        # Input injection (additive)
        hidden_states = hidden_states + input_injection
        
        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


class HRMACTv1_Inner(nn.Module):
    """
    Inner model for HRM-ACT-v1 (without ACT wrapper).
    
    This implements the core hierarchical reasoning with:
    - Token and position embeddings
    - Optional puzzle embeddings
    - Hierarchical high/low-level reasoning
    - Language modeling head
    - Q-value head for halting
    """
    
    def __init__(self, config: HRMACTv1Config):
        """
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # Input/Output embeddings
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype
        )
        
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)  # Q(halt), Q(continue)

        # Puzzle embeddings (optional)
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        
        if self.config.puzzle_emb_ndim > 0:
            # Zero-initialized puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0.0,  # Zero init
                cast_to=self.forward_dtype
            )

        # Position encodings
        total_seq_len = self.config.seq_len + self.puzzle_emb_len
        
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=total_seq_len,
                base=self.config.rope_theta
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                total_seq_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype
            )
        else:
            raise ValueError(f"Unknown pos_encodings: {self.config.pos_encodings}")

        # Hierarchical reasoning modules
        self.H_level = HRMACTv1ReasoningModule(
            layers=[
                HRMTransformerBlock(
                    hidden_size=self.config.hidden_size,
                    num_heads=self.config.num_heads,
                    expansion=self.config.expansion,
                    rms_norm_eps=self.config.rms_norm_eps,
                    causal=False,  # Non-causal attention
                    use_flash=True,
                )
                for _ in range(self.config.H_layers)
            ]
        )
        
        self.L_level = HRMACTv1ReasoningModule(
            layers=[
                HRMTransformerBlock(
                    hidden_size=self.config.hidden_size,
                    num_heads=self.config.num_heads,
                    expansion=self.config.expansion,
                    rms_norm_eps=self.config.rms_norm_eps,
                    causal=False,
                    use_flash=True,
                )
                for _ in range(self.config.L_layers)
            ]
        )
        
        # Initial states for H and L levels
        self.register_buffer(
            "H_init",
            trunc_normal_init_(
                torch.empty(self.config.hidden_size, dtype=self.forward_dtype),
                std=1.0
            ),
            persistent=True
        )
        
        self.register_buffer(
            "L_init",
            trunc_normal_init_(
                torch.empty(self.config.hidden_size, dtype=self.forward_dtype),
                std=1.0
            ),
            persistent=True
        )

        # Q-head special initialization
        # Initialize Q-values to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)  # type: ignore

    def _input_embeddings(
        self,
        input: torch.Tensor,
        puzzle_identifiers: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute input embeddings (tokens + puzzle + position).
        
        Args:
            input: Token IDs (batch_size, seq_len)
            puzzle_identifiers: Puzzle IDs (batch_size,)
            
        Returns:
            Embeddings (batch_size, seq_len + puzzle_emb_len, hidden_size)
        """
        # Token embeddings
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings (if enabled)
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            # Pad to multiple of hidden_size
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            # Reshape and prepend to token embeddings
            puzzle_embedding = puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size)
            embedding = torch.cat((puzzle_embedding, embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # Scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (
                embedding + self.embed_pos.embedding_weight.to(self.forward_dtype)
            )

        # Scale embeddings
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int) -> HRMACTv1InnerCarry:
        """Create empty carry state."""
        seq_len = self.config.seq_len + self.puzzle_emb_len
        
        # Use device of H_init buffer
        device = self.H_init.device
        
        return HRMACTv1InnerCarry(
            z_H=torch.empty(
                batch_size, seq_len, self.config.hidden_size,
                dtype=self.forward_dtype,
                device=device
            ),
            z_L=torch.empty(
                batch_size, seq_len, self.config.hidden_size,
                dtype=self.forward_dtype,
                device=device
            ),
        )
        
    def reset_carry(
        self,
        reset_flag: torch.Tensor,
        carry: HRMACTv1InnerCarry
    ) -> HRMACTv1InnerCarry:
        """
        Reset carry state for sequences that have halted.
        
        Args:
            reset_flag: Boolean tensor (batch_size,)
            carry: Current carry state
            
        Returns:
            Updated carry state
        """
        return HRMACTv1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(
        self,
        carry: HRMACTv1InnerCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[HRMACTv1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with hierarchical reasoning.
        
        Args:
            carry: Current carry state
            batch: Input batch with keys:
                - "inputs": Token IDs (batch_size, seq_len)
                - "puzzle_identifiers": Puzzle IDs (batch_size,)
                
        Returns:
            Tuple of:
            - New carry state (detached)
            - LM logits (batch_size, seq_len, vocab_size)
            - Q-values (Q_halt, Q_continue)
        """
        # Sequence info (RoPE if used)
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(
            batch["inputs"],
            batch["puzzle_identifiers"]
        )

        # Hierarchical reasoning iterations
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            # H cycles (high-level)
            for H_step in range(self.config.H_cycles):
                # L cycles (low-level) within each H cycle
                for L_step in range(self.config.L_cycles):
                    # Skip last L step of last H cycle (will be done with grad)
                    if not ((H_step == self.config.H_cycles - 1) and 
                            (L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

                # Skip last H step (will be done with grad)
                if not (H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, **seq_info)

        assert not z_H.requires_grad and not z_L.requires_grad

        # Final 1-step with gradient
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        # LM outputs (remove puzzle embeddings)
        new_carry = HRMACTv1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        # Q-values for halting (from first position)
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HRMACTv1(nn.Module):
    """
    HRM-ACT-v1: Hierarchical Reasoning Model with Adaptive Computation Time.
    
    This wraps the inner model with ACT (Adaptive Computation Time) halting logic
    based on Q-learning.
    """

    def __init__(self, config_dict: dict):
        """
        Args:
            config_dict: Configuration dictionary
        """
        super().__init__()
        self.config = HRMACTv1Config(**config_dict)
        self.inner = HRMACTv1_Inner(self.config)

    @property
    def puzzle_emb(self):
        """Access to puzzle embeddings (for optimizer setup)."""
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> HRMACTv1Carry:
        """
        Create initial carry state for a new batch.
        
        Args:
            batch: Input batch
            
        Returns:
            Initial carry state (all sequences halted)
        """
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device

        return HRMACTv1Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),  # Start halted
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(
        self,
        carry: HRMACTv1Carry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[HRMACTv1Carry, Dict[str, torch.Tensor]]:
        """
        Forward pass with ACT halting logic.
        
        Args:
            carry: Current carry state
            batch: Input batch
            
        Returns:
            Tuple of:
            - New carry state
            - Outputs dict with keys:
                - "logits": LM logits
                - "q_halt_logits": Q-value for halting
                - "q_continue_logits": Q-value for continuing
                - "target_q_continue" (training only): Target Q for RL
        """
        # Update carry for halted sequences
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v
            )
            for k, v in carry.current_data.items()
        }

        # Forward through inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry,
            new_current_data
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        # ACT halting logic
        with torch.no_grad():
            # Increment steps
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # Training: ACT with exploration
            if self.training and (self.config.halt_max_steps > 1):
                # Halt if Q(halt) > Q(continue)
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration: force minimum steps
                min_halt_steps = (
                    (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) *
                    torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                )
                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q-value (no replay buffer, similar to PQN)
                next_q_halt_logits, next_q_continue_logits = self.inner(
                    new_inner_carry,
                    new_current_data
                )[-1]
                
                outputs["target_q_continue"] = torch.sigmoid(
                    torch.where(
                        is_last_step,
                        next_q_halt_logits,
                        torch.maximum(next_q_halt_logits, next_q_continue_logits)
                    )
                )

        return HRMACTv1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs


__all__ = [
    "HRMACTv1InnerCarry",
    "HRMACTv1Carry",
    "HRMACTv1Config",
    "HRMACTv1ReasoningModule",
    "HRMACTv1_Inner",
    "HRMACTv1",
]

