# HRM-ACT-v1 Usage Guide

## Overview

HRM-ACT-v1 (Hierarchical Reasoning Model with Adaptive Computation Time) is a novel architecture that combines:
- **Hierarchical reasoning**: Two levels of reasoning (High and Low) that iterate
- **Adaptive Computation Time (ACT)**: Learns when to stop reasoning via Q-learning
- **Puzzle embeddings**: Optional sparse embeddings for task-specific context

## Architecture

### Hierarchical Reasoning

```
High Level (H): Broad, abstract reasoning
    ↓ (injects into)
Low Level (L): Detailed, fine-grained reasoning
    ↑ (feeds back to)
High Level (H): Updated abstract understanding
```

**Cycles:**
- `H_cycles`: Number of high-level reasoning iterations
- `L_cycles`: Number of low-level iterations per high cycle

### Adaptive Computation Time

The model learns **when to stop reasoning** using Q-learning:
- **Q(halt)**: Value of stopping now
- **Q(continue)**: Value of continuing reasoning
- Halts when `Q(halt) > Q(continue)` or max steps reached

## Quick Start

### 1. Basic Usage

```python
import torch
from hrm.models import HRMACTv1

# Configuration
config = {
    "batch_size": 8,
    "seq_len": 128,
    "puzzle_emb_ndim": 0,  # Disable puzzle embeddings
    "num_puzzle_identifiers": 10,
    "vocab_size": 10000,
    "H_cycles": 3,
    "L_cycles": 2,
    "H_layers": 4,
    "L_layers": 4,
    "hidden_size": 512,
    "num_heads": 8,
    "pos_encodings": "rope",
    "halt_max_steps": 5,
    "halt_exploration_prob": 0.1,
    "forward_dtype": "bfloat16",
}

# Create model
model = HRMACTv1(config).cuda()

# Prepare batch
batch = {
    "inputs": torch.randint(0, 10000, (8, 128), device="cuda"),
    "puzzle_identifiers": torch.zeros(8, dtype=torch.int32, device="cuda"),
}

# Initialize carry state
carry = model.initial_carry(batch)

# Forward pass
carry, outputs = model(carry, batch)

# Get outputs
logits = outputs["logits"]  # (batch, seq_len, vocab_size)
q_halt = outputs["q_halt_logits"]  # (batch,)
q_continue = outputs["q_continue_logits"]  # (batch,)
```

### 2. Multi-Step Reasoning

```python
# Run until all sequences halt
max_iterations = 10
carry = model.initial_carry(batch)

for step in range(max_iterations):
    carry, outputs = model(carry, batch)
    
    print(f"Step {step+1}:")
    print(f"  Halted: {carry.halted.sum().item()}/{len(carry.halted)}")
    print(f"  Steps: {carry.steps.tolist()}")
    
    # All sequences halted
    if carry.halted.all():
        break

# Final predictions
final_logits = outputs["logits"]
```

### 3. With Puzzle Embeddings

```python
# Enable puzzle embeddings
config["puzzle_emb_ndim"] = 512  # Embedding dimension
config["num_puzzle_identifiers"] = 1000  # Number of unique puzzles

model = HRMACTv1(config).cuda()

# Each sequence can have a different puzzle
batch = {
    "inputs": torch.randint(0, 10000, (8, 128), device="cuda"),
    "puzzle_identifiers": torch.randint(0, 1000, (8,), device="cuda"),  # Different puzzles
}

carry = model.initial_carry(batch)
carry, outputs = model(carry, batch)
```

## Training

### Basic Training Loop

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from hrm.models import HRMACTv1, CastedSparseEmbeddingSignSGD_Distributed

# Create model
model = HRMACTv1(config).cuda()
model.train()

# Optimizer for main model
optimizer = AdamW(
    [p for n, p in model.named_parameters() if "puzzle_emb" not in n],
    lr=3e-4,
    weight_decay=0.01,
)

# Separate optimizer for sparse puzzle embeddings (if used)
if config["puzzle_emb_ndim"] > 0:
    sparse_optimizer = CastedSparseEmbeddingSignSGD_Distributed(
        model.puzzle_emb.parameters(),
        world_size=1,  # Single GPU
        lr=1e-3,
        weight_decay=1e-2,
    )

# Training step
def train_step(batch, targets):
    # Initialize carry
    carry = model.initial_carry(batch)
    
    # Forward pass
    carry, outputs = model(carry, batch)
    
    # Language modeling loss
    logits = outputs["logits"]
    lm_loss = nn.functional.cross_entropy(
        logits.reshape(-1, config["vocab_size"]),
        targets.reshape(-1),
    )
    
    # Q-learning loss (if in training mode with ACT)
    q_loss = 0.0
    if "target_q_continue" in outputs:
        q_continue = torch.sigmoid(outputs["q_continue_logits"])
        target_q = outputs["target_q_continue"]
        q_loss = nn.functional.mse_loss(q_continue, target_q)
    
    # Total loss
    loss = lm_loss + 0.1 * q_loss  # Weight Q-loss
    
    # Backward
    loss.backward()
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Update
    optimizer.step()
    if config["puzzle_emb_ndim"] > 0:
        sparse_optimizer.step()
    
    optimizer.zero_grad()
    if config["puzzle_emb_ndim"] > 0:
        sparse_optimizer.zero_grad()
    
    return {
        "loss": loss.item(),
        "lm_loss": lm_loss.item(),
        "q_loss": q_loss if isinstance(q_loss, float) else q_loss.item(),
    }

# Training loop
for step in range(max_steps):
    batch = get_next_batch()  # Your data loader
    targets = batch["targets"]
    
    metrics = train_step(batch, targets)
    
    if step % 100 == 0:
        print(f"Step {step}: {metrics}")
```

### Evaluation

```python
model.eval()

def evaluate(val_loader):
    total_loss = 0.0
    total_steps = 0
    total_halts = []
    
    with torch.no_grad():
        for batch in val_loader:
            carry = model.initial_carry(batch)
            
            # Run until halt
            for _ in range(config["halt_max_steps"]):
                carry, outputs = model(carry, batch)
                if carry.halted.all():
                    break
            
            # Compute loss
            logits = outputs["logits"]
            targets = batch["targets"]
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, config["vocab_size"]),
                targets.reshape(-1),
            )
            
            total_loss += loss.item()
            total_steps += 1
            total_halts.append(carry.steps.float().mean().item())
    
    return {
        "loss": total_loss / total_steps,
        "avg_steps": sum(total_halts) / len(total_halts),
    }
```

## Configuration Guide

### Key Parameters

#### Model Size
```yaml
hidden_size: 512        # Model width (256, 512, 768, 1024, ...)
num_heads: 8            # Attention heads (must divide hidden_size)
H_layers: 4             # High-level depth
L_layers: 4             # Low-level depth
expansion: 2.6667       # FFN expansion (typically 2.5-4.0)
```

#### Reasoning Iterations
```yaml
H_cycles: 3-5           # High-level iterations (more = deeper reasoning)
L_cycles: 2-4           # Low-level iterations per H cycle
halt_max_steps: 5-10    # Maximum total reasoning steps
```

#### Halting (ACT)
```yaml
halt_max_steps: 8                 # Max iterations before forcing halt
halt_exploration_prob: 0.1        # Exploration for Q-learning (0.05-0.2)
```

#### Position Encodings
```yaml
pos_encodings: "rope"   # "rope" (recommended) or "learned"
rope_theta: 10000.0     # RoPE base frequency
```

#### Puzzle Embeddings
```yaml
puzzle_emb_ndim: 512              # Dimension (0 to disable)
num_puzzle_identifiers: 1000      # Number of unique puzzles
```

## Tips & Best Practices

### 1. Start Small
Begin with a small model to verify your setup:
```python
config = {
    "hidden_size": 256,
    "H_layers": 2,
    "L_layers": 2,
    "H_cycles": 2,
    "L_cycles": 1,
    "halt_max_steps": 3,
    ...
}
```

### 2. Halting Tuning
- **Too many steps**: Increase `halt_exploration_prob`, decrease `halt_max_steps`
- **Too few steps**: Decrease `halt_exploration_prob`, increase `halt_max_steps`
- Monitor average steps during training

### 3. Puzzle Embeddings
- Use when tasks have distinct identities (different puzzles, domains, etc.)
- Zero-initialized by default (learns task-specific information)
- Requires separate optimizer (SignSGD works well for sparse updates)

### 4. Mixed Precision
- Use `bfloat16` on modern GPUs (Ampere/Blackwell)
- Use `float16` on older GPUs with careful loss scaling
- Keep Q-values in `float32` for stability (done automatically)

### 5. Memory Optimization
- Reduce `batch_size` if OOM
- Reduce `H_cycles`, `L_cycles`, or `halt_max_steps`
- Use gradient checkpointing for very deep models

## Advanced Usage

### Custom Forward Logic

```python
class CustomHRMTrainer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, batch):
        carry = self.model.initial_carry(batch)
        
        # Custom multi-step logic
        all_logits = []
        
        for step in range(self.model.config.halt_max_steps):
            carry, outputs = self.model(carry, batch)
            all_logits.append(outputs["logits"])
            
            if carry.halted.all():
                break
        
        # Average predictions across steps
        avg_logits = torch.stack(all_logits).mean(dim=0)
        
        return avg_logits, carry.steps
```

### Distributed Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Setup distributed
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# Create model
model = HRMACTv1(config).cuda()
model = DDP(model, device_ids=[local_rank])

# Sparse embedding optimizer needs world_size
sparse_optimizer = CastedSparseEmbeddingSignSGD_Distributed(
    model.module.puzzle_emb.parameters(),
    world_size=dist.get_world_size(),
    lr=1e-3,
    weight_decay=1e-2,
)
```

## Troubleshooting

### Issue: Model always halts immediately
- Decrease `halt_exploration_prob`
- Check Q-head initialization (should be biased toward continuing initially)
- Increase `halt_max_steps`

### Issue: Model never halts
- Increase `halt_exploration_prob`
- Check Q-learning loss is being optimized
- Verify `halt_max_steps` is being enforced

### Issue: Poor performance
- Increase model size (`hidden_size`, layers)
- Increase reasoning cycles (`H_cycles`, `L_cycles`)
- Check learning rate and warmup schedule
- Verify data preprocessing

### Issue: OOM (Out of Memory)
- Reduce `batch_size`
- Reduce `seq_len`
- Reduce `halt_max_steps` or cycles
- Use gradient accumulation

## References

- Original HRM implementation
- [Adaptive Computation Time](https://arxiv.org/abs/1603.08983)
- [Q-learning for ACT](https://arxiv.org/abs/2407.04811) (PQN-style)

---

For more examples, see `configs/hrm_act_v1_example.yaml` and `tests/test_hrm_act_v1.py`.

