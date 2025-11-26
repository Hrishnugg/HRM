<!-- 8c98cb9b-b0b5-4423-a2e9-27a326c353a4 4c789317-da5e-4deb-9424-f4f63c441ab6 -->
# HRM-v2 Port Review: Discrepancies & Bug Hunt

## Review Strategy

- **Focus**: Find discrepancies and obvious bugs between original and ported code
- **Scope**: Equal attention to core model logic AND training infrastructure
- **Not doing**: Full mathematical proofs or exhaustive edge case testing

## Critical Bug Already Found

**BUG #1: Truncated Normal Init in `/HRM-v2/src/hrm/utils/init.py:48`**

```python
# WRONG (line 48):
pdf_l = c * math.exp(-0.5 * lower ** 2)

# CORRECT (from original):
pdf_l = c * math.exp(-0.5 * upper ** 2)
```

## Review Phases

### Phase 1: Core HRM-ACT-v1 Model Logic

**Files**: `models/hrm/hrm_act_v1.py` (original) vs `HRM-v2/src/hrm/models/hrm_act_v1.py`

Check for discrepancies in:

- Hierarchical reasoning loop structure (H_cycles, L_cycles, skip conditions)
- Carry state initialization, reset, and propagation
- Gradient detachment placement (critical for correctness)
- ACT Q-learning halting logic (exploration, target Q)
- Puzzle embedding padding and concatenation
- Buffer registration (`nn.Buffer` vs `register_buffer`)

### Phase 2: Training Infrastructure (Sparse Embeddings)

**Files**: `models/sparse_embedding.py` vs `HRM-v2/src/hrm/models/sparse_embedding.py`

Check for discrepancies in:

- Local vs global weight copying logic
- Training vs eval mode behavior
- SignSGD gradient aggregation (unique ID handling)
- Distributed all-gather implementation
- Weight decay application order

### Phase 3: Layers & Basic Operations  

**Files**: `models/layers.py` vs `HRM-v2/src/hrm/models/layers.py` + ops files

Check for discrepancies in:

- `CastedLinear` and `CastedEmbedding` weight casting
- Original `Attention` class vs new `AttentionWithRoPE` integration
- SwiGLU implementation
- RoPE application and buffer registration
- RMS normalization calculation
- Transformer block post-norm order

### Phase 4: API & Configuration Compatibility

Check:

- Config parameter names and defaults
- Forward pass signatures (argument order, names)
- Carry state dataclass structures
- Output dictionary keys
- Return value ordering

### Phase 5: Compile Final Report

Generate comprehensive bug report with:

- All bugs found with line numbers and fixes
- Validation that key logic matches
- API compatibility notes
- Modernization improvements that are intentional vs bugs

## Key Files to Compare

**Original (root):**

- `models/hrm/hrm_act_v1.py` (284 lines)
- `models/layers.py` (158 lines)  
- `models/sparse_embedding.py` (133 lines)
- `models/common.py` (33 lines)

**Ported (HRM-v2):**

- `src/hrm/models/hrm_act_v1.py` (506 lines)
- `src/hrm/models/layers.py` (309 lines)
- `src/hrm/models/sparse_embedding.py` (224 lines)
- `src/hrm/utils/init.py` (60 lines)
- `src/hrm/ops/attention.py` (141 lines)
- `src/hrm/ops/rotary.py` (108 lines)
- `src/hrm/ops/norm.py` (62 lines)