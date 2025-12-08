# HRM Cloud Experiments

This folder contains Modal-based cloud training scripts for HRM (Hierarchical Reasoning Module) scaling experiments and Diffusion Planner experiments.

## Fully Automated Experiments

> **All experiments are fully automated.** Each script handles the entire pipeline:
> 1. **Data Collection/Generation** - Expert trajectories are generated in parallel
> 2. **Data Processing** - Trajectories are preprocessed and batched automatically
> 3. **Training** - Model training runs with configured hyperparameters
> 4. **Evaluation** - Performance metrics are computed and reported
>
> Simply run the command once and wait for it to complete. Results will be printed to the console.

---

## Prerequisites: Setting Up Modal

All experiments in this folder run on [Modal](https://modal.com/), a serverless GPU cloud platform.

### 1. Create a Modal Account

1. Go to [modal.com](https://modal.com/)
2. Sign up for an account (free tier includes $30/month in credits)
3. Verify your email address

### 2. Install the Modal CLI

```bash
pip install modal
```

### 3. Authenticate with Modal

```bash
modal setup
```

This will open a browser window to authenticate. Follow the prompts to link your CLI to your Modal account.

### 4. Verify Installation

```bash
modal run --help
```

If this returns help information, you're ready to run experiments.

---

## HRM Scaling Experiments (HRM-Augmented A*)

These experiments test the Hierarchical Reasoning Module at various scales.

### Small Scale (302k parameters, 1 GPU)

| Script | Description |
|--------|-------------|
| `hrm_cloud.py` | Base small-scale HRM experiment |
| `hrm_cloudv2.py` | Small-scale v2 with improvements |
| `hrm_cloudv3.py` | Small-scale v3 iteration |

**Result:** 52% success rate (LSTM baseline: 72%)

**How to run:**
```bash
modal run hrm_cloud.py
modal run hrm_cloudv2.py
modal run hrm_cloudv3.py
```

---

### Mid Scale (3.5M parameters, 1 GPU)

| Script | Description |
|--------|-------------|
| `hrm_cloudMid.py` | Mid-scale HRM experiment |

**Result:** 62% success rate (LSTM baseline: 68%)

**How to run:**
```bash
modal run hrm_cloudMid.py
```

---

### Full Scale (⚠️ LOW PRIORITY)

> **Note:** These experiments are low priority due to timeout and stability issues during execution. They are preserved for reference but not recommended for regular use.

| Script | Description |
|--------|-------------|
| `hrm_cloudFullScale.py` | Full-scale (~28M params) experiment |
| `hrm_cloudFullScaleSplit.py` | Split training approach |
| `hrm_cloudFullScaleRobustFix.py` | Attempted stability fixes |

**Result:** Timed out / stability issues

**How to run (not recommended):**
```bash
modal run hrm_cloudFullScale.py
modal run hrm_cloudFullScaleSplit.py
modal run hrm_cloudFullScaleRobustFix.py
```

---

### 8-GPU DDP (28.97M parameters, 8×H100) 🏆

| Script | Description | Recommended |
|--------|-------------|-------------|
| `hrm_cloud_8gpu.py` | Distributed Data Parallel training on 8 H100 GPUs | ✅ Yes |
| `hrm_cloud_8gpu_v2.py` | 8-GPU v2 with different batch size and optimizer | ❌ No |

**Result:** **68% success rate** (LSTM baseline: 66%) — **Best HRM result!**

**How to run:**
```bash
modal run hrm_cloud_8gpu.py
```

> **Note:** This requires significant GPU resources (8×H100). Ensure your Modal account has sufficient credits.

> ⚠️ **Not Recommended:** `hrm_cloud_8gpu_v2.py` uses a different batch size configuration and optimizer, which produced significantly worse results than the original. Use `hrm_cloud_8gpu.py` instead.

---

## Diffusion Planner Experiments

### Current Version: v2 Only

> **Important:** Only the v2 version is currently available in `diffusion_cloud.py`. Previous versions (v0 and v1) were replaced by direct updates to the file.

| Version | Status | Params | Description | Result |
|---------|--------|--------|-------------|--------|
| v0 | ❌ Replaced | ~500k | 500 trajectories | 20% |
| v1 | ❌ Replaced | ~500k | 2,500 trajectories, 1000 epochs | 64% |
| **v2** | ✅ Current | ~4M | ResBlocks + Attention + EMA | 60% |

**v2 Features:**
- ResBlocks + Self-Attention architecture
- Exponential Moving Average (EMA) for stable training
- 5000 trajectories (50 workers × 100 each)
- 2000 epochs with cosine learning rate schedule
- Multi-sample inference (10 samples)

**How to run:**
```bash
modal run diffusion_cloud.py
```

---

## Quick Reference: All Commands

```bash
# Small Scale HRM
modal run hrm_cloud.py
modal run hrm_cloudv2.py
modal run hrm_cloudv3.py

# Mid Scale HRM
modal run hrm_cloudMid.py

# Full Scale HRM (LOW PRIORITY - not recommended)
modal run hrm_cloudFullScale.py
modal run hrm_cloudFullScaleSplit.py
modal run hrm_cloudFullScaleRobustFix.py

# 8-GPU DDP HRM (Best Results)
modal run hrm_cloud_8gpu.py
# modal run hrm_cloud_8gpu_v2.py  # Not recommended - worse results

# Diffusion Planner (v2)
modal run diffusion_cloud.py
```

---

## Results Summary

| Experiment | Parameters | GPUs | Success Rate |
|------------|------------|------|--------------|
| HRM Small | 302k | 1 | 52% |
| HRM Mid | 3.5M | 1 | 62% |
| HRM Full-Scale | ~28M | 1-4 | ❌ Timeout |
| **HRM 8-GPU DDP** | **28.97M** | **8×H100** | **68%** 🏆 |
| Diffusion v2 | ~4M | 1 | 60% |

*LSTM Baselines: Small/Mid = 68-72%, 8-GPU = 66%*

---

## Troubleshooting

### Modal Authentication Issues
```bash
modal token new
```

### Timeout Errors
Increase the timeout in the `@app.function()` decorator within the script.

### Out of Memory
Reduce batch size or model parameters in the script configuration.

