# Modal Training Quickstart

Train your HRM-Augmented A* model on cloud GPUs using [Modal](https://modal.com/).

## Setup (One-time)

### 1. Install Modal

```bash
pip install modal
```

### 2. Authenticate with Modal

```bash
modal token new
```

This opens a browser window to authenticate.

### 3. (Optional) Set up Weights & Biases

If you want W&B logging, create a secret:

```bash
modal secret create wandb-secret WANDB_API_KEY=<your_api_key>
```

## Training Commands

### Basic Training (Single A100 GPU)

```bash
# Default: 1M timesteps, PPO, 16 parallel envs
python modal_run.py train

# Custom timesteps
python modal_run.py train --timesteps 2000000

# Use SAC instead of PPO
python modal_run.py train --algorithm SAC

# Enable W&B logging
python modal_run.py train --wandb

# Custom run name
python modal_run.py train --name my_experiment
```

### Fast Training (2x A100 GPUs)

```bash
# Default: 5M timesteps, 32 parallel envs
python modal_run.py train-fast

# Long run with W&B
python modal_run.py train-fast --timesteps 10000000 --wandb
```

### 🚀 BLAZING FAST: 4x B200 Blackwell GPUs (GB200-class)

```bash
# Default: 10M timesteps, 128 envs, 4096 batch size
python modal_run.py train-4gpu

# Custom config
python modal_run.py train-4gpu --timesteps 20000000 --n-envs 256 --wandb

# Maximum batch size for Blackwell's memory bandwidth
python modal_run.py train-4gpu --batch-size 8192 --timesteps 50000000
```

### 🔥 MAXIMUM POWER: 8x B200 Blackwell GPUs

```bash
# Default: 50M timesteps, 256 envs, 8192 batch size
python modal_run.py train-8gpu

# With W&B monitoring
python modal_run.py train-8gpu --wandb --timesteps 100000000
```

### Hyperparameter Sweep

```bash
# Run 5 different configurations in parallel
python modal_run.py sweep --n-runs 5

# Longer sweep runs
python modal_run.py sweep --n-runs 10 --timesteps 1000000
```

## Checkpoint Management

### List All Checkpoints

```bash
python modal_run.py list
```

### Evaluate a Checkpoint

```bash
python modal_run.py evaluate <run_id>

# More episodes
python modal_run.py evaluate <run_id> --episodes 200
```

### Download Checkpoint Locally

```bash
python modal_run.py download <run_id>
```

This creates `<run_id>_checkpoint.zip` in your current directory.

## Alternative: Direct Modal Commands

You can also use Modal directly:

```bash
# Train
modal run modal_app.py::train --timesteps 1000000

# Evaluate  
modal run modal_app.py::evaluate --checkpoint-path <run_id>

# List checkpoints
modal run modal_app.py::list_checkpoints

# Deploy for scheduled runs
modal deploy modal_app.py
```

## GPU Options

Modal supports various GPUs. See [Modal GPU docs](https://modal.com/docs/guide/gpu) for details.

| GPU | Memory | Use Case | Command |
|-----|--------|----------|---------|
| `T4` | 16GB | Testing, small runs | `train` |
| `L4` | 24GB | Medium training | `train` |
| `A100` | 40/80GB | Fast training | `train` |
| `A100:2` | 80/160GB | Very fast | `train-fast` |
| `H100` | 80GB | High performance | `train-fast` |
| `H200` | 141GB | Higher bandwidth | `train-fast` |
| **`B200`** | **192GB** | **Blackwell (GB200-class)** | `train-4gpu` |
| `B200:4` | 768GB | 🚀 BLAZING | `train-4gpu` |
| `B200:8` | 1.5TB | 🔥 MAXIMUM | `train-8gpu` |

### B200 Blackwell Architecture (GB200-class)

The B200 is NVIDIA's flagship Blackwell data center GPU - the closest to GB200 available on Modal:
- **192GB HBM3e memory** (vs 80GB on H100)
- **8TB/s bandwidth** (vs 3.35TB/s on H100)
- Perfect for large batch training

## Estimated Training Times

| Timesteps | GPU Config | Approx. Time | Steps/sec |
|-----------|------------|--------------|-----------|
| 1M | A100 | ~40 min | ~400 |
| 5M | A100:2 | ~1.5 hours | ~900 |
| 10M | H100:4 | ~45 min | ~3,500 |
| **10M** | **B200:4** | **~15-20 min** | **~10,000** |
| **50M** | **B200:8** | **~30-45 min** | **~20,000** |
| 100M | B200:8 | ~1.5 hours | ~20,000 |

## Monitoring

### TensorBoard (Local)

After downloading checkpoints:

```bash
tensorboard --logdir ./checkpoints
```

### Weights & Biases

If you enabled W&B logging, view runs at:
https://wandb.ai/<your_username>/hrm-astar-modal

## Troubleshooting

### "No CUDA devices available"

Make sure you're using a GPU-enabled function. Check `modal_app.py` has `gpu="A100"` set.

### "Out of memory"

Reduce batch size or number of environments:

```bash
python modal_run.py train --n-envs 8
```

### "Connection timeout"

Modal has a 12-hour timeout by default. For longer runs, edit `timeout` in `modal_app.py`.

### "Secret not found: wandb-secret"

Either create the secret or disable W&B:

```bash
# Create secret
modal secret create wandb-secret WANDB_API_KEY=<key>

# Or run without W&B
python modal_run.py train  # (no --wandb flag)
```

## Cost Estimation

Modal charges per second of GPU time. Rough estimates:

- A100: ~$3-4/hour
- A100:2: ~$6-8/hour
- T4: ~$0.50/hour

Training 1M timesteps typically costs $2-3 on A100.

Check [Modal pricing](https://modal.com/pricing) for current rates.

