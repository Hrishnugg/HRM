"""
HRM 8-GPU DDP Training v2 - Improved Configuration
===================================================
Changes from v1:
- Smaller batch (4096 vs 16384) for better generalization
- LAMB optimizer (designed for large batch training)
- Fewer epochs (12 vs 30) - more gradient updates per epoch
- Expected: Lower final loss (~0.0001) and better success rate
"""
import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import heapq
from typing import List
from concurrent.futures import ThreadPoolExecutor

app = modal.App("hrm-8gpu-v2-improved")

# Image with LAMB optimizer
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch>=2.4.0", "numpy", "gymnasium", "tqdm", "torch-optimizer")
)

# New volume for v2 with obstacle-prediction format
vol = modal.Volume.from_name("hrm-8gpu-v2-obs-vol", create_if_missing=True)

CONFIG = {
    # --- FULL SCALE ARCHITECTURE ---
    "hidden_dim": 512,
    "num_heads": 8,
    "num_layers": 4,
    
    # --- IMPROVED TRAINING PARAMS ---
    "gpu_count": 8,       
    "per_gpu_batch": 512,  # 512 × 8 = 4096 global (vs 16384) - better generalization
    "lr": 3e-4,            # Base LR for LAMB (handles scaling internally)
    "epochs": 12,          # Fewer epochs (4x more updates/epoch with smaller batch)
    
    # --- DATA & ENV ---
    "data_episodes": 60000,
    "hrm_k_step": 2,
    "grid_size": 20, "n_static": 12, "n_dynamic": 6,
    "obs_history": 20, "pred_horizon": 20, "eval_episodes": 50
}

# UNIQUE PATHS for v2
PATHS = {
    "data_dir": "/data/episodes_v2_obs_fixed",      # New data dir for fixed obstacle format
    "merged_data": "/data/merged_v2_obs_fixed.pt",  # Fresh data with correct format
    "model_lstm": "/data/lstm_v2_obs_fixed.pt",     # Fresh LSTM model
    "model_hrm": "/data/hrm_v2_obs_fixed.pt"        # Fresh HRM model
}

# ==========================================
# 1. ROBUST ARCHITECTURE (BF16 + Gated)
# ==========================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.eps=eps; self.scale=dim**-0.5; self.g=nn.Parameter(torch.ones(dim))
    def forward(self, x):
        x_f32 = x.float()
        norm = x_f32.norm(dim=-1, keepdim=True) * self.scale
        return (x_f32 / norm.clamp(min=self.eps) * self.g).to(x.dtype)

class GatedRecurrentBlock(nn.Module):
    """GTrXL-style gated recurrence for stable training"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.0)
        self.gate = nn.Linear(dim * 2, dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
        self.norm2 = RMSNorm(dim)
    
    def forward(self, x, state):
        # Gated state update (GTrXL style)
        combined = torch.cat([x, state], dim=-1)
        z = torch.sigmoid(self.gate(combined))
        h = (x + state) * 0.7071  # Variance scaling
        
        # Self-attention
        h_norm = self.norm(h)
        attn_out, _ = self.attn(h_norm, h_norm, h_norm)
        h = h + attn_out
        
        # FFN
        h = h + self.ffn(self.norm2(h))
        
        # Gated output
        new_state = z * h + (1 - z) * state
        return h, new_state

class DeepSapientHRM(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, k_step=2, num_layers=4, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_step = k_step
        self.num_layers = num_layers
        
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.L_blocks = nn.ModuleList([GatedRecurrentBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.H_blocks = nn.ModuleList([GatedRecurrentBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.out = nn.Linear(hidden_dim, out_dim)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01)
                if m.bias is not None: nn.init.zeros_(m.bias)
    
    def forward(self, obs_seq):
        B, T, _ = obs_seq.shape
        x = self.embed(obs_seq)
        
        h_L_states = [torch.zeros(B, T, self.hidden_dim, device=x.device, dtype=x.dtype) for _ in range(self.num_layers)]
        h_H_states = [torch.zeros(B, T, self.hidden_dim, device=x.device, dtype=x.dtype) for _ in range(self.num_layers)]
        
        h_L = x
        for i, block in enumerate(self.L_blocks):
            h_L, h_L_states[i] = block(h_L, h_L_states[i])
        
        h_H = h_L
        for step in range(self.k_step):
            for i, block in enumerate(self.H_blocks):
                h_H, h_H_states[i] = block(h_H, h_H_states[i])
        
        return self.out(h_H[:, -1, :])

# ==========================================
# 2. LSTM BASELINE
# ==========================================
class LSTMPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ==========================================
# 3. ENVIRONMENT (v1 format - obstacle prediction)
# ==========================================
class DynamicGridEnv:
    """v1-compatible environment: _get_obs returns obstacle positions"""
    def __init__(self, config):
        self.size = config["grid_size"]
        self.n_dyn = config["n_dynamic"]
        self.reset()
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.static_map = np.zeros((self.size, self.size))
        self.static_map[0, 0] = 0
        self.static_map[self.size-1, self.size-1] = 0
        for _ in range(self.size):
            r, c = np.random.randint(0, self.size, 2)
            if (r, c) != (0, 0) and (r, c) != (self.size-1, self.size-1):
                self.static_map[r, c] = 1.0
        
        self.dynamic_obs = []
        for _ in range(self.n_dyn):
            while True:
                pos = np.random.randint(0, self.size, 2).astype(float)
                if self.static_map[int(pos[0]), int(pos[1])] == 0:
                    vel = np.random.randn(2)
                    vel = vel / np.linalg.norm(vel) * 0.7
                    self.dynamic_obs.append({'pos': pos, 'vel': vel})
                    break
        
        self.agent_pos = np.array([0., 0.])
        self.goal_pos = np.array([self.size-1., self.size-1.])
        return self._get_obs()
    
    def step_physics(self):
        """Move obstacles and return new positions"""
        for o in self.dynamic_obs:
            o['pos'] += o['vel']
            for i in range(2):
                if o['pos'][i] < 0 or o['pos'][i] >= self.size:
                    o['vel'][i] *= -1
                    o['pos'][i] = np.clip(o['pos'][i], 0, self.size - 0.01)
        return self._get_obs()
    
    def _get_obs(self):
        """Returns obstacle positions - shape (n_dynamic, 2)"""
        return np.array([o['pos'] for o in self.dynamic_obs])

# ==========================================
# 4. DDP TRAINING WITH LAMB
# ==========================================

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def ddp_cleanup():
    dist.destroy_process_group()

def train_worker(rank, world_size, merged_path, config):
    """Training loop with LAMB optimizer"""
    import torch_optimizer as lamb_optim
    from tqdm import tqdm
    import time
    
    ddp_setup(rank, world_size)
    
    # 1. Load Data
    if rank == 0: print(f"Rank {rank}: Loading Dataset...")
    X, Y = torch.load(merged_path, weights_only=False, map_location='cpu')
    
    dataset = torch.utils.data.TensorDataset(X, Y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config['per_gpu_batch'], 
        shuffle=False,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    # 2. Model Setup
    model = DeepSapientHRM(2, config['hidden_dim'], 2,  # 2 input features (v1 data: agent_xy only)
                           k_step=config['hrm_k_step'],
                           num_layers=config['num_layers'],
                           num_heads=config['num_heads']).to(rank).to(torch.bfloat16)
    
    # find_unused_parameters=True needed for gated architectures where
    # some parameters may not receive gradients due to gating mechanism
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # 3. LAMB Optimizer (designed for large batch training)
    optimizer = lamb_optim.Lamb(
        model.parameters(), 
        lr=config['lr'],
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing (works well with LAMB)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=len(loader) * config['epochs'],
        eta_min=1e-6
    )
    
    if rank == 0:
        print(f"🚀 DDP v2 Started with LAMB optimizer")
        print(f"   Global Batch: {config['per_gpu_batch']*world_size}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        print(f"   {len(loader)} batches/epoch, {config['epochs']} epochs")
        print(f"   Total gradient updates: {len(loader) * config['epochs']:,}")
        print(f"   DDP find_unused_parameters=True (for gated architecture)")
    
    # 4. Training
    model.train()
    best_loss = float('inf')
    
    for ep in range(config['epochs']):
        sampler.set_epoch(ep)
        ep_start = time.time()
        
        ep_loss = torch.zeros(1).to(rank)
        steps = 0
        
        pbar = tqdm(loader, desc=f"Ep {ep+1}/{config['epochs']}", disable=(rank != 0), leave=False)
        for batch_idx, (bx, by) in enumerate(pbar):
            bx = bx.to(rank, non_blocking=True).to(torch.bfloat16)
            by = by.to(rank, non_blocking=True).to(torch.bfloat16)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred = model(bx)
                loss = nn.MSELoss()(pred, by)
            
            # Check for NaN but still execute backward to keep DDP in sync
            is_nan = torch.isnan(loss)
            if is_nan:
                # Zero the loss to prevent NaN gradients, but still call backward
                loss = loss * 0
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            if not is_nan:
                ep_loss += loss
                steps += 1
            
            if rank == 0:
                pbar.set_postfix(loss=f"{loss.item():.6f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            
            if rank == 0 and batch_idx > 0 and batch_idx % 1000 == 0:
                print(f"  Ep {ep+1} Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.6f}")
            
        # Aggregate Loss
        dist.all_reduce(ep_loss, op=dist.ReduceOp.SUM)
        if rank == 0:
            ep_time = time.time() - ep_start
            avg_loss = ep_loss.item() / (steps * world_size)
            eta = (config['epochs'] - ep - 1) * ep_time / 60
            
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            print(f"Epoch {ep+1}/{config['epochs']} | Loss: {avg_loss:.6f} | Best: {best_loss:.6f} | Time: {ep_time:.1f}s | ETA: {eta:.1f}min")
    
    # 5. Save Model
    if rank == 0:
        print("Saving DDP Model...")
        torch.save(model.module.state_dict(), PATHS['model_hrm'])
        vol.commit()
        print("✅ Training Complete.")
    
    ddp_cleanup()

# ==========================================
# 5. MODAL FUNCTIONS
# ==========================================

@app.function(image=image, volumes={"/data": vol}, cpu=2.0, timeout=600)
def collect_data_chunk(worker_id: int, n_episodes: int = 600):
    """Collect trajectory data - v1 format: per-obstacle samples"""
    import os
    os.makedirs(PATHS['data_dir'], exist_ok=True)
    
    env = DynamicGridEnv(CONFIG)
    X, Y = [], []
    
    for _ in range(n_episodes):
        env.reset()
        hist = []
        for _ in range(70):  # Collect 70 timesteps
            hist.append(env.step_physics())
            if len(hist) > CONFIG['obs_history']:
                # past: (seq_len, n_obstacles, 2), future/prev: (n_obstacles, 2)
                past = np.array(hist[-CONFIG['obs_history']-1:-1]) / env.size
                future = np.array(hist[-1]) / env.size
                prev = np.array(hist[-2]) / env.size
                # Create per-obstacle samples
                for j in range(env.n_dyn):
                    X.append(past[:, j, :])  # Shape: (seq_len, 2)
                    Y.append(future[j, :] - prev[j, :])  # Shape: (2,) - delta
    
    path = f"{PATHS['data_dir']}/chunk_{worker_id}.pt"
    torch.save((X, Y), path)  # v1 format: tuple not dict
    vol.commit()
    print(f"Chunk {worker_id}: {len(X)} samples saved")
    return path

@app.function(image=image, volumes={"/data": vol}, cpu=1.0, timeout=60)
def check_cached_data():
    """Check if merged data already exists"""
    import os
    exists = os.path.exists(PATHS['merged_data'])
    print(f"Checking cache: {PATHS['merged_data']} exists = {exists}")
    return exists

@app.function(image=image, volumes={"/data": vol}, cpu=1.0, timeout=60)
def check_cached_models():
    """Check if both models already exist"""
    import os
    vol.reload()
    return {
        'lstm': os.path.exists(PATHS['model_lstm']),
        'hrm': os.path.exists(PATHS['model_hrm'])
    }

@app.function(image=image, volumes={"/data": vol}, cpu=8.0, memory=65536, timeout=1800)
def merge_chunks():
    """Merge all data chunks into single tensor - v1 format"""
    from concurrent.futures import ThreadPoolExecutor
    import glob
    
    chunk_files = sorted(glob.glob(f"{PATHS['data_dir']}/chunk_*.pt"))
    print(f"Merging {len(chunk_files)} chunks...")
    
    def load(f):
        # v1 format: tuple (X, Y) not dict
        return torch.load(f, weights_only=False)
    
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(load, chunk_files))
    
    X_all = [x for (X, Y) in results for x in X]
    Y_all = [y for (X, Y) in results for y in Y]
    
    X = torch.tensor(np.array(X_all), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_all), dtype=torch.float32)
    
    print(f"Merged: X={X.shape}, Y={Y.shape}")
    torch.save((X, Y), PATHS['merged_data'])
    vol.commit()
    return X.shape[0]

@app.function(image=image, volumes={"/data": vol}, gpu="A10", timeout=28800)
def train_lstm():
    """Train LSTM baseline on single GPU"""
    from tqdm import tqdm
    from torch.amp import autocast, GradScaler
    
    print(f"Loading merged data from {PATHS['merged_data']}...")
    X, Y = torch.load(PATHS['merged_data'], weights_only=False)
    print(f"Dataset Size: {len(X)} samples")
    
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, Y),
        batch_size=4096, shuffle=True, num_workers=8, pin_memory=True
    )
    
    model = LSTMPredictor(2, 256, 2).cuda()  # 2 input features (v1 data: agent_xy only)
    opt = optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    scheduler = optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, steps_per_epoch=len(loader), epochs=30)
    scaler = GradScaler()
    
    print(f"Training LSTM Baseline ({len(loader)} batches/epoch)...")
    for ep in range(30):
        pbar = tqdm(loader, desc=f"LSTM Ep {ep}/{30}", leave=False)
        for bx, by in pbar:
            bx, by = bx.cuda(non_blocking=True), by.cuda(non_blocking=True)
            opt.zero_grad()
            with autocast('cuda'):
                loss = nn.MSELoss()(model(bx), by)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            pbar.set_postfix(loss=f"{loss.item():.6f}")
    
    torch.save(model.state_dict(), PATHS['model_lstm'])
    vol.commit()
    print(f"LSTM saved to {PATHS['model_lstm']}")

@app.function(image=image, volumes={"/data": vol}, gpu="H100:8", timeout=28800)
def train_hrm_multigpu(merged_path: str):
    """Launch 8-GPU DDP training with LAMB"""
    print("🚀 Launching 8-GPU DDP Training v2 (LAMB + Small Batch)...")
    
    X, Y = torch.load(merged_path, weights_only=False)
    print(f"Dataset Size: {len(X)} samples (loaded in background)")
    
    world_size = CONFIG['gpu_count']
    mp.spawn(train_worker, args=(world_size, merged_path, CONFIG), nprocs=world_size, join=True)

# ==========================================
# 6. EVALUATION (Same as v1)
# ==========================================

class SpaceTimeAStar:
    """A* planner with obstacle trajectory prediction - v1 format"""
    def __init__(self, env, model, device):
        self.env = env
        self.model = model
        self.device = device
        self.model.eval()
    
    def get_next_action(self, start, goal, obs_history):
        """
        obs_history: shape (n_obstacles, seq_len, 2) - normalized obstacle positions
        """
        curr = torch.tensor(obs_history / self.env.size, dtype=torch.float32).to(self.device)
        # Match model dtype (FP32 for LSTM, BF16 for HRM)
        curr = curr.to(next(self.model.parameters()).dtype)
        
        future_obs = []
        with torch.no_grad():
            for _ in range(CONFIG['pred_horizon']):
                delta = self.model(curr)
                next_pos_norm = curr[:, -1, :] + delta.to(curr.dtype)  # Keep same dtype
                future_obs.append((next_pos_norm.float().cpu().numpy() * self.env.size))
                curr = torch.cat([curr[:, 1:, :], next_pos_norm.unsqueeze(1)], dim=1)
        future_obs = np.array(future_obs)  # Shape: (horizon, n_obstacles, 2)
        
        start_node = (int(start[0]), int(start[1]), 0)
        pq = [(0, 0, start_node)]
        g_score = {start_node: 0}
        came_from = {}
        best_node, min_h = None, float('inf')
        
        while pq:
            f, g, curr_node = heapq.heappop(pq)
            r, c, t = curr_node
            
            if (r, c) == (int(goal[0]), int(goal[1])):
                return self.trace(came_from, curr_node, start_node)
            
            if t >= CONFIG['pred_horizon'] - 1:
                h = abs(r - goal[0]) + abs(c - goal[1])
                if h < min_h:
                    min_h = h
                    best_node = curr_node
                continue
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]:
                nr, nc, nt = r + dr, c + dc, t + 1
                if not (0 <= nr < self.env.size and 0 <= nc < self.env.size):
                    continue
                if self.env.static_map[nr, nc] == 1:
                    continue
                # Check collision with ANY predicted obstacle position
                if np.any(np.linalg.norm(future_obs[nt] - np.array([nr, nc]), axis=1) < 1.0):
                    continue
                
                new_g = g + 1
                neigh = (nr, nc, nt)
                if new_g < g_score.get(neigh, float('inf')):
                    g_score[neigh] = new_g
                    heapq.heappush(pq, (new_g + abs(nr - goal[0]) + abs(nc - goal[1]), new_g, neigh))
                    came_from[neigh] = curr_node
        
        if best_node:
            return self.trace(came_from, best_node, start_node)
        return (int(start[0]), int(start[1]))
    
    def trace(self, came_from, curr, start):
        """Trace back from goal to find next move"""
        path = []
        while curr in came_from:
            path.append(curr)
            curr = came_from[curr]
        return (path[-1][0], path[-1][1]) if path else (int(start[0]), int(start[1]))

@app.cls(image=image, volumes={"/data": vol}, gpu="A10", max_containers=20)
class Evaluator:
    @modal.enter()
    def setup(self):
        self.device = torch.device("cuda")
        # LSTM in FP32 - trained with hidden_dim=256 (not CONFIG['hidden_dim'])
        self.lstm = LSTMPredictor(2, 256, 2).to(self.device)
        self.lstm.load_state_dict(torch.load(PATHS['model_lstm'], weights_only=True))
        self.lstm.eval()
        
        # HRM in BF16 (matches training)
        self.hrm = DeepSapientHRM(2, CONFIG['hidden_dim'], 2,
                                   k_step=CONFIG['hrm_k_step'],
                                   num_layers=CONFIG['num_layers'],
                                   num_heads=CONFIG['num_heads']).to(self.device).bfloat16()
        self.hrm.load_state_dict(torch.load(PATHS['model_hrm'], weights_only=True))
        self.hrm.eval()
    
    @modal.method()
    def evaluate_episode(self, seed: int):
        """v1 evaluation: obstacle trajectory prediction + A* planning"""
        results = {}
        
        for name, model in [("lstm", self.lstm), ("hrm", self.hrm)]:
            env = DynamicGridEnv(CONFIG)
            env.reset(seed=seed + 100)
            planner = SpaceTimeAStar(env, model, self.device)
            
            # Collect initial obstacle history
            hist = [env.step_physics() for _ in range(CONFIG['obs_history'])]
            
            success = True
            for _ in range(80):  # Max 80 steps
                # obs_history: (n_obstacles, seq_len, 2) - transpose from list format
                h_np = np.array(hist[-CONFIG['obs_history']:]).transpose(1, 0, 2)
                
                # Get next action from planner
                nr, nc = planner.get_next_action(env.agent_pos, env.goal_pos, h_np)
                env.agent_pos = np.array([float(nr), float(nc)])
                
                # Step environment and record obstacle positions
                hist.append(env.step_physics())
                
                # Check goal reached
                if np.linalg.norm(env.agent_pos - env.goal_pos) < 1.5:
                    break
                
                # Check static obstacle collision
                ar, ac = int(env.agent_pos[0]), int(env.agent_pos[1])
                if 0 <= ar < env.size and 0 <= ac < env.size:
                    if env.static_map[ar, ac] == 1:
                        success = False
                        break
                
                # Check dynamic obstacle collision
                for o in env.dynamic_obs:
                    if np.linalg.norm(env.agent_pos - o['pos']) < 1.0:
                        success = False
                        break
                if not success:
                    break
            else:
                # Didn't reach goal in 80 steps
                if np.linalg.norm(env.agent_pos - env.goal_pos) >= 1.5:
                    success = False
            
            results[name] = success
        return results

# ==========================================
# 7. MAIN ENTRYPOINT
# ==========================================

@app.local_entrypoint()
def main():
    print("🚀 Launching 8-GPU DDP HRM Training v2 (Obstacle Prediction - v1 Format)")
    print("   - LAMB optimizer (designed for large batch)")
    print("   - v1-compatible obstacle trajectory prediction")
    print("   - Fresh data collection with per-obstacle samples")
    print("   - Target: Match v1's 68% HRM success rate")
    
    # Check for cached models first
    print("\n🔍 Checking for cached models...")
    cache_status = check_cached_models.remote()
    print(f"Model cache: LSTM={cache_status['lstm']}, HRM={cache_status['hrm']}")
    
    if cache_status['lstm'] and cache_status['hrm']:
        print("✅ Found cached models! Skipping training, going to evaluation...")
    else:
        # Check for cached data
        print("\n🔍 Checking for cached data...")
        if not check_cached_data.remote():
            print("\n📦 Step 1: Collecting data (100 chunks × 600 episodes)...")
            # v1 format: worker_id, n_episodes
            list(collect_data_chunk.map(range(100), kwargs={'n_episodes': 600}))
            
            print("\n🔄 Step 2: Merging chunks...")
            merge_chunks.remote()
        else:
            print("✅ Found cached merged data! Skipping collection & merge.")
        
        # Training
        print("\n🏋️ Step 3: Training models...")
        print("   🏃 Training LSTM...")
        lstm_handle = train_lstm.spawn()
        
        print("   🏃 Training HRM v2 (12 epochs, 8x H100 DDP + LAMB)...")
        hrm_handle = train_hrm_multigpu.spawn(PATHS['merged_data'])
        
        lstm_handle.get()
        print("   ✅ LSTM training complete")
        
        hrm_handle.get()
        print("   ✅ HRM training complete")
    
    # Evaluation
    print("\n📊 Step 4: Evaluation...")
    evaluator = Evaluator()
    results = list(evaluator.evaluate_episode.map(range(CONFIG['eval_episodes'])))
    
    lstm_success = sum(r['lstm'] for r in results)
    hrm_success = sum(r['hrm'] for r in results)
    
    print("\n" + "="*50)
    print("📈 FINAL RESULTS (8-GPU DDP v2 - LAMB + Small Batch)")
    print("="*50)
    print(f"LSTM: {lstm_success}/{CONFIG['eval_episodes']} ({100*lstm_success/CONFIG['eval_episodes']:.1f}%)")
    print(f"HRM:  {hrm_success}/{CONFIG['eval_episodes']} ({100*hrm_success/CONFIG['eval_episodes']:.1f}%)")

