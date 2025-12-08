import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import heapq
import os
from typing import List
from torch.amp import autocast, GradScaler
from concurrent.futures import ThreadPoolExecutor

app = modal.App("hrm-b200-full-scale-split")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch>=2.2.0", "numpy", "gymnasium", "tqdm")
)

vol = modal.Volume.from_name("hrm-research-vol", create_if_missing=True)

CONFIG = {
    # --- SCALE: FULL SIZE (~27M Params) ---
    # To hit ~27M params, we need Depth (Layers) + Width (512)
    "hidden_dim": 512,   # 4x larger width
    "num_layers": 4,     # 4x deeper (Stacks blocks)
    "num_heads": 8,      # Standard for d=512
    # --------------------------------------
    
    # We need significantly MORE data to saturate a 27M param model
    "data_episodes": 60000, 
    "batch_size": 4096,     # Larger batch for 4 GPUs     
    
    "hrm_k_step": 2,     
    "lr": 3e-4,          # Higher LR now safe with gated architecture
    "epochs": 20,        # Reduced from 50 - model converges by ep 17
    
    # Env Params
    "grid_size": 20, "n_static": 12, "n_dynamic": 6,
    "obs_history": 20, "pred_horizon": 20, "eval_episodes": 50
}

PATHS = {
    "data_dir": "/data/episodes_full",
    "merged_data": "/data/merged_full.pt",
    "model_lstm": "/data/lstm_full.pt",
    "model_hrm": "/data/hrm_full.pt"
}

# ==========================================
# 1. DEEP SAPIENT HRM ARCHITECTURE (Gated + AMP-Safe)
# ==========================================
class RMSNorm(nn.Module):
    """AMP-safe RMSNorm that upcasts to FP32 for stability"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = dim ** -0.5
        self.g = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # Upcast to FP32 for numerical stability with AMP
        x_dt = x.dtype
        x_f32 = x.float()
        norm = x_f32.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return ((x_f32 / norm) * self.scale * self.g).to(x_dt)

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
    def forward(self, x): return self.w3(F.silu(self.w1(x)) * self.w2(x))

class GatedRecurrentBlock(nn.Module):
    """
    GTrXL-style Gated Recurrent Block - AMP stable.
    Uses variance scaling + learned gating to prevent state explosion.
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, int(dim * 2.6))
        self.gate = nn.Linear(dim * 2, dim)  # Learned gate
        
    def forward(self, x, state):
        # 1. Variance Scaling (Critical for Deep Recurrence)
        h = (x + state) * 0.7071  # 1/sqrt(2)
        
        # Self-Attention
        res = h
        h_norm = self.norm1(h)
        attn_out, _ = self.attn(h_norm.unsqueeze(1), h_norm.unsqueeze(1), h_norm.unsqueeze(1))
        h = res + attn_out.squeeze(1)
        
        # FFN
        candidate = h + self.ffn(self.norm2(h))
        
        # 2. Gating (prevents unbounded state growth)
        z = torch.sigmoid(self.gate(torch.cat([candidate, state], dim=-1)))
        return z * candidate + (1 - z) * state

class DeepSapientHRM(nn.Module):
    """
    Full-Scale Gated HRM - AMP compatible.
    ~25M Parameters at dim=512, layers=4.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, k_step=3, num_heads=8, num_layers=4):
        super().__init__()
        self.k_step = k_step
        self.hidden_dim = hidden_dim
        self.embed = nn.Linear(input_dim, hidden_dim)
        
        # Gated blocks for stability
        self.L_blocks = nn.ModuleList([
            GatedRecurrentBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.H_blocks = nn.ModuleList([
            GatedRecurrentBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.head = nn.Linear(hidden_dim, output_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Smaller init std for gated architecture
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        b, seq, _ = x.size()
        
        # Zero init - gating handles stability
        h_L_states = [torch.zeros(b, self.hidden_dim, device=x.device, dtype=x.dtype) for _ in range(len(self.L_blocks))]
        h_H_states = [torch.zeros(b, self.hidden_dim, device=x.device, dtype=x.dtype) for _ in range(len(self.H_blocks))]
        
        for t in range(seq):
            current_input = self.embed(x[:, t, :])
            
            # System 2 (H-Module)
            if t % self.k_step == 0:
                h_input = h_L_states[-1].detach()
                for i, block in enumerate(self.H_blocks):
                    h_H_states[i] = block(h_input, h_H_states[i])
                    h_input = h_H_states[i]
            
            # System 1 (L-Module)
            l_input = current_input + h_H_states[-1]
            for i, block in enumerate(self.L_blocks):
                h_L_states[i] = block(l_input, h_L_states[i])
                l_input = h_L_states[i]

        return self.head(h_L_states[-1])

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ==========================================
# 2. ENV & PLANNER
# ==========================================
class DynamicGridEnv:
    def __init__(self, config):
        self.size = config["grid_size"]; self.n_dyn = config["n_dynamic"]; self.reset()
    def reset(self, seed=None):
        if seed is not None: np.random.seed(seed)
        self.static_map = np.zeros((self.size, self.size))
        self.static_map[0,0]=0; self.static_map[self.size-1, self.size-1]=0
        for _ in range(self.size):
            r,c=np.random.randint(0,self.size,2)
            if (r,c)!=(0,0) and (r,c)!=(self.size-1,self.size-1): self.static_map[r,c]=1.0
        self.dynamic_obs=[]
        for _ in range(self.n_dyn):
            while True:
                pos=np.random.randint(0,self.size,2).astype(float)
                if self.static_map[int(pos[0]),int(pos[1])]==0:
                    vel=np.random.randn(2); vel=vel/np.linalg.norm(vel)*0.7
                    self.dynamic_obs.append({'pos':pos,'vel':vel}); break
        self.agent_pos=np.array([0.,0.]); self.goal_pos=np.array([self.size-1.,self.size-1.])
        return self._get_obs()
    def step_physics(self):
        for o in self.dynamic_obs:
            o['pos']+=o['vel']
            for i in range(2):
                if o['pos'][i]<0 or o['pos'][i]>=self.size:
                    o['vel'][i]*=-1; o['pos'][i]=np.clip(o['pos'][i],0,self.size-0.01)
        return self._get_obs()
    def _get_obs(self): return np.array([o['pos'] for o in self.dynamic_obs])

class SpaceTimeAStar:
    def __init__(self, env, model, device):
        self.env=env; self.model=model; self.device=device; self.model.eval()
    def get_next_action(self, start, goal, obs_history):
        curr = torch.tensor(obs_history/self.env.size, dtype=torch.float32).to(self.device)
        # Match model dtype (FP32 for LSTM, BF16 for HRM)
        curr = curr.to(next(self.model.parameters()).dtype)
        future_obs = []
        with torch.no_grad():
            for _ in range(CONFIG['pred_horizon']):
                delta = self.model(curr)
                next_pos_norm = curr[:,-1,:] + delta.to(curr.dtype)  # Keep same dtype
                future_obs.append((next_pos_norm.float().cpu().numpy()*self.env.size))
                curr = torch.cat([curr[:,1:,:], next_pos_norm.unsqueeze(1)], dim=1)
        future_obs = np.array(future_obs)
        
        start_node = (int(start[0]),int(start[1]),0)
        pq=[(0,0,start_node)]; g_score={start_node:0}; came_from={}
        best_node, min_h = None, float('inf')
        while pq:
            f,g,curr_node = heapq.heappop(pq)
            r,c,t = curr_node
            if (r,c)==(int(goal[0]),int(goal[1])): return self.trace(came_from, curr_node, start_node)
            if t>=CONFIG['pred_horizon']-1:
                h = abs(r-goal[0])+abs(c-goal[1])
                if h<min_h: min_h=h; best_node=curr_node
                continue
            for dr,dc in [(0,1),(0,-1),(1,0),(-1,0),(0,0)]:
                nr,nc,nt = r+dr, c+dc, t+1
                if not (0<=nr<self.env.size and 0<=nc<self.env.size): continue
                if self.env.static_map[nr,nc]==1: continue
                if np.any(np.linalg.norm(future_obs[nt]-np.array([nr,nc]), axis=1)<1.0): continue
                new_g = g+1; neigh=(nr,nc,nt)
                if new_g < g_score.get(neigh, float('inf')):
                    g_score[neigh]=new_g
                    heapq.heappush(pq, (new_g+abs(nr-goal[0])+abs(nc-goal[1]), new_g, neigh))
                    came_from[neigh]=curr_node
        if best_node: return self.trace(came_from, best_node, start_node)
        return (int(start[0]),int(start[1]))
    def trace(self, came_from, curr, start):
        path=[]
        while curr in came_from: path.append(curr); curr=came_from[curr]
        return (path[-1][0], path[-1][1]) if path else (int(start[0]), int(start[1]))

# ==========================================
# 3. MODAL CLOUD WORKFLOW (SPLIT ARCHITECTURE)
# ==========================================

@app.function(image=image, volumes={"/data": vol}, cpu=1.0)
def collect_data_chunk(worker_id, n_episodes):
    """Generate trajectory data chunks (100 parallel workers)"""
    env = DynamicGridEnv(CONFIG)
    X, Y = [], []
    for _ in range(n_episodes):
        env.reset()
        hist = []
        for _ in range(70):
            hist.append(env.step_physics())
            if len(hist) > CONFIG['obs_history']:
                past = np.array(hist[-CONFIG['obs_history']-1:-1])/env.size
                future = np.array(hist[-1])/env.size
                prev = np.array(hist[-2])/env.size
                for j in range(env.n_dyn): X.append(past[:, j, :]); Y.append(future[j, :]-prev[j, :])
    os.makedirs(PATHS['data_dir'], exist_ok=True)
    fn = f"{PATHS['data_dir']}/chunk_{worker_id}.pt"
    torch.save((X, Y), fn)
    vol.commit()
    return fn


@app.function(image=image, volumes={"/data": vol}, cpu=1.0)
def check_cached_data() -> bool:
    """Check if merged data already exists on volume"""
    vol.reload()
    exists = os.path.exists(PATHS["merged_data"])
    print(f"Checking cache: {PATHS['merged_data']} exists = {exists}")
    return exists

@app.function(image=image, volumes={"/data": vol}, cpu=1.0)
def check_cached_models() -> dict:
    """Check if trained models already exist on volume"""
    vol.reload()
    return {
        "lstm": os.path.exists(PATHS["model_lstm"]),
        "hrm": os.path.exists(PATHS["model_hrm"])
    }


@app.function(image=image, volumes={"/data": vol}, cpu=8.0, memory=65536, timeout=1800)
def merge_chunks(chunk_files: List[str]) -> str:
    """Merge all chunks into single tensor file (8 CPUs, 64GB RAM)"""
    print(f"--> Merging {len(chunk_files)} chunks into single tensor...")
    
    def load_chunk(f):
        return torch.load(f, weights_only=False)
    
    # Parallel load with 8 workers
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(load_chunk, chunk_files))
    
    X_l, Y_l = [], []
    for x, y in results:
        X_l.extend(x); Y_l.extend(y)
    
    print(f"--> Converting {len(X_l)} samples to tensors...")
    X = torch.tensor(np.array(X_l), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_l), dtype=torch.float32)
    
    print(f"--> Saving merged data: X={X.shape}, Y={Y.shape}")
    torch.save((X, Y), PATHS["merged_data"])
    vol.commit()
    
    print(f"✅ Merged data saved to {PATHS['merged_data']}")
    return PATHS["merged_data"]


@app.function(image=image, gpu="A10", volumes={"/data": vol}, timeout=28800)
def train_lstm(merged_path: str):
    """Train LSTM baseline on A10 GPU (cost-efficient for small model) - 8hr timeout"""
    from tqdm import tqdm
    import time
    
    # Speed optimizations
    torch.backends.cudnn.benchmark = True
    
    print(f"--> Loading merged data from {merged_path}...")
    t0 = time.time()
    X, Y = torch.load(merged_path, weights_only=False)
    print(f"Dataset Size: {len(X)} samples (loaded in {time.time()-t0:.1f}s)")
    
    # DataLoader (optimized for ~17 core Modal container)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, Y), 
        batch_size=4096,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    print(f"Training LSTM Baseline (A10 + AMP, {len(dl)} batches/epoch)...")
    lstm = LSTMPredictor(2, CONFIG['hidden_dim'], 2).cuda()
    opt = optim.AdamW(lstm.parameters(), lr=CONFIG['lr'])
    scaler = GradScaler('cuda')
    
    lstm.train()
    total_epochs = 30
    for ep in range(total_epochs):
        ep_start = time.time()
        ep_loss = 0
        pbar = tqdm(dl, desc=f"LSTM Ep {ep}/{total_epochs}", leave=False)
        for bx, by in pbar:
            bx, by = bx.cuda(non_blocking=True), by.cuda(non_blocking=True)
            opt.zero_grad()
            with autocast('cuda'):
                pred = lstm(bx)
                loss = nn.MSELoss()(pred, by)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ep_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")
        
        ep_time = time.time() - ep_start
        avg_loss = ep_loss / len(dl)
        print(f"LSTM Ep {ep}/{total_epochs} | Loss: {avg_loss:.6f} | Time: {ep_time:.1f}s | ETA: {ep_time*(total_epochs-ep-1)/60:.1f}min")
    
    print(f"✅ LSTM training complete, saving to {PATHS['model_lstm']}")
    torch.save(lstm.state_dict(), PATHS["model_lstm"])
    vol.commit()


@app.function(image=image, gpu="B200", volumes={"/data": vol}, timeout=28800)
def train_hrm(merged_path: str):
    """Train DeepSapientHRM on single B200 GPU (no DataParallel overhead) - 8hr timeout"""
    from tqdm import tqdm
    import time
    
    # Speed optimizations
    torch.backends.cudnn.benchmark = True
    
    print(f"--> Loading merged data from {merged_path}...")
    t0 = time.time()
    X, Y = torch.load(merged_path, weights_only=False)
    print(f"Dataset Size: {len(X)} samples (loaded in {time.time()-t0:.1f}s)")
    
    # DataLoader (optimized for ~17 core Modal container)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, Y), 
        batch_size=4096,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    print(f"Training DeepSapientHRM (Single B200 + AMP, {len(dl)} batches/epoch)")
    print(f"  Config: Dim={CONFIG['hidden_dim']}, Layers={CONFIG['num_layers']}, Heads={CONFIG['num_heads']}")
    
    hrm = DeepSapientHRM(2, CONFIG['hidden_dim'], 2, 
                         k_step=CONFIG['hrm_k_step'],
                         num_layers=CONFIG['num_layers'],
                         num_heads=CONFIG['num_heads']).cuda()
    
    params = sum(p.numel() for p in hrm.parameters())
    print(f"--> HRM Parameter Count: {params:,}")
    
    opt = optim.AdamW(hrm.parameters(), lr=CONFIG['lr'], fused=True)
    total_epochs = CONFIG['epochs']
    scheduler = optim.lr_scheduler.OneCycleLR(opt, max_lr=CONFIG['lr'], steps_per_epoch=len(dl), epochs=total_epochs)
    
    hrm.train()
    print(f"\n{'='*60}")
    print(f"Starting HRM Training: {total_epochs} epochs, {len(dl)} batches/epoch")
    print(f"AMP ENABLED (BF16) - Gated architecture is stable")
    print(f"{'='*60}\n")
    
    for ep in range(total_epochs):
        ep_start = time.time()
        ep_loss = 0
        valid_batches = 0
        pbar = tqdm(dl, desc=f"HRM Ep {ep}/{total_epochs}", leave=False)
        for batch_idx, (bx, by) in enumerate(pbar):
            bx, by = bx.cuda(non_blocking=True), by.cuda(non_blocking=True)
            opt.zero_grad()
            
            # BF16 AMP - safe with gated architecture
            with autocast('cuda', dtype=torch.bfloat16):
                pred = hrm(bx)
                loss = nn.MSELoss()(pred, by)
            
            # Skip NaN batches (rare with gated architecture)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hrm.parameters(), 1.0)
            opt.step()
            scheduler.step()
            ep_loss += loss.item()
            valid_batches += 1
            
            # Update progress bar
            pbar.set_postfix(loss=f"{loss.item():.6f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            
            # Log every 500 batches
            if batch_idx > 0 and batch_idx % 500 == 0:
                print(f"  Ep {ep} Batch {batch_idx}/{len(dl)} | Loss: {loss.item():.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        ep_time = time.time() - ep_start
        avg_loss = ep_loss / max(valid_batches, 1)
        eta_min = ep_time * (total_epochs - ep - 1) / 60
        print(f"HRM Ep {ep}/{total_epochs} | Loss: {avg_loss:.6f} | Time: {ep_time:.1f}s | Valid: {valid_batches}/{len(dl)} | ETA: {eta_min:.1f}min")
    
    print(f"\n{'='*60}")
    print(f"✅ HRM training complete, saving to {PATHS['model_hrm']}")
    print(f"{'='*60}")
    torch.save(hrm.state_dict(), PATHS["model_hrm"])
    vol.commit()


@app.cls(image=image, gpu="A10", volumes={"/data": vol}, max_containers=20)
class Evaluator:
    """Evaluation on A10 GPUs (cost-efficient for inference)"""
    @modal.enter()
    def setup(self):
        self.device = torch.device("cuda")
        self.lstm = LSTMPredictor(2, CONFIG['hidden_dim'], 2).to(self.device)
        self.hrm = DeepSapientHRM(2, CONFIG['hidden_dim'], 2, 
                                  k_step=CONFIG['hrm_k_step'],
                                  num_layers=CONFIG['num_layers'],
                                  num_heads=CONFIG['num_heads']).to(self.device)
        self.lstm.load_state_dict(torch.load(PATHS["model_lstm"], weights_only=True))
        self.hrm.load_state_dict(torch.load(PATHS["model_hrm"], weights_only=True))

    @modal.method()
    def run_episode(self, seed):
        env = DynamicGridEnv(CONFIG); env.reset(seed=seed+100)
        def sim(model):
            e = DynamicGridEnv(CONFIG); e.reset(seed=seed+100)
            p = SpaceTimeAStar(e, model, self.device)
            hist = [e.step_physics() for _ in range(CONFIG['obs_history'])]
            for _ in range(80):
                h_np = np.array(hist[-CONFIG['obs_history']:]).transpose(1,0,2)
                nr, nc = p.get_next_action(e.agent_pos, e.goal_pos, h_np)
                e.agent_pos = np.array([float(nr), float(nc)])
                obs = e.step_physics(); hist.append(obs)
                if np.linalg.norm(e.agent_pos-e.goal_pos)<0.5: return 1
                if e.static_map[int(nr),int(nc)]==1: return 0
                if np.any(np.linalg.norm(obs-e.agent_pos, axis=1)<0.8): return 0
            return 0
        return {"L": sim(self.lstm), "H": sim(self.hrm)}


@app.local_entrypoint()
def main():
    print("🚀 Launching FULL-SCALE HRM Research (Split Architecture v4 - Gated + AMP)")
    print("   - Data Collection: 100x CPU workers (skipped if cached)")
    print("   - Merge: 8 CPUs, 64GB RAM (skipped if cached)")
    print("   - LSTM Training: A10 GPU (8hr timeout)")
    print("   - HRM Training: Single B200 (8hr timeout)")
    print("   - Evaluation: 20x A10 GPUs")
    
    # Check if models already exist (skip training if so)
    print("\n🔍 Checking for cached models...")
    model_cache = check_cached_models.remote()
    
    if model_cache["lstm"] and model_cache["hrm"]:
        print("✅ Found cached models! Skipping all training.")
    else:
        # Check if merged data already exists (skip expensive collection/merge)
        print("\n🔍 Checking for cached data...")
        cached = check_cached_data.remote()
        
        if cached:
            print("✅ Found cached merged data! Skipping collection & merge.")
            merged_path = PATHS["merged_data"]
        else:
            print("❌ No cached data found. Running full pipeline...")
            
            # Step 1: Generate 60,000 Episodes (100 parallel workers)
            print("\n📦 Step 1: Collecting data...")
            chunks = list(collect_data_chunk.map(range(100), kwargs={'n_episodes': 600}))
            
            # Step 2: Merge all chunks into single tensor (fast subsequent loads)
            print("\n🔗 Step 2: Merging chunks...")
            merged_path = merge_chunks.remote(chunks)
        
        # Step 3: Train only what's needed
        print("\n🏋️ Step 3: Training models...")
        handles = []
        
        if not model_cache["lstm"]:
            print("   🏃 Training LSTM...")
            handles.append(("LSTM", train_lstm.spawn(merged_path)))
        else:
            print("   ✅ LSTM already trained, skipping.")
        
        if not model_cache["hrm"]:
            print("   🏃 Training HRM (20 epochs, Gated + BF16 AMP)...")
            handles.append(("HRM", train_hrm.spawn(merged_path)))
        else:
            print("   ✅ HRM already trained, skipping.")
        
        for name, handle in handles:
            handle.get()
            print(f"   ✅ {name} training complete")
    
    # Step 4: Evaluate both models
    print("\n📊 Step 4: Running evaluation...")
    res = list(Evaluator().run_episode.map(range(CONFIG['eval_episodes'])))
    l = sum(r['L'] for r in res); h = sum(r['H'] for r in res)
    
    print("\n" + "="*50)
    print("📈 FINAL RESULTS")
    print("="*50)
    print(f"LSTM: {l}/{len(res)} ({l/len(res)*100:.1f}%)")
    print(f"HRM:  {h}/{len(res)} ({h/len(res)*100:.1f}%)")