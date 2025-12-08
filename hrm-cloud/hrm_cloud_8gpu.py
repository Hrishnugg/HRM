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

app = modal.App("hrm-b200-8gpu-cluster")

# Torch 2.4+ required for H100/B200 optimizations
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch>=2.4.0", "numpy", "gymnasium", "tqdm")
)

# UNIQUE VOLUME to avoid conflicts with other runs
vol = modal.Volume.from_name("hrm-8gpu-vol", create_if_missing=True)

CONFIG = {
    # --- FULL SCALE ARCHITECTURE ---
    "hidden_dim": 512,
    "num_heads": 8,
    "num_layers": 4,
    
    # --- MULTI-GPU TRAINING PARAMS ---
    "gpu_count": 8,       
    "per_gpu_batch": 2048, # 2048 × 8 GPUs = 16384 Global Batch (saturates GPU compute)
    "lr": 6e-4,            # Sqrt scaling: 3e-4 × √4 (safer for recurrent models)
    "epochs": 30,         # 30 epochs as requested
    
    # --- DATA & ENV ---
    "data_episodes": 60000,
    "hrm_k_step": 2,
    "grid_size": 20, "n_static": 12, "n_dynamic": 6,
    "obs_history": 20, "pred_horizon": 20, "eval_episodes": 50
}

# UNIQUE PATHS to avoid conflicts
PATHS = {
    "data_dir": "/data/episodes_8gpu",
    "merged_data": "/data/merged_8gpu.pt",
    "model_lstm": "/data/lstm_8gpu.pt",
    "model_hrm": "/data/hrm_8gpu.pt"
}

# ==========================================
# 1. ROBUST ARCHITECTURE (BF16 + Gated)
# ==========================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.eps=eps; self.scale=dim**-0.5; self.g=nn.Parameter(torch.ones(dim))
    def forward(self, x):
        x_dt=x.dtype; x_f32=x.float(); n=x_f32.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return ((x_f32/n)*self.scale*self.g).to(x_dt)

class SwiGLU(nn.Module):
    def __init__(self, dim, h_dim):
        super().__init__(); self.w1=nn.Linear(dim, h_dim, bias=False); self.w2=nn.Linear(dim, h_dim, bias=False); self.w3=nn.Linear(h_dim, dim, bias=False)
    def forward(self, x): return self.w3(F.silu(self.w1(x))*self.w2(x))

class GatedRecurrentBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1=RMSNorm(dim); self.attn=nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2=RMSNorm(dim); self.ffn=SwiGLU(dim, int(dim*2.6)); self.gate=nn.Linear(dim*2, dim)
    def forward(self, x, state):
        # 1. Variance Scaling (Critical for Deep Recurrence)
        h=(x+state)*0.7071 
        res=h; h_norm=self.norm1(h)
        attn_out,_=self.attn(h_norm.unsqueeze(1), h_norm.unsqueeze(1), h_norm.unsqueeze(1))
        h=res+attn_out.squeeze(1)
        candidate=h+self.ffn(self.norm2(h))
        # 2. Gating
        z=torch.sigmoid(self.gate(torch.cat([candidate, state], dim=-1)))
        return z*candidate+(1-z)*state

class DeepSapientHRM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, k_step=3, num_heads=8, num_layers=4):
        super().__init__()
        self.k_step=k_step; self.hidden_dim=hidden_dim; self.embed=nn.Linear(input_dim, hidden_dim)
        self.L_blocks=nn.ModuleList([GatedRecurrentBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.H_blocks=nn.ModuleList([GatedRecurrentBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.head=nn.Linear(hidden_dim, output_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None: torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        b, seq, _ = x.size()
        h_L = [torch.zeros(b, self.hidden_dim, device=x.device, dtype=x.dtype) for _ in range(len(self.L_blocks))]
        h_H = [torch.zeros(b, self.hidden_dim, device=x.device, dtype=x.dtype) for _ in range(len(self.H_blocks))]
        for t in range(seq):
            curr_in = self.embed(x[:, t, :])
            if t % self.k_step == 0:
                h_in = h_L[-1].detach()
                for i, blk in enumerate(self.H_blocks): h_H[i]=blk(h_in, h_H[i]); h_in=h_H[i]
            l_in = curr_in + h_H[-1]
            for i, blk in enumerate(self.L_blocks): h_L[i]=blk(l_in, h_L[i]); l_in=h_L[i]
        return self.head(h_L[-1])

class LSTMPredictor(nn.Module):
    """LSTM baseline for comparison"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ==========================================
# 2. ENV & PLANNER (from Split)
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
    """A* planner with dtype-safe model inference"""
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
# 3. DDP TRAINING LOGIC
# ==========================================

def ddp_setup(rank, world_size):
    """Initializes the distributed backend"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def ddp_cleanup():
    dist.destroy_process_group()

def train_worker(rank, world_size, merged_path, config):
    """The training loop running on EACH GPU"""
    from tqdm import tqdm
    import time
    
    ddp_setup(rank, world_size)
    
    # 1. Load Data
    if rank == 0: print(f"Rank {rank}: Loading Dataset...")
    X, Y = torch.load(merged_path, weights_only=False, map_location='cpu')
    
    dataset = torch.utils.data.TensorDataset(X, Y)
    
    # 2. Distributed Sampler
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
    
    # 3. Model Setup
    model = DeepSapientHRM(2, config['hidden_dim'], 2, 
                           k_step=config['hrm_k_step'],
                           num_layers=config['num_layers'],
                           num_heads=config['num_heads']).to(rank).to(torch.bfloat16)
    
    model = DDP(model, device_ids=[rank])
    
    # 4. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], fused=True)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['lr'], 
                                              steps_per_epoch=len(loader), 
                                              epochs=config['epochs'])
    
    if rank == 0:
        print(f"🚀 DDP Started. Global Batch: {config['per_gpu_batch']*world_size}")
        print(f"--> Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        print(f"--> {len(loader)} batches/epoch, {config['epochs']} epochs")
    
    # 5. Training with progress logging
    model.train()
    for ep in range(config['epochs']):
        sampler.set_epoch(ep)
        ep_start = time.time()
        
        ep_loss = torch.zeros(1).to(rank)
        steps = 0
        
        # Only rank 0 shows progress bar
        pbar = tqdm(loader, desc=f"Ep {ep+1}/{config['epochs']}", disable=(rank != 0), leave=False)
        for batch_idx, (bx, by) in enumerate(pbar):
            bx = bx.to(rank, non_blocking=True).to(torch.bfloat16)
            by = by.to(rank, non_blocking=True).to(torch.bfloat16)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred = model(bx)
                loss = nn.MSELoss()(pred, by)
            
            if torch.isnan(loss): continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            ep_loss += loss
            steps += 1
            
            # Update progress bar (rank 0 only)
            if rank == 0:
                pbar.set_postfix(loss=f"{loss.item():.6f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            
            # Log every 500 batches (rank 0 only)
            if rank == 0 and batch_idx > 0 and batch_idx % 500 == 0:
                print(f"  Ep {ep+1} Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.6f}")
            
        # Aggregate Loss for Logging
        dist.all_reduce(ep_loss, op=dist.ReduceOp.SUM)
        if rank == 0:
            ep_time = time.time() - ep_start
            avg_loss = ep_loss.item() / (world_size * max(steps, 1))
            eta_min = ep_time * (config['epochs'] - ep - 1) / 60
            print(f"Epoch {ep+1}/{config['epochs']} | Loss: {avg_loss:.6f} | Time: {ep_time:.1f}s | ETA: {eta_min:.1f}min")
            
    # 6. Save (Rank 0 only)
    if rank == 0:
        print("Saving DDP Model...")
        torch.save(model.module.state_dict(), PATHS['model_hrm'])
    
    ddp_cleanup()

# ==========================================
# 4. MODAL FUNCTIONS
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
    from torch.amp import autocast, GradScaler
    import time
    
    torch.backends.cudnn.benchmark = True
    
    print(f"--> Loading merged data from {merged_path}...")
    t0 = time.time()
    X, Y = torch.load(merged_path, weights_only=False)
    print(f"Dataset Size: {len(X)} samples (loaded in {time.time()-t0:.1f}s)")
    
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

@app.function(
    image=image, 
    gpu="H100:8", # Request 8x H100s
    volumes={"/data": vol}, 
    timeout=28800,  # 8 hours
    memory=65536
)
def train_hrm_multigpu(merged_path):
    """Train HRM with 8-GPU DDP - 8hr timeout"""
    print("🚀 Launching 8-GPU Distributed Training...")
    
    world_size = 8
    
    # Spawn 8 processes within the container
    mp.spawn(
        train_worker,
        args=(world_size, merged_path, CONFIG),
        nprocs=world_size,
        join=True
    )
    
    vol.commit()
    print("✅ Training Complete.")

# ==========================================
# 5. EVALUATOR CLASS (from Split)
# ==========================================

@app.cls(image=image, gpu="A10", volumes={"/data": vol}, max_containers=20)
class Evaluator:
    """Evaluation on A10 GPUs (cost-efficient for inference)"""
    @modal.enter()
    def setup(self):
        self.device = torch.device("cuda")
        # LSTM stays in FP32
        self.lstm = LSTMPredictor(2, CONFIG['hidden_dim'], 2).to(self.device)
        # HRM in BF16 (matches training)
        self.hrm = DeepSapientHRM(2, CONFIG['hidden_dim'], 2, 
                                  k_step=CONFIG['hrm_k_step'],
                                  num_layers=CONFIG['num_layers'],
                                  num_heads=CONFIG['num_heads']).to(self.device).to(torch.bfloat16)
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

# ==========================================
# 6. MAIN ENTRYPOINT
# ==========================================

@app.local_entrypoint()
def main():
    print("🚀 Launching 8-GPU DDP HRM Training (30 epochs)")
    print("   - Data Collection: 100x CPU workers (skipped if cached)")
    print("   - Merge: 8 CPUs, 64GB RAM (skipped if cached)")
    print("   - LSTM Training: A10 GPU (8hr timeout)")
    print("   - HRM Training: 8x H100 DDP (8hr timeout)")
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
            print("   🏃 Training HRM (30 epochs, 8x H100 DDP)...")
            handles.append(("HRM", train_hrm_multigpu.spawn(merged_path)))
        else:
            print("   ✅ HRM already trained, skipping.")
        
        # WAIT for all training to complete
        for name, handle in handles:
            handle.get()  # Blocks until complete
            print(f"   ✅ {name} training complete")
    
    # Step 4: Evaluate both models
    print("\n📊 Step 4: Running evaluation...")
    res = list(Evaluator().run_episode.map(range(CONFIG['eval_episodes'])))
    l = sum(r['L'] for r in res); h = sum(r['H'] for r in res)
    
    print("\n" + "="*50)
    print("📈 FINAL RESULTS (8-GPU DDP)")
    print("="*50)
    print(f"LSTM: {l}/{len(res)} ({l/len(res)*100:.1f}%)")
    print(f"HRM:  {h}/{len(res)} ({h/len(res)*100:.1f}%)")
