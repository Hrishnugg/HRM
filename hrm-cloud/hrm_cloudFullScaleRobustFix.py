import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import heapq
import os
from typing import List
from concurrent.futures import ProcessPoolExecutor

# Use Torch 2.4+ for native BFloat16 optimization
app = modal.App("hrm-b200-final-robust")
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch>=2.4.0", "numpy", "gymnasium", "tqdm")
)

vol = modal.Volume.from_name("hrm-robust-vol", create_if_missing=True)

CONFIG = {
    # Full Scale Config
    "hidden_dim": 512, "num_layers": 4, "num_heads": 8,
    "data_episodes": 60000, "batch_size": 8192,  # Doubled for better GPU util
    "hrm_k_step": 2, "lr": 3e-4, "epochs": 20,  # Reduced from 50 - model converges by ep 17
    "grid_size": 20, "n_static": 12, "n_dynamic": 6,
    "obs_history": 20, "pred_horizon": 20, "eval_episodes": 50
}

PATHS = {
    "data_dir": "/data/episodes_robust",
    "merged_data": "/data/merged_robust.pt",
    "model_lstm": "/data/lstm_robust.pt",
    "model_hrm": "/data/hrm_robust.pt"
}

# ==========================================
# CACHE CHECKS
# ==========================================
@app.function(image=image, volumes={"/data": vol}, cpu=1.0)
def check_cached_data() -> bool:
    vol.reload()
    return os.path.exists(PATHS["merged_data"])

@app.function(image=image, volumes={"/data": vol}, cpu=1.0)
def check_cached_models() -> dict:
    vol.reload()
    return {
        "lstm": os.path.exists(PATHS["model_lstm"]),
        "hrm": os.path.exists(PATHS["model_hrm"])
    }

# ==========================================
# 1. ROBUST ARCHITECTURE
# ==========================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = dim ** -0.5
        self.g = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # Upcast to FP32 for stability, then downcast
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
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, int(dim * 2.6))
        
        # Gating to bound energy (Sigmoid 0-1)
        self.gate = nn.Linear(dim * 2, dim)
        
        # CRITICAL FIX: Removed self.out_norm
        # We must allow the state to decay to zero without the norm forcing it back up.
        
    def forward(self, x, state):
        # CRITICAL FIX: Variance Scaling
        # Dampens the "Infinite Accumulator" (x+state)
        h = (x + state) * 0.7071
        
        res = h
        h_norm = self.norm1(h)
        attn_out, _ = self.attn(h_norm.unsqueeze(1), h_norm.unsqueeze(1), h_norm.unsqueeze(1))
        h = res + attn_out.squeeze(1)
        
        candidate = h + self.ffn(self.norm2(h))
        
        # Gated Update
        z = torch.sigmoid(self.gate(torch.cat([candidate, state], dim=-1)))
        new_state = z * candidate + (1 - z) * state
        
        return new_state

class DeepSapientHRM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, k_step=3, num_heads=8, num_layers=4):
        super().__init__()
        self.k_step = k_step
        self.hidden_dim = hidden_dim
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.L_blocks = nn.ModuleList([GatedRecurrentBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.H_blocks = nn.ModuleList([GatedRecurrentBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_dim, output_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Reduced std for stability
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        b, seq, _ = x.size()
        # Init with small noise to avoid initial singularity
        h_L = [torch.randn(b, self.hidden_dim, device=x.device, dtype=x.dtype)*0.01 for _ in range(len(self.L_blocks))]
        h_H = [torch.randn(b, self.hidden_dim, device=x.device, dtype=x.dtype)*0.01 for _ in range(len(self.H_blocks))]
        
        for t in range(seq):
            curr_in = self.embed(x[:, t, :])
            if t % self.k_step == 0:
                h_in = h_L[-1].detach()
                for i, blk in enumerate(self.H_blocks):
                    h_H[i] = blk(h_in, h_H[i])
                    h_in = h_H[i]
            l_in = curr_in + h_H[-1]
            for i, blk in enumerate(self.L_blocks):
                h_L[i] = blk(l_in, h_L[i])
                l_in = h_L[i]
        return self.head(h_L[-1])

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ==========================================
# 2. SANITIZED ENV
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
                    # FIX: Prevent Zero Velocity Initialization (NaN Source)
                    vel = np.random.randn(2)
                    norm = np.linalg.norm(vel)
                    if norm < 1e-4: vel = np.array([0.1, 0.1])
                    else: vel = vel / norm * 0.7
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
        future_obs = []
        with torch.no_grad():
            for _ in range(CONFIG['pred_horizon']):
                # curr shape: [n_dyn, seq, 2] - already batched per obstacle
                delta = self.model(curr)  # Model expects [batch, seq, input_dim]
                next_pos_norm = curr[:,-1,:] + delta.to(curr.dtype)  # Keep same dtype as curr
                future_obs.append((next_pos_norm.float().cpu().numpy()*self.env.size))
                curr = torch.cat([curr[:,1:,:], next_pos_norm.unsqueeze(1)], dim=1)
        future_obs = np.array(future_obs)
        start_node = (int(start[0]), int(start[1]), 0)
        pq=[(0, 0, start_node)]; g_score={start_node: 0}; came_from={}
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
        return (int(start[0]), int(start[1]))
    
    def trace(self, came_from, curr, start):
        path=[]
        while curr in came_from: path.append(curr); curr=came_from[curr]
        return (path[-1][0], path[-1][1]) if path else (int(start[0]), int(start[1]))

# ==========================================
# 3. DATA SANITIZATION PIPELINE
# ==========================================
@app.function(image=image, volumes={"/data": vol}, cpu=1.0)
def collect_data_chunk(worker_id, n_episodes):
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
    torch.save((X, Y), fn); vol.commit()
    return fn

def _load_and_convert(f):
    """Top-level function for ProcessPoolExecutor (must be picklable)"""
    import torch
    import numpy as np
    x, y = torch.load(f, weights_only=False)
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

@app.function(image=image, volumes={"/data": vol}, cpu=8.0, memory=65536, timeout=1800)
def merge_and_sanitize(chunk_files: List[str]):
    print(f"--> Merging {len(chunk_files)} chunks with 8 processes...")
    
    # ProcessPoolExecutor bypasses GIL for true parallelism
    with ProcessPoolExecutor(8) as exe:
        results = list(exe.map(_load_and_convert, chunk_files))
    
    print("--> Concatenating arrays...")
    # Fast numpy concatenation (much faster than list extend + np.array)
    X = np.concatenate([r[0] for r in results], axis=0)
    Y = np.concatenate([r[1] for r in results], axis=0)
    
    print(f"--> Total samples: {len(X):,}")
    
    # --- SANITIZER ---
    print("--> Scanning for NaNs in Dataset...")
    valid_mask = ~np.isnan(X).any(axis=(1,2)) & ~np.isnan(Y).any(axis=1)
    
    if not valid_mask.all():
        n_dropped = (~valid_mask).sum()
        print(f"⚠️ FOUND {n_dropped} CORRUPTED SAMPLES! Dropping them.")
        X = X[valid_mask]
        Y = Y[valid_mask]
    else:
        print("✅ Data integrity check passed.")
    
    # Convert to tensor at the end
    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y)
        
    torch.save((X_t, Y_t), PATHS["merged_data"])
    vol.commit()
    print(f"✅ Saved {len(X_t):,} samples to {PATHS['merged_data']}")
    return PATHS["merged_data"]

@app.function(image=image, gpu="A10", volumes={"/data": vol}, timeout=28800)
def train_lstm(merged_path: str):
    from tqdm import tqdm
    print("Training LSTM (A10)...")
    X, Y = torch.load(merged_path, weights_only=False)
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y), batch_size=4096, shuffle=True, num_workers=16, pin_memory=True)
    lstm = LSTMPredictor(2, CONFIG['hidden_dim'], 2).cuda()
    opt = optim.AdamW(lstm.parameters(), lr=CONFIG['lr'])
    lstm.train()
    for ep in range(30):
        pbar = tqdm(dl, desc=f"LSTM Ep {ep}", leave=False)
        for bx, by in pbar:
            opt.zero_grad()
            loss = nn.MSELoss()(lstm(bx.cuda()), by.cuda())
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=f"{loss.item():.6f}")
        print(f"LSTM Ep {ep}/30 complete")
    torch.save(lstm.state_dict(), PATHS["model_lstm"]); vol.commit()
    print("✅ LSTM training complete")

@app.function(image=image, gpu="B200", volumes={"/data": vol}, timeout=43200)  # 12 hours
def train_hrm(merged_path: str):
    from tqdm import tqdm
    print("Training HRM (B200 BFloat16) - batch=8192, workers=16...")
    X, Y = torch.load(merged_path, weights_only=False)
    
    # Final sanity check
    if torch.isnan(X).any() or torch.isnan(Y).any():
        raise ValueError("Data still contains NaNs after sanitization!")

    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, Y), 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=16,  # Match system limit (~17 max)
        pin_memory=True,
        prefetch_factor=4,  # Prefetch more batches
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    hrm = DeepSapientHRM(2, CONFIG['hidden_dim'], 2, 
                         k_step=CONFIG['hrm_k_step'],
                         num_layers=CONFIG['num_layers'],
                         num_heads=CONFIG['num_heads']).cuda().to(torch.bfloat16)
    
    opt = optim.AdamW(hrm.parameters(), lr=CONFIG['lr'], fused=True)
    scheduler = optim.lr_scheduler.OneCycleLR(opt, max_lr=CONFIG['lr'], steps_per_epoch=len(dl), epochs=CONFIG['epochs'])
    
    hrm.train()
    for ep in range(CONFIG['epochs']):
        ep_loss = 0
        valid_batches = 0
        pbar = tqdm(dl, desc=f"Ep {ep}", leave=False)
        for bx, by in pbar:
            bx, by = bx.cuda(non_blocking=True).to(torch.bfloat16), by.cuda(non_blocking=True).to(torch.bfloat16)
            opt.zero_grad()
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred = hrm(bx)
                loss = nn.MSELoss()(pred, by)
            
            # 1. Forward Guard
            if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️ Forward NaN. Skipping.")
                continue
                
            loss.backward()
            
            # 2. Gradient Guard
            grad_norm = torch.nn.utils.clip_grad_norm_(hrm.parameters(), 1.0)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print("⚠️ Gradient NaN. Skipping Step.")
                opt.zero_grad() 
                continue

            opt.step()
            scheduler.step()
            ep_loss += loss.item()
            valid_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")
            
        print(f"HRM Ep {ep} | Avg Loss: {ep_loss/max(valid_batches, 1):.6f}")

    torch.save(hrm.state_dict(), PATHS["model_hrm"]); vol.commit()

@app.cls(image=image, gpu="A10", volumes={"/data": vol}, max_containers=20)
class Evaluator:
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
        self.hrm = self.hrm.bfloat16() # Inference in BF16

    @modal.method()
    def run_episode(self, seed):
        env = DynamicGridEnv(CONFIG); env.reset(seed=seed+100)
        def sim(model, dtype=torch.float32):
            e = DynamicGridEnv(CONFIG); e.reset(seed=seed+100)
            p = SpaceTimeAStar(e, model, self.device)
            hist = [e.step_physics() for _ in range(CONFIG['obs_history'])]
            for _ in range(80):
                h_np = np.array(hist[-CONFIG['obs_history']:]).transpose(1,0,2)
                if dtype == torch.bfloat16: nr, nc = p.get_next_action(e.agent_pos, e.goal_pos, h_np)
                else: nr, nc = p.get_next_action(e.agent_pos, e.goal_pos, h_np)
                e.agent_pos = np.array([float(nr), float(nc)])
                obs = e.step_physics(); hist.append(obs)
                if np.linalg.norm(e.agent_pos-e.goal_pos)<0.5: return 1
                if e.static_map[int(nr),int(nc)]==1: return 0
                if np.any(np.linalg.norm(obs-e.agent_pos, axis=1)<0.8): return 0
            return 0
        return {"L": sim(self.lstm, torch.float32), "H": sim(self.hrm, torch.bfloat16)}

@app.local_entrypoint()
def main():
    print("🚀 Launching ROBUST HRM Research (27M params, 20 epochs)")
    
    # Check if models already exist (skip training if so)
    model_cache = check_cached_models.remote()
    if model_cache["lstm"] and model_cache["hrm"]:
        print("✅ Found cached models (lstm_robust.pt + hrm_robust.pt). Skipping training!")
    else:
        # Check if data exists
        if check_cached_data.remote():
            print("✅ Found cached data (merged_robust.pt). Reusing.")
            merged = PATHS["merged_data"]
        else:
            print("📦 Collecting and sanitizing data...")
            chunks = list(collect_data_chunk.map(range(100), kwargs={'n_episodes': 600}))
            merged = merge_and_sanitize.remote(chunks)
        
        # Train only what's needed
        handles = []
        if not model_cache["lstm"]:
            print("🏃 Training LSTM...")
            handles.append(("LSTM", train_lstm.spawn(merged)))
        else:
            print("✅ LSTM already trained, skipping.")
        
        if not model_cache["hrm"]:
            print("🏃 Training HRM (27M params, 20 epochs)...")
            handles.append(("HRM", train_hrm.spawn(merged)))
        else:
            print("✅ HRM already trained, skipping.")
        
        for name, handle in handles:
            print(f"⏳ Waiting for {name}...")
            handle.get()
            print(f"✅ {name} complete!")
    
    print("\n📊 Evaluation...")
    res = list(Evaluator().run_episode.map(range(CONFIG['eval_episodes'])))
    l = sum(r['L'] for r in res); h = sum(r['H'] for r in res)
    print(f"LSTM: {l}/{len(res)} ({l/len(res)*100:.1f}%)")
    print(f"HRM:  {h}/{len(res)} ({h/len(res)*100:.1f}%)")