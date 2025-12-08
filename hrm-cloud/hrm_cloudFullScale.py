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

app = modal.App("hrm-b200-full-scale")

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
    "lr": 1e-4,          # Lower LR is required for larger/deeper models
    "epochs": 50,        
    
    # Env Params
    "grid_size": 20, "n_static": 12, "n_dynamic": 6,
    "obs_history": 20, "pred_horizon": 20, "eval_episodes": 50
}

PATHS = {
    "data_dir": "/data/episodes_full",
    "model_lstm": "/data/lstm_full.pt",
    "model_hrm": "/data/hrm_full.pt"
}

# ==========================================
# 1. DEEP SAPIENT HRM ARCHITECTURE
# ==========================================
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.g = nn.Parameter(torch.ones(dim))
    def forward(self, x): return F.normalize(x, dim=-1) * self.scale * self.g

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
    def forward(self, x): return self.w3(F.silu(self.w1(x)) * self.w2(x))

class RecurrentTransformerBlock(nn.Module):
    """
    Standard Pre-Norm Recurrent Block used in Deep HRM
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        # B200 Native FlashAttention via SDPA
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, int(dim * 2.6))
        
    def forward(self, x, state):
        # x: Input Context, state: Recurrent Memory
        h = x + state
        
        # Self-Attention on State (Recurrent Context)
        res = h
        h_norm = self.norm1(h)
        attn_out, _ = self.attn(h_norm.unsqueeze(1), h_norm.unsqueeze(1), h_norm.unsqueeze(1))
        h = res + attn_out.squeeze(1)
        
        # FFN
        return h + self.ffn(self.norm2(h))

class DeepSapientHRM(nn.Module):
    """
    The Full-Scale Implementation.
    Stacks multiple Transformer blocks for deep reasoning.
    ~27M Parameters at dim=512, layers=4.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, k_step=3, num_heads=8, num_layers=4):
        super().__init__()
        self.k_step = k_step
        self.hidden_dim = hidden_dim
        self.embed = nn.Linear(input_dim, hidden_dim)
        
        # DEEP STACKS (ModuleList allows PyTorch to register them)
        self.L_blocks = nn.ModuleList([
            RecurrentTransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.H_blocks = nn.ModuleList([
            RecurrentTransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.head = nn.Linear(hidden_dim, output_dim)
        
        # Deep transformers need explicit initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        b, seq, _ = x.size()
        
        # State is now a LIST of states (one per layer)
        # L_states[i] is the recurrent state for L-Block i
        h_L_states = [torch.zeros(b, self.hidden_dim, device=x.device) for _ in range(len(self.L_blocks))]
        h_H_states = [torch.zeros(b, self.hidden_dim, device=x.device) for _ in range(len(self.H_blocks))]
        
        for t in range(seq):
            # Input Embedding
            current_input = self.embed(x[:, t, :])
            
            # --- System 2 (H-Module) ---
            if t % self.k_step == 0:
                # Top-level input to H is the detached state from L's last layer
                # (Bottom-Up Abstraction from previous timestep)
                h_input = h_L_states[-1].detach() 
                
                # Propagate through H layers
                for i, block in enumerate(self.H_blocks):
                    h_H_states[i] = block(h_input, h_H_states[i])
                    h_input = h_H_states[i] # Output of layer i is input to layer i+1
            
            # --- System 1 (L-Module) ---
            # Input to L is Embed + Top-Down Context from H's last layer
            l_input = current_input + h_H_states[-1]
            
            # Propagate through L layers
            for i, block in enumerate(self.L_blocks):
                h_L_states[i] = block(l_input, h_L_states[i])
                l_input = h_L_states[i]

        # Prediction from the final L layer
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
        future_obs = []
        with torch.no_grad():
            for _ in range(CONFIG['pred_horizon']):
                next_pos_norm = curr[:,-1,:] + self.model(curr)
                future_obs.append((next_pos_norm.cpu().numpy()*self.env.size))
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
# 3. MODAL CLOUD WORKFLOW
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
    torch.save((X, Y), fn)
    vol.commit()
    return fn

@app.function(image=image, gpu="B200:4", volumes={"/data": vol}, timeout=14400)  # 4 GPUs, 4hr timeout
def train_models(chunk_files: List[str]):
    print(f"--> Loading Data from {len(chunk_files)} chunks (parallel)...")
    
    def load_chunk(f):
        return torch.load(f, weights_only=False)
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(load_chunk, chunk_files))
    
    X_l, Y_l = [], []
    for x, y in results:
        X_l.extend(x); Y_l.extend(y)
    
    X = torch.tensor(np.array(X_l), dtype=torch.float32)  # Keep on CPU
    Y = torch.tensor(np.array(Y_l), dtype=torch.float32)  # Keep on CPU
    print(f"Dataset Size: {len(X)} samples")

    # Optimized DataLoader
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, Y), 
        batch_size=CONFIG['batch_size'], 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    print("Training LSTM Baseline (4-GPU + AMP)...")
    lstm = nn.DataParallel(LSTMPredictor(2, CONFIG['hidden_dim'], 2).cuda())
    opt = optim.AdamW(lstm.parameters(), lr=CONFIG['lr'], fused=True)
    scaler_lstm = GradScaler('cuda')
    lstm.train()
    for ep in range(30):
        for bx, by in dl:
            bx, by = bx.cuda(non_blocking=True), by.cuda(non_blocking=True)
            opt.zero_grad()
            with autocast('cuda'):
                pred = lstm(bx)
                loss = nn.MSELoss()(pred, by)
            scaler_lstm.scale(loss).backward()
            scaler_lstm.step(opt)
            scaler_lstm.update()
        if ep % 10 == 0: print(f"LSTM Ep {ep}/30")

    # --- FULL SCALE HRM TRAINING ---
    print(f"Training FULL HRM (4-GPU + AMP, Dim={CONFIG['hidden_dim']}, Layers={CONFIG['num_layers']})...")
    
    hrm = nn.DataParallel(DeepSapientHRM(2, CONFIG['hidden_dim'], 2, 
                         k_step=CONFIG['hrm_k_step'],
                         num_layers=CONFIG['num_layers'],
                         num_heads=CONFIG['num_heads']).cuda())
                         
    # Calculate params for verification
    params = sum(p.numel() for p in hrm.parameters())
    print(f"--> HRM Parameter Count: {params:,}")
    
    opt = optim.AdamW(hrm.parameters(), lr=CONFIG['lr'], fused=True)
    scaler_hrm = GradScaler('cuda')
    scheduler = optim.lr_scheduler.OneCycleLR(opt, max_lr=CONFIG['lr'], steps_per_epoch=len(dl), epochs=CONFIG['epochs'])
    
    hrm.train()
    for ep in range(CONFIG['epochs']):
        ep_loss = 0
        for bx, by in dl:
            bx, by = bx.cuda(non_blocking=True), by.cuda(non_blocking=True)
            opt.zero_grad()
            with autocast('cuda'):
                pred = hrm(bx)
                loss = nn.MSELoss()(pred, by)
            scaler_hrm.scale(loss).backward()
            scaler_hrm.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(hrm.parameters(), 1.0)
            scaler_hrm.step(opt)
            scaler_hrm.update()
            scheduler.step()
            ep_loss += loss.item()
        if ep % 5 == 0: print(f"HRM Ep {ep}: Loss {ep_loss/len(dl):.6f}")

    # Save underlying model (unwrap DataParallel)
    torch.save(lstm.module.state_dict(), PATHS["model_lstm"])
    torch.save(hrm.module.state_dict(), PATHS["model_hrm"])
    vol.commit()

@app.cls(image=image, gpu="B200", volumes={"/data": vol}, max_containers=20)
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
    print("🚀 Launching FULL-SCALE HRM Research (Dim 512, Deep Stack)")
    # Generate 60,000 Episodes (More data for big brain)
    chunks = list(collect_data_chunk.map(range(100), kwargs={'n_episodes': 600}))
    train_models.remote(chunks)
    res = list(Evaluator().run_episode.map(range(CONFIG['eval_episodes'])))
    l = sum(r['L'] for r in res); h = sum(r['H'] for r in res)
    print(f"LSTM: {l}/{len(res)} ({l/len(res)*100:.1f}%)")
    print(f"HRM:  {h}/{len(res)} ({h/len(res)*100:.1f}%)")