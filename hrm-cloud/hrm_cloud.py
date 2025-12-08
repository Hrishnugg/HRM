import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import heapq
import time
import os
from typing import List

# ==========================================
# 1. MODAL INFRASTRUCTURE
# ==========================================
app = modal.App("hrm-b200-research-v2")

# torch>=2.2.0 required for stable native FlashAttention (SDPA)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.2.0", 
        "numpy",
        "gymnasium",
        "tqdm"
    )
)

vol = modal.Volume.from_name("hrm-research-vol", create_if_missing=True)

CONFIG = {
    "grid_size": 20,
    "n_static": 15,
    "n_dynamic": 8,
    "obs_history": 10,
    "pred_horizon": 15,
    "hrm_k_step": 3,
    "hidden_dim": 256,   # Matches Sapient 'Small' config
    "num_heads": 4,      # Standard for this dim
    "batch_size": 512,
    "lr": 3e-4,          # Transformers require lower LR than GRUs
    "epochs": 40,
    "data_episodes": 2000, 
    "eval_episodes": 50
}

PATHS = {
    "data_dir": "/data/episodes",
    "model_lstm": "/data/lstm.pt",
    "model_hrm": "/data/hrm.pt"
}

# ==========================================
# 2. EXACT SAPIENT HRM ARCHITECTURE
# ==========================================

class RMSNorm(nn.Module):
    """Root Mean Square Normalization (Standard in HRMv2)"""
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g

class SwiGLU(nn.Module):
    """Gated Linear Unit (SiLU) - Replaces standard MLP"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class RecurrentTransformerBlock(nn.Module):
    """
    The 'Cell' of the HRM. It is an Encoder-only Transformer Block 
    that treats the recurrent state as its sequence context.
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        # batch_first=True for compatibility
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, int(dim * 2.6)) # 2.6x expansion ratio

    def forward(self, x, state):
        # x: Input injection (Observation or Context)
        # state: The Recurrent State (Z_t-1)
        
        # 1. State + Input Mixing (Element-wise addition)
        h = x + state 
        
        # 2. Self-Attention (State Mixing)
        # In a recurrent step, Q=K=V=h. This mixes the feature dimensions.
        # Natively uses FlashAttention (SDPA) backend on B200.
        res = h
        h_norm = self.norm1(h)
        attn_out, _ = self.attn(h_norm.unsqueeze(1), h_norm.unsqueeze(1), h_norm.unsqueeze(1))
        h = res + attn_out.squeeze(1)
        
        # 3. SwiGLU FFN
        res = h
        h = res + self.ffn(self.norm2(h))
        
        return h

class SapientHRM(nn.Module):
    """
    Exact implementation of the Sapient Inc HRM.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, k_step=3, num_heads=4):
        super().__init__()
        self.k_step = k_step
        self.hidden_dim = hidden_dim
        
        self.embed = nn.Linear(input_dim, hidden_dim)
        
        # The two "Brains" (System 1 & System 2)
        self.L_block = RecurrentTransformerBlock(hidden_dim, num_heads)
        self.H_block = RecurrentTransformerBlock(hidden_dim, num_heads)
        
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (Batch, Seq, Feat)
        b, seq, _ = x.size()
        
        # Latent States
        h_L = torch.zeros(b, self.hidden_dim, device=x.device)
        h_H = torch.zeros(b, self.hidden_dim, device=x.device)
        
        outputs = []
        
        for t in range(seq):
            x_t = self.embed(x[:, t, :])
            
            # --- System 2 (H-Module) ---
            # Updates every k steps. Attends to L (Bottom-Up Abstraction).
            if t % self.k_step == 0:
                h_H = self.H_block(h_L, h_H)
            
            # --- System 1 (L-Module) ---
            # Updates every step. Attends to H (Top-Down Context).
            # Context is injected via addition before the block logic.
            context = x_t + h_H
            h_L = self.L_block(context, h_L)
            
            outputs.append(h_L)
            
        # Return final state prediction
        return self.head(h_L)

# ==========================================
# 3. ENV & UTILS
# ==========================================
class DynamicGridEnv:
    def __init__(self, config):
        self.size = config["grid_size"]; self.n_dyn = config["n_dynamic"]; self.reset()
    def reset(self, seed=None):
        if seed is not None: np.random.seed(seed)
        self.static_map = np.zeros((self.size, self.size))
        for _ in range(self.size*2): self.static_map[np.random.randint(0,self.size,2).tolist()] = 1.0
        self.dynamic_obs = [{'pos':np.random.rand(2)*self.size,'vel':np.random.randn(2)*0.5} for _ in range(self.n_dyn)]
        self.agent_pos=np.array([0.,0.]); self.goal_pos=np.array([self.size-1.,self.size-1.])
        return self._get_obs()
    def step_physics(self):
        res=[]; 
        for o in self.dynamic_obs:
            o['pos']+=o['vel']
            if not(0<=o['pos'][0]<self.size): o['vel'][0]*=-1; o['pos'][0]=np.clip(o['pos'][0],0,self.size-0.1)
            if not(0<=o['pos'][1]<self.size): o['vel'][1]*=-1; o['pos'][1]=np.clip(o['pos'][1],0,self.size-0.1)
            res.append(o['pos'].copy())
        return np.array(res)
    def _get_obs(self): return np.array([o['pos'] for o in self.dynamic_obs])

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class SpaceTimeAStar:
    def __init__(self, env, model, device):
        self.env = env; self.model = model; self.device = device; self.model.eval()
    def search(self, start, goal, obs_history):
        curr = torch.tensor(obs_history, dtype=torch.float32).to(self.device)
        future_obs = []
        with torch.no_grad():
            for _ in range(CONFIG['pred_horizon']):
                delta = self.model(curr)
                next_pos = curr[:, -1, :] + delta
                future_obs.append(next_pos.cpu().numpy())
                curr = torch.cat([curr[:, 1:, :], next_pos.unsqueeze(1)], dim=1)
        future_obs = np.array(future_obs)
        
        start_node = (int(start[0]), int(start[1]), 0)
        pq = [(0, 0, start_node)]; g_score = {start_node: 0}
        while pq:
            f, g, current = heapq.heappop(pq)
            r, c, t = current
            if (r, c) == (int(goal[0]), int(goal[1])): return True
            if t >= CONFIG['pred_horizon'] - 1: continue
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0), (0,0)]:
                nr, nc, nt = r+dr, c+dc, t+1
                if not (0 <= nr < self.env.size and 0 <= nc < self.env.size): continue
                if self.env.static_map[nr, nc] == 1: continue
                if np.any(np.linalg.norm(future_obs[nt] - np.array([nr, nc]), axis=1) < 1.1): continue
                new_g = g + 1
                neigh = (nr, nc, nt)
                if new_g < g_score.get(neigh, float('inf')):
                    g_score[neigh] = new_g
                    heapq.heappush(pq, (new_g + abs(nr-goal[0]) + abs(nc-goal[1]), new_g, neigh))
        return False

# ==========================================
# 4. DISTRIBUTED CLOUD FUNCTIONS
# ==========================================

@app.function(image=image, volumes={"/data": vol}, cpu=1.0)
def collect_data_chunk(worker_id, n_episodes):
    env = DynamicGridEnv(CONFIG)
    X, Y = [], []
    for i in range(n_episodes):
        env.reset(seed=worker_id*10000 + i)
        hist = []
        for _ in range(60):
            hist.append(env.step_physics())
            if len(hist) > CONFIG['obs_history']:
                past = np.array(hist[-CONFIG['obs_history']-1 : -1])
                future = np.array(hist[-1])
                prev = np.array(hist[-2])
                for j in range(env.n_dyn):
                    X.append(past[:, j, :])
                    Y.append(future[j, :] - prev[j, :])
    os.makedirs(PATHS['data_dir'], exist_ok=True)
    fn = f"{PATHS['data_dir']}/chunk_{worker_id}.pt"
    torch.save((X, Y), fn)
    vol.commit()
    return fn

@app.function(image=image, gpu="B200", volumes={"/data": vol}, timeout=3600)
def train_models(chunk_files: List[str]):
    print(f"--> Loading Data...")
    X_list, Y_list = [], []
    for f in chunk_files:
        x, y = torch.load(f, weights_only=False)
        X_list.extend(x); Y_list.extend(y)
    X = torch.tensor(np.array(X_list), dtype=torch.float32).cuda()
    Y = torch.tensor(np.array(Y_list), dtype=torch.float32).cuda()
    
    # 1. Train LSTM
    print("Training LSTM Baseline...")
    lstm = LSTMPredictor(2, CONFIG['hidden_dim'], 2).cuda()
    opt = optim.AdamW(lstm.parameters(), lr=CONFIG['lr'])
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y), batch_size=CONFIG['batch_size'], shuffle=True)
    lstm.train()
    for ep in range(CONFIG['epochs']):
        for bx, by in dl:
            opt.zero_grad(); loss = nn.MSELoss()(lstm(bx), by); loss.backward(); opt.step()
            
    # 2. Train HRM (Transformer Architecture)
    print("Training HRM (Recurrent Transformer)...")
    hrm = SapientHRM(2, CONFIG['hidden_dim'], 2, 
                    k_step=CONFIG['hrm_k_step'], 
                    num_heads=CONFIG['num_heads']).cuda()
                    
    opt = optim.AdamW(hrm.parameters(), lr=CONFIG['lr']) # No weight decay
    hrm.train()
    
    for ep in range(CONFIG['epochs']):
        ep_loss = 0
        for bx, by in dl:
            opt.zero_grad()
            pred = hrm(bx)
            loss = nn.MSELoss()(pred, by)
            loss.backward() # Using BPTT for Playground stability
            opt.step()
            ep_loss += loss.item()
        if ep % 5 == 0: print(f"HRM Epoch {ep}: {ep_loss/len(dl):.5f}")

    torch.save(lstm.state_dict(), PATHS["model_lstm"])
    torch.save(hrm.state_dict(), PATHS["model_hrm"])
    vol.commit()
    return "Training Complete"

@app.cls(image=image, gpu="B200", volumes={"/data": vol}, max_containers=10)
class Evaluator:
    @modal.enter()
    def setup(self):
        self.device = torch.device("cuda")
        self.lstm = LSTMPredictor(2, CONFIG['hidden_dim'], 2).to(self.device)
        self.hrm = SapientHRM(2, CONFIG['hidden_dim'], 2, 
                             k_step=CONFIG['hrm_k_step'], 
                             num_heads=CONFIG['num_heads']).to(self.device)
        self.lstm.load_state_dict(torch.load(PATHS["model_lstm"], weights_only=True))
        self.hrm.load_state_dict(torch.load(PATHS["model_hrm"], weights_only=True))
        
    @modal.method()
    def run_episode(self, seed):
        env = DynamicGridEnv(CONFIG); env.reset(seed=seed + 9999)
        hist = []
        for _ in range(CONFIG['obs_history']): hist.append(env.step_physics())
        hist_np = np.array(hist).transpose(1, 0, 2)
        
        start = time.time()
        res_l = SpaceTimeAStar(env, self.lstm, self.device).search(env.agent_pos, env.goal_pos, hist_np)
        t_l = time.time() - start
        
        start = time.time()
        res_h = SpaceTimeAStar(env, self.hrm, self.device).search(env.agent_pos, env.goal_pos, hist_np)
        t_h = time.time() - start
        return {"L": 1 if res_l else 0, "H": 1 if res_h else 0, "TL": t_l, "TH": t_h}

@app.local_entrypoint()
def main():
    print("🚀 Launching Official HRM Architecture Cluster")
    chunks = list(collect_data_chunk.map(range(50), kwargs={'n_episodes': 40}))
    train_models.remote(chunks)
    res = list(Evaluator().run_episode.map(range(CONFIG['eval_episodes'])))
    print(f"LSTM Success: {sum(r['L'] for r in res)}/{len(res)}")
    print(f"HRM Success:  {sum(r['H'] for r in res)}/{len(res)}")