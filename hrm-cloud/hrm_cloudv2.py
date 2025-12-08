import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import heapq
import os
from typing import List

app = modal.App("hrm-b200-research-fixed")

# torch>=2.2.0 required for native FlashAttention (SDPA)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch>=2.2.0", "numpy", "gymnasium", "tqdm")
)

vol = modal.Volume.from_name("hrm-research-vol", create_if_missing=True)

CONFIG = {
    "grid_size": 20,
    "n_static": 12,      # Reduced slightly to ensure valid paths exist
    "n_dynamic": 6,
    "obs_history": 10,
    "pred_horizon": 20,  # Look 20 steps ahead (Partial Planning)
    "hrm_k_step": 3,
    "hidden_dim": 256,
    "num_heads": 4,
    "batch_size": 1024,
    "lr": 3e-4,
    "epochs": 50,
    "data_episodes": 3000, 
    "eval_episodes": 50
}

PATHS = {
    "data_dir": "/data/episodes",
    "model_lstm": "/data/lstm_v2.pt",
    "model_hrm": "/data/hrm_v2.pt"
}

# ==========================================
# 1. EXACT SAPIENT HRM ARCHITECTURE
# ==========================================

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.g = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class RecurrentTransformerBlock(nn.Module):
    """
    The core 'Reasoning Unit'. 
    Uses FlashAttention implicitly via F.scaled_dot_product_attention 
    inside nn.MultiheadAttention.
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, int(dim * 2.6))

    def forward(self, x, state):
        # x: Input Context, state: Recurrent Memory
        h = x + state
        
        # Self-Attention on State (Recurrent Context)
        res = h
        h_norm = self.norm1(h)
        # In recurrent mode, Q=K=V=state
        attn_out, _ = self.attn(h_norm.unsqueeze(1), h_norm.unsqueeze(1), h_norm.unsqueeze(1))
        h = res + attn_out.squeeze(1)
        
        # FFN
        return h + self.ffn(self.norm2(h))

class SapientHRM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, k_step=3, num_heads=4):
        super().__init__()
        self.k_step = k_step
        self.hidden_dim = hidden_dim
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.L_block = RecurrentTransformerBlock(hidden_dim, num_heads)
        self.H_block = RecurrentTransformerBlock(hidden_dim, num_heads)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        b, seq, _ = x.size()
        h_L = torch.zeros(b, self.hidden_dim, device=x.device)
        h_H = torch.zeros(b, self.hidden_dim, device=x.device)
        
        for t in range(seq):
            x_t = self.embed(x[:, t, :])
            
            # System 2 (Slow)
            if t % self.k_step == 0:
                h_H = self.H_block(h_L, h_H)
            
            # System 1 (Fast)
            context = x_t + h_H
            h_L = self.L_block(context, h_L)
            
        return self.head(h_L)

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ==========================================
# 2. ENV & PLANNER (RECEDING HORIZON)
# ==========================================

class DynamicGridEnv:
    def __init__(self, config):
        self.size = config["grid_size"]
        self.n_dyn = config["n_dynamic"]
        self.reset()

    def reset(self, seed=None):
        if seed is not None: np.random.seed(seed)
        self.static_map = np.zeros((self.size, self.size))
        # Clear start/goal
        self.static_map[0,0] = 0; self.static_map[self.size-1, self.size-1] = 0
        # Random blocks
        for _ in range(self.size):
            r, c = np.random.randint(0, self.size, 2)
            if (r,c) != (0,0) and (r,c) != (self.size-1, self.size-1):
                self.static_map[r, c] = 1.0
                
        self.dynamic_obs = []
        for _ in range(self.n_dyn):
            while True:
                pos = np.random.randint(0, self.size, 2).astype(float)
                if self.static_map[int(pos[0]), int(pos[1])] == 0:
                    vel = np.random.randn(2)
                    vel = vel / np.linalg.norm(vel) * 0.7 # Limit speed
                    self.dynamic_obs.append({'pos': pos, 'vel': vel})
                    break
        self.agent_pos = np.array([0., 0.])
        self.goal_pos = np.array([self.size-1., self.size-1.])
        return self._get_obs()

    def step_physics(self):
        for o in self.dynamic_obs:
            o['pos'] += o['vel']
            for i in range(2):
                if o['pos'][i] < 0 or o['pos'][i] >= self.size:
                    o['vel'][i] *= -1
                    o['pos'][i] = np.clip(o['pos'][i], 0, self.size-0.01)
        return self._get_obs()

    def _get_obs(self):
        # Return RAW coords (Normalized in Planner)
        return np.array([o['pos'] for o in self.dynamic_obs])

class SpaceTimeAStar:
    """RHC Planner: Plans partial paths if goal is unreachable within horizon"""
    def __init__(self, env, model, device):
        self.env = env; self.model = model; self.device = device; self.model.eval()

    def get_next_action(self, start, goal, obs_history):
        # 1. NORMALIZE INPUT [0, 1]
        curr = torch.tensor(obs_history / self.env.size, dtype=torch.float32).to(self.device)
        
        future_obs = []
        with torch.no_grad():
            for _ in range(CONFIG['pred_horizon']):
                # Model predicts Normalized Delta
                norm_delta = self.model(curr)
                
                # Next Normalized Pos
                next_pos_norm = curr[:, -1, :] + norm_delta
                
                # Denormalize for Collision Check
                future_obs.append((next_pos_norm.cpu().numpy() * self.env.size))
                
                # Update Buffer
                curr = torch.cat([curr[:, 1:, :], next_pos_norm.unsqueeze(1)], dim=1)
        
        future_obs = np.array(future_obs)

        # 2. A* Search
        start_node = (int(start[0]), int(start[1]), 0)
        pq = [(0, 0, start_node)]
        g_score = {start_node: 0}
        came_from = {}
        
        best_node_at_horizon = None
        min_h = float('inf')

        while pq:
            f, g, current = heapq.heappop(pq)
            r, c, t = current
            
            # Goal Reached?
            if (r, c) == (int(goal[0]), int(goal[1])):
                return self.reconstruct_first_step(came_from, current, start_node)
            
            # Horizon Limit? Track best partial path
            if t >= CONFIG['pred_horizon'] - 1:
                h = abs(r - goal[0]) + abs(c - goal[1])
                if h < min_h:
                    min_h = h
                    best_node_at_horizon = current
                continue

            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0), (0,0)]:
                nr, nc, nt = r+dr, c+dc, t+1
                if not (0 <= nr < self.env.size and 0 <= nc < self.env.size): continue
                if self.env.static_map[nr, nc] == 1: continue
                
                # Dynamic Check (Predicted)
                dists = np.linalg.norm(future_obs[nt] - np.array([nr, nc]), axis=1)
                if np.any(dists < 1.0): continue 
                
                new_g = g + 1
                neigh = (nr, nc, nt)
                if new_g < g_score.get(neigh, float('inf')):
                    g_score[neigh] = new_g
                    h = abs(nr - goal[0]) + abs(nc - goal[1])
                    heapq.heappush(pq, (new_g + h, new_g, neigh))
                    came_from[neigh] = current
                    
        # If no path to goal, move towards best intermediate node
        if best_node_at_horizon:
            return self.reconstruct_first_step(came_from, best_node_at_horizon, start_node)
        
        return (int(start[0]), int(start[1])) # Wait

    def reconstruct_first_step(self, came_from, current, start):
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        if not path: return (int(start[0]), int(start[1]))
        first_step = path[-1]
        return (first_step[0], first_step[1])

# ==========================================
# 3. MODAL WORKFLOW
# ==========================================

@app.function(image=image, volumes={"/data": vol}, cpu=1.0)
def collect_data_chunk(worker_id, n_episodes):
    env = DynamicGridEnv(CONFIG)
    X, Y = [], []
    for _ in range(n_episodes):
        env.reset()
        hist = []
        # Run longer than horizon to ensure targets exist
        for _ in range(70):
            hist.append(env.step_physics())
            if len(hist) > CONFIG['obs_history']:
                # NORMALIZE INPUTS (0-1)
                past = np.array(hist[-CONFIG['obs_history']-1 : -1]) / env.size
                future = np.array(hist[-1]) / env.size
                prev = np.array(hist[-2]) / env.size
                
                # Target is Normalized Delta
                delta = future - prev
                for j in range(env.n_dyn):
                    X.append(past[:, j, :])
                    Y.append(delta[j, :])
    
    os.makedirs(PATHS['data_dir'], exist_ok=True)
    fn = f"{PATHS['data_dir']}/chunk_{worker_id}.pt"
    torch.save((X, Y), fn)
    vol.commit()
    return fn

@app.function(image=image, gpu="B200", volumes={"/data": vol}, timeout=3600)
def train_models(chunk_files: List[str]):
    print(f"--> Loading Data...")
    X_l, Y_l = [], []
    for f in chunk_files:
        x, y = torch.load(f, weights_only=False)
        X_l.extend(x); Y_l.extend(y)
    
    X = torch.tensor(np.array(X_l), dtype=torch.float32).cuda()
    Y = torch.tensor(np.array(Y_l), dtype=torch.float32).cuda()
    print(f"Dataset: {len(X)} samples.")
    
    models = {
        "LSTM": LSTMPredictor(2, CONFIG['hidden_dim'], 2).cuda(),
        "HRM": SapientHRM(2, CONFIG['hidden_dim'], 2, k_step=CONFIG['hrm_k_step']).cuda()
    }
    
    for name, model in models.items():
        print(f"Training {name}...")
        opt = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
        dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y), batch_size=CONFIG['batch_size'], shuffle=True)
        model.train()
        for ep in range(CONFIG['epochs']):
            ep_loss = 0
            for bx, by in dl:
                opt.zero_grad()
                pred = model(bx)
                loss = nn.MSELoss()(pred, by)
                loss.backward()
                # Clip Gradients for Transformers
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item()
            if ep % 10 == 0: print(f"[{name}] Ep {ep}: Loss {ep_loss/len(dl):.6f}")

    torch.save(models["LSTM"].state_dict(), PATHS["model_lstm"])
    torch.save(models["HRM"].state_dict(), PATHS["model_hrm"])
    vol.commit()

@app.cls(image=image, gpu="B200", volumes={"/data": vol}, max_containers=10)
class Evaluator:
    @modal.enter()
    def setup(self):
        self.device = torch.device("cuda")
        self.lstm = LSTMPredictor(2, CONFIG['hidden_dim'], 2).to(self.device)
        self.hrm = SapientHRM(2, CONFIG['hidden_dim'], 2, k_step=CONFIG['hrm_k_step']).to(self.device)
        self.lstm.load_state_dict(torch.load(PATHS["model_lstm"], weights_only=True))
        self.hrm.load_state_dict(torch.load(PATHS["model_hrm"], weights_only=True))

    @modal.method()
    def run_episode(self, seed):
        # 1. Setup Environment
        env = DynamicGridEnv(CONFIG)
        env.reset(seed=seed + 5000)
        
        # 2. Simulation Loop (Receding Horizon)
        def simulate(model):
            # Reset env to same state for fairness
            sim_env = DynamicGridEnv(CONFIG)
            sim_env.reset(seed=seed + 5000)
            
            # Planner
            planner = SpaceTimeAStar(sim_env, model, self.device)
            
            # History Buffer
            hist = []
            for _ in range(CONFIG['obs_history']): hist.append(sim_env.step_physics())
            
            for step in range(100): # Max 100 steps to cross
                # Plan
                hist_np = np.array(hist[-CONFIG['obs_history']:]).transpose(1, 0, 2)
                nr, nc = planner.get_next_action(sim_env.agent_pos, sim_env.goal_pos, hist_np)
                
                # Move
                sim_env.agent_pos = np.array([float(nr), float(nc)])
                
                # Update Physics
                new_obs = sim_env.step_physics()
                hist.append(new_obs)
                
                # Check Goal
                if np.linalg.norm(sim_env.agent_pos - sim_env.goal_pos) < 0.5:
                    return 1 # Success
                
                # Check Crash (Static or Dynamic)
                if sim_env.static_map[int(nr), int(nc)] == 1: return 0
                if np.any(np.linalg.norm(new_obs - sim_env.agent_pos, axis=1) < 0.8): return 0
            
            return 0 # Timeout

        return {"L": simulate(self.lstm), "H": simulate(self.hrm)}

@app.local_entrypoint()
def main():
    print("🚀 Launching Fixed HRM Cluster (RHC + Norm)")
    # Data Collection
    chunks = list(collect_data_chunk.map(range(50), kwargs={'n_episodes': 60}))
    # Training
    train_models.remote(chunks)
    # Evaluation
    print("Running Receding Horizon Simulation...")
    res = list(Evaluator().run_episode.map(range(CONFIG['eval_episodes'])))
    
    l_succ = sum(r['L'] for r in res)
    h_succ = sum(r['H'] for r in res)
    
    print(f"\nFINAL METRICS (Execution Success Rate)")
    print(f"LSTM: {l_succ}/{len(res)} ({l_succ/len(res)*100:.1f}%)")
    print(f"HRM:  {h_succ}/{len(res)} ({h_succ/len(res)*100:.1f}%)")