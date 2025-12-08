import modal

# ==============================================================================
# 1. MODAL APP CONFIGURATION
# ==============================================================================
app = modal.App("rl-hybrid-diffusion-playground")

image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch",
        "numpy",
        "matplotlib",
        "tqdm",
        "scikit-image",
        "scipy"
    )
)

# Constants (available to all functions)
MAP_SIZE = 64
HORIZON = 64
OBS_RADIUS = 2

# ==============================================================================
# 2. PARALLEL TRAJECTORY GENERATION (CPU WORKERS)
# ==============================================================================

@app.function(image=image, cpu=2.0, timeout=600)
def generate_trajectory_chunk(worker_id: int, n_trajectories: int):
    """Generate a chunk of expert trajectories using Space-Time A* (CPU only)"""
    import numpy as np
    import heapq
    
    class DynamicEnv:
        def __init__(self, size=MAP_SIZE, n_dynamic=8, seed=None):
            self.size = size
            if seed: np.random.seed(seed)
            
            self.static_grid = np.zeros((size, size))
            for _ in range(10):
                cx, cy = np.random.randint(0, size, 2)
                r = np.random.randint(2, 6)
                y, x = np.ogrid[-r:r+1, -r:r+1]
                mask = x**2 + y**2 <= r**2
                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        if mask[i, j]:
                            px, py = cx + i - r, cy + j - r
                            if 0 <= px < size and 0 <= py < size:
                                self.static_grid[px, py] = 1.0

            self.dynamic_obs = []
            for _ in range(n_dynamic):
                pos = np.random.rand(2) * size
                vel = (np.random.rand(2) - 0.5) * 1.5
                self.dynamic_obs.append({'pos': pos, 'vel': vel})
                
            self.start = self._find_free_point()
            self.goal = self._find_free_point()

        def _find_free_point(self):
            while True:
                p = np.random.randint(0, self.size, 2)
                if self.static_grid[p[0], p[1]] == 0:
                    return p

        def get_grid(self, t_offset=0):
            grid = self.static_grid.copy()
            for obs in self.dynamic_obs:
                future_pos = obs['pos'] + obs['vel'] * t_offset
                cx, cy = int(future_pos[0]), int(future_pos[1])
                r = OBS_RADIUS
                y, x = np.ogrid[-r:r+1, -r:r+1]
                mask = x**2 + y**2 <= r**2
                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        if mask[i, j]:
                            px, py = cx + i - r, cy + j - r
                            if 0 <= px < self.size and 0 <= py < self.size:
                                grid[px, py] = 1.0
            return grid

        def step(self):
            for obs in self.dynamic_obs:
                obs['pos'] += obs['vel']
                for i in range(2):
                    if obs['pos'][i] < 0 or obs['pos'][i] >= self.size:
                        obs['vel'][i] *= -1
                        obs['pos'][i] = np.clip(obs['pos'][i], 0, self.size-0.01)

    class SpaceTimeAStar:
        def __init__(self, env):
            self.env = env

        def plan(self):
            start = tuple(self.env.start)
            goal = tuple(self.env.goal)
            
            pq = [(0, 0, start[0], start[1], 0, [start])]
            visited = set()
            
            future_grids = {}
            temp_obs = [{'pos': o['pos'].copy(), 'vel': o['vel'].copy()} for o in self.env.dynamic_obs]
            
            for t in range(HORIZON + 20):
                future_grids[t] = [o['pos'].copy() for o in temp_obs]
                for o in temp_obs:
                    o['pos'] += o['vel']
                    for i in range(2):
                        if o['pos'][i] < 0 or o['pos'][i] >= self.env.size:
                            o['vel'][i] *= -1
            
            while pq:
                f, g, r, c, t, path = heapq.heappop(pq)
                
                if t >= HORIZON: continue
                if np.linalg.norm([r - goal[0], c - goal[1]]) < 2:
                    return np.array(path)
                
                state = (r, c, t)
                if state in visited: continue
                visited.add(state)
                
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        nt = t + 1
                        
                        if 0 <= nr < self.env.size and 0 <= nc < self.env.size:
                            if self.env.static_grid[nr, nc] == 1:
                                continue
                            
                            dynamic_collision = False
                            for d_pos in future_grids[nt]:
                                if np.linalg.norm([nr - d_pos[0], nc - d_pos[1]]) < OBS_RADIUS + 1:
                                    dynamic_collision = True
                                    break
                            
                            if not dynamic_collision:
                                h = np.linalg.norm([nr - goal[0], nc - goal[1]])
                                new_g = g + np.linalg.norm([dr, dc])
                                heapq.heappush(pq, (new_g + h, new_g, nr, nc, nt, path + [(nr, nc)]))
            return None

    # Generate trajectories
    np.random.seed(worker_id * 1000)  # Unique seed per worker
    
    data_grids, data_sg, data_paths = [], [], []
    attempts = 0
    max_attempts = n_trajectories * 3  # Allow some failures
    
    while len(data_grids) < n_trajectories and attempts < max_attempts:
        attempts += 1
        env = DynamicEnv()
        planner = SpaceTimeAStar(env)
        path = planner.plan()
        
        if path is not None and len(path) > 10:
            path_interp = np.zeros((HORIZON, 2))
            old_idx = np.linspace(0, 1, len(path))
            new_idx = np.linspace(0, 1, HORIZON)
            path_interp[:, 0] = np.interp(new_idx, old_idx, path[:, 0])
            path_interp[:, 1] = np.interp(new_idx, old_idx, path[:, 1])
            
            data_grids.append(env.get_grid(t_offset=0))
            data_sg.append(np.concatenate([env.start, env.goal]) / MAP_SIZE)
            data_paths.append(path_interp / MAP_SIZE)
    
    print(f"Worker {worker_id}: Generated {len(data_grids)}/{n_trajectories} trajectories")
    return {
        'grids': np.array(data_grids),
        'sg': np.array(data_sg),
        'paths': np.array(data_paths)
    }

# ==============================================================================
# 3. GPU TRAINING AND EVALUATION
# ==============================================================================

@app.function(image=image, gpu="A10G", timeout=3600)
def train_and_evaluate(all_chunks: list):
    """Train diffusion model on GPU and evaluate"""
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from scipy.ndimage import distance_transform_edt
    from tqdm import tqdm
    
    DEVICE = "cuda"
    NUM_EPOCHS = 2000   # 2x more training for better convergence
    N_SAMPLES = 10      # More samples for safer path selection
    N_DIFFUSION_STEPS = 100  # Higher quality generation
    N_REFINE_STEPS = 40      # More SDF refinement iterations
    EMA_DECAY = 0.999   # Exponential moving average for stability
    
    # --- Merge all chunks ---
    print(f"[Phase 2] Merging {len(all_chunks)} chunks...")
    all_grids = np.concatenate([c['grids'] for c in all_chunks], axis=0)
    all_sg = np.concatenate([c['sg'] for c in all_chunks], axis=0)
    all_paths = np.concatenate([c['paths'] for c in all_chunks], axis=0)
    
    print(f"Total trajectories: {len(all_grids)}")
    
    # Convert to tensors
    t_grids = torch.tensor(all_grids, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    t_sg = torch.tensor(all_sg, dtype=torch.float32).to(DEVICE)
    t_paths = torch.tensor(all_paths, dtype=torch.float32).transpose(1, 2).to(DEVICE)
    
    # --- Define Model (Enhanced with Attention + Residual Blocks) ---
    class ResBlock1D(nn.Module):
        """Residual block with time conditioning"""
        def __init__(self, in_ch, out_ch, time_dim):
            super().__init__()
            self.conv1 = nn.Conv1d(in_ch, out_ch, 3, 1, 1)
            self.conv2 = nn.Conv1d(out_ch, out_ch, 3, 1, 1)
            self.time_proj = nn.Linear(time_dim, out_ch)
            self.norm1 = nn.GroupNorm(8, out_ch)
            self.norm2 = nn.GroupNorm(8, out_ch)
            self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
        def forward(self, x, t_emb):
            h = F.silu(self.norm1(self.conv1(x)))
            h = h + self.time_proj(t_emb).unsqueeze(-1)
            h = F.silu(self.norm2(self.conv2(h)))
            return h + self.skip(x)
    
    class SelfAttention1D(nn.Module):
        """Self-attention for sequence modeling"""
        def __init__(self, dim, heads=4):
            super().__init__()
            self.heads = heads
            self.scale = (dim // heads) ** -0.5
            self.qkv = nn.Conv1d(dim, dim * 3, 1)
            self.proj = nn.Conv1d(dim, dim, 1)
            self.norm = nn.GroupNorm(8, dim)
        
        def forward(self, x):
            B, C, L = x.shape
            qkv = self.qkv(self.norm(x)).reshape(B, 3, self.heads, C // self.heads, L)
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
            attn = (q.transpose(-1, -2) @ k) * self.scale
            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-1, -2)).transpose(-1, -2).reshape(B, C, L)
            return x + self.proj(out)
    
    class ConditionalUnet1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.map_enc = nn.Sequential(
                nn.Conv2d(1, 32, 5, 2, 2), nn.SiLU(),
                nn.Conv2d(32, 64, 5, 2, 2), nn.SiLU(),
                nn.Conv2d(64, 128, 3, 2, 1), nn.SiLU(),  # Deeper
                nn.Flatten(),
                nn.Linear(128*8*8, 256)
            )
            self.coord_enc = nn.Linear(4, 256)
            self.time_mlp = nn.Sequential(
                nn.Linear(1, 256), nn.SiLU(), 
                nn.Linear(256, 256), nn.SiLU(),
                nn.Linear(256, 256)
            )
            
            # Encoder
            self.down1 = ResBlock1D(2, 64, 256)
            self.down2 = ResBlock1D(64, 128, 256)
            self.attn1 = SelfAttention1D(128)
            
            # Middle (h2=128 + cond_partial=128 = 256 input channels)
            self.mid1 = ResBlock1D(256, 256, 256)  # Fixed: 128+128=256 input
            self.mid_attn = SelfAttention1D(256)
            self.mid2 = ResBlock1D(256, 256, 256)
            
            # Decoder
            self.up1 = ResBlock1D(256 + 128, 128, 256)  # Skip connection
            self.up2 = ResBlock1D(128 + 64, 64, 256)
            self.attn2 = SelfAttention1D(64)
            self.out = nn.Conv1d(64, 2, 3, 1, 1)

        def forward(self, x, t, grid, sg):
            B, _, H = x.shape
            
            # Embeddings
            emb_map = self.map_enc(grid)  # (B, 256)
            emb_sg = self.coord_enc(sg)    # (B, 256)
            emb_t = self.time_mlp(t)       # (B, 256)
            cond = torch.cat([emb_map, emb_sg], dim=1)  # (B, 512)
            
            # Encoder
            h1 = self.down1(x, emb_t)      # (B, 64, H)
            h2 = self.down2(h1, emb_t)     # (B, 128, H)
            h2 = self.attn1(h2)
            
            # Inject conditioning
            cond_expanded = cond.unsqueeze(-1).expand(-1, -1, H)  # (B, 512, H)
            h_cond = torch.cat([h2, cond_expanded[:, :128, :]], dim=1)  # Add partial conditioning
            
            # Middle
            h_mid = self.mid1(h_cond, emb_t)
            h_mid = self.mid_attn(h_mid)
            h_mid = self.mid2(h_mid, emb_t)
            
            # Decoder with skip connections
            h_up = self.up1(torch.cat([h_mid, h2], dim=1), emb_t)
            h_up = self.up2(torch.cat([h_up, h1], dim=1), emb_t)
            h_up = self.attn2(h_up)
            
            return self.out(h_up)

    # --- Train with EMA ---
    print(f"\n[Phase 3] Training Enhanced Diffusion Model for {NUM_EPOCHS} epochs...")
    model = ConditionalUnet1D().to(DEVICE)
    
    # EMA model for inference (smoother, higher quality)
    ema_model = ConditionalUnet1D().to(DEVICE)
    ema_model.load_state_dict(model.state_dict())
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_loss = float('inf')
    for epoch in tqdm(range(NUM_EPOCHS), desc="Training"):
        t_flat = torch.rand(t_paths.shape[0], 1).to(DEVICE)  # (B, 1) for model
        t = t_flat.unsqueeze(-1)  # (B, 1, 1) for broadcasting
        noise = torch.randn_like(t_paths)
        noisy_path = t_paths * (1-t) + noise * t
        pred_noise = model(noisy_path, t_flat, t_grids, t_sg)
        loss = F.mse_loss(pred_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        scheduler.step()
        
        # Update EMA model
        with torch.no_grad():
            for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                ema_p.data.mul_(EMA_DECAY).add_(p.data, alpha=1 - EMA_DECAY)
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}, LR {scheduler.get_last_lr()[0]:.2e}, Best {best_loss:.4f}")
    
    # Use EMA model for inference
    model = ema_model
    print(f"Training complete! Using EMA model for inference.")
    
    # --- Evaluation ---
    print("\n[Phase 4] Evaluation: Hybrid Diffusion vs Oracle A*")
    
    # DynamicEnv for evaluation (redefined here for GPU function)
    class DynamicEnv:
        def __init__(self, size=MAP_SIZE, n_dynamic=8, seed=None):
            self.size = size
            if seed: np.random.seed(seed)
            self.static_grid = np.zeros((size, size))
            for _ in range(10):
                cx, cy = np.random.randint(0, size, 2)
                r = np.random.randint(2, 6)
                y, x = np.ogrid[-r:r+1, -r:r+1]
                mask = x**2 + y**2 <= r**2
                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        if mask[i, j]:
                            px, py = cx + i - r, cy + j - r
                            if 0 <= px < size and 0 <= py < size:
                                self.static_grid[px, py] = 1.0
            self.dynamic_obs = []
            for _ in range(n_dynamic):
                pos = np.random.rand(2) * size
                vel = (np.random.rand(2) - 0.5) * 1.5
                self.dynamic_obs.append({'pos': pos, 'vel': vel})
            self.start = self._find_free_point()
            self.goal = self._find_free_point()

        def _find_free_point(self):
            while True:
                p = np.random.randint(0, self.size, 2)
                if self.static_grid[p[0], p[1]] == 0:
                    return p

        def get_grid(self, t_offset=0):
            grid = self.static_grid.copy()
            for obs in self.dynamic_obs:
                future_pos = obs['pos'] + obs['vel'] * t_offset
                cx, cy = int(future_pos[0]), int(future_pos[1])
                r = OBS_RADIUS
                y, x = np.ogrid[-r:r+1, -r:r+1]
                mask = x**2 + y**2 <= r**2
                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        if mask[i, j]:
                            px, py = cx + i - r, cy + j - r
                            if 0 <= px < self.size and 0 <= py < self.size:
                                grid[px, py] = 1.0
            return grid

        def step(self):
            for obs in self.dynamic_obs:
                obs['pos'] += obs['vel']
                for i in range(2):
                    if obs['pos'][i] < 0 or obs['pos'][i] >= self.size:
                        obs['vel'][i] *= -1
                        obs['pos'][i] = np.clip(obs['pos'][i], 0, self.size-0.01)

    class HybridPlanner:
        def __init__(self, model, n_samples=N_SAMPLES):
            self.model = model
            self.n_steps = N_DIFFUSION_STEPS  # 100 steps for higher quality
            self.n_samples = n_samples  # Multi-sample inference
            self.n_refine = N_REFINE_STEPS  # More refinement iterations
            # Cosine noise schedule (better than linear)
            s = 0.008
            steps = torch.linspace(0, self.n_steps, self.n_steps + 1)
            alphas_bar = torch.cos(((steps / self.n_steps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_bar = alphas_bar / alphas_bar[0]
            self.betas = torch.clip(1 - alphas_bar[1:] / alphas_bar[:-1], 0.0001, 0.02).to(DEVICE)
            self.alphas = (1 - self.betas).to(DEVICE)
            self.alphas_bar = torch.cumprod(self.alphas, 0).to(DEVICE)

        def _generate_single(self, grid_t, sg_t):
            """Generate a single path sample"""
            x = torch.randn(1, 2, HORIZON).to(DEVICE)
            
            for i in reversed(range(self.n_steps)):
                t = torch.tensor([[i/self.n_steps]]).to(DEVICE)
                with torch.no_grad():
                    noise_pred = self.model(x, t, grid_t, sg_t)
                alpha = self.alphas[i]
                alpha_bar = self.alphas_bar[i]
                beta = self.betas[i]
                noise = torch.randn_like(x) if i > 0 else 0
                x = (1/torch.sqrt(alpha)) * (x - ((1-alpha)/torch.sqrt(1-alpha_bar))*noise_pred) + torch.sqrt(beta)*noise
                x[:, :, 0] = sg_t[:, :2]
                x[:, :, -1] = sg_t[:, 2:]
            return x.squeeze().cpu().numpy().T * MAP_SIZE

        def _refine_path(self, traj, sdf):
            """SDF-based trajectory refinement with more iterations"""
            traj_opt = traj.copy()
            lr = 0.5
            for _ in range(self.n_refine):  # More iterations
                for t in range(1, HORIZON-1):
                    px, py = int(traj_opt[t, 0]), int(traj_opt[t, 1])
                    if 0 <= px < MAP_SIZE and 0 <= py < MAP_SIZE:
                        dist = sdf[px, py]
                        if dist < OBS_RADIUS + 1:
                            dx = sdf[min(px+1, MAP_SIZE-1), py] - sdf[max(px-1, 0), py]
                            dy = sdf[px, min(py+1, MAP_SIZE-1)] - sdf[px, max(py-1, 0)]
                            traj_opt[t, 0] += dx * lr
                            traj_opt[t, 1] += dy * lr
                traj_opt[1:-1] = 0.5 * traj_opt[1:-1] + 0.25 * traj_opt[:-2] + 0.25 * traj_opt[2:]
            return traj_opt

        def _score_path(self, traj, sdf):
            """Score a path based on collision risk (lower = better)"""
            score = 0
            for t in range(len(traj)):
                px, py = int(np.clip(traj[t, 0], 0, MAP_SIZE-1)), int(np.clip(traj[t, 1], 0, MAP_SIZE-1))
                dist = sdf[px, py]
                if dist < OBS_RADIUS + 1:
                    score += (OBS_RADIUS + 1 - dist) ** 2  # Penalty for being close to obstacles
            return score

        def generate_and_refine(self, grid_np, start, goal):
            """Generate N samples, refine all, pick the best"""
            self.model.eval()
            grid_t = torch.tensor(grid_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            sg_norm = np.array([start[0], start[1], goal[0], goal[1]]) / MAP_SIZE
            sg_t = torch.tensor(sg_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            sdf = distance_transform_edt(1 - grid_np)
            
            # Multi-sample inference: generate N paths, refine, pick best
            best_path = None
            best_score = float('inf')
            
            for _ in range(self.n_samples):
                raw_path = self._generate_single(grid_t, sg_t)
                refined_path = self._refine_path(raw_path, sdf)
                score = self._score_path(refined_path, sdf)
                
                if score < best_score:
                    best_score = score
                    best_path = refined_path
            
            return best_path

    import heapq
    class SpaceTimeAStar:
        def __init__(self, env):
            self.env = env
        def plan(self):
            start = tuple(self.env.start)
            goal = tuple(self.env.goal)
            pq = [(0, 0, start[0], start[1], 0, [start])]
            visited = set()
            future_grids = {}
            temp_obs = [{'pos': o['pos'].copy(), 'vel': o['vel'].copy()} for o in self.env.dynamic_obs]
            for t in range(HORIZON + 20):
                future_grids[t] = [o['pos'].copy() for o in temp_obs]
                for o in temp_obs:
                    o['pos'] += o['vel']
                    for i in range(2):
                        if o['pos'][i] < 0 or o['pos'][i] >= self.env.size:
                            o['vel'][i] *= -1
            while pq:
                f, g, r, c, t, path = heapq.heappop(pq)
                if t >= HORIZON: continue
                if np.linalg.norm([r - goal[0], c - goal[1]]) < 2:
                    return np.array(path)
                state = (r, c, t)
                if state in visited: continue
                visited.add(state)
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        nt = t + 1
                        if 0 <= nr < self.env.size and 0 <= nc < self.env.size:
                            if self.env.static_grid[nr, nc] == 1: continue
                            dynamic_collision = False
                            for d_pos in future_grids[nt]:
                                if np.linalg.norm([nr - d_pos[0], nc - d_pos[1]]) < OBS_RADIUS + 1:
                                    dynamic_collision = True; break
                            if not dynamic_collision:
                                h = np.linalg.norm([nr - goal[0], nc - goal[1]])
                                new_g = g + np.linalg.norm([dr, dc])
                                heapq.heappush(pq, (new_g + h, new_g, nr, nc, nt, path + [(nr, nc)]))
            return None

    results = {"Hybrid-Diffusion": 0, "Oracle-A*": 0}
    n_eval = 50  # More episodes for statistical significance
    hybrid_planner = HybridPlanner(model, n_samples=N_SAMPLES)
    print(f"Evaluating with {N_SAMPLES} samples per episode...")
    
    for i in tqdm(range(n_eval), desc="Evaluating"):
        env = DynamicEnv(seed=1000+i)
        path_diff = hybrid_planner.generate_and_refine(env.get_grid(), env.start, env.goal)
        
        sim_env = DynamicEnv(seed=1000+i)
        success_diff = True
        for pt in path_diff:
            sim_env.step()
            if not (0 <= pt[0] < MAP_SIZE and 0 <= pt[1] < MAP_SIZE): success_diff = False; break
            if sim_env.static_grid[int(pt[0]), int(pt[1])] == 1: success_diff = False; break
            for d in sim_env.dynamic_obs:
                if np.linalg.norm(pt - d['pos']) < OBS_RADIUS: success_diff = False; break
        if success_diff: results["Hybrid-Diffusion"] += 1
        
        sim_env = DynamicEnv(seed=1000+i)
        planner_base = SpaceTimeAStar(sim_env)
        path_base = planner_base.plan()
        if path_base is not None: results["Oracle-A*"] += 1
    
    print("\n" + "="*50)
    print("=== FINAL RESEARCH RESULTS ===")
    print("="*50)
    print(f"Success Rate over {n_eval} complex dynamic episodes:")
    print(f"Hybrid Diffusion Planner: {results['Hybrid-Diffusion']/n_eval * 100:.1f}%")
    print(f"Oracle A* (Perfect LSTM): {results['Oracle-A*']/n_eval * 100:.1f}%")
    
    return results

# ==============================================================================
# 4. MAIN ENTRYPOINT
# ==============================================================================

@app.local_entrypoint()
def main():
    print("🚀 Starting ENHANCED Diffusion Research Playground v2")
    print("   - Phase 1: 50x CPU workers generating 100 trajectories each (5000 total)")
    print("   - Phase 2: 2000 epochs training with EMA + cosine LR")
    print("   - Phase 3: Deeper model with attention + residual blocks")
    print("   - Phase 4: Multi-sample inference (10 samples, cosine noise schedule)")
    
    # Phase 1: Parallel trajectory generation (50 workers × 100 = 5000 trajectories)
    print("\n[Phase 1] Launching parallel trajectory generation...")
    chunks = list(generate_trajectory_chunk.map(
        range(50),  # 50 workers (2x more)
        kwargs={'n_trajectories': 100}  # 100 each = 5000 total
    ))
    
    total_trajs = sum(len(c['grids']) for c in chunks)
    print(f"✅ Generated {total_trajs} trajectories in parallel!")
    
    # Phase 2-4: Train and evaluate on GPU
    results = train_and_evaluate.remote(chunks)
    
    print("\n🎉 Research complete!")
    return results
