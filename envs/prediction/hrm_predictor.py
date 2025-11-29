"""
HRM-based Obstacle Trajectory Predictor.

Combines HRM predictions with Kalman filtering for robust
obstacle trajectory forecasting.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from .kalman import AdaptiveKalmanFilter, MultiObstacleKalmanTracker


@dataclass
class PredictionResult:
    """Result of trajectory prediction."""
    
    # Predicted positions: (num_obstacles, prediction_horizon, 2)
    positions: np.ndarray
    
    # Prediction uncertainties: (num_obstacles, prediction_horizon, 2)
    uncertainties: np.ndarray
    
    # HRM contribution weight (for debugging)
    hrm_weight: float
    
    # Kalman contribution weight
    kalman_weight: float


class TrajectoryEncoder(nn.Module):
    """
    Encodes obstacle trajectory history for HRM input.
    
    Takes raw (x, y) positions and produces embeddings.
    """
    
    def __init__(
        self,
        hidden_size: int = 128,
        history_length: int = 10,
        num_obstacles: int = 5,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.history_length = history_length
        self.num_obstacles = num_obstacles
        
        # Position embedding
        self.pos_embed = nn.Sequential(
            nn.Linear(2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        
        # Temporal encoding
        self.temporal_embed = nn.Embedding(history_length, hidden_size)
        
        # Obstacle ID embedding
        self.obstacle_embed = nn.Embedding(num_obstacles, hidden_size)
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self,
        obstacle_history: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode obstacle trajectories.
        
        Args:
            obstacle_history: (batch, num_obstacles, history_length, 2)
            
        Returns:
            Encoded trajectories: (batch, num_obstacles * history_length, hidden_size)
        """
        batch_size = obstacle_history.shape[0]
        
        # Position embeddings
        pos_emb = self.pos_embed(obstacle_history)  # (B, N, T, H)
        
        # Temporal embeddings
        time_ids = torch.arange(
            self.history_length,
            device=obstacle_history.device
        )
        temp_emb = self.temporal_embed(time_ids)  # (T, H)
        temp_emb = temp_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, T, H)
        
        # Obstacle embeddings
        obs_ids = torch.arange(
            self.num_obstacles,
            device=obstacle_history.device
        )
        obs_emb = self.obstacle_embed(obs_ids)  # (N, H)
        obs_emb = obs_emb.unsqueeze(0).unsqueeze(2)  # (1, N, 1, H)
        
        # Combine embeddings
        combined = pos_emb + temp_emb + obs_emb  # (B, N, T, H)
        
        # Reshape to sequence
        combined = combined.view(batch_size, -1, self.hidden_size)
        
        return self.norm(combined)


class TrajectoryDecoder(nn.Module):
    """
    Decodes HRM output to predicted trajectories.
    """
    
    def __init__(
        self,
        hidden_size: int = 128,
        prediction_horizon: int = 5,
        num_obstacles: int = 5,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.prediction_horizon = prediction_horizon
        self.num_obstacles = num_obstacles
        
        # Position prediction head
        self.pos_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, prediction_horizon * 2),
        )
        
        # Uncertainty prediction head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, prediction_horizon * 2),
            nn.Softplus(),  # Ensure positive uncertainty
        )
    
    def forward(
        self,
        hrm_output: torch.Tensor,
        num_obstacles: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode HRM output to predictions.
        
        Args:
            hrm_output: (batch, seq_len, hidden_size)
            num_obstacles: Number of obstacles
            
        Returns:
            positions: (batch, num_obstacles, prediction_horizon, 2)
            uncertainties: (batch, num_obstacles, prediction_horizon, 2)
        """
        batch_size = hrm_output.shape[0]
        
        # Take the obstacle-specific outputs
        # Assuming first N tokens correspond to N obstacles
        obstacle_outputs = hrm_output[:, :num_obstacles]  # (B, N, H)
        
        # Predict positions
        pos_pred = self.pos_head(obstacle_outputs)  # (B, N, T*2)
        pos_pred = pos_pred.view(
            batch_size, num_obstacles, self.prediction_horizon, 2
        )
        
        # Predict uncertainties
        unc_pred = self.uncertainty_head(obstacle_outputs)  # (B, N, T*2)
        unc_pred = unc_pred.view(
            batch_size, num_obstacles, self.prediction_horizon, 2
        )
        
        return pos_pred, unc_pred


class HRMObstaclePredictor:
    """
    Hybrid predictor combining HRM with Kalman filtering.
    
    Uses HRM for pattern recognition and Kalman filter for
    robustness to noise and rapid changes.
    """
    
    def __init__(
        self,
        hrm_model: Optional[nn.Module] = None,
        hidden_size: int = 128,
        history_length: int = 10,
        prediction_horizon: int = 5,
        num_obstacles: int = 5,
        kalman_weight: float = 0.3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            hrm_model: Pre-trained HRM model (optional, uses MLP if None)
            hidden_size: Hidden dimension for trajectory encoding
            history_length: Number of past timesteps to use
            prediction_horizon: Number of future steps to predict
            num_obstacles: Maximum number of obstacles
            kalman_weight: Weight for Kalman predictions (0-1)
            device: Device to run on
        """
        self.hidden_size = hidden_size
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.num_obstacles = num_obstacles
        self.kalman_weight = kalman_weight
        self.device = device
        
        # Initialize encoder/decoder
        self.encoder = TrajectoryEncoder(
            hidden_size=hidden_size,
            history_length=history_length,
            num_obstacles=num_obstacles,
        ).to(device)
        
        self.decoder = TrajectoryDecoder(
            hidden_size=hidden_size,
            prediction_horizon=prediction_horizon,
            num_obstacles=num_obstacles,
        ).to(device)
        
        # HRM model (or simple transformer if none provided)
        if hrm_model is not None:
            self.hrm_model = hrm_model.to(device)
        else:
            # Simple transformer for standalone use
            self.hrm_model = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=4,
                    dim_feedforward=hidden_size * 4,
                    dropout=0.1,
                    batch_first=True,
                ),
                num_layers=3,
            ).to(device)
        
        # Kalman tracker
        self.kalman_tracker = MultiObstacleKalmanTracker(
            num_obstacles=num_obstacles,
            adaptive=True,
        )
        
        # Adaptive fusion weight
        self._hrm_error_history = []
        self._kalman_error_history = []
        self._adaptive_hrm_weight = 1.0 - kalman_weight
    
    def predict(
        self,
        obstacle_history: np.ndarray,
        current_positions: Optional[np.ndarray] = None,
        steps_ahead: int = 5,
    ) -> PredictionResult:
        """
        Predict future obstacle positions.
        
        Args:
            obstacle_history: (num_obstacles, history_length, 2) historical positions
            current_positions: (num_obstacles, 2) current positions for Kalman update
            steps_ahead: Number of steps to predict
            
        Returns:
            PredictionResult with predicted positions and uncertainties
        """
        num_obs = obstacle_history.shape[0]
        steps = min(steps_ahead, self.prediction_horizon)
        
        # Update Kalman filters with current positions
        if current_positions is not None:
            self.kalman_tracker.update(current_positions)
        
        # Get Kalman predictions
        kalman_predictions = self.kalman_tracker.predict_trajectories(steps)
        kalman_uncertainties = self.kalman_tracker.get_uncertainties()
        
        # Expand uncertainties to match prediction shape
        kalman_unc_expanded = np.tile(
            kalman_uncertainties[:, np.newaxis, :],
            (1, steps, 1)
        )
        
        # Get HRM predictions
        with torch.no_grad():
            # Prepare input
            history_tensor = torch.from_numpy(
                obstacle_history
            ).float().unsqueeze(0).to(self.device)  # (1, N, T, 2)
            
            # Encode
            encoded = self.encoder(history_tensor)
            
            # Process through HRM
            hrm_output = self.hrm_model(encoded)
            
            # Decode
            hrm_positions, hrm_uncertainties = self.decoder(hrm_output, num_obs)
            
            hrm_positions = hrm_positions[0, :, :steps].cpu().numpy()
            hrm_uncertainties = hrm_uncertainties[0, :, :steps].cpu().numpy()
        
        # Fuse predictions (weighted average based on uncertainties)
        hrm_weight = self._adaptive_hrm_weight
        kalman_weight = self.kalman_weight
        
        # Normalize weights
        total_weight = hrm_weight + kalman_weight
        hrm_weight /= total_weight
        kalman_weight /= total_weight
        
        # Combine predictions
        # Use inverse variance weighting
        hrm_var = hrm_uncertainties + 1e-6
        kalman_var = kalman_unc_expanded + 1e-6
        
        combined_var = 1.0 / (1.0 / hrm_var + 1.0 / kalman_var)
        
        fused_positions = (
            hrm_positions / hrm_var + 
            kalman_predictions / kalman_var
        ) * combined_var
        
        return PredictionResult(
            positions=fused_positions,
            uncertainties=combined_var,
            hrm_weight=hrm_weight,
            kalman_weight=kalman_weight,
        )
    
    def update_with_ground_truth(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
    ):
        """
        Update adaptive weights based on prediction errors.
        
        Args:
            predicted: (num_obstacles, steps, 2) predicted positions
            actual: (num_obstacles, steps, 2) actual positions
        """
        error = np.mean(np.square(predicted - actual))
        
        # This would track HRM vs Kalman errors separately
        # for more sophisticated weight adaptation
        self._hrm_error_history.append(error)
        if len(self._hrm_error_history) > 100:
            self._hrm_error_history.pop(0)
    
    def reset(self):
        """Reset predictor state."""
        self.kalman_tracker.reset()
        self._hrm_error_history = []
        self._kalman_error_history = []
        self._adaptive_hrm_weight = 1.0 - self.kalman_weight
    
    def train_mode(self, mode: bool = True):
        """Set training mode for neural components."""
        self.encoder.train(mode)
        self.decoder.train(mode)
        self.hrm_model.train(mode)
    
    def eval_mode(self):
        """Set evaluation mode."""
        self.train_mode(False)
    
    def get_parameters(self):
        """Get all trainable parameters."""
        params = list(self.encoder.parameters())
        params += list(self.decoder.parameters())
        params += list(self.hrm_model.parameters())
        return params
    
    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'hrm_model': self.hrm_model.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.hrm_model.load_state_dict(checkpoint['hrm_model'])


class HRMPredictorWrapper(nn.Module):
    """
    PyTorch module wrapper for HRM predictor.
    
    Useful for end-to-end training with the RL agent.
    """
    
    def __init__(
        self,
        hidden_size: int = 128,
        history_length: int = 10,
        prediction_horizon: int = 5,
        num_obstacles: int = 5,
    ):
        super().__init__()
        
        self.encoder = TrajectoryEncoder(
            hidden_size=hidden_size,
            history_length=history_length,
            num_obstacles=num_obstacles,
        )
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=3,
        )
        
        self.decoder = TrajectoryDecoder(
            hidden_size=hidden_size,
            prediction_horizon=prediction_horizon,
            num_obstacles=num_obstacles,
        )
        
        self.num_obstacles = num_obstacles
    
    def forward(
        self,
        obstacle_history: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for trajectory prediction.
        
        Args:
            obstacle_history: (batch, num_obstacles, history_length, 2)
            
        Returns:
            positions: (batch, num_obstacles, prediction_horizon, 2)
            uncertainties: (batch, num_obstacles, prediction_horizon, 2)
        """
        encoded = self.encoder(obstacle_history)
        transformed = self.transformer(encoded)
        return self.decoder(transformed, self.num_obstacles)

