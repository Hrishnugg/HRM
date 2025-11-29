"""
Kalman Filter for obstacle trajectory prediction.

Implements adaptive Kalman filtering for robust trajectory prediction
as described in the LSTM-A* paper.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class KalmanState:
    """State of a Kalman filter for 2D position tracking."""
    
    # State vector: [x, y, vx, vy]
    x: np.ndarray = field(default_factory=lambda: np.zeros(4))
    
    # State covariance matrix
    P: np.ndarray = field(default_factory=lambda: np.eye(4) * 100.0)
    
    # Process noise covariance
    Q: np.ndarray = field(default_factory=lambda: np.eye(4) * 0.1)
    
    # Measurement noise covariance
    R: np.ndarray = field(default_factory=lambda: np.eye(2) * 1.0)


class KalmanFilter:
    """
    Standard Kalman Filter for 2D obstacle tracking.
    
    State vector: [x, y, vx, vy] (position and velocity)
    Measurement: [x, y] (position only)
    
    Uses constant velocity model.
    """
    
    def __init__(
        self,
        initial_position: Optional[Tuple[float, float]] = None,
        process_noise: float = 0.1,
        measurement_noise: float = 1.0,
        dt: float = 1.0,
    ):
        """
        Initialize Kalman filter.
        
        Args:
            initial_position: Starting (x, y) position
            process_noise: Process noise variance
            measurement_noise: Measurement noise variance
            dt: Time step
        """
        self.dt = dt
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        
        # Measurement matrix (observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        
        # Process noise
        self.Q = np.eye(4, dtype=np.float32) * process_noise
        # Higher noise for velocity components
        self.Q[2, 2] = process_noise * 2
        self.Q[3, 3] = process_noise * 2
        
        # Measurement noise
        self.R = np.eye(2, dtype=np.float32) * measurement_noise
        
        # Initialize state
        self.x = np.zeros(4, dtype=np.float32)
        if initial_position is not None:
            self.x[0] = initial_position[0]
            self.x[1] = initial_position[1]
        
        # State covariance (high initial uncertainty)
        self.P = np.eye(4, dtype=np.float32) * 100.0
        
        # Innovation (for adaptive filtering)
        self.innovation = np.zeros(2, dtype=np.float32)
        self.innovation_covariance = np.eye(2, dtype=np.float32)
    
    def predict(self) -> np.ndarray:
        """
        Predict next state.
        
        Returns:
            Predicted state vector [x, y, vx, vy]
        """
        # State prediction
        self.x = self.F @ self.x
        
        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x.copy()
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update state with measurement.
        
        Args:
            measurement: Observed position [x, y]
            
        Returns:
            Updated state vector [x, y, vx, vy]
        """
        measurement = np.asarray(measurement, dtype=np.float32)
        
        # Innovation (measurement residual)
        y = measurement - self.H @ self.x
        self.innovation = y.copy()
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        self.innovation_covariance = S.copy()
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + K @ y
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        return self.x.copy()
    
    def predict_trajectory(self, steps: int) -> np.ndarray:
        """
        Predict future trajectory.
        
        Args:
            steps: Number of future steps to predict
            
        Returns:
            Predicted positions of shape (steps, 2)
        """
        trajectory = np.zeros((steps, 2), dtype=np.float32)
        
        # Save current state
        x_save = self.x.copy()
        P_save = self.P.copy()
        
        for i in range(steps):
            self.predict()
            trajectory[i] = self.x[:2]
        
        # Restore state
        self.x = x_save
        self.P = P_save
        
        return trajectory
    
    def get_position(self) -> np.ndarray:
        """Get current position estimate."""
        return self.x[:2].copy()
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate."""
        return self.x[2:4].copy()
    
    def get_uncertainty(self) -> np.ndarray:
        """Get position uncertainty (diagonal of P for position)."""
        return np.array([self.P[0, 0], self.P[1, 1]], dtype=np.float32)
    
    def reset(self, position: Optional[Tuple[float, float]] = None):
        """Reset filter state."""
        self.x = np.zeros(4, dtype=np.float32)
        if position is not None:
            self.x[0] = position[0]
            self.x[1] = position[1]
        self.P = np.eye(4, dtype=np.float32) * 100.0


class AdaptiveKalmanFilter(KalmanFilter):
    """
    Adaptive Kalman Filter that adjusts noise parameters based on innovation.
    
    Uses the innovation-based adaptive estimation (IAE) approach
    to handle changing dynamics and measurement noise.
    """
    
    def __init__(
        self,
        initial_position: Optional[Tuple[float, float]] = None,
        process_noise: float = 0.1,
        measurement_noise: float = 1.0,
        dt: float = 1.0,
        adaptation_window: int = 10,
        min_noise: float = 0.01,
        max_noise: float = 10.0,
    ):
        """
        Args:
            initial_position: Starting position
            process_noise: Initial process noise
            measurement_noise: Initial measurement noise
            dt: Time step
            adaptation_window: Window size for noise estimation
            min_noise: Minimum allowed noise value
            max_noise: Maximum allowed noise value
        """
        super().__init__(initial_position, process_noise, measurement_noise, dt)
        
        self.adaptation_window = adaptation_window
        self.min_noise = min_noise
        self.max_noise = max_noise
        
        # Innovation history for adaptation
        self.innovation_history = []
        
        # Store initial noise values
        self.initial_Q = self.Q.copy()
        self.initial_R = self.R.copy()
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update state with measurement and adapt noise parameters.
        
        Args:
            measurement: Observed position [x, y]
            
        Returns:
            Updated state vector
        """
        # Standard Kalman update
        state = super().update(measurement)
        
        # Store innovation for adaptation
        self.innovation_history.append(self.innovation.copy())
        if len(self.innovation_history) > self.adaptation_window:
            self.innovation_history.pop(0)
        
        # Adapt noise if we have enough history
        if len(self.innovation_history) >= self.adaptation_window:
            self._adapt_noise()
        
        return state
    
    def _adapt_noise(self):
        """Adapt noise parameters based on innovation sequence."""
        innovations = np.array(self.innovation_history)
        
        # Estimate innovation covariance
        innovation_cov = np.cov(innovations.T)
        if innovation_cov.ndim == 0:
            innovation_cov = np.array([[innovation_cov]])
        
        # Expected innovation covariance from filter
        expected_cov = self.H @ self.P @ self.H.T + self.R
        
        # Ratio indicates if we need more/less noise
        ratio = np.mean(np.diag(innovation_cov)) / np.mean(np.diag(expected_cov))
        
        # Adapt measurement noise
        if ratio > 1.5:
            # Innovation too large -> increase R
            scale = min(ratio, 2.0)
            self.R = np.clip(
                self.R * scale,
                self.min_noise,
                self.max_noise
            )
        elif ratio < 0.5:
            # Innovation too small -> decrease R
            scale = max(ratio, 0.5)
            self.R = np.clip(
                self.R * scale,
                self.min_noise,
                self.max_noise
            )
        
        # Also adapt process noise based on prediction errors
        velocity_change = np.abs(innovations[-1] - innovations[-2]) if len(innovations) > 1 else 0
        avg_velocity_change = np.mean(velocity_change)
        
        if avg_velocity_change > 2.0:
            # Rapid changes -> increase Q
            self.Q = np.clip(
                self.Q * 1.2,
                self.min_noise,
                self.max_noise
            )
        elif avg_velocity_change < 0.5:
            # Smooth motion -> decrease Q
            self.Q = np.clip(
                self.Q * 0.9,
                self.min_noise,
                self.max_noise
            )
    
    def reset(self, position: Optional[Tuple[float, float]] = None):
        """Reset filter including noise adaptation."""
        super().reset(position)
        self.innovation_history = []
        self.Q = self.initial_Q.copy()
        self.R = self.initial_R.copy()
    
    def get_adapted_noise(self) -> Tuple[float, float]:
        """Get current adapted noise values."""
        return (
            float(np.mean(np.diag(self.Q))),
            float(np.mean(np.diag(self.R)))
        )


class MultiObstacleKalmanTracker:
    """
    Tracks multiple obstacles with individual Kalman filters.
    """
    
    def __init__(
        self,
        num_obstacles: int,
        adaptive: bool = True,
        **kalman_kwargs
    ):
        """
        Args:
            num_obstacles: Number of obstacles to track
            adaptive: Use adaptive Kalman filters
            **kalman_kwargs: Arguments passed to Kalman filter constructors
        """
        self.num_obstacles = num_obstacles
        self.adaptive = adaptive
        
        FilterClass = AdaptiveKalmanFilter if adaptive else KalmanFilter
        self.filters = [FilterClass(**kalman_kwargs) for _ in range(num_obstacles)]
    
    def update(self, measurements: np.ndarray) -> np.ndarray:
        """
        Update all filters with measurements.
        
        Args:
            measurements: Positions of shape (num_obstacles, 2)
            
        Returns:
            Updated states of shape (num_obstacles, 4)
        """
        states = np.zeros((self.num_obstacles, 4), dtype=np.float32)
        
        for i, (filter_, measurement) in enumerate(zip(self.filters, measurements)):
            filter_.predict()
            states[i] = filter_.update(measurement)
        
        return states
    
    def predict_trajectories(self, steps: int) -> np.ndarray:
        """
        Predict future trajectories for all obstacles.
        
        Args:
            steps: Number of steps to predict
            
        Returns:
            Predicted positions of shape (num_obstacles, steps, 2)
        """
        trajectories = np.zeros(
            (self.num_obstacles, steps, 2), dtype=np.float32
        )
        
        for i, filter_ in enumerate(self.filters):
            trajectories[i] = filter_.predict_trajectory(steps)
        
        return trajectories
    
    def get_positions(self) -> np.ndarray:
        """Get current position estimates for all obstacles."""
        return np.array([f.get_position() for f in self.filters], dtype=np.float32)
    
    def get_velocities(self) -> np.ndarray:
        """Get current velocity estimates for all obstacles."""
        return np.array([f.get_velocity() for f in self.filters], dtype=np.float32)
    
    def get_uncertainties(self) -> np.ndarray:
        """Get position uncertainties for all obstacles."""
        return np.array([f.get_uncertainty() for f in self.filters], dtype=np.float32)
    
    def reset(self, positions: Optional[np.ndarray] = None):
        """Reset all filters."""
        for i, filter_ in enumerate(self.filters):
            pos = tuple(positions[i]) if positions is not None else None
            filter_.reset(pos)

