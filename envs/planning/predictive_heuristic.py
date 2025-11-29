"""
Predictive Heuristic for HRM-Augmented A*.

Integrates obstacle trajectory predictions into the A* heuristic
to enable proactive collision avoidance.
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass

from .astar import octile_distance


@dataclass
class CollisionRiskConfig:
    """Configuration for collision risk estimation."""
    
    # Spatial influence radius
    influence_radius: float = 3.0
    
    # Maximum penalty for collision risk
    max_penalty: float = 10.0
    
    # Decay factor for uncertainty
    uncertainty_decay: float = 0.5
    
    # Minimum safety margin
    safety_margin: float = 1.0
    
    # Temporal discount (future risks weighted less)
    temporal_discount: float = 0.9


class CollisionRiskEstimator:
    """
    Estimates collision risk based on predicted obstacle trajectories.
    
    Used by the predictive heuristic to inflate costs near
    predicted obstacle positions.
    """
    
    def __init__(self, config: Optional[CollisionRiskConfig] = None):
        """
        Args:
            config: Risk estimation configuration
        """
        self.config = config or CollisionRiskConfig()
    
    def estimate_risk(
        self,
        position: Tuple[int, int],
        predicted_obstacles: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
        arrival_time: int = 0,
    ) -> float:
        """
        Estimate collision risk at a position.
        
        Args:
            position: (x, y) position to evaluate
            predicted_obstacles: (num_obstacles, prediction_horizon, 2) predictions
            uncertainties: (num_obstacles, prediction_horizon, 2) prediction uncertainties
            arrival_time: Estimated arrival time at this position
            
        Returns:
            Collision risk value (higher = more dangerous)
        """
        if predicted_obstacles is None or len(predicted_obstacles) == 0:
            return 0.0
        
        pos = np.array(position, dtype=np.float32)
        num_obstacles, horizon, _ = predicted_obstacles.shape
        
        # Time to consider (clamped to prediction horizon)
        t = min(arrival_time, horizon - 1)
        
        total_risk = 0.0
        
        for i in range(num_obstacles):
            # Get predicted position at arrival time
            pred_pos = predicted_obstacles[i, t]
            
            # Distance to predicted obstacle position
            distance = np.linalg.norm(pos - pred_pos)
            
            # Within influence radius?
            if distance < self.config.influence_radius:
                # Base risk from proximity
                proximity_risk = 1.0 - (distance / self.config.influence_radius)
                
                # Scale by uncertainty if available
                if uncertainties is not None:
                    unc = np.mean(uncertainties[i, t])
                    # Higher uncertainty = spread risk over larger area
                    # but reduce peak risk
                    uncertainty_factor = np.exp(-unc * self.config.uncertainty_decay)
                    proximity_risk *= uncertainty_factor
                
                # Apply temporal discount (future predictions less reliable)
                temporal_factor = self.config.temporal_discount ** t
                proximity_risk *= temporal_factor
                
                # Also consider nearby timesteps
                for dt in [-1, 1]:
                    t_adj = t + dt
                    if 0 <= t_adj < horizon:
                        pred_adj = predicted_obstacles[i, t_adj]
                        dist_adj = np.linalg.norm(pos - pred_adj)
                        if dist_adj < self.config.influence_radius:
                            adj_risk = (1.0 - dist_adj / self.config.influence_radius) * 0.5
                            proximity_risk = max(proximity_risk, adj_risk)
                
                total_risk += proximity_risk
        
        # Scale and clamp
        return min(total_risk * self.config.max_penalty, self.config.max_penalty)
    
    def get_risk_map(
        self,
        grid_shape: Tuple[int, int],
        predicted_obstacles: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
        time_step: int = 0,
    ) -> np.ndarray:
        """
        Generate a risk map for visualization.
        
        Args:
            grid_shape: (height, width) of the grid
            predicted_obstacles: Predicted obstacle positions
            uncertainties: Prediction uncertainties
            time_step: Time step to visualize
            
        Returns:
            Risk map of shape (height, width)
        """
        height, width = grid_shape
        risk_map = np.zeros((height, width), dtype=np.float32)
        
        for y in range(height):
            for x in range(width):
                risk_map[y, x] = self.estimate_risk(
                    (x, y),
                    predicted_obstacles,
                    uncertainties,
                    time_step,
                )
        
        return risk_map


class PredictiveHeuristic:
    """
    Predictive heuristic function for A*.
    
    Combines base heuristic (e.g., octile distance) with
    collision risk from predicted obstacle trajectories.
    """
    
    def __init__(
        self,
        predictor,  # HRMObstaclePredictor
        base_heuristic: Callable = octile_distance,
        risk_weight: float = 1.0,
        risk_config: Optional[CollisionRiskConfig] = None,
    ):
        """
        Args:
            predictor: HRM obstacle predictor
            base_heuristic: Base heuristic function(node, goal) -> float
            risk_weight: Weight for collision risk penalty
            risk_config: Collision risk configuration
        """
        self.predictor = predictor
        self.base_heuristic = base_heuristic
        self.risk_weight = risk_weight
        self.risk_estimator = CollisionRiskEstimator(risk_config)
        
        # Cache for predictions (avoid recomputing)
        self._cached_predictions = None
        self._cached_uncertainties = None
        self._cache_time = -1
    
    def update_predictions(
        self,
        obstacle_history: np.ndarray,
        current_positions: np.ndarray,
        current_time: int,
    ):
        """
        Update cached obstacle predictions.
        
        Call this when obstacle positions change.
        
        Args:
            obstacle_history: Historical positions
            current_positions: Current obstacle positions
            current_time: Current simulation time
        """
        result = self.predictor.predict(
            obstacle_history,
            current_positions,
        )
        
        self._cached_predictions = result.positions
        self._cached_uncertainties = result.uncertainties
        self._cache_time = current_time
    
    def __call__(
        self,
        node: Tuple[int, int],
        goal: Tuple[int, int],
        time_step: int = 0,
    ) -> float:
        """
        Compute predictive heuristic value.
        
        Args:
            node: Current position
            goal: Goal position
            time_step: Current time step (for arrival time estimation)
            
        Returns:
            Heuristic value (base + risk penalty)
        """
        # Base heuristic
        base_h = self.base_heuristic(node, goal)
        
        # No predictions available
        if self._cached_predictions is None:
            return base_h
        
        # Estimate arrival time at this node
        # (assuming movement at approximately 1 cell per timestep)
        arrival_time = time_step + int(base_h)
        
        # Compute collision risk
        risk = self.risk_estimator.estimate_risk(
            node,
            self._cached_predictions,
            self._cached_uncertainties,
            arrival_time,
        )
        
        # Combined heuristic
        return base_h + self.risk_weight * risk
    
    def get_heuristic_function(
        self,
        time_step: int = 0,
    ) -> Callable:
        """
        Get a heuristic function for use with A*.
        
        Args:
            time_step: Current time step
            
        Returns:
            Callable heuristic function(node, goal) -> float
        """
        def heuristic(node: Tuple[int, int], goal: Tuple[int, int]) -> float:
            return self(node, goal, time_step)
        
        return heuristic
    
    def needs_replan(
        self,
        current_position: Tuple[int, int],
        current_path: list,
        threshold: float = 5.0,
    ) -> bool:
        """
        Check if replanning is needed based on risk changes.
        
        Args:
            current_position: Agent's current position
            current_path: Current planned path
            threshold: Risk threshold for triggering replan
            
        Returns:
            True if replanning is recommended
        """
        if self._cached_predictions is None or not current_path:
            return False
        
        # Check risk along remaining path
        for i, pos in enumerate(current_path):
            risk = self.risk_estimator.estimate_risk(
                pos,
                self._cached_predictions,
                self._cached_uncertainties,
                i,
            )
            
            if risk > threshold:
                return True
        
        return False
    
    def set_risk_weight(self, weight: float):
        """Update risk weight."""
        self.risk_weight = weight


def create_predictive_heuristic(
    predictor,
    risk_weight: float = 1.0,
    influence_radius: float = 3.0,
    max_penalty: float = 10.0,
) -> PredictiveHeuristic:
    """
    Factory function for creating predictive heuristic.
    
    Args:
        predictor: HRM obstacle predictor
        risk_weight: Weight for collision risk
        influence_radius: Spatial influence of obstacles
        max_penalty: Maximum penalty for collision risk
        
    Returns:
        Configured PredictiveHeuristic instance
    """
    config = CollisionRiskConfig(
        influence_radius=influence_radius,
        max_penalty=max_penalty,
    )
    
    return PredictiveHeuristic(
        predictor=predictor,
        risk_weight=risk_weight,
        risk_config=config,
    )

