"""
Evaluation Metrics for HRM-Augmented A*.

Implements metrics from the LSTM-A* paper:
- Path efficiency
- Computational speed
- Prediction accuracy
- Path smoothness
- Safety margin
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import time


@dataclass
class PathMetrics:
    """Metrics for a single path."""
    
    # Path length (number of steps)
    length: int = 0
    
    # Total Euclidean distance traveled
    distance: float = 0.0
    
    # Path efficiency (actual / optimal)
    efficiency: float = 0.0
    
    # Path smoothness (lower = smoother)
    smoothness: float = 0.0
    
    # Minimum distance to obstacles
    min_safety_margin: float = float('inf')
    
    # Average distance to obstacles
    avg_safety_margin: float = 0.0
    
    # Number of near-misses (close calls)
    near_misses: int = 0
    
    # Whether goal was reached
    goal_reached: bool = False
    
    # Whether collision occurred
    collision: bool = False


@dataclass
class PredictionMetrics:
    """Metrics for trajectory predictions."""
    
    # Mean squared error
    mse: float = 0.0
    
    # Mean absolute error
    mae: float = 0.0
    
    # Root mean squared error
    rmse: float = 0.0
    
    # R-squared (coefficient of determination)
    r_squared: float = 0.0
    
    # Average prediction horizon accuracy
    horizon_accuracy: List[float] = field(default_factory=list)


@dataclass
class ComputationMetrics:
    """Metrics for computational performance."""
    
    # Average planning time per step (seconds)
    avg_planning_time: float = 0.0
    
    # Total planning time
    total_planning_time: float = 0.0
    
    # Number of replans
    num_replans: int = 0
    
    # Average nodes expanded per plan
    avg_nodes_expanded: float = 0.0


@dataclass
class EvaluationResult:
    """Complete evaluation result for an episode or set of episodes."""
    
    # Path metrics
    path_metrics: PathMetrics = field(default_factory=PathMetrics)
    
    # Prediction metrics
    prediction_metrics: PredictionMetrics = field(default_factory=PredictionMetrics)
    
    # Computation metrics
    computation_metrics: ComputationMetrics = field(default_factory=ComputationMetrics)
    
    # Episode statistics
    total_reward: float = 0.0
    episode_length: int = 0
    success_rate: float = 0.0
    
    # Additional info
    num_episodes: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "path/length": self.path_metrics.length,
            "path/distance": self.path_metrics.distance,
            "path/efficiency": self.path_metrics.efficiency,
            "path/smoothness": self.path_metrics.smoothness,
            "path/min_safety_margin": self.path_metrics.min_safety_margin,
            "path/avg_safety_margin": self.path_metrics.avg_safety_margin,
            "path/near_misses": self.path_metrics.near_misses,
            "path/goal_reached": self.path_metrics.goal_reached,
            "path/collision": self.path_metrics.collision,
            "prediction/mse": self.prediction_metrics.mse,
            "prediction/mae": self.prediction_metrics.mae,
            "prediction/rmse": self.prediction_metrics.rmse,
            "prediction/r_squared": self.prediction_metrics.r_squared,
            "computation/avg_planning_time": self.computation_metrics.avg_planning_time,
            "computation/total_planning_time": self.computation_metrics.total_planning_time,
            "computation/num_replans": self.computation_metrics.num_replans,
            "computation/avg_nodes_expanded": self.computation_metrics.avg_nodes_expanded,
            "total_reward": self.total_reward,
            "episode_length": self.episode_length,
            "success_rate": self.success_rate,
            "num_episodes": self.num_episodes,
        }


def compute_path_efficiency(
    path: List[Tuple[int, int]],
    optimal_length: float,
) -> float:
    """
    Compute path efficiency ratio.
    
    Args:
        path: List of (x, y) positions
        optimal_length: Optimal path length (e.g., from static A*)
        
    Returns:
        Efficiency ratio (1.0 = optimal, <1.0 = suboptimal)
    """
    if not path or optimal_length <= 0:
        return 0.0
    
    # Compute actual path length
    actual_length = 0.0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        actual_length += np.sqrt(dx*dx + dy*dy)
    
    if actual_length <= 0:
        return 1.0 if optimal_length <= 0 else 0.0
    
    return optimal_length / actual_length


def compute_path_smoothness(path: List[Tuple[int, int]]) -> float:
    """
    Compute path smoothness based on direction changes.
    
    Lower values indicate smoother paths.
    
    Args:
        path: List of (x, y) positions
        
    Returns:
        Smoothness metric (sum of absolute angle changes)
    """
    if len(path) < 3:
        return 0.0
    
    total_angle_change = 0.0
    
    for i in range(1, len(path) - 1):
        # Direction vectors
        v1 = np.array([
            path[i][0] - path[i-1][0],
            path[i][1] - path[i-1][1]
        ], dtype=np.float32)
        
        v2 = np.array([
            path[i+1][0] - path[i][0],
            path[i+1][1] - path[i][1]
        ], dtype=np.float32)
        
        # Normalize
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 > 0 and norm2 > 0:
            v1 = v1 / norm1
            v2 = v2 / norm2
            
            # Angle between vectors
            cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            total_angle_change += abs(angle)
    
    return total_angle_change


def compute_safety_margin(
    agent_positions: List[Tuple[int, int]],
    obstacle_positions: List[List[Tuple[float, float]]],
    near_miss_threshold: float = 2.0,
) -> Tuple[float, float, int]:
    """
    Compute safety metrics for a trajectory.
    
    Args:
        agent_positions: Agent positions at each timestep
        obstacle_positions: Obstacle positions at each timestep
        near_miss_threshold: Distance threshold for near-miss
        
    Returns:
        (min_distance, avg_distance, near_miss_count)
    """
    min_distance = float('inf')
    total_distance = 0.0
    near_misses = 0
    count = 0
    
    for t, agent_pos in enumerate(agent_positions):
        if t >= len(obstacle_positions):
            break
        
        obs_at_t = obstacle_positions[t]
        
        for obs_pos in obs_at_t:
            distance = np.sqrt(
                (agent_pos[0] - obs_pos[0])**2 +
                (agent_pos[1] - obs_pos[1])**2
            )
            
            min_distance = min(min_distance, distance)
            total_distance += distance
            count += 1
            
            if distance < near_miss_threshold:
                near_misses += 1
    
    avg_distance = total_distance / count if count > 0 else 0.0
    
    return min_distance, avg_distance, near_misses


def compute_prediction_accuracy(
    predictions: np.ndarray,
    actual: np.ndarray,
) -> PredictionMetrics:
    """
    Compute prediction accuracy metrics.
    
    Args:
        predictions: (N, T, 2) predicted positions
        actual: (N, T, 2) actual positions
        
    Returns:
        PredictionMetrics with accuracy statistics
    """
    metrics = PredictionMetrics()
    
    if predictions.size == 0 or actual.size == 0:
        return metrics
    
    # Flatten for overall metrics
    pred_flat = predictions.flatten()
    actual_flat = actual.flatten()
    
    # MSE
    metrics.mse = float(np.mean((pred_flat - actual_flat) ** 2))
    
    # MAE
    metrics.mae = float(np.mean(np.abs(pred_flat - actual_flat)))
    
    # RMSE
    metrics.rmse = float(np.sqrt(metrics.mse))
    
    # R-squared
    ss_res = np.sum((actual_flat - pred_flat) ** 2)
    ss_tot = np.sum((actual_flat - np.mean(actual_flat)) ** 2)
    metrics.r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Per-horizon accuracy
    num_horizons = predictions.shape[1] if len(predictions.shape) > 1 else 1
    for t in range(num_horizons):
        if len(predictions.shape) > 1:
            pred_t = predictions[:, t]
            actual_t = actual[:, t]
        else:
            pred_t = predictions
            actual_t = actual
        
        mse_t = float(np.mean((pred_t - actual_t) ** 2))
        metrics.horizon_accuracy.append(np.exp(-mse_t))  # Exponential accuracy
    
    return metrics


def evaluate_episode(
    env,
    agent,
    predictor=None,
    max_steps: int = 500,
    record_trajectory: bool = True,
) -> EvaluationResult:
    """
    Evaluate a single episode.
    
    Args:
        env: Environment to evaluate in
        agent: Agent to evaluate (must have .act(obs) method)
        predictor: Optional predictor for prediction metrics
        max_steps: Maximum episode length
        record_trajectory: Whether to record full trajectory
        
    Returns:
        EvaluationResult with all metrics
    """
    result = EvaluationResult()
    
    # Reset environment
    obs, info = env.reset()
    
    # Storage for metrics computation
    agent_positions = []
    obstacle_positions = []
    predictions = []
    actual_obstacles = []
    planning_times = []
    
    total_reward = 0.0
    goal_reached = False
    collision = False
    
    for step in range(max_steps):
        # Record positions
        if record_trajectory:
            agent_positions.append(tuple(obs["agent_position"].astype(int)))
            obstacle_positions.append([
                tuple(pos) for pos in obs["obstacle_positions"]
            ])
        
        # Get prediction if predictor available
        if predictor is not None:
            pred_start = time.perf_counter()
            pred_result = predictor.predict(
                obs["obstacle_history"],
                obs["obstacle_positions"],
            )
            predictions.append(pred_result.positions)
            planning_times.append(time.perf_counter() - pred_start)
        
        # Get action
        action_start = time.perf_counter()
        action = agent.act(obs)
        action_time = time.perf_counter() - action_start
        planning_times.append(action_time)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Record actual positions for prediction comparison
        if predictor is not None:
            actual_obstacles.append(obs["obstacle_positions"].copy())
        
        # Check termination
        if terminated:
            goal_reached = info.get("distance_to_goal", 1.0) < 1.0
            collision = info.get("obstacle_collision", False) or info.get("wall_collision", False)
            break
        
        if truncated:
            break
    
    # Compute path metrics
    result.episode_length = len(agent_positions)
    result.total_reward = total_reward
    
    if agent_positions:
        result.path_metrics.length = len(agent_positions)
        result.path_metrics.goal_reached = goal_reached
        result.path_metrics.collision = collision
        
        # Distance traveled
        total_distance = 0.0
        for i in range(1, len(agent_positions)):
            dx = agent_positions[i][0] - agent_positions[i-1][0]
            dy = agent_positions[i][1] - agent_positions[i-1][1]
            total_distance += np.sqrt(dx*dx + dy*dy)
        result.path_metrics.distance = total_distance
        
        # Smoothness
        result.path_metrics.smoothness = compute_path_smoothness(agent_positions)
        
        # Safety metrics
        if obstacle_positions:
            min_safety, avg_safety, near_misses = compute_safety_margin(
                agent_positions, obstacle_positions
            )
            result.path_metrics.min_safety_margin = min_safety
            result.path_metrics.avg_safety_margin = avg_safety
            result.path_metrics.near_misses = near_misses
    
    # Compute prediction metrics
    if predictions and actual_obstacles:
        # Compare predictions at t to actual at t+1, t+2, etc.
        pred_array = np.array(predictions[:-1]) if len(predictions) > 1 else np.array([])
        actual_array = np.array(actual_obstacles[1:]) if len(actual_obstacles) > 1 else np.array([])
        
        if pred_array.size > 0 and actual_array.size > 0:
            # Align shapes (take first step of prediction vs actual)
            min_len = min(len(pred_array), len(actual_array))
            pred_first = pred_array[:min_len, :, 0, :] if pred_array.ndim > 2 else pred_array[:min_len]
            actual_first = actual_array[:min_len]
            
            result.prediction_metrics = compute_prediction_accuracy(
                pred_first, actual_first
            )
    
    # Computation metrics
    if planning_times:
        result.computation_metrics.avg_planning_time = np.mean(planning_times)
        result.computation_metrics.total_planning_time = sum(planning_times)
    
    result.success_rate = 1.0 if goal_reached else 0.0
    
    return result


def evaluate_model(
    env,
    agent,
    predictor=None,
    num_episodes: int = 100,
    max_steps: int = 500,
    verbose: bool = True,
) -> EvaluationResult:
    """
    Evaluate model over multiple episodes.
    
    Args:
        env: Environment
        agent: Agent to evaluate
        predictor: Optional predictor
        num_episodes: Number of episodes
        max_steps: Max steps per episode
        verbose: Print progress
        
    Returns:
        Aggregated EvaluationResult
    """
    all_results = []
    
    for ep in range(num_episodes):
        if verbose and (ep + 1) % 10 == 0:
            print(f"Evaluating episode {ep + 1}/{num_episodes}")
        
        result = evaluate_episode(
            env, agent, predictor, max_steps,
            record_trajectory=True,
        )
        all_results.append(result)
    
    # Aggregate results
    aggregated = EvaluationResult()
    aggregated.num_episodes = num_episodes
    
    # Path metrics (averages)
    aggregated.path_metrics.length = np.mean([r.path_metrics.length for r in all_results])
    aggregated.path_metrics.distance = np.mean([r.path_metrics.distance for r in all_results])
    aggregated.path_metrics.smoothness = np.mean([r.path_metrics.smoothness for r in all_results])
    aggregated.path_metrics.min_safety_margin = np.mean([r.path_metrics.min_safety_margin for r in all_results])
    aggregated.path_metrics.avg_safety_margin = np.mean([r.path_metrics.avg_safety_margin for r in all_results])
    aggregated.path_metrics.near_misses = np.mean([r.path_metrics.near_misses for r in all_results])
    
    # Success rate
    successes = sum(1 for r in all_results if r.path_metrics.goal_reached)
    aggregated.success_rate = successes / num_episodes
    aggregated.path_metrics.goal_reached = aggregated.success_rate > 0.5
    
    # Collision rate
    collisions = sum(1 for r in all_results if r.path_metrics.collision)
    aggregated.path_metrics.collision = collisions / num_episodes > 0.5
    
    # Prediction metrics
    aggregated.prediction_metrics.mse = np.mean([r.prediction_metrics.mse for r in all_results])
    aggregated.prediction_metrics.mae = np.mean([r.prediction_metrics.mae for r in all_results])
    aggregated.prediction_metrics.rmse = np.mean([r.prediction_metrics.rmse for r in all_results])
    aggregated.prediction_metrics.r_squared = np.mean([r.prediction_metrics.r_squared for r in all_results])
    
    # Computation metrics
    aggregated.computation_metrics.avg_planning_time = np.mean([
        r.computation_metrics.avg_planning_time for r in all_results
    ])
    aggregated.computation_metrics.total_planning_time = sum([
        r.computation_metrics.total_planning_time for r in all_results
    ])
    
    # Episode stats
    aggregated.total_reward = np.mean([r.total_reward for r in all_results])
    aggregated.episode_length = np.mean([r.episode_length for r in all_results])
    
    if verbose:
        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        print(f"Episodes: {num_episodes}")
        print(f"Success Rate: {aggregated.success_rate:.2%}")
        print(f"Avg Path Length: {aggregated.path_metrics.length:.1f}")
        print(f"Avg Distance: {aggregated.path_metrics.distance:.1f}")
        print(f"Path Smoothness: {aggregated.path_metrics.smoothness:.3f}")
        print(f"Min Safety Margin: {aggregated.path_metrics.min_safety_margin:.2f}")
        print(f"Near Misses: {aggregated.path_metrics.near_misses:.1f}")
        print(f"Prediction MSE: {aggregated.prediction_metrics.mse:.4f}")
        print(f"Avg Planning Time: {aggregated.computation_metrics.avg_planning_time*1000:.2f}ms")
        print(f"Avg Reward: {aggregated.total_reward:.2f}")
    
    return aggregated

