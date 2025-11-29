"""
Stable-Baselines3 Integration for HRM-Augmented Pathfinding.

Provides:
- Custom feature extractor using HRM
- Reward wrapper for prediction-based rewards
- Curriculum learning wrapper
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback


class HRMFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that uses HRM for processing observations.
    
    Encodes obstacle history using the HRM predictor's encoder and
    combines with map/position features.
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        hrm_predictor=None,
        use_predictor_encoder: bool = True,
    ):
        """
        Args:
            observation_space: Environment observation space
            features_dim: Output feature dimension
            hrm_predictor: HRM predictor (optional, creates new if None)
            use_predictor_encoder: Whether to use HRM's encoder
        """
        super().__init__(observation_space, features_dim)
        
        self.hrm_predictor = hrm_predictor
        self.use_predictor_encoder = use_predictor_encoder
        
        # Get observation shapes
        map_shape = observation_space["map"].shape  # (H, W)
        num_obstacles = observation_space["obstacle_positions"].shape[0]
        history_length = observation_space["obstacle_history"].shape[1]
        
        # Map encoder (CNN)
        self.map_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            sample_map = torch.zeros(1, 1, *map_shape)
            map_features_size = self.map_encoder(sample_map).shape[1]
        
        # Position encoder
        self.position_encoder = nn.Sequential(
            nn.Linear(4, 64),  # agent (2) + goal (2)
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        
        # Obstacle encoder (if not using HRM)
        if not use_predictor_encoder or hrm_predictor is None:
            self.obstacle_encoder = nn.Sequential(
                nn.Linear(num_obstacles * history_length * 2, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
            )
            obstacle_features_size = 128
        else:
            # Use HRM encoder's output size
            obstacle_features_size = hrm_predictor.hidden_size
        
        # Velocity encoder
        self.velocity_encoder = nn.Sequential(
            nn.Linear(num_obstacles * 2, 64),
            nn.ReLU(),
        )
        
        # Final combination
        total_input_size = map_features_size + 64 + obstacle_features_size + 64
        
        self.combiner = nn.Sequential(
            nn.Linear(total_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )
        
        self._features_dim = features_dim
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract features from observations.
        
        Args:
            observations: Dict with map, positions, obstacle info
            
        Returns:
            Feature tensor of shape (batch_size, features_dim)
        """
        batch_size = observations["map"].shape[0]
        
        # Encode map
        map_obs = observations["map"].unsqueeze(1)  # Add channel dim
        map_features = self.map_encoder(map_obs)
        
        # Encode positions
        positions = torch.cat([
            observations["agent_position"],
            observations["goal_position"],
        ], dim=-1)
        position_features = self.position_encoder(positions)
        
        # Encode obstacle history
        if self.use_predictor_encoder and self.hrm_predictor is not None:
            # Use HRM encoder
            history = observations["obstacle_history"]
            with torch.no_grad():
                encoded = self.hrm_predictor.encoder(history)
                # Pool across sequence dimension
                obstacle_features = encoded.mean(dim=1)
        else:
            # Simple MLP encoder
            history = observations["obstacle_history"]
            history_flat = history.view(batch_size, -1)
            obstacle_features = self.obstacle_encoder(history_flat)
        
        # Encode velocities
        velocities = observations["obstacle_velocities"]
        velocity_flat = velocities.view(batch_size, -1)
        velocity_features = self.velocity_encoder(velocity_flat)
        
        # Combine all features
        combined = torch.cat([
            map_features,
            position_features,
            obstacle_features,
            velocity_features,
        ], dim=-1)
        
        return self.combiner(combined)


class PredictorRewardWrapper(gym.Wrapper):
    """
    Reward wrapper that adds prediction-based rewards.
    
    Rewards the agent for:
    - Accurate obstacle predictions
    - Maintaining safe distances from obstacles
    """
    
    def __init__(
        self,
        env: gym.Env,
        predictor=None,
        prediction_weight: float = 0.1,
        safety_weight: float = 0.2,
        safety_threshold: float = 2.0,
    ):
        """
        Args:
            env: Base environment
            predictor: HRM predictor for evaluating predictions
            prediction_weight: Weight for prediction accuracy reward
            safety_weight: Weight for safety margin reward
            safety_threshold: Distance threshold for safety bonus
        """
        super().__init__(env)
        
        self.predictor = predictor
        self.prediction_weight = prediction_weight
        self.safety_weight = safety_weight
        self.safety_threshold = safety_threshold
        
        # Track previous predictions
        self._prev_predictions = None
        self._prev_positions = None
    
    def reset(self, **kwargs):
        """Reset environment and prediction tracking."""
        obs, info = self.env.reset(**kwargs)
        
        self._prev_predictions = None
        self._prev_positions = None
        
        return obs, info
    
    def step(self, action):
        """Step with augmented rewards."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add prediction reward
        if self.predictor is not None and self._prev_predictions is not None:
            prediction_reward = self._compute_prediction_reward(obs)
            reward += self.prediction_weight * prediction_reward
            info["prediction_reward"] = prediction_reward
        
        # Add safety reward
        safety_reward = self._compute_safety_reward(obs)
        reward += self.safety_weight * safety_reward
        info["safety_reward"] = safety_reward
        
        # Update predictions for next step
        if self.predictor is not None:
            self._update_predictions(obs)
        
        self._prev_positions = obs["obstacle_positions"].copy()
        
        return obs, reward, terminated, truncated, info
    
    def _compute_prediction_reward(self, obs: Dict[str, np.ndarray]) -> float:
        """Compute reward based on prediction accuracy."""
        if self._prev_predictions is None or self._prev_positions is None:
            return 0.0
        
        # Compare predictions to actual positions
        actual = obs["obstacle_positions"]
        predicted = self._prev_predictions[:, 0]  # First timestep prediction
        
        # Mean squared error
        mse = np.mean((actual - predicted) ** 2)
        
        # Convert to reward (lower error = higher reward)
        reward = np.exp(-mse)  # Exponential reward, max 1.0
        
        return float(reward)
    
    def _compute_safety_reward(self, obs: Dict[str, np.ndarray]) -> float:
        """Compute reward based on safety margins."""
        agent_pos = obs["agent_position"]
        obstacle_pos = obs["obstacle_positions"]
        
        # Compute distances to all obstacles
        distances = np.linalg.norm(obstacle_pos - agent_pos, axis=1)
        min_distance = np.min(distances) if len(distances) > 0 else float('inf')
        
        # Reward for maintaining safe distance
        if min_distance > self.safety_threshold:
            return 0.1  # Small bonus for safe distance
        else:
            # Penalty scales with how close we are
            return -0.1 * (1.0 - min_distance / self.safety_threshold)
    
    def _update_predictions(self, obs: Dict[str, np.ndarray]):
        """Update predictions for next step comparison."""
        history = obs["obstacle_history"]
        current = obs["obstacle_positions"]
        
        result = self.predictor.predict(
            obstacle_history=history,
            current_positions=current,
            steps_ahead=5,
        )
        
        self._prev_predictions = result.positions


class CurriculumEnvWrapper(gym.Wrapper):
    """
    Curriculum learning wrapper.
    
    Gradually increases difficulty by adjusting:
    - Number of obstacles
    - Obstacle speed
    - Obstacle behavior complexity
    """
    
    def __init__(
        self,
        env: gym.Env,
        curriculum_stages: List[Dict[str, Any]],
        stage_thresholds: List[int],
    ):
        """
        Args:
            env: Base environment
            curriculum_stages: List of config dicts for each stage
            stage_thresholds: Episode counts at which to advance stages
        """
        super().__init__(env)
        
        self.curriculum_stages = curriculum_stages
        self.stage_thresholds = stage_thresholds
        
        self.current_stage = 0
        self.episode_count = 0
        self.total_reward_history = []
        
        # Apply initial stage
        self._apply_stage(0)
    
    def reset(self, **kwargs):
        """Reset and check for curriculum advancement."""
        self.episode_count += 1
        
        # Check if we should advance
        if (self.current_stage < len(self.stage_thresholds) and
            self.episode_count >= self.stage_thresholds[self.current_stage]):
            
            self._advance_stage()
        
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """Step and track performance."""
        return self.env.step(action)
    
    def _apply_stage(self, stage: int):
        """Apply curriculum stage settings."""
        if stage >= len(self.curriculum_stages):
            return
        
        config = self.curriculum_stages[stage]
        
        # Update environment settings
        if hasattr(self.env, 'set_obstacles_config'):
            self.env.set_obstacles_config(
                num_obstacles=config.get("num_obstacles"),
                speed_range=config.get("speed_range"),
                type_weights=config.get("type_weights"),
            )
        
        print(f"Curriculum: Advanced to stage {stage + 1}/{len(self.curriculum_stages)}")
        print(f"  Config: {config}")
    
    def _advance_stage(self):
        """Advance to next curriculum stage."""
        self.current_stage += 1
        
        if self.current_stage < len(self.curriculum_stages):
            self._apply_stage(self.current_stage)
    
    def get_current_stage(self) -> int:
        """Get current curriculum stage."""
        return self.current_stage
    
    def force_stage(self, stage: int):
        """Force a specific curriculum stage."""
        self.current_stage = stage
        self._apply_stage(stage)


class CurriculumCallback(BaseCallback):
    """
    SB3 callback for curriculum learning.
    
    Monitors performance and can adjust curriculum based on success rate.
    """
    
    def __init__(
        self,
        curriculum_wrapper: CurriculumEnvWrapper,
        success_threshold: float = 0.7,
        evaluation_window: int = 100,
        verbose: int = 1,
    ):
        """
        Args:
            curriculum_wrapper: The curriculum environment wrapper
            success_threshold: Success rate to advance stages
            evaluation_window: Episodes to consider for evaluation
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        self.curriculum_wrapper = curriculum_wrapper
        self.success_threshold = success_threshold
        self.evaluation_window = evaluation_window
        
        self.episode_rewards = []
        self.episode_successes = []
    
    def _on_step(self) -> bool:
        """Called at each step."""
        # Check for episode end
        if self.locals.get("dones") is not None:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals["infos"][i]
                    reward = info.get("episode", {}).get("r", 0)
                    success = info.get("is_success", reward > 0)
                    
                    self.episode_rewards.append(reward)
                    self.episode_successes.append(success)
                    
                    # Keep window size
                    if len(self.episode_rewards) > self.evaluation_window:
                        self.episode_rewards.pop(0)
                        self.episode_successes.pop(0)
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at end of rollout."""
        if len(self.episode_successes) >= self.evaluation_window:
            success_rate = np.mean(self.episode_successes)
            
            if self.verbose > 0:
                print(f"Curriculum: Success rate = {success_rate:.2%}")
            
            # Log to tensorboard
            self.logger.record("curriculum/success_rate", success_rate)
            self.logger.record("curriculum/stage", 
                             self.curriculum_wrapper.get_current_stage())


def create_curriculum_stages(
    base_obstacles: int = 3,
    max_obstacles: int = 10,
    base_speed: float = 0.5,
    max_speed: float = 2.0,
    num_stages: int = 5,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Create curriculum stages with linear progression.
    
    Args:
        base_obstacles: Starting number of obstacles
        max_obstacles: Maximum number of obstacles
        base_speed: Starting obstacle speed
        max_speed: Maximum obstacle speed
        num_stages: Number of curriculum stages
        
    Returns:
        (stages, thresholds) tuple
    """
    stages = []
    thresholds = []
    
    for i in range(num_stages):
        progress = i / (num_stages - 1) if num_stages > 1 else 1.0
        
        num_obs = int(base_obstacles + progress * (max_obstacles - base_obstacles))
        speed_min = base_speed + progress * (max_speed - base_speed) * 0.5
        speed_max = base_speed + progress * (max_speed - base_speed)
        
        # Start with simpler obstacle types, add complexity
        if progress < 0.33:
            type_weights = {"linear": 0.8, "patrol": 0.2}
        elif progress < 0.66:
            type_weights = {"linear": 0.5, "patrol": 0.3, "circular": 0.2}
        else:
            type_weights = {
                "linear": 0.3,
                "patrol": 0.25,
                "circular": 0.15,
                "oscillating": 0.15,
                "intelligent": 0.15,
            }
        
        stages.append({
            "num_obstacles": num_obs,
            "speed_range": (speed_min, speed_max),
            "type_weights": type_weights,
        })
        
        # Episodes per stage (more episodes for harder stages)
        episodes = 10000 + i * 20000
        thresholds.append(episodes)
    
    return stages, thresholds


def make_env(
    config: Dict[str, Any],
    predictor=None,
    use_curriculum: bool = True,
    seed: Optional[int] = None,
) -> gym.Env:
    """
    Factory function for creating the training environment.
    
    Args:
        config: Environment configuration
        predictor: HRM predictor for reward shaping
        use_curriculum: Whether to use curriculum learning
        seed: Random seed
        
    Returns:
        Configured environment
    """
    from envs.dynamic_pathfinding_env import DynamicPathfindingEnv, EnvConfig
    
    # Create base environment
    env_config = EnvConfig(**config)
    env = DynamicPathfindingEnv(env_config)
    
    if seed is not None:
        env.reset(seed=seed)
    
    # Add reward shaping
    if predictor is not None:
        env = PredictorRewardWrapper(env, predictor=predictor)
    
    # Add curriculum
    if use_curriculum:
        stages, thresholds = create_curriculum_stages()
        env = CurriculumEnvWrapper(env, stages, thresholds)
    
    return env

