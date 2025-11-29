"""
Dynamic Pathfinding Environment.

A Gymnasium environment for training HRM-augmented A* pathfinding
with dynamic obstacles.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path

from .maps.map_parser import MapParser, GridMap, create_simple_map
from .obstacles import (
    Obstacle,
    ObstacleFactory,
    ObstacleType,
)


@dataclass
class EnvConfig:
    """Configuration for the dynamic pathfinding environment."""
    
    # Map settings
    map_name: Optional[str] = None  # Name of benchmark map to load
    map_size: int = 32  # Size if generating random map
    obstacle_density: float = 0.15  # For random maps
    
    # Dynamic obstacle settings
    num_obstacles: int = 5
    obstacle_speed_range: Tuple[float, float] = (0.5, 2.0)
    obstacle_type_weights: Optional[Dict[str, float]] = None
    
    # History settings (for prediction)
    history_length: int = 10  # Number of past timesteps to track
    
    # Agent settings
    eight_connected: bool = True  # 8-directional movement
    
    # Reward settings
    step_penalty: float = -0.01
    collision_penalty: float = -1.0
    goal_reward: float = 10.0
    distance_reward_scale: float = 0.1
    near_miss_penalty: float = -0.1
    near_miss_threshold: float = 1.5
    
    # Episode settings
    max_steps: int = 500
    
    # Rendering
    render_mode: Optional[str] = None  # "human", "rgb_array", or None


class DynamicPathfindingEnv(gym.Env):
    """
    Dynamic Pathfinding Environment with moving obstacles.
    
    Observation Space (Dict):
        - map: (H, W) float32 - static obstacle map (0=free, 1=wall)
        - agent_position: (2,) float32 - current agent (x, y)
        - goal_position: (2,) float32 - goal (x, y)
        - obstacle_positions: (N, 2) float32 - current obstacle positions
        - obstacle_history: (N, T, 2) float32 - obstacle position history
        - obstacle_velocities: (N, 2) float32 - current obstacle velocities
    
    Action Space:
        - Discrete(5): STAY, UP, DOWN, LEFT, RIGHT
        - Discrete(8): + diagonals (if eight_connected=True)
    
    Rewards:
        - Step penalty: small negative per step
        - Collision with obstacle/wall: large negative, episode ends
        - Reaching goal: large positive, episode ends
        - Distance reduction: small positive proportional to distance decrease
        - Near miss: small negative for being too close to obstacles
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    # Action mappings
    ACTIONS_4 = {
        0: (0, 0),   # STAY
        1: (0, -1),  # UP
        2: (0, 1),   # DOWN
        3: (-1, 0),  # LEFT
        4: (1, 0),   # RIGHT
    }
    
    ACTIONS_8 = {
        0: (0, 0),    # STAY
        1: (0, -1),   # UP
        2: (0, 1),    # DOWN
        3: (-1, 0),   # LEFT
        4: (1, 0),    # RIGHT
        5: (-1, -1),  # UP-LEFT
        6: (1, -1),   # UP-RIGHT
        7: (-1, 1),   # DOWN-LEFT
        8: (1, 1),    # DOWN-RIGHT
    }
    
    def __init__(self, config: Optional[EnvConfig] = None, **kwargs):
        """
        Initialize the environment.
        
        Args:
            config: Environment configuration
            **kwargs: Override config parameters
        """
        super().__init__()
        
        # Initialize config
        self.config = config or EnvConfig()
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self.render_mode = self.config.render_mode
        
        # Initialize map parser
        self.map_parser = MapParser()
        
        # Load or create map
        self._load_map()
        
        # Action space
        self.actions = self.ACTIONS_8 if self.config.eight_connected else self.ACTIONS_4
        self.action_space = spaces.Discrete(len(self.actions))
        
        # Observation space
        self.observation_space = spaces.Dict({
            "map": spaces.Box(
                low=0, high=1,
                shape=(self.grid_map.height, self.grid_map.width),
                dtype=np.float32
            ),
            "agent_position": spaces.Box(
                low=0, high=max(self.grid_map.height, self.grid_map.width),
                shape=(2,), dtype=np.float32
            ),
            "goal_position": spaces.Box(
                low=0, high=max(self.grid_map.height, self.grid_map.width),
                shape=(2,), dtype=np.float32
            ),
            "obstacle_positions": spaces.Box(
                low=0, high=max(self.grid_map.height, self.grid_map.width),
                shape=(self.config.num_obstacles, 2),
                dtype=np.float32
            ),
            "obstacle_history": spaces.Box(
                low=0, high=max(self.grid_map.height, self.grid_map.width),
                shape=(self.config.num_obstacles, self.config.history_length, 2),
                dtype=np.float32
            ),
            "obstacle_velocities": spaces.Box(
                low=-10, high=10,
                shape=(self.config.num_obstacles, 2),
                dtype=np.float32
            ),
        })
        
        # Initialize obstacle factory
        type_weights = None
        if self.config.obstacle_type_weights:
            type_weights = {
                ObstacleType(k): v 
                for k, v in self.config.obstacle_type_weights.items()
            }
        
        self.obstacle_factory = ObstacleFactory(
            type_weights=type_weights,
            speed_range=self.config.obstacle_speed_range,
        )
        
        # State variables (initialized in reset)
        self.agent_position: np.ndarray = np.zeros(2, dtype=np.float32)
        self.goal_position: np.ndarray = np.zeros(2, dtype=np.float32)
        self.obstacles: List[Obstacle] = []
        self.current_step: int = 0
        self._rng: np.random.Generator = np.random.default_rng()
        
        # Rendering
        self._renderer = None
    
    def _load_map(self):
        """Load or create the grid map."""
        if self.config.map_name:
            try:
                self.grid_map = self.map_parser.load_benchmark(self.config.map_name)
            except FileNotFoundError:
                print(f"Map '{self.config.map_name}' not found, creating random map")
                self.grid_map = self.map_parser.create_random_map(
                    height=self.config.map_size,
                    width=self.config.map_size,
                    obstacle_ratio=self.config.obstacle_density,
                    name="random"
                )
        else:
            self.grid_map = self.map_parser.create_random_map(
                height=self.config.map_size,
                width=self.config.map_size,
                obstacle_ratio=self.config.obstacle_density,
                name="random"
            )
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Reset options
                - agent_position: Fixed starting position
                - goal_position: Fixed goal position
                - map_name: Change map for this episode
                
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self.obstacle_factory.reset(seed)
        
        # Handle options
        options = options or {}
        
        # Optionally load a different map
        if "map_name" in options:
            self.config.map_name = options["map_name"]
            self._load_map()
        
        # Reset step counter
        self.current_step = 0
        
        # Set agent position
        if "agent_position" in options:
            self.agent_position = np.array(options["agent_position"], dtype=np.float32)
        else:
            self.agent_position = np.array(
                self.grid_map.random_passable_position(self._rng),
                dtype=np.float32
            )
        
        # Set goal position (must be different from agent)
        if "goal_position" in options:
            self.goal_position = np.array(options["goal_position"], dtype=np.float32)
        else:
            for _ in range(100):
                self.goal_position = np.array(
                    self.grid_map.random_passable_position(self._rng),
                    dtype=np.float32
                )
                if np.linalg.norm(self.goal_position - self.agent_position) > 5:
                    break
        
        # Create obstacles
        avoid_positions = [
            tuple(self.agent_position.astype(int)),
            tuple(self.goal_position.astype(int)),
        ]
        
        self.obstacles = self.obstacle_factory.create_obstacles(
            num_obstacles=self.config.num_obstacles,
            grid=self.grid_map.grid,
            avoid_positions=avoid_positions,
            min_distance=5.0,
        )
        
        # Get observation
        observation = self._get_observation()
        
        # Info
        info = {
            "distance_to_goal": np.linalg.norm(self.goal_position - self.agent_position),
            "map_name": self.grid_map.name,
            "num_obstacles": len(self.obstacles),
        }
        
        return observation, info
    
    def step(
        self,
        action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            observation: New observation
            reward: Step reward
            terminated: Episode ended (goal/collision)
            truncated: Episode truncated (max steps)
            info: Additional information
        """
        self.current_step += 1
        
        # Get action delta
        dx, dy = self.actions[action]
        
        # Calculate new agent position
        old_position = self.agent_position.copy()
        new_position = self.agent_position + np.array([dx, dy], dtype=np.float32)
        
        # Check wall collision
        grid_x, grid_y = int(round(new_position[0])), int(round(new_position[1]))
        wall_collision = not self.grid_map.is_passable(grid_x, grid_y)
        
        # Update obstacles
        agent_grid_pos = (int(self.agent_position[0]), int(self.agent_position[1]))
        for obstacle in self.obstacles:
            obstacle.update(
                grid=self.grid_map.grid,
                agent_position=agent_grid_pos,
                rng=self._rng,
            )
        
        # Check obstacle collision
        obstacle_collision = False
        min_obstacle_distance = float('inf')
        
        for obstacle in self.obstacles:
            obs_pos = obstacle.position
            distance = np.linalg.norm(new_position - obs_pos)
            min_obstacle_distance = min(min_obstacle_distance, distance)
            
            if distance < (1.0 + obstacle.radius):  # Agent radius + obstacle radius
                obstacle_collision = True
                break
        
        # Calculate reward
        reward = self.config.step_penalty
        terminated = False
        truncated = False
        
        if wall_collision or obstacle_collision:
            # Collision
            reward += self.config.collision_penalty
            terminated = True
        else:
            # Update agent position
            self.agent_position = new_position
            
            # Check goal reached
            if np.linalg.norm(self.agent_position - self.goal_position) < 1.0:
                reward += self.config.goal_reward
                terminated = True
            else:
                # Distance reward
                old_dist = np.linalg.norm(old_position - self.goal_position)
                new_dist = np.linalg.norm(self.agent_position - self.goal_position)
                reward += self.config.distance_reward_scale * (old_dist - new_dist)
                
                # Near miss penalty
                if min_obstacle_distance < self.config.near_miss_threshold:
                    reward += self.config.near_miss_penalty
        
        # Check truncation
        if self.current_step >= self.config.max_steps:
            truncated = True
        
        # Get observation
        observation = self._get_observation()
        
        # Info
        info = {
            "distance_to_goal": np.linalg.norm(self.goal_position - self.agent_position),
            "min_obstacle_distance": min_obstacle_distance,
            "wall_collision": wall_collision,
            "obstacle_collision": obstacle_collision,
            "step": self.current_step,
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        # Obstacle positions and history
        obstacle_positions = np.zeros(
            (self.config.num_obstacles, 2), dtype=np.float32
        )
        obstacle_history = np.zeros(
            (self.config.num_obstacles, self.config.history_length, 2),
            dtype=np.float32
        )
        obstacle_velocities = np.zeros(
            (self.config.num_obstacles, 2), dtype=np.float32
        )
        
        for i, obstacle in enumerate(self.obstacles):
            if i >= self.config.num_obstacles:
                break
            obstacle_positions[i] = obstacle.position
            obstacle_history[i] = obstacle.get_history(self.config.history_length)
            obstacle_velocities[i] = obstacle.velocity
        
        return {
            "map": self.grid_map.to_observation(),
            "agent_position": self.agent_position.copy(),
            "goal_position": self.goal_position.copy(),
            "obstacle_positions": obstacle_positions,
            "obstacle_history": obstacle_history,
            "obstacle_velocities": obstacle_velocities,
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None
        
        return self._render_frame()
    
    def _render_frame(self) -> Optional[np.ndarray]:
        """Render a single frame."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Draw map
        ax.imshow(
            self.grid_map.grid,
            cmap='binary',
            origin='upper',
            extent=[0, self.grid_map.width, self.grid_map.height, 0]
        )
        
        # Draw obstacles
        for obstacle in self.obstacles:
            circle = Circle(
                (obstacle.position[0], obstacle.position[1]),
                radius=obstacle.radius,
                color='red',
                alpha=0.7
            )
            ax.add_patch(circle)
            
            # Draw velocity vector
            ax.arrow(
                obstacle.position[0],
                obstacle.position[1],
                obstacle.velocity[0] * 2,
                obstacle.velocity[1] * 2,
                head_width=0.3,
                head_length=0.2,
                fc='orange',
                ec='orange'
            )
        
        # Draw agent
        agent_circle = Circle(
            (self.agent_position[0], self.agent_position[1]),
            radius=0.5,
            color='blue',
            alpha=0.9
        )
        ax.add_patch(agent_circle)
        
        # Draw goal
        goal_circle = Circle(
            (self.goal_position[0], self.goal_position[1]),
            radius=0.5,
            color='green',
            alpha=0.9
        )
        ax.add_patch(goal_circle)
        
        ax.set_xlim(0, self.grid_map.width)
        ax.set_ylim(self.grid_map.height, 0)
        ax.set_aspect('equal')
        ax.set_title(f'Step: {self.current_step}')
        
        if self.render_mode == "human":
            plt.show(block=False)
            plt.pause(0.01)
            plt.close(fig)
            return None
        else:
            # Return RGB array
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return data
    
    def close(self):
        """Clean up resources."""
        if self._renderer is not None:
            self._renderer = None
    
    def get_agent_grid_position(self) -> Tuple[int, int]:
        """Get agent position as integer grid coordinates."""
        return (int(round(self.agent_position[0])), int(round(self.agent_position[1])))
    
    def get_goal_grid_position(self) -> Tuple[int, int]:
        """Get goal position as integer grid coordinates."""
        return (int(round(self.goal_position[0])), int(round(self.goal_position[1])))
    
    def get_obstacle_grid_positions(self) -> List[Tuple[int, int]]:
        """Get all obstacle positions as integer grid coordinates."""
        return [
            (int(round(obs.position[0])), int(round(obs.position[1])))
            for obs in self.obstacles
        ]
    
    def set_obstacles_config(
        self,
        num_obstacles: Optional[int] = None,
        speed_range: Optional[Tuple[float, float]] = None,
        type_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Update obstacle configuration for curriculum learning.
        
        Args:
            num_obstacles: New number of obstacles
            speed_range: New speed range
            type_weights: New type distribution
        """
        if num_obstacles is not None:
            self.config.num_obstacles = num_obstacles
            # Update observation space
            self.observation_space["obstacle_positions"] = spaces.Box(
                low=0, high=max(self.grid_map.height, self.grid_map.width),
                shape=(num_obstacles, 2),
                dtype=np.float32
            )
            self.observation_space["obstacle_history"] = spaces.Box(
                low=0, high=max(self.grid_map.height, self.grid_map.width),
                shape=(num_obstacles, self.config.history_length, 2),
                dtype=np.float32
            )
            self.observation_space["obstacle_velocities"] = spaces.Box(
                low=-10, high=10,
                shape=(num_obstacles, 2),
                dtype=np.float32
            )
        
        if speed_range is not None:
            self.config.obstacle_speed_range = speed_range
            self.obstacle_factory.speed_range = speed_range
        
        if type_weights is not None:
            self.config.obstacle_type_weights = type_weights
            self.obstacle_factory.type_weights = {
                ObstacleType(k): v for k, v in type_weights.items()
            }


# Register the environment with Gymnasium
def register_env():
    """Register the environment with Gymnasium."""
    from gymnasium.envs.registration import register
    
    register(
        id="DynamicPathfinding-v0",
        entry_point="envs.dynamic_pathfinding_env:DynamicPathfindingEnv",
        max_episode_steps=500,
    )


if __name__ == "__main__":
    # Quick test
    env = DynamicPathfindingEnv(
        config=EnvConfig(
            map_size=32,
            num_obstacles=5,
            render_mode="human",
        )
    )
    
    obs, info = env.reset()
    print(f"Initial observation shapes:")
    for key, value in obs.items():
        print(f"  {key}: {value.shape}")
    
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()

