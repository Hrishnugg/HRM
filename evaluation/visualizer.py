"""
Visualization Tools for HRM-Augmented Pathfinding.

Provides real-time rendering and replay capabilities.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import io


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    
    # Figure size
    figsize: Tuple[int, int] = (10, 10)
    
    # Colors
    wall_color: str = "#2c3e50"
    free_color: str = "#ecf0f1"
    agent_color: str = "#3498db"
    goal_color: str = "#2ecc71"
    obstacle_color: str = "#e74c3c"
    path_color: str = "#9b59b6"
    prediction_color: str = "#f39c12"
    
    # Sizes
    agent_size: float = 0.4
    goal_size: float = 0.4
    obstacle_size: float = 0.35
    
    # Animation
    fps: int = 10
    interval: int = 100  # ms between frames
    
    # Features
    show_predictions: bool = True
    show_path: bool = True
    show_velocities: bool = True
    show_risk_map: bool = False


class EnvironmentVisualizer:
    """
    Visualizer for the dynamic pathfinding environment.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        
        self.fig = None
        self.ax = None
        self._artists = {}
    
    def setup(self, grid_shape: Tuple[int, int]):
        """
        Set up the visualization figure.
        
        Args:
            grid_shape: (height, width) of the grid
        """
        self.fig, self.ax = plt.subplots(1, 1, figsize=self.config.figsize)
        
        self.ax.set_xlim(-0.5, grid_shape[1] - 0.5)
        self.ax.set_ylim(grid_shape[0] - 0.5, -0.5)
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Create colormap for risk
        self.risk_cmap = LinearSegmentedColormap.from_list(
            'risk',
            [(0, self.config.free_color), (1, '#e74c3c')]
        )
    
    def render(
        self,
        grid: np.ndarray,
        agent_position: Tuple[float, float],
        goal_position: Tuple[float, float],
        obstacle_positions: List[Tuple[float, float]],
        obstacle_velocities: Optional[List[Tuple[float, float]]] = None,
        predicted_trajectories: Optional[np.ndarray] = None,
        planned_path: Optional[List[Tuple[int, int]]] = None,
        risk_map: Optional[np.ndarray] = None,
        step: int = 0,
    ):
        """
        Render a single frame.
        
        Args:
            grid: Occupancy grid
            agent_position: Current agent (x, y)
            goal_position: Goal (x, y)
            obstacle_positions: List of obstacle (x, y) positions
            obstacle_velocities: List of obstacle (vx, vy) velocities
            predicted_trajectories: (N, T, 2) predicted positions
            planned_path: Planned path as list of positions
            risk_map: Risk values for each cell
            step: Current step number
        """
        if self.fig is None:
            self.setup(grid.shape)
        
        self.ax.clear()
        
        # Draw grid
        self._draw_grid(grid)
        
        # Draw risk map if available
        if self.config.show_risk_map and risk_map is not None:
            self._draw_risk_map(risk_map)
        
        # Draw planned path
        if self.config.show_path and planned_path:
            self._draw_path(planned_path)
        
        # Draw predicted trajectories
        if self.config.show_predictions and predicted_trajectories is not None:
            self._draw_predictions(obstacle_positions, predicted_trajectories)
        
        # Draw obstacles
        self._draw_obstacles(obstacle_positions, obstacle_velocities)
        
        # Draw goal
        self._draw_goal(goal_position)
        
        # Draw agent
        self._draw_agent(agent_position)
        
        # Title
        self.ax.set_title(f"Step: {step}", fontsize=14)
        
        plt.tight_layout()
    
    def _draw_grid(self, grid: np.ndarray):
        """Draw the occupancy grid."""
        # Create color array
        colors = np.where(
            grid == 1,
            0.2,  # Wall
            1.0   # Free
        )
        
        self.ax.imshow(
            colors,
            cmap='gray',
            origin='upper',
            extent=[-0.5, grid.shape[1]-0.5, grid.shape[0]-0.5, -0.5],
            vmin=0, vmax=1,
        )
    
    def _draw_risk_map(self, risk_map: np.ndarray):
        """Draw risk overlay."""
        # Normalize risk
        max_risk = np.max(risk_map) if np.max(risk_map) > 0 else 1.0
        normalized = risk_map / max_risk
        
        self.ax.imshow(
            normalized,
            cmap=self.risk_cmap,
            origin='upper',
            extent=[-0.5, risk_map.shape[1]-0.5, risk_map.shape[0]-0.5, -0.5],
            alpha=0.5,
            vmin=0, vmax=1,
        )
    
    def _draw_agent(self, position: Tuple[float, float]):
        """Draw the agent."""
        circle = plt.Circle(
            position,
            self.config.agent_size,
            color=self.config.agent_color,
            zorder=10,
        )
        self.ax.add_patch(circle)
        
        # Add direction indicator (small triangle)
        triangle = patches.RegularPolygon(
            (position[0], position[1] - self.config.agent_size * 0.3),
            numVertices=3,
            radius=self.config.agent_size * 0.3,
            orientation=np.pi,
            color='white',
            zorder=11,
        )
        self.ax.add_patch(triangle)
    
    def _draw_goal(self, position: Tuple[float, float]):
        """Draw the goal."""
        # Outer circle
        circle = plt.Circle(
            position,
            self.config.goal_size,
            color=self.config.goal_color,
            zorder=5,
        )
        self.ax.add_patch(circle)
        
        # Inner star pattern
        star = patches.RegularPolygon(
            position,
            numVertices=5,
            radius=self.config.goal_size * 0.6,
            orientation=np.pi / 2,
            color='white',
            zorder=6,
        )
        self.ax.add_patch(star)
    
    def _draw_obstacles(
        self,
        positions: List[Tuple[float, float]],
        velocities: Optional[List[Tuple[float, float]]] = None,
    ):
        """Draw obstacles and optionally their velocities."""
        for i, pos in enumerate(positions):
            # Obstacle circle
            circle = plt.Circle(
                pos,
                self.config.obstacle_size,
                color=self.config.obstacle_color,
                zorder=8,
            )
            self.ax.add_patch(circle)
            
            # Velocity arrow
            if self.config.show_velocities and velocities is not None and i < len(velocities):
                vel = velocities[i]
                speed = np.sqrt(vel[0]**2 + vel[1]**2)
                
                if speed > 0.1:
                    self.ax.arrow(
                        pos[0], pos[1],
                        vel[0] * 2, vel[1] * 2,
                        head_width=0.2,
                        head_length=0.15,
                        fc='orange',
                        ec='orange',
                        zorder=9,
                    )
    
    def _draw_path(self, path: List[Tuple[int, int]]):
        """Draw the planned path."""
        if len(path) < 2:
            return
        
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        
        self.ax.plot(
            xs, ys,
            color=self.config.path_color,
            linewidth=2,
            linestyle='--',
            alpha=0.7,
            zorder=3,
        )
        
        # Mark waypoints
        self.ax.scatter(
            xs[1:-1], ys[1:-1],
            color=self.config.path_color,
            s=30,
            alpha=0.5,
            zorder=4,
        )
    
    def _draw_predictions(
        self,
        current_positions: List[Tuple[float, float]],
        predictions: np.ndarray,
    ):
        """Draw predicted obstacle trajectories."""
        num_obstacles = min(len(current_positions), predictions.shape[0])
        prediction_horizon = predictions.shape[1]
        
        for i in range(num_obstacles):
            # Draw prediction trajectory
            xs = [current_positions[i][0]] + [predictions[i, t, 0] for t in range(prediction_horizon)]
            ys = [current_positions[i][1]] + [predictions[i, t, 1] for t in range(prediction_horizon)]
            
            # Fade alpha along prediction
            for t in range(len(xs) - 1):
                alpha = 0.7 * (1 - t / prediction_horizon)
                self.ax.plot(
                    [xs[t], xs[t+1]], [ys[t], ys[t+1]],
                    color=self.config.prediction_color,
                    linewidth=2,
                    alpha=alpha,
                    zorder=2,
                )
            
            # Mark predicted positions
            self.ax.scatter(
                xs[1:], ys[1:],
                color=self.config.prediction_color,
                s=20,
                alpha=0.5,
                marker='x',
                zorder=2,
            )
    
    def save_frame(self, path: str):
        """Save current frame to file."""
        if self.fig is not None:
            self.fig.savefig(path, dpi=100, bbox_inches='tight')
    
    def get_frame_array(self) -> np.ndarray:
        """Get current frame as numpy array."""
        if self.fig is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        self.fig.canvas.draw()
        data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return data
    
    def close(self):
        """Close the figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


def create_animation(
    episode_data: Dict[str, Any],
    config: Optional[VisualizationConfig] = None,
    save_path: Optional[str] = None,
) -> FuncAnimation:
    """
    Create an animation from episode data.
    
    Args:
        episode_data: Dictionary containing:
            - grid: Occupancy grid
            - agent_positions: List of agent positions
            - goal_position: Goal position
            - obstacle_positions: List of obstacle position lists
            - obstacle_velocities: List of obstacle velocity lists (optional)
            - predicted_trajectories: List of prediction arrays (optional)
            - planned_paths: List of planned paths (optional)
        config: Visualization config
        save_path: Path to save animation (MP4)
        
    Returns:
        FuncAnimation object
    """
    config = config or VisualizationConfig()
    
    # Extract data
    grid = episode_data["grid"]
    agent_positions = episode_data["agent_positions"]
    goal_position = episode_data["goal_position"]
    obstacle_positions = episode_data["obstacle_positions"]
    
    num_frames = len(agent_positions)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=config.figsize)
    
    visualizer = EnvironmentVisualizer(config)
    visualizer.fig = fig
    visualizer.ax = ax
    visualizer.setup(grid.shape)
    
    def update(frame):
        ax.clear()
        
        # Get data for this frame
        obs_pos = obstacle_positions[frame] if frame < len(obstacle_positions) else []
        obs_vel = None
        if "obstacle_velocities" in episode_data:
            obs_vel = episode_data["obstacle_velocities"][frame] if frame < len(episode_data["obstacle_velocities"]) else None
        
        predictions = None
        if "predicted_trajectories" in episode_data:
            preds = episode_data["predicted_trajectories"]
            if frame < len(preds) and preds[frame] is not None:
                predictions = preds[frame]
        
        path = None
        if "planned_paths" in episode_data:
            paths = episode_data["planned_paths"]
            if frame < len(paths):
                path = paths[frame]
        
        # Render
        visualizer.render(
            grid=grid,
            agent_position=agent_positions[frame],
            goal_position=goal_position,
            obstacle_positions=obs_pos,
            obstacle_velocities=obs_vel,
            predicted_trajectories=predictions,
            planned_path=path,
            step=frame,
        )
        
        return []
    
    anim = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=config.interval,
        blit=False,
    )
    
    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='ffmpeg', fps=config.fps)
        print("Done!")
    
    return anim


def visualize_single_step(
    env,
    predictor=None,
    planner=None,
    config: Optional[VisualizationConfig] = None,
) -> np.ndarray:
    """
    Visualize a single environment step.
    
    Args:
        env: Environment (must be reset)
        predictor: Optional predictor for showing predictions
        planner: Optional planner for showing planned path
        config: Visualization config
        
    Returns:
        Frame as numpy array
    """
    config = config or VisualizationConfig()
    visualizer = EnvironmentVisualizer(config)
    
    # Get current observation
    obs = env._get_observation() if hasattr(env, '_get_observation') else None
    
    if obs is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Get predictions
    predictions = None
    if predictor is not None and config.show_predictions:
        result = predictor.predict(
            obs["obstacle_history"],
            obs["obstacle_positions"],
        )
        predictions = result.positions
    
    # Get planned path
    path = None
    if planner is not None and config.show_path:
        path = planner.get_current_path()
    
    # Render
    visualizer.render(
        grid=obs["map"],
        agent_position=tuple(obs["agent_position"]),
        goal_position=tuple(obs["goal_position"]),
        obstacle_positions=[tuple(p) for p in obs["obstacle_positions"]],
        obstacle_velocities=[tuple(v) for v in obs["obstacle_velocities"]],
        predicted_trajectories=predictions,
        planned_path=path,
    )
    
    frame = visualizer.get_frame_array()
    visualizer.close()
    
    return frame

