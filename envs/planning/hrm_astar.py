"""
HRM-Augmented A* Pathfinding.

Combines A* with HRM-based obstacle predictions for
proactive collision avoidance in dynamic environments.
"""

import numpy as np
from typing import Tuple, List, Optional, Set, Dict, Any
from dataclasses import dataclass
import time

from .astar import AStar, AStarResult, HeuristicType
from .predictive_heuristic import PredictiveHeuristic, CollisionRiskConfig


@dataclass
class HRMAStarConfig:
    """Configuration for HRM-augmented A*."""
    
    # Base A* settings
    eight_connected: bool = True
    base_heuristic: HeuristicType = HeuristicType.OCTILE
    heuristic_weight: float = 1.0
    
    # Predictive heuristic settings
    risk_weight: float = 1.0
    influence_radius: float = 3.0
    max_penalty: float = 10.0
    
    # Re-planning settings
    replan_interval: int = 5  # Steps between replans
    replan_risk_threshold: float = 5.0  # Risk threshold for emergency replan
    
    # Safety settings
    safety_margin: float = 1.5  # Min distance from obstacles
    
    # Smoothing
    smooth_path: bool = True
    smoothing_iterations: int = 3


@dataclass
class HRMAStarResult(AStarResult):
    """Extended result for HRM-augmented A*."""
    
    # Path risk score
    path_risk: float = 0.0
    
    # Number of replans
    num_replans: int = 0
    
    # Predictions used
    predictions_updated: int = 0


class HRMAugmentedAStar:
    """
    HRM-Augmented A* Pathfinder.
    
    Uses HRM predictions of obstacle trajectories to:
    1. Inform the heuristic with collision risk
    2. Trigger proactive replanning when risks change
    3. Avoid predicted future obstacle positions
    """
    
    def __init__(
        self,
        predictor,  # HRMObstaclePredictor
        config: Optional[HRMAStarConfig] = None,
    ):
        """
        Args:
            predictor: HRM-based obstacle trajectory predictor
            config: Algorithm configuration
        """
        self.predictor = predictor
        self.config = config or HRMAStarConfig()
        
        # Create base A* planner
        self.astar = AStar(
            heuristic=self.config.base_heuristic,
            eight_connected=self.config.eight_connected,
            heuristic_weight=self.config.heuristic_weight,
        )
        
        # Create predictive heuristic
        risk_config = CollisionRiskConfig(
            influence_radius=self.config.influence_radius,
            max_penalty=self.config.max_penalty,
            safety_margin=self.config.safety_margin,
        )
        
        self.predictive_heuristic = PredictiveHeuristic(
            predictor=predictor,
            risk_weight=self.config.risk_weight,
            risk_config=risk_config,
        )
        
        # State
        self._current_path: List[Tuple[int, int]] = []
        self._path_index: int = 0
        self._last_replan_step: int = 0
        self._steps_since_replan: int = 0
    
    def plan(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacle_history: np.ndarray,
        current_obstacle_positions: np.ndarray,
        current_time: int = 0,
    ) -> HRMAStarResult:
        """
        Plan a path from start to goal.
        
        Args:
            grid: Occupancy grid (0 = free, 1 = obstacle)
            start: Start position (x, y)
            goal: Goal position (x, y)
            obstacle_history: Historical obstacle positions
            current_obstacle_positions: Current obstacle positions
            current_time: Current simulation time step
            
        Returns:
            HRMAStarResult with path and statistics
        """
        start_time = time.perf_counter()
        
        # Update predictions
        self.predictive_heuristic.update_predictions(
            obstacle_history,
            current_obstacle_positions,
            current_time,
        )
        
        # Get dynamic obstacle positions (current + predicted near-future)
        dynamic_obstacles = self._get_dynamic_obstacle_set(
            current_obstacle_positions,
            self.predictive_heuristic._cached_predictions,
            lookahead=2,
        )
        
        # Set predictive heuristic
        self.astar.set_heuristic(
            heuristic=self.predictive_heuristic.get_heuristic_function(current_time)
        )
        
        # Run A*
        result = self.astar.find_path(
            grid=grid,
            start=start,
            goal=goal,
            dynamic_obstacles=dynamic_obstacles,
        )
        
        # Convert to HRM result
        hrm_result = HRMAStarResult(
            path=result.path,
            success=result.success,
            cost=result.cost,
            nodes_expanded=result.nodes_expanded,
            computation_time=time.perf_counter() - start_time,
            open_set_max_size=result.open_set_max_size,
            closed_set_size=result.closed_set_size,
            predictions_updated=1,
        )
        
        if result.success:
            # Compute path risk
            hrm_result.path_risk = self._compute_path_risk(result.path)
            
            # Smooth path if enabled
            if self.config.smooth_path:
                hrm_result.path = self._smooth_path(
                    result.path,
                    grid,
                    dynamic_obstacles,
                )
        
        # Store path
        self._current_path = hrm_result.path
        self._path_index = 0
        self._last_replan_step = current_time
        self._steps_since_replan = 0
        
        return hrm_result
    
    def get_next_action(
        self,
        current_position: Tuple[int, int],
        grid: np.ndarray,
        obstacle_history: np.ndarray,
        current_obstacle_positions: np.ndarray,
        goal: Tuple[int, int],
        current_time: int,
    ) -> Tuple[Tuple[int, int], bool]:
        """
        Get next position to move to.
        
        Args:
            current_position: Agent's current position
            grid: Occupancy grid
            obstacle_history: Historical obstacle positions
            current_obstacle_positions: Current obstacle positions
            goal: Goal position
            current_time: Current time step
            
        Returns:
            (next_position, replanned) tuple
        """
        self._steps_since_replan += 1
        
        # Check if we need to replan
        needs_replan = False
        
        # Periodic replan
        if self._steps_since_replan >= self.config.replan_interval:
            needs_replan = True
        
        # Empty path
        if not self._current_path:
            needs_replan = True
        
        # Reached end of path but not goal
        if self._path_index >= len(self._current_path) - 1:
            if current_position != goal:
                needs_replan = True
        
        # Risk-based replan
        if not needs_replan and self._current_path:
            self.predictive_heuristic.update_predictions(
                obstacle_history,
                current_obstacle_positions,
                current_time,
            )
            
            remaining_path = self._current_path[self._path_index:]
            if self.predictive_heuristic.needs_replan(
                current_position,
                remaining_path,
                self.config.replan_risk_threshold,
            ):
                needs_replan = True
        
        # Replan if needed
        replanned = False
        if needs_replan:
            result = self.plan(
                grid=grid,
                start=current_position,
                goal=goal,
                obstacle_history=obstacle_history,
                current_obstacle_positions=current_obstacle_positions,
                current_time=current_time,
            )
            replanned = True
            
            if not result.success:
                # Stay in place if no path found
                return current_position, replanned
        
        # Get next position from path
        if self._current_path and self._path_index < len(self._current_path) - 1:
            self._path_index += 1
            return self._current_path[self._path_index], replanned
        
        return current_position, replanned
    
    def _get_dynamic_obstacle_set(
        self,
        current_positions: np.ndarray,
        predictions: Optional[np.ndarray],
        lookahead: int = 2,
    ) -> Set[Tuple[int, int]]:
        """
        Get set of positions blocked by dynamic obstacles.
        
        Includes current positions and predicted near-future positions.
        """
        obstacles = set()
        
        # Add current positions (with safety margin)
        for pos in current_positions:
            x, y = int(round(pos[0])), int(round(pos[1]))
            
            # Add position and neighbors within safety margin
            margin = int(np.ceil(self.config.safety_margin))
            for dx in range(-margin, margin + 1):
                for dy in range(-margin, margin + 1):
                    if dx * dx + dy * dy <= self.config.safety_margin ** 2:
                        obstacles.add((x + dx, y + dy))
        
        # Add predicted positions
        if predictions is not None:
            for i in range(min(lookahead, predictions.shape[1])):
                for obs_idx in range(predictions.shape[0]):
                    pos = predictions[obs_idx, i]
                    x, y = int(round(pos[0])), int(round(pos[1]))
                    obstacles.add((x, y))
        
        return obstacles
    
    def _compute_path_risk(self, path: List[Tuple[int, int]]) -> float:
        """Compute total risk along a path."""
        if not path or self.predictive_heuristic._cached_predictions is None:
            return 0.0
        
        total_risk = 0.0
        
        for t, pos in enumerate(path):
            risk = self.predictive_heuristic.risk_estimator.estimate_risk(
                pos,
                self.predictive_heuristic._cached_predictions,
                self.predictive_heuristic._cached_uncertainties,
                t,
            )
            total_risk += risk
        
        return total_risk
    
    def _smooth_path(
        self,
        path: List[Tuple[int, int]],
        grid: np.ndarray,
        dynamic_obstacles: Set[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """
        Smooth path using iterative refinement.
        
        Tries to shortcut unnecessary waypoints while maintaining validity.
        """
        if len(path) <= 2:
            return path
        
        smoothed = list(path)
        
        for _ in range(self.config.smoothing_iterations):
            i = 0
            while i < len(smoothed) - 2:
                # Try to skip intermediate point
                if self._has_line_of_sight(
                    smoothed[i],
                    smoothed[i + 2],
                    grid,
                    dynamic_obstacles,
                ):
                    smoothed.pop(i + 1)
                else:
                    i += 1
        
        return smoothed
    
    def _has_line_of_sight(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        grid: np.ndarray,
        dynamic_obstacles: Set[Tuple[int, int]],
    ) -> bool:
        """Check if there's a clear line of sight between two points."""
        x0, y0 = start
        x1, y1 = end
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        
        err = dx - dy
        
        while True:
            # Check current position
            if not (0 <= x0 < grid.shape[1] and 0 <= y0 < grid.shape[0]):
                return False
            
            if grid[y0, x0] != 0:
                return False
            
            if (x0, y0) in dynamic_obstacles:
                return False
            
            if x0 == x1 and y0 == y1:
                break
            
            e2 = 2 * err
            
            if e2 > -dy:
                err -= dy
                x0 += sx
            
            if e2 < dx:
                err += dx
                y0 += sy
        
        return True
    
    def reset(self):
        """Reset planner state."""
        self._current_path = []
        self._path_index = 0
        self._last_replan_step = 0
        self._steps_since_replan = 0
        self.predictor.reset()
    
    def get_current_path(self) -> List[Tuple[int, int]]:
        """Get the current planned path."""
        return self._current_path
    
    def get_remaining_path(self) -> List[Tuple[int, int]]:
        """Get remaining portion of the path."""
        if not self._current_path:
            return []
        return self._current_path[self._path_index:]


class HRMAStarAgent:
    """
    Agent that uses HRM-augmented A* for navigation.
    
    Wraps HRMAugmentedAStar with environment interaction logic.
    """
    
    def __init__(
        self,
        predictor,
        config: Optional[HRMAStarConfig] = None,
    ):
        self.planner = HRMAugmentedAStar(predictor, config)
        self.goal: Optional[Tuple[int, int]] = None
        self.current_step: int = 0
    
    def set_goal(self, goal: Tuple[int, int]):
        """Set navigation goal."""
        self.goal = goal
        self.planner.reset()
    
    def act(
        self,
        observation: Dict[str, np.ndarray],
    ) -> int:
        """
        Get action for current observation.
        
        Args:
            observation: Environment observation dict
            
        Returns:
            Action index (0=STAY, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, etc.)
        """
        if self.goal is None:
            return 0  # Stay
        
        current_pos = tuple(observation["agent_position"].astype(int))
        goal_pos = tuple(observation["goal_position"].astype(int))
        
        # Update goal if changed
        if goal_pos != self.goal:
            self.set_goal(goal_pos)
        
        # Get grid
        grid = observation["map"]
        
        # Get obstacle info
        obstacle_history = observation["obstacle_history"]
        current_obstacles = observation["obstacle_positions"]
        
        # Get next position
        next_pos, _ = self.planner.get_next_action(
            current_position=current_pos,
            grid=grid,
            obstacle_history=obstacle_history,
            current_obstacle_positions=current_obstacles,
            goal=self.goal,
            current_time=self.current_step,
        )
        
        self.current_step += 1
        
        # Convert to action
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]
        
        # Action mapping
        action_map = {
            (0, 0): 0,    # STAY
            (0, -1): 1,   # UP
            (0, 1): 2,    # DOWN
            (-1, 0): 3,   # LEFT
            (1, 0): 4,    # RIGHT
            (-1, -1): 5,  # UP-LEFT
            (1, -1): 6,   # UP-RIGHT
            (-1, 1): 7,   # DOWN-LEFT
            (1, 1): 8,    # DOWN-RIGHT
        }
        
        return action_map.get((dx, dy), 0)
    
    def reset(self):
        """Reset agent state."""
        self.planner.reset()
        self.goal = None
        self.current_step = 0

