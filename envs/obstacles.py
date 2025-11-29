"""
Dynamic Obstacle System.

Implements various obstacle motion patterns for the dynamic pathfinding environment:
- Linear: constant velocity with random direction changes
- Patrol: waypoint-following routes
- Circular/Oscillating: periodic motion patterns
- Intelligent: basic pursuit/avoidance behavior
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum


class ObstacleType(Enum):
    """Types of obstacle motion patterns."""
    LINEAR = "linear"
    PATROL = "patrol"
    CIRCULAR = "circular"
    OSCILLATING = "oscillating"
    INTELLIGENT = "intelligent"


@dataclass
class ObstacleState:
    """Current state of an obstacle."""
    position: np.ndarray  # (x, y) float position
    velocity: np.ndarray  # (vx, vy) velocity
    
    @property
    def grid_position(self) -> Tuple[int, int]:
        """Get integer grid position."""
        return (int(round(self.position[0])), int(round(self.position[1])))


class Obstacle(ABC):
    """Abstract base class for dynamic obstacles."""
    
    def __init__(
        self,
        initial_position: Tuple[float, float],
        speed: float = 1.0,
        radius: float = 0.5,
        obstacle_id: int = 0,
    ):
        """
        Initialize obstacle.
        
        Args:
            initial_position: Starting (x, y) position
            speed: Movement speed (cells per timestep)
            radius: Collision radius
            obstacle_id: Unique identifier
        """
        self.initial_position = np.array(initial_position, dtype=np.float32)
        self.position = self.initial_position.copy()
        self.velocity = np.zeros(2, dtype=np.float32)
        self.speed = speed
        self.radius = radius
        self.obstacle_id = obstacle_id
        self.time_step = 0
        
        # History tracking for prediction
        self.position_history: List[np.ndarray] = []
        self.max_history = 50
    
    @abstractmethod
    def update(
        self,
        grid: np.ndarray,
        agent_position: Optional[Tuple[int, int]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Update obstacle position for one timestep.
        
        Args:
            grid: Current map grid (0 = passable, 1 = wall)
            agent_position: Current agent position (for intelligent obstacles)
            rng: Random number generator
        """
        pass
    
    def reset(self) -> None:
        """Reset obstacle to initial state."""
        self.position = self.initial_position.copy()
        self.velocity = np.zeros(2, dtype=np.float32)
        self.time_step = 0
        self.position_history.clear()
    
    def record_position(self) -> None:
        """Record current position to history."""
        self.position_history.append(self.position.copy())
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
    
    def get_history(self, steps: int = 10) -> np.ndarray:
        """
        Get position history.
        
        Args:
            steps: Number of past steps to return
            
        Returns:
            Array of shape (steps, 2) with historical positions
        """
        history = self.position_history[-steps:] if self.position_history else []
        
        # Pad if not enough history
        if len(history) < steps:
            padding = [self.position.copy()] * (steps - len(history))
            history = padding + list(history)
        
        return np.array(history, dtype=np.float32)
    
    def get_state(self) -> ObstacleState:
        """Get current obstacle state."""
        return ObstacleState(
            position=self.position.copy(),
            velocity=self.velocity.copy()
        )
    
    def _is_valid_position(self, pos: np.ndarray, grid: np.ndarray) -> bool:
        """Check if position is valid (within bounds and not in wall)."""
        x, y = int(round(pos[0])), int(round(pos[1]))
        h, w = grid.shape
        
        if 0 <= x < w and 0 <= y < h:
            return grid[y, x] == 0
        return False
    
    @property
    def obstacle_type(self) -> ObstacleType:
        """Return the type of this obstacle."""
        raise NotImplementedError


class LinearObstacle(Obstacle):
    """
    Obstacle with linear motion.
    
    Moves in a straight line at constant velocity.
    Changes direction randomly when hitting walls.
    """
    
    def __init__(
        self,
        initial_position: Tuple[float, float],
        initial_direction: Optional[Tuple[float, float]] = None,
        speed: float = 1.0,
        direction_change_prob: float = 0.1,
        **kwargs
    ):
        """
        Args:
            initial_position: Starting position
            initial_direction: Initial movement direction (normalized)
            speed: Movement speed
            direction_change_prob: Probability of random direction change
        """
        super().__init__(initial_position, speed, **kwargs)
        
        if initial_direction is None:
            # Random initial direction
            angle = np.random.uniform(0, 2 * np.pi)
            initial_direction = (np.cos(angle), np.sin(angle))
        
        direction = np.array(initial_direction, dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        
        self.velocity = direction * speed
        self.direction_change_prob = direction_change_prob
    
    @property
    def obstacle_type(self) -> ObstacleType:
        return ObstacleType.LINEAR
    
    def update(
        self,
        grid: np.ndarray,
        agent_position: Optional[Tuple[int, int]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        rng = rng or np.random.default_rng()
        
        # Record current position
        self.record_position()
        
        # Try to move
        new_position = self.position + self.velocity
        
        # Check for collision with walls
        if not self._is_valid_position(new_position, grid):
            # Bounce: reflect velocity
            # Try reflecting x
            test_pos = np.array([self.position[0] - self.velocity[0], new_position[1]])
            if self._is_valid_position(test_pos, grid):
                self.velocity[0] = -self.velocity[0]
            else:
                # Try reflecting y
                test_pos = np.array([new_position[0], self.position[1] - self.velocity[1]])
                if self._is_valid_position(test_pos, grid):
                    self.velocity[1] = -self.velocity[1]
                else:
                    # Reflect both
                    self.velocity = -self.velocity
            
            new_position = self.position + self.velocity
            
            # If still invalid, pick a random valid direction
            if not self._is_valid_position(new_position, grid):
                self._pick_random_direction(grid, rng)
                new_position = self.position + self.velocity
        
        # Random direction change
        if rng.random() < self.direction_change_prob:
            self._pick_random_direction(grid, rng)
        
        # Update position if valid
        if self._is_valid_position(new_position, grid):
            self.position = new_position
        
        self.time_step += 1
    
    def _pick_random_direction(self, grid: np.ndarray, rng: np.random.Generator):
        """Pick a random valid direction."""
        for _ in range(8):  # Try up to 8 times
            angle = rng.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])
            self.velocity = direction * self.speed
            
            test_pos = self.position + self.velocity
            if self._is_valid_position(test_pos, grid):
                break


class PatrolObstacle(Obstacle):
    """
    Obstacle that follows a patrol route between waypoints.
    """
    
    def __init__(
        self,
        initial_position: Tuple[float, float],
        waypoints: List[Tuple[float, float]],
        speed: float = 1.0,
        loop: bool = True,
        **kwargs
    ):
        """
        Args:
            initial_position: Starting position
            waypoints: List of (x, y) waypoints to visit
            speed: Movement speed
            loop: Whether to loop back to start after reaching last waypoint
        """
        super().__init__(initial_position, speed, **kwargs)
        
        self.waypoints = [np.array(wp, dtype=np.float32) for wp in waypoints]
        self.current_waypoint_idx = 0
        self.loop = loop
        self.direction = 1  # 1 = forward, -1 = backward (for non-loop)
        
        if not self.waypoints:
            self.waypoints = [self.initial_position.copy()]
    
    @property
    def obstacle_type(self) -> ObstacleType:
        return ObstacleType.PATROL
    
    def update(
        self,
        grid: np.ndarray,
        agent_position: Optional[Tuple[int, int]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.record_position()
        
        if len(self.waypoints) < 2:
            self.time_step += 1
            return
        
        # Get current target
        target = self.waypoints[self.current_waypoint_idx]
        
        # Move towards target
        direction = target - self.position
        distance = np.linalg.norm(direction)
        
        if distance < self.speed:
            # Reached waypoint
            self.position = target.copy()
            
            # Move to next waypoint
            if self.loop:
                self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)
            else:
                # Ping-pong between waypoints
                next_idx = self.current_waypoint_idx + self.direction
                if next_idx >= len(self.waypoints) or next_idx < 0:
                    self.direction *= -1
                    next_idx = self.current_waypoint_idx + self.direction
                self.current_waypoint_idx = next_idx
        else:
            # Move towards waypoint
            direction = direction / distance
            self.velocity = direction * self.speed
            new_position = self.position + self.velocity
            
            if self._is_valid_position(new_position, grid):
                self.position = new_position
        
        self.time_step += 1
    
    def reset(self) -> None:
        super().reset()
        self.current_waypoint_idx = 0
        self.direction = 1


class CircularObstacle(Obstacle):
    """
    Obstacle that moves in a circular pattern around a center point.
    """
    
    def __init__(
        self,
        initial_position: Tuple[float, float],
        center: Optional[Tuple[float, float]] = None,
        orbit_radius: float = 3.0,
        angular_speed: float = 0.1,
        clockwise: bool = True,
        **kwargs
    ):
        """
        Args:
            initial_position: Starting position
            center: Center of circular motion (defaults to initial position)
            orbit_radius: Radius of circular path
            angular_speed: Angular velocity (radians per timestep)
            clockwise: Direction of rotation
        """
        super().__init__(initial_position, **kwargs)
        
        if center is None:
            center = initial_position
        self.center = np.array(center, dtype=np.float32)
        self.orbit_radius = orbit_radius
        self.angular_speed = angular_speed
        self.clockwise = clockwise
        self.current_angle = 0.0
        
        # Set initial position on the orbit
        self.position = self.center + np.array([
            self.orbit_radius * np.cos(self.current_angle),
            self.orbit_radius * np.sin(self.current_angle)
        ])
    
    @property
    def obstacle_type(self) -> ObstacleType:
        return ObstacleType.CIRCULAR
    
    def update(
        self,
        grid: np.ndarray,
        agent_position: Optional[Tuple[int, int]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.record_position()
        
        # Update angle
        delta = self.angular_speed if not self.clockwise else -self.angular_speed
        self.current_angle += delta
        
        # Calculate new position
        new_position = self.center + np.array([
            self.orbit_radius * np.cos(self.current_angle),
            self.orbit_radius * np.sin(self.current_angle)
        ])
        
        # Update velocity for prediction purposes
        self.velocity = new_position - self.position
        
        # Only move if valid
        if self._is_valid_position(new_position, grid):
            self.position = new_position
        
        self.time_step += 1
    
    def reset(self) -> None:
        super().reset()
        self.current_angle = 0.0
        self.position = self.center + np.array([
            self.orbit_radius * np.cos(self.current_angle),
            self.orbit_radius * np.sin(self.current_angle)
        ])


class OscillatingObstacle(Obstacle):
    """
    Obstacle that oscillates back and forth along a line.
    """
    
    def __init__(
        self,
        initial_position: Tuple[float, float],
        direction: Tuple[float, float] = (1.0, 0.0),
        amplitude: float = 5.0,
        frequency: float = 0.1,
        **kwargs
    ):
        """
        Args:
            initial_position: Center of oscillation
            direction: Direction of oscillation (will be normalized)
            amplitude: Maximum displacement from center
            frequency: Oscillation frequency
        """
        super().__init__(initial_position, **kwargs)
        
        direction = np.array(direction, dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        
        self.oscillation_direction = direction
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = 0.0
    
    @property
    def obstacle_type(self) -> ObstacleType:
        return ObstacleType.OSCILLATING
    
    def update(
        self,
        grid: np.ndarray,
        agent_position: Optional[Tuple[int, int]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.record_position()
        
        old_position = self.position.copy()
        
        # Update phase
        self.phase += self.frequency * 2 * np.pi
        
        # Calculate new position
        displacement = self.amplitude * np.sin(self.phase)
        new_position = self.initial_position + self.oscillation_direction * displacement
        
        # Update velocity
        self.velocity = new_position - old_position
        
        # Only move if valid
        if self._is_valid_position(new_position, grid):
            self.position = new_position
        
        self.time_step += 1
    
    def reset(self) -> None:
        super().reset()
        self.phase = 0.0


class IntelligentObstacle(Obstacle):
    """
    Obstacle with basic pursuit/avoidance behavior.
    
    Can either pursue the agent or move away from it.
    """
    
    def __init__(
        self,
        initial_position: Tuple[float, float],
        speed: float = 0.8,
        pursue: bool = True,
        awareness_radius: float = 10.0,
        **kwargs
    ):
        """
        Args:
            initial_position: Starting position
            speed: Movement speed
            pursue: If True, pursues agent; if False, avoids agent
            awareness_radius: Distance at which obstacle reacts to agent
        """
        super().__init__(initial_position, speed, **kwargs)
        
        self.pursue = pursue
        self.awareness_radius = awareness_radius
        self.wander_direction = np.array([1.0, 0.0])
    
    @property
    def obstacle_type(self) -> ObstacleType:
        return ObstacleType.INTELLIGENT
    
    def update(
        self,
        grid: np.ndarray,
        agent_position: Optional[Tuple[int, int]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        rng = rng or np.random.default_rng()
        self.record_position()
        
        if agent_position is None:
            # Wander randomly
            self._wander(grid, rng)
            self.time_step += 1
            return
        
        agent_pos = np.array(agent_position, dtype=np.float32)
        distance_to_agent = np.linalg.norm(agent_pos - self.position)
        
        if distance_to_agent > self.awareness_radius:
            # Agent not in awareness range, wander
            self._wander(grid, rng)
        else:
            # React to agent
            direction = agent_pos - self.position
            if not self.pursue:
                direction = -direction  # Move away
            
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            
            self.velocity = direction * self.speed
            new_position = self.position + self.velocity
            
            if self._is_valid_position(new_position, grid):
                self.position = new_position
            else:
                # Can't move towards/away from agent, try perpendicular
                perp = np.array([-direction[1], direction[0]])
                new_position = self.position + perp * self.speed
                if self._is_valid_position(new_position, grid):
                    self.position = new_position
        
        self.time_step += 1
    
    def _wander(self, grid: np.ndarray, rng: np.random.Generator):
        """Random wandering behavior."""
        # Occasionally change direction
        if rng.random() < 0.1:
            angle = rng.uniform(-np.pi / 4, np.pi / 4)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            new_dir = np.array([
                cos_a * self.wander_direction[0] - sin_a * self.wander_direction[1],
                sin_a * self.wander_direction[0] + cos_a * self.wander_direction[1]
            ])
            self.wander_direction = new_dir
        
        self.velocity = self.wander_direction * self.speed
        new_position = self.position + self.velocity
        
        if self._is_valid_position(new_position, grid):
            self.position = new_position
        else:
            # Pick new random direction
            angle = rng.uniform(0, 2 * np.pi)
            self.wander_direction = np.array([np.cos(angle), np.sin(angle)])


class ObstacleFactory:
    """
    Factory for creating obstacles with configurable distributions.
    """
    
    DEFAULT_WEIGHTS = {
        ObstacleType.LINEAR: 0.4,
        ObstacleType.PATROL: 0.2,
        ObstacleType.CIRCULAR: 0.15,
        ObstacleType.OSCILLATING: 0.1,
        ObstacleType.INTELLIGENT: 0.15,
    }
    
    def __init__(
        self,
        type_weights: Optional[Dict[ObstacleType, float]] = None,
        speed_range: Tuple[float, float] = (0.5, 2.0),
        seed: Optional[int] = None,
    ):
        """
        Args:
            type_weights: Probability weights for each obstacle type
            speed_range: (min_speed, max_speed) range for obstacles
            seed: Random seed
        """
        self.type_weights = type_weights or self.DEFAULT_WEIGHTS
        self.speed_range = speed_range
        self.rng = np.random.default_rng(seed)
        self._obstacle_counter = 0
    
    def create_obstacle(
        self,
        position: Tuple[float, float],
        grid: np.ndarray,
        obstacle_type: Optional[ObstacleType] = None,
        **kwargs
    ) -> Obstacle:
        """
        Create an obstacle at the given position.
        
        Args:
            position: (x, y) position
            grid: Map grid for generating valid waypoints
            obstacle_type: Specific type to create (or random if None)
            **kwargs: Additional arguments for the obstacle
            
        Returns:
            Obstacle instance
        """
        if obstacle_type is None:
            obstacle_type = self._sample_type()
        
        speed = kwargs.pop('speed', self.rng.uniform(*self.speed_range))
        obstacle_id = self._obstacle_counter
        self._obstacle_counter += 1
        
        if obstacle_type == ObstacleType.LINEAR:
            return LinearObstacle(
                initial_position=position,
                speed=speed,
                obstacle_id=obstacle_id,
                **kwargs
            )
        
        elif obstacle_type == ObstacleType.PATROL:
            waypoints = kwargs.pop('waypoints', None)
            if waypoints is None:
                waypoints = self._generate_patrol_waypoints(position, grid)
            return PatrolObstacle(
                initial_position=position,
                waypoints=waypoints,
                speed=speed,
                obstacle_id=obstacle_id,
                **kwargs
            )
        
        elif obstacle_type == ObstacleType.CIRCULAR:
            return CircularObstacle(
                initial_position=position,
                center=position,
                orbit_radius=kwargs.pop('orbit_radius', self.rng.uniform(2, 5)),
                angular_speed=kwargs.pop('angular_speed', self.rng.uniform(0.05, 0.15)),
                obstacle_id=obstacle_id,
                **kwargs
            )
        
        elif obstacle_type == ObstacleType.OSCILLATING:
            angle = self.rng.uniform(0, 2 * np.pi)
            direction = (np.cos(angle), np.sin(angle))
            return OscillatingObstacle(
                initial_position=position,
                direction=kwargs.pop('direction', direction),
                amplitude=kwargs.pop('amplitude', self.rng.uniform(3, 8)),
                frequency=kwargs.pop('frequency', self.rng.uniform(0.05, 0.15)),
                obstacle_id=obstacle_id,
                **kwargs
            )
        
        elif obstacle_type == ObstacleType.INTELLIGENT:
            return IntelligentObstacle(
                initial_position=position,
                speed=speed,
                pursue=kwargs.pop('pursue', self.rng.random() > 0.3),  # 70% pursuers
                obstacle_id=obstacle_id,
                **kwargs
            )
        
        raise ValueError(f"Unknown obstacle type: {obstacle_type}")
    
    def create_obstacles(
        self,
        num_obstacles: int,
        grid: np.ndarray,
        avoid_positions: Optional[List[Tuple[int, int]]] = None,
        min_distance: float = 3.0,
    ) -> List[Obstacle]:
        """
        Create multiple obstacles with valid positions.
        
        Args:
            num_obstacles: Number of obstacles to create
            grid: Map grid
            avoid_positions: Positions to avoid (e.g., agent start, goal)
            min_distance: Minimum distance from avoid_positions
            
        Returns:
            List of obstacles
        """
        obstacles = []
        avoid_positions = avoid_positions or []
        
        # Get all passable positions
        passable = np.argwhere(grid == 0)
        
        for _ in range(num_obstacles):
            # Try to find valid position
            for _ in range(100):  # Max attempts
                idx = self.rng.integers(0, len(passable))
                y, x = passable[idx]
                pos = (float(x), float(y))
                
                # Check distance from avoid positions
                valid = True
                for avoid in avoid_positions:
                    dist = np.sqrt((pos[0] - avoid[0])**2 + (pos[1] - avoid[1])**2)
                    if dist < min_distance:
                        valid = False
                        break
                
                # Check distance from other obstacles
                for obs in obstacles:
                    dist = np.sqrt(
                        (pos[0] - obs.position[0])**2 + 
                        (pos[1] - obs.position[1])**2
                    )
                    if dist < min_distance / 2:
                        valid = False
                        break
                
                if valid:
                    obstacle = self.create_obstacle(pos, grid)
                    obstacles.append(obstacle)
                    break
        
        return obstacles
    
    def _sample_type(self) -> ObstacleType:
        """Sample an obstacle type based on weights."""
        types = list(self.type_weights.keys())
        weights = [self.type_weights[t] for t in types]
        total = sum(weights)
        weights = [w / total for w in weights]
        
        return self.rng.choice(types, p=weights)
    
    def _generate_patrol_waypoints(
        self,
        start: Tuple[float, float],
        grid: np.ndarray,
        num_waypoints: int = 4,
        max_distance: float = 10.0,
    ) -> List[Tuple[float, float]]:
        """Generate valid patrol waypoints."""
        waypoints = [start]
        current = np.array(start)
        
        for _ in range(num_waypoints - 1):
            # Try to find a valid waypoint
            for _ in range(20):
                angle = self.rng.uniform(0, 2 * np.pi)
                distance = self.rng.uniform(3, max_distance)
                new_point = current + np.array([
                    np.cos(angle) * distance,
                    np.sin(angle) * distance
                ])
                
                x, y = int(round(new_point[0])), int(round(new_point[1]))
                h, w = grid.shape
                
                if 0 <= x < w and 0 <= y < h and grid[y, x] == 0:
                    waypoints.append((float(new_point[0]), float(new_point[1])))
                    current = new_point
                    break
        
        return waypoints
    
    def reset(self, seed: Optional[int] = None):
        """Reset the factory with a new seed."""
        self.rng = np.random.default_rng(seed)
        self._obstacle_counter = 0

