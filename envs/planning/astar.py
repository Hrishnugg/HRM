"""
A* Pathfinding Algorithm.

Standard A* implementation with support for:
- Multiple heuristic functions
- 4-connected and 8-connected grids
- Dynamic re-planning triggers
"""

import heapq
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Dict, Set
from enum import Enum
import time


class HeuristicType(Enum):
    """Built-in heuristic types."""
    MANHATTAN = "manhattan"
    EUCLIDEAN = "euclidean"
    OCTILE = "octile"
    CHEBYSHEV = "chebyshev"


@dataclass
class AStarResult:
    """Result of A* pathfinding."""
    
    # The computed path as list of (x, y) coordinates
    path: List[Tuple[int, int]] = field(default_factory=list)
    
    # Whether a path was found
    success: bool = False
    
    # Total path cost
    cost: float = float('inf')
    
    # Number of nodes expanded
    nodes_expanded: int = 0
    
    # Computation time in seconds
    computation_time: float = 0.0
    
    # Additional statistics
    open_set_max_size: int = 0
    closed_set_size: int = 0


@dataclass(order=True)
class Node:
    """A* search node."""
    
    f_score: float  # Total estimated cost (g + h)
    position: Tuple[int, int] = field(compare=False)
    g_score: float = field(compare=False, default=float('inf'))
    h_score: float = field(compare=False, default=0.0)
    parent: Optional['Node'] = field(compare=False, default=None)


class AStar:
    """
    A* pathfinding algorithm.
    
    Supports various heuristics and grid connectivity options.
    """
    
    # Movement costs for 8-directional movement
    STRAIGHT_COST = 1.0
    DIAGONAL_COST = 1.41421356  # sqrt(2)
    
    # 4-directional movements (dx, dy)
    MOVES_4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    # 8-directional movements
    MOVES_8 = [
        (0, 1), (0, -1), (1, 0), (-1, 0),  # Cardinal
        (1, 1), (1, -1), (-1, 1), (-1, -1),  # Diagonal
    ]
    
    def __init__(
        self,
        heuristic: HeuristicType = HeuristicType.OCTILE,
        eight_connected: bool = True,
        heuristic_weight: float = 1.0,
        custom_heuristic: Optional[Callable] = None,
    ):
        """
        Initialize A* planner.
        
        Args:
            heuristic: Type of heuristic to use
            eight_connected: Use 8-directional movement
            heuristic_weight: Weight for heuristic (>1 for weighted A*)
            custom_heuristic: Custom heuristic function(node, goal) -> float
        """
        self.heuristic_type = heuristic
        self.eight_connected = eight_connected
        self.heuristic_weight = heuristic_weight
        self.custom_heuristic = custom_heuristic
        
        self.moves = self.MOVES_8 if eight_connected else self.MOVES_4
    
    def _heuristic(
        self,
        node: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> float:
        """
        Compute heuristic value.
        
        Args:
            node: Current position
            goal: Goal position
            
        Returns:
            Estimated cost to goal
        """
        if self.custom_heuristic is not None:
            return self.custom_heuristic(node, goal)
        
        dx = abs(node[0] - goal[0])
        dy = abs(node[1] - goal[1])
        
        if self.heuristic_type == HeuristicType.MANHATTAN:
            return dx + dy
        
        elif self.heuristic_type == HeuristicType.EUCLIDEAN:
            return np.sqrt(dx * dx + dy * dy)
        
        elif self.heuristic_type == HeuristicType.OCTILE:
            # Optimal for 8-connected grids
            return max(dx, dy) + (self.DIAGONAL_COST - 1) * min(dx, dy)
        
        elif self.heuristic_type == HeuristicType.CHEBYSHEV:
            return max(dx, dy)
        
        return 0.0
    
    def _movement_cost(self, dx: int, dy: int) -> float:
        """Get cost for a movement."""
        if dx != 0 and dy != 0:
            return self.DIAGONAL_COST
        return self.STRAIGHT_COST
    
    def find_path(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        dynamic_obstacles: Optional[Set[Tuple[int, int]]] = None,
        max_iterations: int = 100000,
    ) -> AStarResult:
        """
        Find path from start to goal.
        
        Args:
            grid: Occupancy grid (0 = free, 1 = obstacle)
            start: Start position (x, y)
            goal: Goal position (x, y)
            dynamic_obstacles: Set of positions blocked by dynamic obstacles
            max_iterations: Maximum search iterations
            
        Returns:
            AStarResult with path and statistics
        """
        start_time = time.perf_counter()
        
        height, width = grid.shape
        dynamic_obstacles = dynamic_obstacles or set()
        
        # Validate start and goal
        if not self._is_valid(start, grid, dynamic_obstacles):
            return AStarResult(success=False, computation_time=time.perf_counter() - start_time)
        
        if not self._is_valid(goal, grid, dynamic_obstacles):
            return AStarResult(success=False, computation_time=time.perf_counter() - start_time)
        
        # Initialize
        start_h = self._heuristic(start, goal) * self.heuristic_weight
        start_node = Node(
            f_score=start_h,
            position=start,
            g_score=0.0,
            h_score=start_h,
        )
        
        # Priority queue (min-heap)
        open_set: List[Node] = [start_node]
        heapq.heapify(open_set)
        
        # Position -> best g_score found
        g_scores: Dict[Tuple[int, int], float] = {start: 0.0}
        
        # Position -> node (for path reconstruction)
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # Closed set
        closed_set: Set[Tuple[int, int]] = set()
        
        # Statistics
        nodes_expanded = 0
        open_set_max_size = 1
        
        while open_set and nodes_expanded < max_iterations:
            # Get node with lowest f_score
            current = heapq.heappop(open_set)
            
            # Skip if already processed with better cost
            if current.position in closed_set:
                continue
            
            closed_set.add(current.position)
            nodes_expanded += 1
            
            # Goal check
            if current.position == goal:
                path = self._reconstruct_path(came_from, current.position)
                return AStarResult(
                    path=path,
                    success=True,
                    cost=current.g_score,
                    nodes_expanded=nodes_expanded,
                    computation_time=time.perf_counter() - start_time,
                    open_set_max_size=open_set_max_size,
                    closed_set_size=len(closed_set),
                )
            
            # Expand neighbors
            for dx, dy in self.moves:
                neighbor_pos = (current.position[0] + dx, current.position[1] + dy)
                
                # Skip invalid positions
                if not self._is_valid(neighbor_pos, grid, dynamic_obstacles):
                    continue
                
                # Skip if already fully explored
                if neighbor_pos in closed_set:
                    continue
                
                # Check diagonal movement validity (no corner cutting)
                if dx != 0 and dy != 0:
                    if not self._can_move_diagonal(
                        current.position, dx, dy, grid, dynamic_obstacles
                    ):
                        continue
                
                # Calculate costs
                move_cost = self._movement_cost(dx, dy)
                tentative_g = current.g_score + move_cost
                
                # Skip if we have a better path to this neighbor
                if neighbor_pos in g_scores and tentative_g >= g_scores[neighbor_pos]:
                    continue
                
                # This is the best path to neighbor so far
                g_scores[neighbor_pos] = tentative_g
                came_from[neighbor_pos] = current.position
                
                h_score = self._heuristic(neighbor_pos, goal) * self.heuristic_weight
                f_score = tentative_g + h_score
                
                neighbor_node = Node(
                    f_score=f_score,
                    position=neighbor_pos,
                    g_score=tentative_g,
                    h_score=h_score,
                )
                
                heapq.heappush(open_set, neighbor_node)
                open_set_max_size = max(open_set_max_size, len(open_set))
        
        # No path found
        return AStarResult(
            success=False,
            nodes_expanded=nodes_expanded,
            computation_time=time.perf_counter() - start_time,
            open_set_max_size=open_set_max_size,
            closed_set_size=len(closed_set),
        )
    
    def _is_valid(
        self,
        pos: Tuple[int, int],
        grid: np.ndarray,
        dynamic_obstacles: Set[Tuple[int, int]],
    ) -> bool:
        """Check if position is valid."""
        x, y = pos
        height, width = grid.shape
        
        if not (0 <= x < width and 0 <= y < height):
            return False
        
        if grid[y, x] != 0:
            return False
        
        if pos in dynamic_obstacles:
            return False
        
        return True
    
    def _can_move_diagonal(
        self,
        pos: Tuple[int, int],
        dx: int,
        dy: int,
        grid: np.ndarray,
        dynamic_obstacles: Set[Tuple[int, int]],
    ) -> bool:
        """Check if diagonal movement is valid (no corner cutting)."""
        # Check adjacent cells
        adj1 = (pos[0] + dx, pos[1])
        adj2 = (pos[0], pos[1] + dy)
        
        return (
            self._is_valid(adj1, grid, dynamic_obstacles) and
            self._is_valid(adj2, grid, dynamic_obstacles)
        )
    
    def _reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from dictionary."""
        path = [current]
        
        while current in came_from:
            current = came_from[current]
            path.append(current)
        
        path.reverse()
        return path
    
    def set_heuristic(
        self,
        heuristic: Optional[Callable] = None,
        heuristic_type: Optional[HeuristicType] = None,
    ):
        """Update heuristic function."""
        if heuristic is not None:
            self.custom_heuristic = heuristic
        if heuristic_type is not None:
            self.heuristic_type = heuristic_type


def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Euclidean distance heuristic."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return np.sqrt(dx * dx + dy * dy)


def octile_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Octile distance (optimal for 8-connected grids)."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)

