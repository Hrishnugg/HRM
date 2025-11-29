"""
MovingAI Benchmark Map Parser.

Parses .map files from the MovingAI benchmark suite.
Format specification: https://movingai.com/benchmarks/formats.html
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List
import requests
import zipfile
import io


@dataclass
class GridMap:
    """Represents a parsed grid map."""
    
    grid: np.ndarray  # 2D array: 0 = passable, 1 = obstacle
    height: int
    width: int
    name: str
    
    def is_passable(self, x: int, y: int) -> bool:
        """Check if a cell is passable (within bounds and not an obstacle)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y, x] == 0
        return False
    
    def get_passable_cells(self) -> List[Tuple[int, int]]:
        """Get all passable cell coordinates."""
        passable = np.argwhere(self.grid == 0)
        return [(int(x), int(y)) for y, x in passable]
    
    def random_passable_position(self, rng: np.random.Generator) -> Tuple[int, int]:
        """Get a random passable position."""
        passable = self.get_passable_cells()
        idx = rng.integers(0, len(passable))
        return passable[idx]
    
    def to_observation(self) -> np.ndarray:
        """Convert to observation array (float32 for neural networks)."""
        return self.grid.astype(np.float32)


class MapParser:
    """Parser for MovingAI .map format files."""
    
    # Character to terrain type mapping
    PASSABLE_CHARS = {'.', 'G', 'S'}  # Ground, Grass, Swamp
    OBSTACLE_CHARS = {'@', 'O', 'T', 'W'}  # Wall, Out of bounds, Trees, Water
    
    # URLs for benchmark maps
    BENCHMARK_URLS = {
        "berlin": "https://movingai.com/benchmarks/street/street-maps.zip",
        "paris": "https://movingai.com/benchmarks/street/street-maps.zip",
        "boston": "https://movingai.com/benchmarks/street/street-maps.zip",
        "dao": "https://movingai.com/benchmarks/dao/dao-maps.zip",
        "bg512": "https://movingai.com/benchmarks/bg512/bg512-maps.zip",
    }
    
    def __init__(self, maps_dir: Optional[Path] = None):
        """
        Initialize the map parser.
        
        Args:
            maps_dir: Directory to store/load benchmark maps.
                     Defaults to envs/maps/benchmark/
        """
        if maps_dir is None:
            maps_dir = Path(__file__).parent / "benchmark"
        self.maps_dir = Path(maps_dir)
        self.maps_dir.mkdir(parents=True, exist_ok=True)
    
    def parse_file(self, filepath: Path) -> GridMap:
        """
        Parse a .map file.
        
        Args:
            filepath: Path to the .map file
            
        Returns:
            GridMap object with parsed data
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        header = {}
        line_idx = 0
        
        for line in lines:
            line = line.strip()
            if line.startswith('type'):
                header['type'] = line.split()[1] if len(line.split()) > 1 else 'octile'
            elif line.startswith('height'):
                header['height'] = int(line.split()[1])
            elif line.startswith('width'):
                header['width'] = int(line.split()[1])
            elif line.startswith('map'):
                line_idx = lines.index(line + '\n') + 1
                break
            line_idx += 1
        
        height = header.get('height', 0)
        width = header.get('width', 0)
        
        # Parse grid
        grid = np.zeros((height, width), dtype=np.uint8)
        
        for y in range(height):
            if line_idx + y < len(lines):
                row = lines[line_idx + y].rstrip('\n')
                for x, char in enumerate(row):
                    if x < width:
                        if char in self.OBSTACLE_CHARS:
                            grid[y, x] = 1
                        # Passable chars remain 0
        
        return GridMap(
            grid=grid,
            height=height,
            width=width,
            name=filepath.stem
        )
    
    def parse_string(self, map_string: str, name: str = "custom") -> GridMap:
        """
        Parse a map from a string representation.
        
        Args:
            map_string: Map in string format (# for walls, . for passable)
            name: Name for the map
            
        Returns:
            GridMap object
        """
        lines = [line for line in map_string.strip().split('\n') if line]
        height = len(lines)
        width = max(len(line) for line in lines) if lines else 0
        
        grid = np.zeros((height, width), dtype=np.uint8)
        
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                if char in {'#', '@', 'O', 'T', 'W', '1'}:
                    grid[y, x] = 1
        
        return GridMap(grid=grid, height=height, width=width, name=name)
    
    def create_random_map(
        self,
        height: int,
        width: int,
        obstacle_ratio: float = 0.2,
        seed: Optional[int] = None,
        name: str = "random"
    ) -> GridMap:
        """
        Create a random grid map.
        
        Args:
            height: Map height
            width: Map width
            obstacle_ratio: Fraction of cells that are obstacles
            seed: Random seed
            name: Map name
            
        Returns:
            GridMap with random obstacles
        """
        rng = np.random.default_rng(seed)
        grid = (rng.random((height, width)) < obstacle_ratio).astype(np.uint8)
        
        # Ensure corners are passable (for start/goal)
        grid[0, 0] = 0
        grid[height-1, width-1] = 0
        
        return GridMap(grid=grid, height=height, width=width, name=name)
    
    def create_maze(
        self,
        height: int,
        width: int,
        seed: Optional[int] = None,
        name: str = "maze"
    ) -> GridMap:
        """
        Create a maze using recursive backtracking.
        
        Args:
            height: Maze height (should be odd)
            width: Maze width (should be odd)
            seed: Random seed
            name: Map name
            
        Returns:
            GridMap with maze structure
        """
        # Ensure odd dimensions for proper maze
        height = height if height % 2 == 1 else height + 1
        width = width if width % 2 == 1 else width + 1
        
        rng = np.random.default_rng(seed)
        
        # Start with all walls
        grid = np.ones((height, width), dtype=np.uint8)
        
        # Carve passages using recursive backtracking
        def carve(y: int, x: int):
            grid[y, x] = 0
            directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
            rng.shuffle(directions)
            
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and grid[ny, nx] == 1:
                    grid[y + dy // 2, x + dx // 2] = 0
                    carve(ny, nx)
        
        # Start carving from (1, 1)
        import sys
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old_limit, height * width))
        
        try:
            carve(1, 1)
        finally:
            sys.setrecursionlimit(old_limit)
        
        return GridMap(grid=grid, height=height, width=width, name=name)
    
    def load_benchmark(self, map_name: str) -> GridMap:
        """
        Load a benchmark map by name.
        
        Args:
            map_name: Name of the map (e.g., "Berlin_0_256", "Paris_1_512")
            
        Returns:
            GridMap object
        """
        # Check if map exists locally
        map_path = self.maps_dir / f"{map_name}.map"
        
        if map_path.exists():
            return self.parse_file(map_path)
        
        # Try to find in subdirectories
        for subdir in self.maps_dir.iterdir():
            if subdir.is_dir():
                potential_path = subdir / f"{map_name}.map"
                if potential_path.exists():
                    return self.parse_file(potential_path)
        
        raise FileNotFoundError(
            f"Map '{map_name}' not found. Please download it first using "
            f"download_benchmark() or place the .map file in {self.maps_dir}"
        )
    
    def download_benchmark(self, benchmark_name: str) -> List[Path]:
        """
        Download a benchmark map set.
        
        Args:
            benchmark_name: Name of benchmark (berlin, paris, boston, dao, bg512)
            
        Returns:
            List of paths to downloaded map files
        """
        benchmark_name = benchmark_name.lower()
        
        if benchmark_name not in self.BENCHMARK_URLS:
            raise ValueError(
                f"Unknown benchmark: {benchmark_name}. "
                f"Available: {list(self.BENCHMARK_URLS.keys())}"
            )
        
        url = self.BENCHMARK_URLS[benchmark_name]
        
        print(f"Downloading {benchmark_name} benchmark maps...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Extract zip
        extracted_paths = []
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            for name in zf.namelist():
                if name.endswith('.map'):
                    # Extract to benchmark directory
                    target_path = self.maps_dir / Path(name).name
                    with open(target_path, 'wb') as f:
                        f.write(zf.read(name))
                    extracted_paths.append(target_path)
        
        print(f"Downloaded {len(extracted_paths)} maps to {self.maps_dir}")
        return extracted_paths
    
    def list_available_maps(self) -> List[str]:
        """List all available map names in the maps directory."""
        maps = []
        for path in self.maps_dir.rglob("*.map"):
            maps.append(path.stem)
        return sorted(maps)


# Convenience function for creating simple test maps
def create_simple_map(size: int = 16, obstacle_density: float = 0.15) -> GridMap:
    """
    Create a simple test map with random obstacles.
    
    Args:
        size: Map size (square)
        obstacle_density: Fraction of cells that are obstacles
        
    Returns:
        GridMap for testing
    """
    parser = MapParser()
    return parser.create_random_map(
        height=size,
        width=size,
        obstacle_ratio=obstacle_density,
        name=f"simple_{size}x{size}"
    )

