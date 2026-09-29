import heapq
from importlib.resources import path
from typing import List, Tuple, Dict, Optional
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tabulate import tabulate
import numpy as np

class MazePathfinder:
    def __init__(self, maze: List[List[int]], objects: Dict[str, Dict[str, Tuple[int, int]]]):
        """
        Initialize the pathfinder with maze and object locations.
        
        Args:
            maze: 2D list where 0 = walkable, 1 = wall/obstacle, 2 = object (blocks movement)
            objects: Dictionary mapping object names to their locations and starting positions as (row, col) coordinates
        """
        self.maze = maze
        self.objects = objects
        self.rows = len(maze)
        self.cols = len(maze[0]) if maze else 0
        
        # Directions: up, down, left, right
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Manhattan distance heuristic for A*"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def is_walkable_for_player(self, row: int, col: int) -> bool:
        """Check if position is walkable for the player (only 0 values allowed)"""
        return (0 <= row < self.rows and 
                0 <= col < self.cols and 
                self.maze[row][col] == 0)
    
    def is_valid(self, row: int, col: int) -> bool:
        """Check if position is valid for pathfinding (can walk on 0, but not 1 or 2)"""
        return (0 <= row < self.rows and 
                0 <= col < self.cols and 
                self.maze[row][col] == 0)
    
    def get_adjacent_walkable_positions(self, obj_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get all walkable positions adjacent to an object.
        
        Args:
            obj_pos: Position of the object as (row, col)
            
        Returns:
            List of adjacent walkable positions
        """
        adjacent_positions = []
        obj_row, obj_col = obj_pos
        
        # Check all four directions around the object
        for dr, dc in self.directions:
            adj_row = obj_row + dr
            adj_col = obj_col + dc
            
            # Check if this adjacent position is walkable
            if self.is_walkable_for_player(adj_row, adj_col):
                adjacent_positions.append((adj_row, adj_col))
        
        return adjacent_positions
    
    def find_nearest_walkable_to_object(self, obj_name: str) -> Optional[Tuple[int, int]]:
        """Find the nearest walkable position to an object from player's current position"""
        if obj_name not in self.objects:
            return None
            
        obj_pos = self.objects[obj_name]['object_location']
        adjacent_positions = self.get_adjacent_walkable_positions(obj_pos)
        
        if not adjacent_positions:
            return None
        
        # Find the adjacent position closest to the player
        min_distance = float('inf')
        nearest_pos = None
        
        for adj_pos in adjacent_positions:
            distance = self.heuristic(self.objects[obj_name]['starting_location'], adj_pos)
            if distance < min_distance:
                min_distance = distance
                nearest_pos = adj_pos
                
        return nearest_pos
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions"""
        row, col = pos
        neighbors = []
        for dr, dc in self.directions:
            new_row, new_col = row + dr, col + dc
            if self.is_valid(new_row, new_col):
                neighbors.append((new_row, new_col))
        return neighbors
    
    def a_star(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find shortest path using A* algorithm.
        
        Args:
            start: Starting position (must be walkable)
            goal: Goal position (must be walkable)
        
        Returns:
            List of (row, col) coordinates representing the path, or None if no path exists
        """
        if start == goal:
            return [start]
        
        if not self.is_walkable_for_player(start[0], start[1]) or not self.is_walkable_for_player(goal[0], goal[1]):
            return None
        
        # Priority queue: (f_score, g_score, position)
        open_set = [(0, 0, start)]
        
        # Track visited nodes
        closed_set = set()
        
        # g_score: cost from start to current node
        g_score = {start: 0}
        
        # Parent tracking for path reconstruction
        came_from = {}
        
        while open_set:
            f_score, current_g, current_pos = heapq.heappop(open_set)
            
            if current_pos in closed_set:
                continue
                
            closed_set.add(current_pos)
            
            # Found the goal
            if current_pos == goal:
                return self.reconstruct_path(came_from, current_pos)
            
            # Explore neighbors
            for neighbor in self.get_neighbors(current_pos):
                if neighbor in closed_set:
                    continue
                
                tentative_g = current_g + 1  # Each step costs 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal)
                    came_from[neighbor] = current_pos
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
        
        return None  # No path found
    
    def reconstruct_path(self, came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from goal to start"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # Reverse to get start->goal path
    
    def find_path_from_start_to_object(self, obj_name: str) -> Optional[List[Tuple[int, int]]]:
        """
        Find shortest path between an object and its corresponding starting position.
        
        Args:
            start_pos: Starting position as (row, col)
            obj_name: Name of the target object
            
        Returns:
            Path from the starting position to the nearest walkable position adjacent to the object, or None if no path exists
        """
        start_pos = self.objects[obj_name]['starting_location']
        row, col = start_pos
        if not (0 <= row < len(maze) and 0 <= col < len(maze[0]) and maze[row][col] == 0):
            raise ValueError(f"Position {start_pos} is an invalid starting position. It must be a walkable tile (0).")
        
        if obj_name not in self.objects:
            raise ValueError(f"Object '{obj_name}' not found. Available objects: {list(self.objects.keys())}")
        
        # Find the nearest walkable position to the object
        target_pos = self.find_nearest_walkable_to_object(obj_name)
        
        if target_pos is None:
            raise ValueError(f"No walkable positions found adjacent to object '{obj_name}'")
        
        # Find path from start position to the nearest walkable position
        return self.a_star(start_pos, target_pos)
    
    def find_all_paths_to_object(self) -> List[Optional[List[Tuple[int, int]]]]:
        """
        Find all paths from each object's starting position to its corresponding object.
        
        Returns:
            List of paths, where each path is a list of (row, col) coordinates
        """
        all_paths = []
        for obj_name in self.objects.keys():
            path = self.find_path_from_start_to_object(obj_name)
            if path is not None:
                all_paths.append(path)
        return all_paths

    def get_path_info(self, path: Optional[List[Tuple[int, int]]]) -> Dict:
        """Get information about a path"""
        if path is None:
            return {"exists": False, "length": 0, "path": []}
        
        return {
            "exists": True,
            "length": len(path) - 1,  # Number of steps (edges)
            "path": path,
            "coordinates": path
        }
        
    def plot_all_paths(self, figsize: Tuple[int, int] = (14, 14)) -> None:
        """
        Plot all object paths in a 3x4 grid.
        """
        fig, axes = plt.subplots(3, 4, figsize=figsize)
        axes = axes.flatten()

        colors = ["white", "black", "lightblue", "gray"]
        cmap = ListedColormap(colors)

        for ax, obj_name in zip(axes, self.objects):
            path = self.find_path_from_start_to_object(obj_name)
            path_info = self.get_path_info(path)

            display_maze = np.array(self.maze, dtype=float)

            if path_info["exists"]:
                for row, col in path_info["path"]:
                    if display_maze[row, col] == 0:
                        display_maze[row, col] = 3

            ax.imshow(
                display_maze,
                cmap=cmap,
                interpolation="nearest",
                vmin=0,
                vmax=3
            )

            # Plot all object locations
            for other_obj_name, object_data in self.objects.items():
                row, col = object_data["object_location"]

                if other_obj_name == obj_name:
                    ax.plot(
                        col,
                        row,
                        "ro",
                        markersize=10,
                        markeredgecolor="blue",
                        markeredgewidth=1
                    )
                else:
                    ax.plot(
                        col,
                        row,
                        "bo",
                        markersize=7,
                        alpha=0.7,
                        markeredgecolor="black",
                        markeredgewidth=1
                    )

            # Plot starting position
            player_row, player_col = self.objects[obj_name]["starting_location"]
            ax.plot(
                player_col,
                player_row,
                "go",
                markersize=10,
                markeredgecolor="blue",
                markeredgewidth=1
            )

            # Plot path
            if path_info["exists"] and len(path_info["path"]) > 1:
                path_rows = [position[0] for position in path_info["path"]]
                path_cols = [position[1] for position in path_info["path"]]

                ax.plot(
                    path_cols,
                    path_rows,
                    "r-",
                    linewidth=2,
                    alpha=0.8
                )

            ax.set_xlim(-0.5, self.cols - 0.5)
            ax.set_ylim(self.rows - 0.5, -0.5)
            ax.set_xticks(range(self.cols))
            ax.set_yticks(range(self.rows))
            ax.xaxis.tick_top()
            ax.grid(True, alpha=0.3)

            ax.set_aspect("equal", adjustable="box")

            ax.set_title(
                f"{obj_name}\n"
                f"Path length: {path_info['length']} steps",
                fontsize=8,
                pad=8
            )
            
            ax.tick_params(axis="both", labelsize=7)

        # Create one shared legend for the entire figure
        legend_elements = [
            plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor="green",
                markersize=8,
                label="Starting position"
            ),
            plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=8,
                label="Target object"
            ),
            plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor="blue",
                markersize=8,
                label="Other object"
            ),
            plt.Line2D(
                [0], [0],
                color="red",
                linewidth=2,
                label="Shortest path"
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="white",
                edgecolor="black",
                label="Walkable"
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="black",
                label="Wall"
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="lightblue",
                label="Object tile"
            )
        ]

        fig.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            ncol=1,
            borderaxespad=0
        )

        fig.subplots_adjust(
            left=0.16,
            right=0.99,
            bottom=0.03,
            top=0.94,
            wspace=0.20,
            hspace=0.40
        )
        
        plt.show()
        
        

# Example usage and maze representation
def create_example_maze():
    """
    Create an example maze
    """
    # (0 = walkable, 1 = wall)
    maze = [
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 0],
        [0, 2, 1, 0, 2, 1, 0, 2, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 2, 0, 2, 1, 1, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 1, 1, 0, 1, 1, 0, 1, 1, 0],
        [0, 1, 1, 1, 0, 2, 2, 0, 1, 2, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
        [0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 1, 2, 1, 1]
    ]
    
    # Object locations
    objects = {
        'suitcase': {
            'object_location': (1, 1),
            'starting_location': (10, 0)
        },
        'guitar': {
            'object_location': (1, 4),
            'starting_location': (5, 10)
        },
        'fire_extinguisher': {
            'object_location': (1, 7),
            'starting_location': (4, 3)
        },
        'stove': {
            'object_location': (0, 9),
            'starting_location': (0, 4)
        },
        'cat_statue': {
            'object_location': (4, 5),
            'starting_location': (2, 10)
        },
        'chair': {
            'object_location': (4, 7),
            'starting_location': (4, 0)
        },
        'basketball': {
            'object_location': (6, 1),
            'starting_location': (10, 5)
        },
        'treasure_chest': {
            'object_location': (7, 5),
            'starting_location': (5, 10)
        },
        'vase': {
            'object_location': (7, 6),
            'starting_location': (4, 3)
        },
        'watermelon': {
            'object_location': (7, 9),
            'starting_location': (0, 4)
        },
        'bedside_lamp': {
            'object_location': (9, 2),
            'starting_location': (2, 10)
        },
        'potted_plant': {
            'object_location': (10, 8),
            'starting_location': (10, 0)
        }
    }
    
    return maze, objects

# Example usage
if __name__ == "__main__":
    # Create example maze
    maze, objects = create_example_maze()
    
    # Initialize pathfinder
    pathfinder = MazePathfinder(maze, objects)

    # Print maze with positions and contents in a well-aligned table

    # Build table data
    table_data = []
    header = ["(row,col)"] + [f"{col}" for col in range(len(maze[0]))]
    for row_idx, row in enumerate(maze):
        row_cells = [f"{row_idx}"]
        for col_idx, val in enumerate(row):
            if val == 1:
                cell = "Wall"
            elif val == 2:
                obj_name = next((name for name, pos in objects.items() if pos == (row_idx, col_idx)), "Object")
                cell = obj_name
            else:
                cell = "Free"
            row_cells.append(cell)
        table_data.append(row_cells)

    print("Maze layout (row, col):")
    print(tabulate(table_data, headers=header, tablefmt="grid", stralign="center"))
    
    paths_to_objects = pathfinder.find_all_paths_to_object()
    
    for obj_name, path in zip(objects.keys(), paths_to_objects):
        path_info = pathfinder.get_path_info(path)
        print(f"\nObject: {obj_name}")
        print(f"Path Exists?: {path_info['exists']}")
        print(f"Path Length: {path_info['length']} steps")
        if path_info['exists']:
            print("Path coordinates:")
            for i, (row, col) in enumerate(path_info['path']):
                print(f"  Step {i}: ({row}, {col})")
    
    plot_all_paths = input("\nDo you want to visualize all paths? (y/n): ").strip().lower()
    if plot_all_paths == 'y':
        pathfinder.plot_all_paths()   
