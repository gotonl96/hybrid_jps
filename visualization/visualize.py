from matplotlib import pyplot as plt
import numpy as np
import random

class Visualizer:

    def __init__(self):
        self.grid_map = None  # 2D numpy array
        self.start = None  # (x, y)
        self.goal = None  # (x, y)
        self.path = {}

    def set_start_goal(self, start, goal):
        self.start = start
        self.goal = goal

    def set_grid_map(self, grid_map):
        self.grid_map = grid_map

    def set_path(self, path, path_type="hybrid_a_star"):
        self.path[path_type] = path

    def draw(self):
        if self.grid_map is None:
            raise ValueError("Grid map not set.")

        plt.imshow(self.grid_map, cmap='Greys', origin='lower')

        if self.start:
            plt.plot(self.start[0], self.start[1], "go", label="Start")
        if self.goal:
            plt.plot(self.goal[0], self.goal[1], "ro", label="Goal")

        if self.path:
            for path_type, path in self.path.items():
                if path:
                    path = np.array(path)
                    # 랜덤 색상 생성 (예: RGB)
                    color = [random.random() for _ in range(3)]
                    plt.plot(path[:, 0], path[:, 1], label=path_type, color=color, linewidth=3)

        plt.legend()
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title("Path Planning Visualization")
        plt.grid(True)
        plt.show()

def main():
    # Example usage of Visualizer
    vis = Visualizer()

    # Create a sample grid map
    grid_map = np.zeros((100, 100))
    grid_map[30:70, 40:60] = 1  # Add an obstacle
    vis.set_grid_map(grid_map)

    # Set start and goal positions
    vis.start = (10, 10)
    vis.goal = (90, 90)

    # Sample path (replace with actual path data)
    sample_path1 = [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50), (60, 60), (70, 70), (80, 80), (90, 90)]
    vis.set_path(sample_path1, path_type="hybrid_a_star")

    sample_path2 = [(10, 10), (15, 25), (25, 35), (35, 45), (45, 55), (55, 65), (65, 75), (75, 85), (90, 90)]
    vis.set_path(sample_path2, path_type="jps")

    # Draw the visualization
    vis.draw()

if __name__ == "__main__":
    main()
