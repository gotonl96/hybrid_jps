import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime
import os

class Visualizer:
    def __init__(self):
        self.grid_map = None
        self.start = None
        self.goal = None
        self.path = {}
    
    def set_grid_map(self, grid_map):
        self.grid_map = grid_map
    
    def set_start_goal(self, start, goal):
        self.start = start
        self.goal = goal
    
    def set_path(self, path, path_type="Path"):
        self.path[path_type] = path
    
    def draw(self, save_dir="results"):
        if self.grid_map is None:
            raise ValueError("Grid map not set.")

        plt.figure(figsize=(10, 10))
        plt.imshow(self.grid_map, cmap='Greys', origin='lower')

        if self.start:
            plt.plot(self.start[0], self.start[1], "go", label="Start", markersize=10)
        if self.goal:
            plt.plot(self.goal[0], self.goal[1], "ro", label="Goal", markersize=10)

        if self.path:
            for path_type, path in self.path.items():
                if path:
                    path = np.array(path)
                    # 랜덤 색상 생성
                    color = [random.random() for _ in range(3)]
                    plt.plot(path[:, 0], path[:, 1], label=path_type, color=color, linewidth=3)

        plt.legend()
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title("Path Planning Visualization")
        plt.grid(True)
        
        # 저장 디렉토리 생성
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 현재 시간으로 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"path_planning_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)
        
        # 파일 저장
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ Figure saved: {filepath}")
        
        plt.show()