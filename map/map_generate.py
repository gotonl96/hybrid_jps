'''
large scale path planning test map generate class 

parameter : map size, obstacle cnt, obstacle size
output : numpy array

+ 맵 저장/불러오기 기능 추가 (numpy .npz 포맷)
'''

import numpy as np
import random
import matplotlib.pyplot as plt
import os

class MapGenerator:
    def __init__(self, width, height, obstacle_count=0, obstacle_size_range=(5, 15), seed=None):
        self.width = width
        self.height = height
        self.obstacle_count = obstacle_count
        self.obstacle_size_range = obstacle_size_range
        self.map = np.zeros((height, width), dtype=np.uint8)  # 0: free, 1: obstacle
        self.start = (10, 10)
        self.goal = (width - 10, height - 10)
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_obstacles(self):
        random.seed(self.seed)
        np.random.seed(self.seed if self.seed is not None else random.randint(0, 2**32-1))
        
        for _ in range(self.obstacle_count):
            while True:
                obs_width = random.randint(self.obstacle_size_range[0], self.obstacle_size_range[1])
                obs_height = random.randint(self.obstacle_size_range[0], self.obstacle_size_range[1])
                x_pos = random.randint(0, self.width - obs_width - 1)
                y_pos = random.randint(0, self.height - obs_height - 1)
                
                obs_area = {(x, y) for x in range(x_pos, x_pos + obs_width)
                                      for y in range(y_pos, y_pos + obs_height)}
                
                if self.start in obs_area or self.goal in obs_area:
                    continue
                
                self.map[y_pos:y_pos + obs_height, x_pos:x_pos + obs_width] = 1
                break

    def save_map(self, filename="generated_map.npz"):
        """
        맵 + start + goal + 기타 정보 저장
        """
        np.savez_compressed(
            filename,
            map=self.map,
            start=self.start,
            goal=self.goal,
            width=self.width,
            height=self.height,
            obstacle_count=self.obstacle_count,
            seed=self.seed
        )
        print(f"맵이 저장되었습니다: {os.path.abspath(filename)}")

    @staticmethod
    def load_map(filename="generated_map.npz"):
        """
        저장된 맵 불러오기 (정적 메서드)
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filename}")
        
        data = np.load(filename, allow_pickle=True)
        
        map_gen = MapGenerator(
            width=int(data['width']),
            height=int(data['height']),
            obstacle_count=int(data['obstacle_count']) if 'obstacle_count' in data else 0,
            obstacle_size_range=(5, 15),  # 기본값
            seed=data['seed'].item() if 'seed' in data else None
        )
        map_gen.map = data['map']
        map_gen.start = tuple(data['start'])
        map_gen.goal = tuple(data['goal'])
        
        print(f"맵이 로드되었습니다: {filename}")
        print(f"   크기: {map_gen.width} x {map_gen.height}")
        print(f"   장애물 개수: {map_gen.obstacle_count}")
        print(f"   시작점: {map_gen.start}, 목표점: {map_gen.goal}")
        return map_gen

    def get_map(self):
        return self.map.copy()  # 안전하게 복사본 반환

    def get_obstacle_points(self, resolution=1.0):
        iy, ix = np.where(self.map == 1)
        ox = (ix + 0.5) * resolution
        oy = (iy + 0.5) * resolution
        return ox.tolist(), oy.tolist()

    def display_map(self):
        plt.figure(figsize=(12, 12))
        ax = plt.gca()
        ax.imshow(
            self.map,
            cmap='binary',
            origin='lower',
            extent=[0, self.width, 0, self.height],
            alpha=0.8
        )
        major_grid = max(self.width // 10, 50)
        ax.set_xticks(np.arange(0, self.width + 1, major_grid))
        ax.set_yticks(np.arange(0, self.height + 1, major_grid))
        ax.grid(which='major', color='gray', linestyle='-', linewidth=0.8, alpha=0.7)
        ax.set_aspect('equal')
        
        # 시작점
        ax.scatter(self.start[0] + 0.5, self.start[1] + 0.5,
                   c='lime', s=500, marker='o', edgecolors='black', linewidths=3, zorder=10, label='Start')
        # 목표점
        ax.scatter(self.goal[0] + 0.5, self.goal[1] + 0.5,
                   c='red', s=500, marker='X', edgecolors='black', linewidths=3, zorder=10, label='Goal')
        
        ax.legend(loc='upper right')
        plt.title(f'Path Planning Map ({self.width}×{self.height}, {self.obstacle_count} obstacles)', fontsize=16)
        plt.tight_layout()
        plt.show()


# 사용 예시
if __name__ == "__main__":
    
    MAP_WIDTH = 1000
    MAP_HEIGHT = 1000
    
    # 1. 맵 생성
    # map_gen = MapGenerator(1000, 1000, 1000, (5, 20))
    # map_gen.generate_obstacles()
    # map_gen.display_map()

    # 자동으로 파일명 생성: my_large_map_1000x1000.npz
    filename = f"my_large_map_{MAP_WIDTH}x{MAP_HEIGHT}.npz"
    # map_gen.save_map(filename)
    # print(f"저장 완료 → {filename}")

    # 2. 저장된 맵 불러오기 (파일명도 자동 생성)
    loaded_map = MapGenerator.load_map(filename)
    loaded_map.display_map()