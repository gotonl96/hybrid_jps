'''
large scale path planning test map generate class 

parameter : map size, obstacle cnt, obstacle size

output : numpy array

'''

import numpy as np
import random
import matplotlib.pyplot as plt

class MapGenerator:
    def __init__(self, width, height, obstacle_count, obstacle_size_range):
        self.width = width
        self.height = height
        self.obstacle_count = obstacle_count
        self.obstacle_size_range = obstacle_size_range
        self.map = np.zeros((height, width), dtype=int)
        self.start = (10, 10)
        self.goal = (width - 10, height - 10)

    def generate_obstacles(self):
        for _ in range(self.obstacle_count):
            while True:
                obs_width = random.randint(self.obstacle_size_range[0], self.obstacle_size_range[1])
                obs_height = random.randint(self.obstacle_size_range[0], self.obstacle_size_range[1])
                x_pos = random.randint(0, self.width - obs_width - 1)
                y_pos = random.randint(0, self.height - obs_height - 1)
                # 장애물 영역 좌표 집합
                obs_area = set(
                    (x, y)
                    for x in range(x_pos, x_pos + obs_width)
                    for y in range(y_pos, y_pos + obs_height)
                )
                # 시작점과 목표점이 장애물에 포함되면 다시 생성
                if (self.start in obs_area) or (self.goal in obs_area):
                    continue
                # 장애물 생성
                self.map[y_pos:y_pos + obs_height, x_pos:x_pos + obs_width] = 1
                break
            
    def get_map(self):
        return self.map
    
    def get_obstacle_points(self, resolution=1):
        """
        self.map에서 1인 위치를 world 좌표(m)로 변환해서 반환
        반환: ox, oy (리스트)
        """
        # 1인 위치의 (row, col) = (iy, ix)
        iy, ix = np.where(self.map == 1)           # iy: row, ix: col
        ox = (ix + 0.5) * resolution               # 셀 중심 x
        oy = (iy + 0.5) * resolution               # 셀 중심 y
        return ox.tolist(), oy.tolist()

    # visualize start, goal points and map
    # 격자를 좀 논문용으로 쓸거니까 고급스럽게 시각화
    def display_map(self):
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.imshow(
            self.map,
            cmap='Greys',
            origin='lower',
            extent=[0, self.width, 0, self.height],
            interpolation='none'
        )
        # 격자선: 100마다만 표시 (논문 스타일)
        major_grid_interval = max(self.width // 10, 50)
        ax.set_xticks(np.arange(0, self.width+1, major_grid_interval), minor=False)
        ax.set_yticks(np.arange(0, self.height+1, major_grid_interval), minor=False)
        ax.grid(which='major', color='gray', linestyle='-', linewidth=1)
        ax.set_aspect('equal')
        # 시작점: 굵은 초록 원 + 흰 테두리
        ax.scatter(self.start[0]+0.5, self.start[1]+0.5, 
                   c='lime', s=300, marker='o', edgecolors='white', linewidths=3, zorder=10)
        # 목표점: 굵은 빨간 X + 흰 테두리
        ax.scatter(self.goal[0]+0.5, self.goal[1]+0.5, 
                   c='red', s=300, marker='X', edgecolors='white', linewidths=3, zorder=10)
        plt.title('Generated Map with Obstacles')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    map_gen = MapGenerator(width=1000, height=1000, obstacle_count=200, obstacle_size_range=(5, 15))
    map_gen.generate_obstacles()
    generated_map = map_gen.get_map()
    map_gen.display_map()
