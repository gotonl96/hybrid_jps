import sys
import pathlib
import numpy as np
import os
import matplotlib.pyplot as plt


from map.map_generate import MapGenerator
from path_planner.hybrid_astar.hybrid_a_star import hybrid_a_star_planning
from path_planner.hybrid_jps.hybrid_jps import HybridJPS
import path_planner.jps.jps

XY_GRID_RESOLUTION = 2.0  # [m]
YAW_GRID_RESOLUTION = np.deg2rad(15.0)  # [rad]

def compute_path_length(path):
    if not path or len(path) < 2:
        return 0.0
    length = 0.0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        length += np.hypot(dx, dy)
    return length

def main():
    # 맵 생성
    width, height = 100, 100
    obstacle_count = 20
    obstacle_size_range = (3, 8)
    map_gen = MapGenerator(width, height, obstacle_count, obstacle_size_range)
    map_gen.generate_obstacles()
    grid_map = map_gen.get_map()
    start = map_gen.start
    goal = map_gen.goal

    # # print map 
    # obstacle_points = np.argwhere(grid_map == 1)
    # obstacle_xy = [(int(x), int(y)) for y, x in obstacle_points]
    # print(obstacle_xy)

    # # Hybrid A* 실행 (격자 장애물 좌표 변환)
    # obstacle_points = np.argwhere(grid_map == 1)  # (y, x
    # ox = [int(x) for y, x in obstacle_points]
    # oy = [int(y) for y, x in obstacle_points]

    # # Hybrid A*는 [x, y, yaw] 형태의 시작/목표 필요
    # start_hybrid = [float(start[0]), float(start[1]), np.deg2rad(90.0)]
    # goal_hybrid = [float(goal[0]), float(goal[1]), np.deg2rad(-90.0)]
    # path_obj = hybrid_a_star_planning(
    #     start_hybrid, goal_hybrid, ox, oy, 2.0, np.deg2rad(15.0)
    # )


if __name__ == "__main__":
    main()