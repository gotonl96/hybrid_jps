
import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 이제 원본 코드 수정 없이 import 가능
from path_planner.hybrid_jps.hybrid_jps import VehicleKinematicJPS
import path_planner.jps.jps as jps
from path_planner.hybrid_astar.hybrid_a_star import (
    hybrid_a_star_planning,
    XY_GRID_RESOLUTION,
    YAW_GRID_RESOLUTION
)

from map.map_generate import MapGenerator
from visualization.visualize import Visualizer

import time

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
    MAP_SIZE = 200
    obstacle_count = 20
    obstacle_size_range = (3, 8)
    map_gen = MapGenerator(MAP_SIZE, MAP_SIZE, obstacle_count, obstacle_size_range)
    map_gen.generate_obstacles()
    grid_map = map_gen.get_map()
    start = map_gen.start
    goal = map_gen.goal


    # Hybrid A*는 [x, y, yaw] 형태의 시작/목표 필요
    start_hybrid = [float(start[0]), float(start[1]), np.deg2rad(45.0)]
    goal_hybrid = [float(goal[0]), float(goal[1]), np.deg2rad(45.0)]

    ox, oy = map_gen.get_obstacle_points()
    path_obj = hybrid_a_star_planning(
        start_hybrid, goal_hybrid, ox, oy, 2.0, np.deg2rad(15.0)
    )

    hybrid_astar_start_time = time.time()
    hybrid_astar_path = list(zip(path_obj.x_list, path_obj.y_list))
    hybrid_astar_end_time = round(time.time() - hybrid_astar_start_time, 6)
    print(f"Hybrid A* path length: {compute_path_length(hybrid_astar_path):.2f}")
    print(f"Hybrid A* planning time: {hybrid_astar_end_time} seconds")


    # hybrid jps path planner
    hybrid_jps_start_time = time.time()
    hybrid_jps_path = VehicleKinematicJPS.plan(
        grid=grid_map,
        start=(10.5, 10.5),
        goal=(MAP_SIZE - 10, MAP_SIZE - 10),
        resolution=1.0
    )
    hybrid_jps_end_time = round(time.time() - hybrid_jps_start_time, 6)
    print(f"Hybrid JPS path length: {compute_path_length(hybrid_jps_path):.2f}")
    print(f"Hybrid JPS planning time: {hybrid_jps_end_time} seconds")

    # jps path planner
    jps_start_time = time.time()
    jps_result = jps.method(grid_map, start, goal, 2)
    jps_path = [(x, y) for y, x in jps_result[0]]
    jps_end_time = round(time.time() - jps_start_time, 6)
    print(f"JPS path length: {compute_path_length(jps_path):.2f}")
    print(f"JPS planning time: {jps_end_time} seconds")

    # 시각화
    # 경로 생성 시간 빠른 순 출력
    print("\nPath Planning Times:")
    times = {"Hybrid A*": hybrid_astar_end_time,
             "Hybrid JPS": hybrid_jps_end_time,
             "JPS": jps_end_time}

    visualizer = Visualizer()
    visualizer.set_grid_map(grid_map)
    visualizer.set_start_goal(start, goal)
    visualizer.set_path(hybrid_astar_path, "Hybrid A*")
    visualizer.set_path(hybrid_jps_path, "Hybrid JPS")
    visualizer.set_path(jps_path, "JPS")
    visualizer.draw()


if __name__ == "__main__":
    main()