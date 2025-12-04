# hybrid_jps.py
import numpy as np
import math
from heapq import heappush, heappop
from collections import defaultdict
from typing import List, Tuple, Optional

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from map.map_generate import MapGenerator
from visualization.visualize import Visualizer

# 차량 파라미터
VEHICLE_L = 2.8
VEHICLE_WIDTH = 2.0
VEHICLE_LENGTH = 4.5
SAFETY_MARGIN = 0.3
MAX_STEER_DEG = 30
DT = 0.2
SPEED = 5.0

# 차량 경계 박스
def get_vehicle_corners(x, y, theta):
    half_length = (VEHICLE_LENGTH + SAFETY_MARGIN) / 2
    half_width = (VEHICLE_WIDTH + SAFETY_MARGIN) / 2
    
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    corners = [
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, half_width)
    ]
    
    world_corners = []
    for cx, cy in corners:
        wx = x + cx * cos_t - cy * sin_t
        wy = y + cx * sin_t + cy * cos_t
        world_corners.append((wx, wy))
    
    return world_corners


def _generate_primitives():
    """차량 역학 기반 모션 프리미티브 - Forced Neighbor에서 사용"""
    angles = [-15, -10, -5, 0, 5, 10, 15]
    primitives = []
    primitive_length = 3.0  # 짧은 프리미티브 (세밀한 제어)
    
    for deg in angles:
        rad = math.radians(deg)
        path = [(0.0, 0.0, 0.0)]
        x = y = theta = 0.0
        t = 0.0
        
        while t < primitive_length:
            t += DT
            dx = SPEED * math.cos(theta) * DT
            dy = SPEED * math.sin(theta) * DT
            dtheta = (SPEED / VEHICLE_L) * math.tan(rad) * DT
            x += dx
            y += dy
            theta += dtheta
            path.append((x, y, theta))
        
        distance_cost = math.hypot(x, y)
        rotation_penalty = abs(deg) * 0.15
        
        cost = distance_cost + rotation_penalty
        
        primitives.append({
            'path': path,
            'end': (x, y, theta),
            'cost': cost,
            'steer': deg
        })
    
    return primitives

PRIMITIVES = _generate_primitives()


class HybridJPS:
    @staticmethod
    def plan(grid: np.ndarray,
             start: Tuple[float, float],
             goal: Tuple[float, float],
             resolution: float = 1.0) -> Optional[List[Tuple[float, float]]]:
        """
        Hybrid JPS: JPS의 빠른 점프 + Forced Neighbor에서 차량 프리미티브 적용
        """
        h, w = grid.shape
        
        # 시작 방향을 목표를 향하도록 설정
        start_theta = math.atan2(goal[1] - start[1], goal[0] - start[0])
        start_node = (start[0], start[1], start_theta)
        goal_node = (goal[0], goal[1], 0.0)

        def world_to_grid(x, y):
            return int(x / resolution), int(y / resolution)

        def check_vehicle_collision(x, y, theta):
            """차량 크기 고려 충돌 검사"""
            corners = get_vehicle_corners(x, y, theta)
            
            for cx, cy in corners:
                ix, iy = world_to_grid(cx, cy)
                if not (0 <= ix < w and 0 <= iy < h):
                    return True
                if grid[iy, ix] == 1:
                    return True
            
            # 차량 외곽선 샘플링
            for i in range(len(corners)):
                c1 = corners[i]
                c2 = corners[(i + 1) % len(corners)]
                num_samples = int(math.hypot(c2[0] - c1[0], c2[1] - c1[1]) / (resolution * 0.5)) + 1
                for j in range(num_samples):
                    t = j / max(num_samples - 1, 1)
                    sx = c1[0] + (c2[0] - c1[0]) * t
                    sy = c1[1] + (c2[1] - c1[1]) * t
                    ix, iy = world_to_grid(sx, sy)
                    if not (0 <= ix < w and 0 <= iy < h):
                        return True
                    if grid[iy, ix] == 1:
                        return True
            return False

        def heuristic(a, b=goal_node):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        THETA_BINS = 16
        def discretize_pose(x, y, theta):
            ix = int(x / resolution)
            iy = int(y / resolution)
            itheta = int((theta % (2 * math.pi)) / (2 * math.pi) * THETA_BINS)
            return (ix, iy, itheta)

        def has_forced_neighbor(curr_grid, direction):
            """JPS Forced Neighbor 검사"""
            x, y = curr_grid
            dx, dy = direction
            if dx != 0 and dy != 0:
                left_block = False
                right_block = False
                if 0 <= (y + dy) < h and 0 <= x < w:
                    left_block = (grid[y + dy, x] == 1)
                else:
                    left_block = True
                if 0 <= y < h and 0 <= (x + dx) < w:
                    right_block = (grid[y, x + dx] == 1)
                else:
                    right_block = True
                return left_block or right_block
            else:
                # 직선 이동시 양옆 체크
                if dx != 0:  # 수평 이동
                    for side in [-1, 1]:
                        ny = y + side
                        if 0 <= ny < h and 0 <= x < w:
                            if grid[ny, x] == 1:  # 옆에 장애물
                                nx = x + dx
                                if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 0:
                                    return True
                else:  # 수직 이동
                    for side in [-1, 1]:
                        nx = x + side
                        if 0 <= nx < w and 0 <= y < h:
                            if grid[y, nx] == 1:
                                ny = y + dy
                                if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 0:
                                    return True
            return False

        def jump(curr, dir_vec):
            """JPS 점프 (빠른 탐색) - 더 작은 스텝으로"""
            cx, cy, ctheta = curr
            dx, dy = dir_vec
            step = resolution * 1.0  # 2.0 -> 1.0 (더 세밀하게)
            max_steps = 100

            for i in range(max_steps):
                nx, ny = cx + dx * step, cy + dy * step
                
                # 간단한 점 충돌만 체크
                ix, iy = world_to_grid(nx, ny)
                if not (0 <= ix < w and 0 <= iy < h) or grid[iy, ix] == 1:
                    # 막혔으면 이전 위치를 프리미티브 지점으로
                    if i > 0:
                        return (cx, cy, ctheta, True)
                    return None
                
                # Forced Neighbor 발견!
                if has_forced_neighbor((ix, iy), (dx, dy)):
                    return (nx, ny, ctheta, True)
                
                # 목표 근처 또는 일정 거리마다 프리미티브 체크
                dist_to_goal = math.hypot(nx - goal[0], ny - goal[1])
                if dist_to_goal < 10.0 or i % 5 == 0:  # 5스텝마다 프리미티브 옵션
                    return (nx, ny, ctheta, True)
                
                cx, cy = nx, ny
            
            return None

        open_set = []
        heappush(open_set, (heuristic(start_node), 0.0, start_node, False))
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start_node] = 0.0
        visited = {}  # dict로 변경: best g_score 저장

        directions = [(1,0), (0,1), (-1,0), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]

        while open_set:
            _, cost, current, use_primitives = heappop(open_set)
            x, y, theta = current

            dkey = discretize_pose(x, y, theta)
            # 더 나은 경로가 있으면 스킵
            if dkey in visited and visited[dkey] <= cost:
                continue
            visited[dkey] = cost

            # 목표 도달 (범위 확대)
            if math.hypot(x - goal[0], y - goal[1]) < 5.0:
                return _reconstruct_path(came_from, current)

            # Forced Neighbor 지점이거나 목표 근처 -> 프리미티브 사용
            if use_primitives:
                for prim in PRIMITIVES:
                    dx, dy, dtheta = prim['end']
                    cos_t = math.cos(theta)
                    sin_t = math.sin(theta)
                    nx = x + dx * cos_t - dy * sin_t
                    ny = y + dx * sin_t + dy * cos_t
                    ntheta = (theta + dtheta) % (2 * math.pi)

                    # 프리미티브 경로 충돌 검사 (샘플링으로 간소화)
                    collision = False
                    sample_rate = 3  # 3개마다 하나씩만 체크
                    for idx, (px, py, ptheta) in enumerate(prim['path'][1:]):
                        if idx % sample_rate != 0:
                            continue
                        wx = x + px * cos_t - py * sin_t
                        wy = y + px * sin_t + py * cos_t
                        wtheta = (theta + ptheta) % (2 * math.pi)
                        if check_vehicle_collision(wx, wy, wtheta):
                            collision = True
                            break
                    
                    if collision:
                        continue

                    neighbor = (nx, ny, ntheta)
                    tent_g = cost + prim['cost']
                    
                    if tent_g < g_score[neighbor]:
                        g_score[neighbor] = tent_g
                        priority = tent_g + heuristic(neighbor)
                        # 프리미티브 후 다시 점프 모드로 전환 가능
                        heappush(open_set, (priority, tent_g, neighbor, False))
                        came_from[neighbor] = (current, prim)
            
            # 일반 구간 -> JPS 점프
            else:
                for dx, dy in directions:
                    jumped = jump(current, (dx, dy))
                    if jumped:
                        jx, jy, jtheta, needs_primitives = jumped
                        neighbor = (jx, jy, jtheta)
                        dist = math.hypot(x - jx, y - jy)
                        tent_g = cost + dist
                        
                        if tent_g < g_score[neighbor]:
                            g_score[neighbor] = tent_g
                            priority = tent_g + heuristic(neighbor)
                            heappush(open_set, (priority, tent_g, neighbor, needs_primitives))
                            came_from[neighbor] = current

        return None


def _reconstruct_path(came_from, current):
    """경로 재구성"""
    path = []
    while current in came_from:
        prev = came_from[current]
        if isinstance(prev, tuple) and len(prev) == 2:
            curr_node, prim = prev
            for px, py, ptheta in reversed(prim['path']):
                wx = curr_node[0] + px * math.cos(curr_node[2]) - py * math.sin(curr_node[2])
                wy = curr_node[1] + px * math.sin(curr_node[2]) + py * math.cos(curr_node[2])
                path.append((wx, wy))
            current = curr_node
        else:
            path.append((current[0], current[1]))
            current = prev
    path.append((current[0], current[1]))
    return path[::-1]


if __name__ == "__main__":
    map_gen = MapGenerator(100, 100, 20, (5, 15))
    map_gen.generate_obstacles()
    grid = map_gen.get_map()
    
    path = HybridJPS.plan(
        grid=grid,
        start=(10.5, 10.5),
        goal=(89.5, 89.5),
        resolution=1.0
    )

    visualizer = Visualizer()
    visualizer.set_grid_map(grid)
    visualizer.set_start_goal((10.5, 10.5), (89.5, 89.5))
    visualizer.set_path(path, "Hybrid JPS (Jump + Primitives)")
    visualizer.draw()