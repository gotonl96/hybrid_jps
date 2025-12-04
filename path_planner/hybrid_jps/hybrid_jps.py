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

# 차량 파라미터 (필요시 외부에서 주입 가능)
VEHICLE_L = 2.8          # 휠베이스 (m)
MAX_STEER_DEG = 30       # 최대 조향각
PRIMITIVE_LENGTH = 8.0   # 한 프리미티브당 이동 거리 (격자 8칸 정도)
DT = 0.2
SPEED = 5.0              # m/s (시뮬레이션 속도)

# 부드러운 프리미티브 생성 (한 번만 생성)
def _generate_primitives():
    angles = [-20, -12, -6, 0, 6, 12, 20]  # degree (너무 크면 안 됨!)
    primitives = []
    for deg in angles:
        rad = math.radians(deg)
        path = [(0.0, 0.0, 0.0)]
        x = y = theta = 0.0
        t = 0.0
        while t < PRIMITIVE_LENGTH:
            t += DT
            dx = SPEED * math.cos(theta) * DT
            dy = SPEED * math.sin(theta) * DT
            dtheta = (SPEED / VEHICLE_L) * math.tan(rad) * DT
            x += dx
            y += dy
            theta += dtheta
            path.append((x, y, theta))
        cost = math.hypot(x, y) + abs(deg) * 0.05  # 회전 페널티
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
        Hybrid JPS 경로 탐색 (당신이 원하던 그 함수!)

        Parameters
        ----------
        grid : np.ndarray
            2D 장애물 맵 (0=통행가능, 1=장애물), shape=(height, width)
        start : (float, float)
            시작 위치 (x, y) - world 좌표 (m)
        goal : (float, float)
            목표 위치 (x, y) - world 좌표 (m)
        resolution : float
            격자 한 칸의 실제 길이 (기본 1m)

        Returns
        -------
        path : List[(x,y)] or None
            부드러운 경로 (실제 차량이 갈 수 있는 연속 경로)
            실패시 None 반환
        """
        h, w = grid.shape
        start_node = (start[0], start[1], 0.0)   # (x, y, theta)
        goal_node = (goal[0], goal[1], 0.0)

        def world_to_grid(x, y):
            return int(x / resolution), int(y / resolution)

        def is_collision(x, y):
            ix, iy = world_to_grid(x, y)
            if not (0 <= ix < w and 0 <= iy < h):
                return True
            return grid[iy, ix] == 1

        def line_has_collision(x0, y0, x1, y1):
            """샘플링으로 선분 충돌 검사: 두 점 사이를 resolution/2 간격으로 샘플링합니다."""
            dx = x1 - x0
            dy = y1 - y0
            dist = math.hypot(dx, dy)
            if dist == 0:
                return is_collision(x0, y0)
            step_len = max(resolution * 0.5, 0.001)
            steps = int(math.ceil(dist / step_len))
            for i in range(1, steps + 1):
                t = i / steps
                sx = x0 + dx * t
                sy = y0 + dy * t
                if is_collision(sx, sy):
                    return True
            return False

        def heuristic(a, b=goal_node):
            dist = math.hypot(a[0] - b[0], a[1] - b[1])
            return dist

        # discretize pose to avoid revisiting very similar continuous states
        THETA_BINS = 16
        def discretize_pose(x, y, theta):
            ix = int(x / resolution)
            iy = int(y / resolution)
            itheta = int((theta % (2 * math.pi)) / (2 * math.pi) * THETA_BINS)
            return (ix, iy, itheta)

        # JPS용 강제 이웃 검사 (간단 버전)
        def has_forced_neighbor(curr_grid, direction):
            x, y = curr_grid
            dx, dy = direction
            # Ensure indices are within grid before indexing. Treat out-of-bounds
            # adjacent cells as blocking (so jumps stop at map borders).
            if dx != 0 and dy != 0:  # 대각선
                left_block = False
                right_block = False
                # check grid[y + dy, x]
                if 0 <= (y + dy) < h and 0 <= x < w:
                    left_block = (grid[y + dy, x] == 1)
                else:
                    left_block = True
                # check grid[y, x + dx]
                if 0 <= y < h and 0 <= (x + dx) < w:
                    right_block = (grid[y, x + dx] == 1)
                else:
                    right_block = True
                return left_block or right_block
            return False

        # 직진 점프
        def jump(curr, dir_vec):
            cx, cy, ctheta = curr
            dx, dy = dir_vec
            step = resolution * 2.0
            max_steps = 50

            for _ in range(max_steps):
                nx, ny = cx + dx * step, cy + dy * step
                # 샘플링 충돌 검사: 선분 상의 장애물을 놓치지 않도록 함
                if line_has_collision(cx, cy, nx, ny):
                    return None
                gx, gy = world_to_grid(nx, ny)
                if has_forced_neighbor((gx, gy), (dx, dy)):
                    return (nx, ny, ctheta)
                if math.hypot(nx - goal[0], ny - goal[1]) < 4.0:
                    return (nx, ny, ctheta)
                # advance the current position for the next step
                cx, cy = nx, ny
            return None

        open_set = []
        heappush(open_set, (heuristic(start_node), 0.0, start_node))
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start_node] = 0.0
        visited = {}  # maps discretized pose -> best g seen
        visited[discretize_pose(*start_node)] = 0.0

        directions = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]

        while open_set:
            _, cost, current = heappop(open_set)
            x, y, theta = current

            if math.hypot(x - goal[0], y - goal[1]) < 3.0:
                return _reconstruct_path(came_from, current)

            # 1. JPS 직진 점프
            for dx, dy in directions:
                if dx == 0 and dy == 0: continue
                jumped = jump(current, (dx, dy))
                if jumped:
                    neighbor = (jumped[0], jumped[1], theta)
                    dist = math.hypot(x - jumped[0], y - jumped[1])
                    tent_g = cost + dist
                    # discretized pruning: skip if we already have a better state
                    dkey = discretize_pose(neighbor[0], neighbor[1], neighbor[2])
                    prev_best = visited.get(dkey, float('inf'))
                    if tent_g >= prev_best:
                        continue
                    visited[dkey] = tent_g
                    if tent_g < g_score[neighbor]:
                        g_score[neighbor] = tent_g
                        priority = tent_g + heuristic(neighbor)
                        heappush(open_set, (priority, tent_g, neighbor))
                        came_from[neighbor] = current

            # 2. 프리미티브 기반 부드러운 이동 (Hybrid 핵심!)
            for prim in PRIMITIVES:
                dx, dy, dtheta = prim['end']
                nx = x + dx * math.cos(theta) - dy * math.sin(theta)
                ny = y + dx * math.sin(theta) + dy * math.cos(theta)
                ntheta = (theta + dtheta) % (2 * math.pi)

                # 충돌 체크 (프리미티브 경로 전체)
                if any(is_collision(
                    x + px * math.cos(theta) - py * math.sin(theta),
                    y + px * math.sin(theta) + py * math.cos(theta)
                ) for px, py, _ in prim['path'][1:]):
                    continue

                # slightly increase turn penalty to discourage tight circling
                turn_penalty = abs(prim['steer']) * 0.08
                neighbor = (nx, ny, ntheta)
                tent_g = cost + prim['cost'] + turn_penalty
                # discretized pruning for primitives as well
                dkey = discretize_pose(neighbor[0], neighbor[1], neighbor[2])
                prev_best = visited.get(dkey, float('inf'))
                if tent_g >= prev_best:
                    continue
                visited[dkey] = tent_g
                if tent_g < g_score[neighbor]:
                    g_score[neighbor] = tent_g
                    priority = tent_g + heuristic(neighbor)
                    heappush(open_set, (priority, tent_g, neighbor))
                    came_from[neighbor] = (current, prim)

        return None  # 실패


def _reconstruct_path(came_from, current):
    """부드러운 경로 재구성 (프리미티브 역추적 포함)"""
    path = []
    while current in came_from:
        prev = came_from[current]
        if isinstance(prev, tuple) and len(prev) == 2:  # primitive 사용됨
            curr_node, prim = prev
            # 프리미티브 경로를 현재 자세에 맞춰 변환
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


def smooth_path_with_beziers(path: List[Tuple[float, float]],
                             radius: float = 3.0,
                             angle_threshold_deg: float = 15.0,
                             samples: int = 12) -> List[Tuple[float, float]]:
    """Replace sharp corners in `path` with smooth cubic Bézier arcs.

    - Detects corners at triples of consecutive points.
    - If turning angle is sharper than `angle_threshold_deg`, compute tangent
      points and insert a sampled cubic Bézier between them.
    - If the requested `radius` does not fit on the neighboring segments,
      it is reduced to the maximum feasible value.
    """
    if not path or len(path) < 3:
        return path

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    new_path: List[Tuple[float, float]] = [path[0]]
    ang_thr = math.radians(angle_threshold_deg)

    for i in range(1, len(path) - 1):
        p_prev = path[i - 1]
        p_cur = path[i]
        p_next = path[i + 1]

        v1x, v1y = p_cur[0] - p_prev[0], p_cur[1] - p_prev[1]
        v2x, v2y = p_next[0] - p_cur[0], p_next[1] - p_cur[1]
        l1 = math.hypot(v1x, v1y)
        l2 = math.hypot(v2x, v2y)
        if l1 < 1e-6 or l2 < 1e-6:
            new_path.append(p_cur)
            continue

        u1x, u1y = v1x / l1, v1y / l1
        u2x, u2y = v2x / l2, v2y / l2
        cosang = max(-1.0, min(1.0, u1x * u2x + u1y * u2y))
        angle = math.acos(cosang)
        if angle < ang_thr:
            new_path.append(p_cur)
            continue

        half_theta = angle / 2.0
        tan_half = math.tan(half_theta) if abs(math.tan(half_theta)) > 1e-6 else 1e6
        t = radius / tan_half

        max_t = min(l1, l2) * 0.9
        if t > max_t:
            t = max_t
            radius_eff = t * tan_half
        else:
            radius_eff = radius

        p_t1 = (p_cur[0] - u1x * t, p_cur[1] - u1y * t)
        p_t2 = (p_cur[0] + u2x * t, p_cur[1] + u2y * t)

        control_scale = t * 0.6
        p0 = p_t1
        p3 = p_t2
        p1 = (p0[0] + u1x * control_scale, p0[1] + u1y * control_scale)
        p2 = (p3[0] - u2x * control_scale, p3[1] - u2y * control_scale)

        if dist(new_path[-1], p0) > 1e-6:
            new_path.append(p0)

        for s_i in range(1, samples + 1):
            t_s = s_i / float(samples)
            b0 = (1 - t_s) ** 3
            b1 = 3 * (1 - t_s) ** 2 * t_s
            b2 = 3 * (1 - t_s) * (t_s ** 2)
            b3 = t_s ** 3
            bx = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0]
            by = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]
            new_path.append((bx, by))

    new_path.append(path[-1])
    # remove near-duplicates
    cleaned: List[Tuple[float, float]] = [new_path[0]]
    for pt in new_path[1:]:
        if dist(cleaned[-1], pt) > 1e-4:
            cleaned.append(pt)
    return cleaned

if __name__ == "__main__":
    map_gen = MapGenerator(100, 100, 20, (5, 15))
    map_gen.generate_obstacles()
    grid = map_gen.get_map()
    
    # 한 줄로 경로 탐색!
    path = HybridJPS.plan(
        grid=grid,
        start=(10.5, 10.5),
            # goal must be inside the generated map (size 100x100) — was out of bounds
            goal=(89.5, 89.5),
        resolution=1.0
    )

    # post-process: replace sharp corners with smooth Bézier arcs
    if path:
        path = smooth_path_with_beziers(path, radius=3.0, angle_threshold_deg=20.0, samples=16)

    visualizer = Visualizer()
    visualizer.set_grid_map(grid)
    visualizer.set_start_goal((10.5, 10.5), (89.5, 89.5))
    visualizer.set_path(path, "Hybrid JPS Path ")
    visualizer.draw()
    