import numpy as np
import math
from heapq import heappush, heappop
from collections import defaultdict
from typing import List, Tuple, Optional

# 차량 파라미터
VEHICLE_L = 2.8
VEHICLE_WIDTH = 2.0
VEHICLE_LENGTH = 4.5
SAFETY_MARGIN = 0.3
DT = 0.2
SPEED = 5.0

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
    """차량 역학 기반 모션 프리미티브"""
    angles = [-15, -10, -5, 0, 5, 10, 15]
    primitives = []
    primitive_length = 2.5
    
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
        rotation_penalty = abs(deg) * 0.05
        
        cost = distance_cost + rotation_penalty
        
        primitives.append({
            'path': path,
            'end': (x, y, theta),
            'cost': cost,
            'steer': deg,
            'direction': _classify_direction(x, y)  # 방향 분류
        })
    
    return primitives


def _classify_direction(dx, dy):
    """프리미티브의 방향을 분류 (직진/대각선 판별용)"""
    angle = math.atan2(dy, dx)
    angle_deg = math.degrees(angle)
    
    # 8방향으로 분류
    if -22.5 <= angle_deg < 22.5:
        return (1, 0)  # 동
    elif 22.5 <= angle_deg < 67.5:
        return (1, 1)  # 북동
    elif 67.5 <= angle_deg < 112.5:
        return (0, 1)  # 북
    elif 112.5 <= angle_deg < 157.5:
        return (-1, 1)  # 북서
    elif angle_deg >= 157.5 or angle_deg < -157.5:
        return (-1, 0)  # 서
    elif -157.5 <= angle_deg < -112.5:
        return (-1, -1)  # 남서
    elif -112.5 <= angle_deg < -67.5:
        return (0, -1)  # 남
    else:  # -67.5 <= angle_deg < -22.5
        return (1, -1)  # 남동


PRIMITIVES = _generate_primitives()


class VehicleKinematicJPS:
    @staticmethod
    def plan(grid: np.ndarray,
             start: Tuple[float, float],
             goal: Tuple[float, float],
             resolution: float = 1.0) -> Optional[List[Tuple[float, float]]]:
        """
        Vehicle Kinematic JPS
        - JPS의 모든 동작을 차량 프리미티브로 수행
        - Neighbor 탐색: 프리미티브 기반
        - Jump: 같은 방향 프리미티브 반복
        - 차량 크기 기반 충돌 검사
        """
        h, w = grid.shape
        
        start_theta = math.atan2(goal[1] - start[1], goal[0] - start[0])
        start_node = (start[0], start[1], start_theta)
        goal_node = (goal[0], goal[1], 0.0)
        
        grid_get = grid.__getitem__
        
        def world_to_grid(x, y):
            return int(x / resolution), int(y / resolution)
        
        # 충돌 검사 캐시
        collision_cache = {}
        
        def check_vehicle_collision(x, y, theta):
            """차량 크기 기반 충돌 검사"""
            cache_key = (round(x, 1), round(y, 1), round(theta, 2))
            if cache_key in collision_cache:
                return collision_cache[cache_key]
            
            corners = get_vehicle_corners(x, y, theta)
            
            # 코너 체크
            for cx, cy in corners:
                ix, iy = world_to_grid(cx, cy)
                if not (0 <= ix < w and 0 <= iy < h):
                    collision_cache[cache_key] = True
                    return True
                if grid_get((iy, ix)) == 1:
                    collision_cache[cache_key] = True
                    return True
            
            # 외곽선 샘플링
            for i in range(len(corners)):
                c1 = corners[i]
                c2 = corners[(i + 1) % len(corners)]
                
                edge_length = math.hypot(c2[0] - c1[0], c2[1] - c1[1])
                num_samples = max(2, int(edge_length / resolution * 0.8))
                
                for j in range(1, num_samples):
                    t = j / num_samples
                    sx = c1[0] + (c2[0] - c1[0]) * t
                    sy = c1[1] + (c2[1] - c1[1]) * t
                    ix, iy = world_to_grid(sx, sy)
                    if not (0 <= ix < w and 0 <= iy < h):
                        collision_cache[cache_key] = True
                        return True
                    if grid_get((iy, ix)) == 1:
                        collision_cache[cache_key] = True
                        return True
            
            collision_cache[cache_key] = False
            return False
        
        def check_primitive_collision(x, y, theta, primitive):
            """프리미티브 경로 전체의 충돌 검사"""
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            
            # 샘플링하여 체크
            for idx, (px, py, ptheta) in enumerate(primitive['path'][1::2]):
                wx = x + px * cos_t - py * sin_t
                wy = y + px * sin_t + py * cos_t
                wtheta = (theta + ptheta) % (2 * math.pi)
                
                if check_vehicle_collision(wx, wy, wtheta):
                    return True
            
            return False
        
        def has_forced_neighbor_grid(gx, gy, direction):
            """그리드 기반 Forced Neighbor 검사"""
            dx, dy = direction
            
            if dx != 0 and dy != 0:
                # 대각선
                left_block = (not (0 <= gy + dy < h and 0 <= gx < w)) or grid_get((gy + dy, gx)) == 1
                right_block = (not (0 <= gy < h and 0 <= gx + dx < w)) or grid_get((gy, gx + dx)) == 1
                return left_block or right_block
            else:
                # 직선
                if dx != 0:
                    for side in [-1, 1]:
                        ny = gy + side
                        if 0 <= ny < h and 0 <= gx < w and grid_get((ny, gx)) == 1:
                            nx = gx + dx
                            if 0 <= nx < w and 0 <= ny < h and grid_get((ny, nx)) == 0:
                                return True
                else:
                    for side in [-1, 1]:
                        nx = gx + side
                        if 0 <= nx < w and 0 <= gy < h and grid_get((gy, nx)) == 1:
                            ny = gy + dy
                            if 0 <= nx < w and 0 <= ny < h and grid_get((ny, nx)) == 0:
                                return True
            return False
        
        def jump_with_primitive(curr, primitive, prev_direction):
            """
            프리미티브를 반복 적용하여 점프
            - 같은 방향으로 계속 진행
            - Forced Neighbor 발견 시 중단
            """
            x, y, theta = curr
            direction = primitive['direction']
            
            # 방향 변경 감지
            if prev_direction is not None and prev_direction != direction:
                # 방향이 바뀌면 프리미티브 적용 (부드러운 전환)
                return _apply_primitive(curr, primitive)
            
            # 같은 방향으로 점프
            max_jumps = 15  # 최대 점프 횟수
            jumped = False
            
            for i in range(max_jumps):
                # 프리미티브 적용
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                dx, dy, dtheta = primitive['end']
                
                nx = x + dx * cos_t - dy * sin_t
                ny = y + dx * sin_t + dy * cos_t
                ntheta = (theta + dtheta) % (2 * math.pi)
                
                # 충돌 체크
                if check_primitive_collision(x, y, theta, primitive):
                    if jumped:
                        return (x, y, theta, direction, True)
                    return None
                
                # 그리드 위치
                gx, gy = world_to_grid(nx, ny)
                
                # Forced Neighbor 체크
                if has_forced_neighbor_grid(gx, gy, direction):
                    return (nx, ny, ntheta, direction, True)
                
                # 목표 근처
                if math.hypot(nx - goal[0], ny - goal[1]) < 25.0:
                    return (nx, ny, ntheta, direction, True)
                
                # 다음 점프 준비
                x, y, theta = nx, ny, ntheta
                jumped = True
            
            # 최대 점프 도달
            if jumped:
                return (x, y, theta, direction, True)
            return None
        
        def _apply_primitive(curr, primitive):
            """프리미티브를 한 번만 적용 (방향 전환용)"""
            x, y, theta = curr
            
            if check_primitive_collision(x, y, theta, primitive):
                return None
            
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            dx, dy, dtheta = primitive['end']
            
            nx = x + dx * cos_t - dy * sin_t
            ny = y + dx * sin_t + dy * cos_t
            ntheta = (theta + dtheta) % (2 * math.pi)
            
            return (nx, ny, ntheta, primitive['direction'], False)
        
        def heuristic(a, b=goal_node):
            euclidean = math.hypot(a[0] - b[0], a[1] - b[1])
            angle_to_goal = math.atan2(b[1] - a[1], b[0] - a[0])
            angle_diff = abs(math.atan2(math.sin(a[2] - angle_to_goal), 
                                       math.cos(a[2] - angle_to_goal)))
            return euclidean + angle_diff * 0.3
        
        THETA_BINS = 16
        def discretize_pose(x, y, theta):
            ix = int(x / resolution)
            iy = int(y / resolution)
            itheta = int((theta % (2 * math.pi)) / (2 * math.pi) * THETA_BINS)
            return (ix, iy, itheta)
        
        # A* 탐색
        open_set = []
        heappush(open_set, (heuristic(start_node), 0.0, start_node, None))
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start_node] = 0.0
        visited = {}
        
        # 목표 방향 우선순위
        goal_dx = 1 if goal[0] > start[0] else (-1 if goal[0] < start[0] else 0)
        goal_dy = 1 if goal[1] > start[1] else (-1 if goal[1] < start[1] else 0)
        
        iterations = 0
        max_iterations = 15000
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            _, cost, current, prev_dir = heappop(open_set)
            x, y, theta = current
            
            dkey = discretize_pose(x, y, theta)
            if dkey in visited and visited[dkey] <= cost:
                continue
            visited[dkey] = cost
            
            # 목표 도달
            if math.hypot(x - goal[0], y - goal[1]) < resolution * 1.5:
                return _reconstruct_path(came_from, current)
            
            # 목표 방향 계산
            goal_angle = math.atan2(goal[1] - y, goal[0] - x)
            current_dist = math.hypot(goal[0] - x, goal[1] - y)
            
            # 모든 프리미티브로 Neighbor 탐색
            for prim in PRIMITIVES:
                # 프리미티브 방향과 목표 방향의 정렬도
                prim_dir = prim['direction']
                alignment = prim_dir[0] * goal_dx + prim_dir[1] * goal_dy
                
                # 목표와 많이 어긋나면 스킵
                if alignment < -0.5:
                    continue
                
                # 점프 또는 프리미티브 적용
                result = jump_with_primitive(current, prim, prev_dir)
                
                if result:
                    nx, ny, ntheta, new_dir, needs_expansion = result
                    
                    # 목표에서 멀어지는지 체크
                    new_dist = math.hypot(goal[0] - nx, goal[1] - ny)
                    if new_dist > current_dist * 1.1:
                        continue
                    
                    neighbor = (nx, ny, ntheta)
                    tent_g = cost + prim['cost']
                    
                    # 목표 방향 정렬 보너스
                    result_angle = math.atan2(ny - y, nx - x)
                    result_alignment = math.cos(result_angle - goal_angle)
                    if result_alignment > 0.95:
                        tent_g *= 0.95
                    
                    if tent_g < g_score[neighbor]:
                        g_score[neighbor] = tent_g
                        priority = tent_g + heuristic(neighbor)
                        heappush(open_set, (priority, tent_g, neighbor, new_dir))
                        came_from[neighbor] = (current, prim)
        
        return None


def _reconstruct_path(came_from, current):
    """경로 재구성"""
    path = []
    while current in came_from:
        prev_tuple = came_from[current]
        if isinstance(prev_tuple, tuple) and len(prev_tuple) == 2:
            curr_node, prim = prev_tuple
            # 프리미티브 경로 추가
            for px, py, ptheta in reversed(prim['path']):
                wx = curr_node[0] + px * math.cos(curr_node[2]) - py * math.sin(curr_node[2])
                wy = curr_node[1] + px * math.sin(curr_node[2]) + py * math.cos(curr_node[2])
                path.append((wx, wy))
            current = curr_node
        else:
            path.append((current[0], current[1]))
            current = prev_tuple
    
    path.append((current[0], current[1]))
    return path[::-1]

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from map.map_generate import MapGenerator
from visualization.visualize import Visualizer

if __name__ == "__main__":
    
    MAP_SIZE = 100
    
    map_gen = MapGenerator(MAP_SIZE, MAP_SIZE, 20, (5, 15))
    map_gen.generate_obstacles()
    grid = map_gen.get_map()
    
    path = VehicleKinematicJPS.plan(
        grid=grid,
        start=(10.5, 10.5),
        goal=(MAP_SIZE - 10, MAP_SIZE - 10),
        resolution=1.0
    )

    visualizer = Visualizer()
    visualizer.set_grid_map(grid)
    visualizer.set_start_goal((10.5, 10.5), (MAP_SIZE - 10, MAP_SIZE - 10))
    visualizer.set_path(path, "Hybrid JPS (Jump + Primitives)")
    visualizer.draw()