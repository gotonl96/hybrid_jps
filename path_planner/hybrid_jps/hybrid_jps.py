import numpy as np
import math
from heapq import heappush, heappop
from collections import defaultdict
from typing import List, Tuple, Optional

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

# 차량 파라미터 (새 값 적용)
WB = 3.0  # rear to front wheel
W = 2.0  # width of car
LF = 3.3  # distance from rear to vehicle front end
LB = 1.0  # distance from rear to vehicle back end
MAX_STEER = 0.6  # [rad] maximum steering angle
BUBBLE_DIST = (LF - LB) / 2.0  # distance from rear to center of vehicle.
BUBBLE_R = np.hypot((LF + LB) / 2.0, W / 2.0)  # bubble radius
VRX = [LF, LF, -LB, -LB, LF]
VRY = [W / 2, -W / 2, -W / 2, W / 2, W / 2]
DT = 0.2
SPEED = 5.0

def get_vehicle_corners(x, y, theta):
    corners = []
    for rx, ry in zip(VRX, VRY):
        wx = x + rx * math.cos(theta) - ry * math.sin(theta)
        wy = y + rx * math.sin(theta) + ry * math.cos(theta)
        corners.append((wx, wy))
    return corners


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
            dtheta = (SPEED / WB) * math.tan(rad) * DT
            x += dx
            y += dy
            theta += dtheta
            path.append((x, y, theta))
        
        distance_cost = math.hypot(x, y)
        rotation_penalty = abs(deg) * 0.05
        
        cost = distance_cost + rotation_penalty
        
        # 직진/회전 구분
        is_straight = abs(deg) <= 5  # ±5도 이내는 직진으로 간주
        
        primitives.append({
            'path': path,
            'end': (x, y, theta),
            'cost': cost,
            'steer': deg,
            'direction': _classify_direction(x, y),
            'is_straight': is_straight
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

from scipy.spatial import cKDTree

def rot_mat_2d(yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s], [s, c]])

def rectangle_check(x, y, yaw, ox, oy):
    rot = rot_mat_2d(yaw)
    for iox, ioy in zip(ox, oy):
        tx = iox - x
        ty = ioy - y
        converted_xy = np.stack([tx, ty]).T @ rot
        rx, ry = converted_xy[0], converted_xy[1]
        if not (rx > LF or rx < -LB or ry > W / 2.0 or ry < -W / 2.0):
            return False  # collision
    return True  # no collision

def check_car_collision(x_list, y_list, yaw_list, ox, oy, kd_tree):
    """차량 충돌 검사 - False를 반환하면 충돌"""
    for i_x, i_y, i_yaw in zip(x_list, y_list, yaw_list):
        cx = i_x + BUBBLE_DIST * math.cos(i_yaw)
        cy = i_y + BUBBLE_DIST * math.sin(i_yaw)
        ids = kd_tree.query_ball_point([cx, cy], BUBBLE_R)
        if not ids:
            continue
        if not rectangle_check(i_x, i_y, i_yaw,
                               [ox[i] for i in ids], [oy[i] for i in ids]):
            return False  # collision
    return True  # no collision


def try_analytic_shot(curr_pose, goal_pose, ox, oy, kd_tree, resolution):
    """
    목표 지점이 가까우면 직선/곡선으로 직접 연결 시도
    
    간단한 Analytic Expansion:
    1. 목표까지의 거리가 threshold 이내인지 확인
    2. 직선 경로를 생성
    3. 경로상의 모든 점에서 충돌 검사
    4. 충돌이 없으면 해당 경로 반환
    
    Returns:
        path: 성공시 [(x, y), ...] 리스트, 실패시 None
    """
    x, y, theta = curr_pose
    gx, gy, gtheta = goal_pose
    
    # 거리 체크
    dist = math.hypot(gx - x, gy - y)
    
    # 너무 멀면 시도하지 않음
    if dist > 30.0:
        return None
    
    # 각도 차이 체크
    target_angle = math.atan2(gy - y, gx - x)
    angle_diff = abs(math.atan2(math.sin(theta - target_angle), 
                                math.cos(theta - target_angle)))
    
    # 각도가 너무 크면 직선 연결 불가
    if angle_diff > math.radians(45):
        return None
    
    # 직선 경로 생성 (세밀하게 샘플링)
    num_samples = max(10, int(dist / resolution))
    path_samples = []
    
    for i in range(num_samples + 1):
        t = i / num_samples
        px = x + (gx - x) * t
        py = y + (gy - y) * t
        ptheta = theta + (gtheta - theta) * t
        path_samples.append((px, py, ptheta))
    
    # 전체 경로에 대해 충돌 검사
    for px, py, ptheta in path_samples:
        x_list = [px]
        y_list = [py]
        yaw_list = [ptheta]
        
        if not check_car_collision(x_list, y_list, yaw_list, ox, oy, kd_tree):
            return None  # 충돌 발생
    
    # 충돌이 없으면 경로 반환
    return [(px, py) for px, py, _ in path_samples]


class HybridJPS:
    @staticmethod
    def plan(grid: np.ndarray,
             start_node: Tuple[float, float, float],
             goal_node: Tuple[float, float, float],
             resolution: float = 1.0) -> Optional[List[Tuple[float, float]]]:
        """
        Vehicle Kinematic JPS with Adaptive Step and Analytic Expansion
        
        개선사항:
        1. Adaptive Step: 직진은 멀리 점프, 회전은 기본 단위만 이동
        2. Analytic Expansion: 목표 근처에서 직접 연결 시도
        """
        h, w = grid.shape
        
        grid_get = grid.__getitem__
        
        def world_to_grid(x, y):
            gx, gy = int(x / resolution), int(y / resolution)
            if not (0 <= gx < w and 0 <= gy < h):
                return None, None
            return gx, gy
        
        # 장애물 좌표 추출
        oy, ox = np.where(grid == 1)
        ox = ox * resolution
        oy = oy * resolution
        kd_tree = cKDTree(np.vstack((ox, oy)).T)
        
        # 충돌 검사 캐시
        collision_cache = {}
        
        def is_collision(x, y, theta):
            """차량 크기 기반 충돌 검사. Returns: True면 충돌, False면 안전"""
            cache_key = (round(x, 1), round(y, 1), round(theta, 2))
            if cache_key in collision_cache:
                return collision_cache[cache_key]
            
            gx, gy = world_to_grid(x, y)
            if gx is None or gy is None:
                collision_cache[cache_key] = True
                return True
            
            x_list = [x]
            y_list = [y]
            yaw_list = [theta]
            
            no_collision = check_car_collision(x_list, y_list, yaw_list, ox, oy, kd_tree)
            has_collision = not no_collision
            
            collision_cache[cache_key] = has_collision
            return has_collision
        
        def check_primitive_path_collision(x, y, theta, primitive):
            """프리미티브 경로 전체의 충돌 검사. Returns: True면 충돌, False면 안전"""
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            
            for idx, (px, py, ptheta) in enumerate(primitive['path']):
                if idx == 0:
                    continue
                    
                wx = x + px * cos_t - py * sin_t
                wy = y + px * sin_t + py * cos_t
                wtheta = (theta + ptheta) % (2 * math.pi)
                
                if is_collision(wx, wy, wtheta):
                    return True
            
            return False
        
        def has_forced_neighbor_grid(gx, gy, direction):
            """그리드 기반 Forced Neighbor 검사"""
            if gx is None or gy is None:
                return False
                
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
            Adaptive Step을 적용한 점프 함수
            
            - 직진(Straight): 장애물이 없으면 멀리 점프 (max_jumps 높음)
            - 회전(Turn): 기본 단위만 이동 (점프 없음)
            """
            x, y, theta = curr
            direction = primitive['direction']
            is_straight = primitive['is_straight']
            
            # 방향 변경 감지
            if prev_direction is not None and prev_direction != direction:
                return _apply_primitive(curr, primitive)
            
            # Adaptive Step: 직진은 멀리 점프, 회전은 점프 안 함
            if is_straight:
                max_jumps = 20  # 직진: 멀리 점프
            else:
                max_jumps = 1   # 회전: 점프 안 함 (기본 단위만)
            
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
                if check_primitive_path_collision(x, y, theta, primitive):
                    if jumped:
                        return (x, y, theta, direction, True)
                    return None
                
                # 그리드 위치
                gx, gy = world_to_grid(nx, ny)
                
                # 경계 체크
                if gx is None or gy is None:
                    if jumped:
                        return (x, y, theta, direction, True)
                    return None
                
                # Forced Neighbor 체크
                if has_forced_neighbor_grid(gx, gy, direction):
                    return (nx, ny, ntheta, direction, True)
                
                # 목표 근처
                if math.hypot(nx - goal_node[0], ny - goal_node[1]) < 25.0:
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
            
            if check_primitive_path_collision(x, y, theta, primitive):
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
        
        # 시작점 충돌 체크
        if is_collision(start_node[0], start_node[1], start_node[2]):
            print("경고: 시작점이 장애물과 충돌합니다!")
            return None
        
        # 목표점 충돌 체크
        if is_collision(goal_node[0], goal_node[1], goal_node[2]):
            print("경고: 목표점이 장애물과 충돌합니다!")
            return None
        
        # A* 탐색
        open_set = []
        heappush(open_set, (heuristic(start_node), 0.0, start_node, None))
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start_node] = 0.0
        visited = {}
        
        # 목표 방향 우선순위
        goal_dx = 1 if goal_node[0] > start_node[0] else (-1 if goal_node[0] < start_node[0] else 0)
        goal_dy = 1 if goal_node[1] > start_node[1] else (-1 if goal_node[1] < start_node[1] else 0)
        
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
            if math.hypot(x - goal_node[0], y - goal_node[1]) < resolution * 1.5:
                print(f"경로 탐색 완료! (반복 횟수: {iterations})")
                return _reconstruct_path(came_from, current)
            
            # ===== Analytic Expansion 시도 =====
            # 목표가 가까우면 직접 연결 시도
            dist_to_goal = math.hypot(x - goal_node[0], y - goal_node[1])
            if dist_to_goal < 30.0:
                analytic_path = try_analytic_shot(
                    current, goal_node, ox, oy, kd_tree, resolution
                )
                if analytic_path is not None:
                    print(f"Analytic Expansion 성공! (반복 횟수: {iterations})")
                    # 현재까지의 경로 + analytic 경로
                    base_path = _reconstruct_path(came_from, current)
                    return base_path + analytic_path[1:]  # 중복 제거
            
            # 목표 방향 계산
            goal_angle = math.atan2(goal_node[1] - y, goal_node[0] - x)
            current_dist = math.hypot(goal_node[0] - x, goal_node[1] - y)
            
            # 모든 프리미티브로 Neighbor 탐색
            for prim in PRIMITIVES:
                # 프리미티브 방향과 목표 방향의 정렬도
                prim_dir = prim['direction']
                alignment = prim_dir[0] * goal_dx + prim_dir[1] * goal_dy
                
                # 목표와 많이 어긋나면 스킵
                if alignment < -0.5:
                    continue
                
                # 점프 또는 프리미티브 적용 (Adaptive Step 적용됨)
                result = jump_with_primitive(current, prim, prev_dir)
                
                if result:
                    nx, ny, ntheta, new_dir, needs_expansion = result
                    
                    # 목표에서 멀어지는지 체크
                    new_dist = math.hypot(goal_node[0] - nx, goal_node[1] - ny)
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
        
        print(f"경로를 찾지 못했습니다. (반복 횟수: {iterations})")
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
    
    MAP_SIZE = 300
    
    map_gen = MapGenerator(MAP_SIZE, MAP_SIZE, int(MAP_SIZE / 3), (5, 10))
    map_gen.generate_obstacles()
    map_gen.save_map()
    grid = map_gen.get_map()
    
    import time
    start_time = time.time()
    
    path = HybridJPS.plan(
        grid=grid,
        start_node=(10.5, 10.5, 0.0),
        goal_node=(MAP_SIZE - 10, MAP_SIZE - 10, 0.0),
        resolution=1.0
    )

    elapsed = time.time() - start_time
    print(f"경로 탐색 시간: {elapsed:.2f} 초")

    if path:
        visualizer = Visualizer()
        visualizer.set_grid_map(grid)
        visualizer.set_start_goal((10.5, 10.5), (MAP_SIZE - 10, MAP_SIZE - 10))
        visualizer.set_path(path, "Hybrid JPS (Adaptive + Analytic)")
        visualizer.draw()
    else:
        print("경로를 찾지 못했습니다.")