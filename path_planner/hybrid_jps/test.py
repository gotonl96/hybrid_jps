import numpy as np
import math
from heapq import heappush, heappop
from collections import defaultdict
from typing import List, Tuple, Optional

# 차량 파라미터
VEHICLE_L = 2.8
VEHICLE_WIDTH = 1.8
VEHICLE_LENGTH = 4.0
SAFETY_MARGIN = 0.2
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
    angles = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
    primitives = []
    primitive_length = 3.0
    
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
        rotation_penalty = abs(deg) * 0.03
        
        cost = distance_cost + rotation_penalty
        
        primitives.append({
            'path': path,
            'end': (x, y, theta),
            'cost': cost,
            'steer': deg
        })
    
    return primitives

PRIMITIVES = _generate_primitives()


class DiagnosticKinematicAStar:
    @staticmethod
    def plan(grid: np.ndarray,
             start: Tuple[float, float],
             goal: Tuple[float, float],
             resolution: float = 1.0) -> Optional[List[Tuple[float, float]]]:
        """
        진단 기능이 추가된 Kinematic A*
        """
        h, w = grid.shape
        
        print(f"\n{'='*60}")
        print(f"Path Planning Diagnostics")
        print(f"{'='*60}")
        print(f"Grid size: {w}x{h}")
        print(f"Start: {start}")
        print(f"Goal: {goal}")
        print(f"Distance: {math.hypot(goal[0] - start[0], goal[1] - start[1]):.2f}m")
        
        # 시작/목표 지점 충돌 체크
        start_theta = math.atan2(goal[1] - start[1], goal[0] - start[0])
        start_node = (start[0], start[1], start_theta)
        
        grid_get = grid.__getitem__
        
        def world_to_grid(x, y):
            return int(x / resolution), int(y / resolution)
        
        # 시작점 체크
        start_gx, start_gy = world_to_grid(start[0], start[1])
        if not (0 <= start_gx < w and 0 <= start_gy < h):
            print(f"❌ START OUT OF BOUNDS: grid({start_gx}, {start_gy})")
            return None
        if grid[start_gy, start_gx] == 1:
            print(f"❌ START IN OBSTACLE: grid({start_gx}, {start_gy})")
            return None
        print(f"✓ Start position valid: grid({start_gx}, {start_gy})")
        
        # 목표점 체크
        goal_gx, goal_gy = world_to_grid(goal[0], goal[1])
        if not (0 <= goal_gx < w and 0 <= goal_gy < h):
            print(f"❌ GOAL OUT OF BOUNDS: grid({goal_gx}, {goal_gy})")
            return None
        if grid[goal_gy, goal_gx] == 1:
            print(f"❌ GOAL IN OBSTACLE: grid({goal_gx}, {goal_gy})")
            return None
        print(f"✓ Goal position valid: grid({goal_gx}, {goal_gy})")
        
        collision_cache = {}
        collision_checks = 0
        
        def check_vehicle_collision(x, y, theta):
            nonlocal collision_checks
            collision_checks += 1
            
            cache_key = (round(x, 1), round(y, 1), round(theta, 2))
            if cache_key in collision_cache:
                return collision_cache[cache_key]
            
            corners = get_vehicle_corners(x, y, theta)
            
            for cx, cy in corners:
                ix, iy = world_to_grid(cx, cy)
                if not (0 <= ix < w and 0 <= iy < h):
                    collision_cache[cache_key] = True
                    return True
                if grid_get((iy, ix)) == 1:
                    collision_cache[cache_key] = True
                    return True
            
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
        
        # 시작점 차량 충돌 체크
        if check_vehicle_collision(start[0], start[1], start_theta):
            print(f"❌ START POSITION: Vehicle collision detected!")
            return None
        print(f"✓ Start position: No vehicle collision")
        
        def check_primitive_collision(x, y, theta, primitive):
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            
            for idx, (px, py, ptheta) in enumerate(primitive['path'][1::2]):
                wx = x + px * cos_t - py * sin_t
                wy = y + px * sin_t + py * cos_t
                wtheta = (theta + ptheta) % (2 * math.pi)
                
                if check_vehicle_collision(wx, wy, wtheta):
                    return True
            
            return False
        
        # 시작점에서 프리미티브 적용 가능 여부 체크
        valid_primitives = 0
        for prim in PRIMITIVES:
            if not check_primitive_collision(start[0], start[1], start_theta, prim):
                valid_primitives += 1
        
        print(f"✓ Valid primitives from start: {valid_primitives}/{len(PRIMITIVES)}")
        
        if valid_primitives == 0:
            print(f"❌ CRITICAL: No valid primitives from start! Vehicle is stuck!")
            return None
        
        def heuristic(a, b=(goal[0], goal[1], 0.0)):
            euclidean = math.hypot(a[0] - b[0], a[1] - b[1])
            angle_to_goal = math.atan2(b[1] - a[1], b[0] - a[0])
            angle_diff = abs(math.atan2(math.sin(a[2] - angle_to_goal), 
                                       math.cos(a[2] - angle_to_goal)))
            return euclidean * 1.2 + angle_diff * 0.2
        
        THETA_BINS = 12
        def discretize_pose(x, y, theta):
            ix = int(x / resolution)
            iy = int(y / resolution)
            itheta = int((theta % (2 * math.pi)) / (2 * math.pi) * THETA_BINS)
            return (ix, iy, itheta)
        
        open_set = []
        heappush(open_set, (heuristic(start_node), 0.0, start_node))
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start_node] = 0.0
        visited = {}
        
        iterations = 0
        max_iterations = 30000
        nodes_expanded = 0
        
        print(f"\n{'='*60}")
        print(f"Starting A* search...")
        print(f"{'='*60}")
        
        best_distance = float('inf')
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            if iterations % 1000 == 0:
                print(f"Iteration {iterations}: Open set size = {len(open_set)}, "
                      f"Visited = {len(visited)}, Best dist = {best_distance:.2f}m")
            
            _, cost, current = heappop(open_set)
            x, y, theta = current
            
            dkey = discretize_pose(x, y, theta)
            if dkey in visited and visited[dkey] <= cost:
                continue
            visited[dkey] = cost
            nodes_expanded += 1
            
            # 현재 최단 거리 업데이트
            current_dist_to_goal = math.hypot(x - goal[0], y - goal[1])
            if current_dist_to_goal < best_distance:
                best_distance = current_dist_to_goal
            
            # 목표 도달
            if current_dist_to_goal < resolution * 1.5:
                print(f"\n{'='*60}")
                print(f"✓ PATH FOUND!")
                print(f"{'='*60}")
                print(f"Iterations: {iterations}")
                print(f"Nodes expanded: {nodes_expanded}")
                print(f"Collision checks: {collision_checks}")
                print(f"Cache hits: {len(collision_cache)}")
                return _reconstruct_path(came_from, current)
            
            # 목표 방향 계산
            current_dist = math.hypot(goal[0] - x, goal[1] - y)
            goal_angle = math.atan2(goal[1] - y, goal[0] - x)
            
            candidates = []
            valid_neighbors = 0
            
            for prim in PRIMITIVES:
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                dx, dy, dtheta = prim['end']
                
                nx = x + dx * cos_t - dy * sin_t
                ny = y + dx * sin_t + dy * cos_t
                ntheta = (theta + dtheta) % (2 * math.pi)
                
                if check_primitive_collision(x, y, theta, prim):
                    continue
                
                valid_neighbors += 1
                
                new_dist = math.hypot(goal[0] - nx, goal[1] - ny)
                
                if new_dist > current_dist * 1.3:
                    continue
                
                move_angle = math.atan2(ny - y, nx - x)
                alignment = math.cos(move_angle - goal_angle)
                
                if alignment < -0.5:
                    continue
                
                neighbor = (nx, ny, ntheta)
                tent_g = cost + prim['cost']
                
                if alignment > 0.98:
                    tent_g *= 0.90
                elif alignment > 0.9:
                    tent_g *= 0.95
                elif alignment > 0.7:
                    tent_g *= 0.98
                
                if new_dist < current_dist * 0.95:
                    tent_g *= 0.98
                
                if tent_g < g_score[neighbor]:
                    g_score[neighbor] = tent_g
                    priority = tent_g + heuristic(neighbor)
                    candidates.append((priority, tent_g, neighbor, prim))
            
            if iterations % 1000 == 0 and valid_neighbors == 0:
                print(f"⚠ Warning: No valid neighbors at iteration {iterations}")
            
            candidates.sort(key=lambda x: x[0])
            for priority, tent_g, neighbor, prim in candidates[:10]:
                heappush(open_set, (priority, tent_g, neighbor))
                came_from[neighbor] = (current, prim)
        
        print(f"\n{'='*60}")
        print(f"❌ PATH NOT FOUND")
        print(f"{'='*60}")
        print(f"Reason: Max iterations ({max_iterations}) reached")
        print(f"Iterations: {iterations}")
        print(f"Nodes expanded: {nodes_expanded}")
        print(f"Best distance achieved: {best_distance:.2f}m")
        print(f"Collision checks: {collision_checks}")
        print(f"Open set size at termination: {len(open_set)}")
        print(f"Visited nodes: {len(visited)}")
        
        return None


def _reconstruct_path(came_from, current):
    """경로 재구성"""
    path = []
    while current in came_from:
        prev_tuple = came_from[current]
        if isinstance(prev_tuple, tuple) and len(prev_tuple) == 2:
            curr_node, prim = prev_tuple
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
    
    MAP_SIZE = 1000
    
    map_gen = MapGenerator(MAP_SIZE, MAP_SIZE, 1500, (5, 10))
    map_gen.generate_obstacles()
    grid = map_gen.get_map()
    
    path = DiagnosticKinematicAStar.plan(
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