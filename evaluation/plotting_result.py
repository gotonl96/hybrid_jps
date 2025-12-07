import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 이제 원본 코드 수정 없이 import 가능
from path_planner.hybrid_jps.hybrid_jps import HybridJPS
import path_planner.jps.jps as jps
from path_planner.hybrid_astar.hybrid_a_star import (
    hybrid_a_star_planning,
    XY_GRID_RESOLUTION,
    YAW_GRID_RESOLUTION
)

from map.map_generate import MapGenerator
from visualization.visualize import Visualizer


def compute_path_length(path):
    if not path or len(path) < 2:
        return 0.0
    length = 0.0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        length += np.hypot(dx, dy)
    return length


def run_single_test(map_size, obstacle_count, obstacle_size_range):
    """단일 테스트 실행 - 모든 플래너가 성공하면 True 반환"""
    
    map_gen = MapGenerator(map_size, map_size, obstacle_count, obstacle_size_range)
    map_gen.generate_obstacles()
    grid_map = map_gen.get_map()
    start = map_gen.start
    goal = map_gen.goal
    
    temp_results = {
        'Hybrid JPS': {'time': None, 'length': None, 'success': False},
        'JPS': {'time': None, 'length': None, 'success': False},
        'Hybrid A*': {'time': None, 'length': None, 'success': False}
    }
    
    # Hybrid JPS
    try:
        hybrid_jps_start_time = time.time()
        hybrid_jps_path = HybridJPS.plan(
            grid=grid_map,
            start_node=(10.5, 10.5, np.deg2rad(45.0)),
            goal_node=(map_size - 10, map_size - 10, np.deg2rad(45.0)),
            resolution=1.0
        )
        hybrid_jps_time = time.time() - hybrid_jps_start_time
        
        if hybrid_jps_path and len(hybrid_jps_path) > 0:
            hybrid_jps_length = compute_path_length(hybrid_jps_path)
            temp_results['Hybrid JPS']['time'] = hybrid_jps_time
            temp_results['Hybrid JPS']['length'] = hybrid_jps_length
            temp_results['Hybrid JPS']['success'] = True
            print(f"  Hybrid JPS - Time: {hybrid_jps_time:.4f}s, Length: {hybrid_jps_length:.2f}")
        else:
            print(f"  Hybrid JPS - 경로를 찾지 못했습니다")
    except Exception as e:
        print(f"  Hybrid JPS - 오류 발생: {e}")
    
    # JPS
    try:
        jps_start_time = time.time()
        jps_result = jps.method(grid_map, start, goal, 2)
        jps_time = time.time() - jps_start_time
        
        if jps_result and len(jps_result[0]) > 0:
            jps_path = [(x, y) for y, x in jps_result[0]]
            jps_length = compute_path_length(jps_path)
            temp_results['JPS']['time'] = jps_time
            temp_results['JPS']['length'] = jps_length
            temp_results['JPS']['success'] = True
            print(f"  JPS - Time: {jps_time:.4f}s, Length: {jps_length:.2f}")
        else:
            print(f"  JPS - 경로를 찾지 못했습니다")
    except Exception as e:
        print(f"  JPS - 오류 발생: {e}")
    
    # Hybrid A*
    try:
        start_hybrid = [float(start[0]), float(start[1]), np.deg2rad(45.0)]
        goal_hybrid = [float(goal[0]), float(goal[1]), np.deg2rad(45.0)]
        ox, oy = map_gen.get_obstacle_points()
        
        hybrid_astar_start_time = time.time()
        path_obj = hybrid_a_star_planning(
            start_hybrid, goal_hybrid, ox, oy, 2.0, np.deg2rad(15.0)
        )
        hybrid_astar_time = time.time() - hybrid_astar_start_time
        
        if path_obj and len(path_obj.x_list) > 0:
            hybrid_astar_path = list(zip(path_obj.x_list, path_obj.y_list))
            hybrid_astar_length = compute_path_length(hybrid_astar_path)
            temp_results['Hybrid A*']['time'] = hybrid_astar_time
            temp_results['Hybrid A*']['length'] = hybrid_astar_length
            temp_results['Hybrid A*']['success'] = True
            print(f"  Hybrid A* - Time: {hybrid_astar_time:.4f}s, Length: {hybrid_astar_length:.2f}")
        else:
            print(f"  Hybrid A* - 경로를 찾지 못했습니다")
    except Exception as e:
        print(f"  Hybrid A* - 오류 발생: {e}")
    
    # 모든 플래너가 성공했는지 확인
    all_success = all(temp_results[planner]['success'] for planner in temp_results)
    
    return all_success, temp_results


def test_map_size_variation():
    """맵 크기를 변화시키면서 성능 측정 (장애물은 맵 크기의 1/2)"""
    print("=" * 70)
    print("실험 1: 맵 크기 변화에 따른 성능 분석")
    print("=" * 70)
    
    map_sizes = [10, 50, 100, 200, 300]
    obstacle_size_range = (1, 5)
    max_retries = 5  # 최대 재시도 횟수
    
    results = {
        'Hybrid A*': {'times': [], 'lengths': []},
        'Hybrid JPS': {'times': [], 'lengths': []},
        'JPS': {'times': [], 'lengths': []}
    }
    
    for map_size in map_sizes:
        obstacle_count = map_size // 2
        print(f"\n[맵 크기: {map_size}x{map_size}, 장애물 수: {obstacle_count}]")
        
        # 성공할 때까지 재시도
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"  🔄 재시도 {attempt}/{max_retries-1}...")
            
            all_success, temp_results = run_single_test(map_size, obstacle_count, obstacle_size_range)
            
            if all_success:
                # 모두 성공하면 결과 저장하고 다음 맵으로
                results['Hybrid JPS']['times'].append(temp_results['Hybrid JPS']['time'])
                results['Hybrid JPS']['lengths'].append(temp_results['Hybrid JPS']['length'])
                results['JPS']['times'].append(temp_results['JPS']['time'])
                results['JPS']['lengths'].append(temp_results['JPS']['length'])
                results['Hybrid A*']['times'].append(temp_results['Hybrid A*']['time'])
                results['Hybrid A*']['lengths'].append(temp_results['Hybrid A*']['length'])
                print(f"  ✅ 모든 플래너 성공!")
                break
            else:
                # 실패한 플래너가 있으면 재시도
                failed_planners = [name for name, data in temp_results.items() if not data['success']]
                print(f"  ❌ 실패한 플래너: {', '.join(failed_planners)}")
                
                if attempt == max_retries - 1:
                    # 최대 재시도 횟수 도달
                    print(f"  ⚠️  최대 재시도 횟수 도달. None 값으로 저장합니다.")
                    results['Hybrid JPS']['times'].append(temp_results['Hybrid JPS']['time'])
                    results['Hybrid JPS']['lengths'].append(temp_results['Hybrid JPS']['length'])
                    results['JPS']['times'].append(temp_results['JPS']['time'])
                    results['JPS']['lengths'].append(temp_results['JPS']['length'])
                    results['Hybrid A*']['times'].append(temp_results['Hybrid A*']['time'])
                    results['Hybrid A*']['lengths'].append(temp_results['Hybrid A*']['length'])
    
    # 그래프 생성
    plot_results(map_sizes, results, 
                 xlabel='Map Size', 
                 title_time='Planning Time vs Map Size (Obstacles = Map Size / 2)',
                 title_length='Path Length vs Map Size (Obstacles = Map Size / 2)',
                 filename_prefix='map_size')
    
    return results


def test_obstacle_density_variation():
    """장애물 밀도를 변화시키면서 성능 측정 (맵 크기는 50으로 고정)"""
    print("\n" + "=" * 70)
    print("실험 2: 장애물 밀도 변화에 따른 성능 분석")
    print("=" * 70)
    
    map_size = 50
    obstacle_counts = [5, 10, 20, 30, 40]
    obstacle_size_range = (1, 5)
    max_retries = 5  # 최대 재시도 횟수
    
    results = {
        'Hybrid A*': {'times': [], 'lengths': []},
        'Hybrid JPS': {'times': [], 'lengths': []},
        'JPS': {'times': [], 'lengths': []}
    }
    
    for obstacle_count in obstacle_counts:
        print(f"\n[맵 크기: {map_size}x{map_size}, 장애물 수: {obstacle_count}]")
        
        # 성공할 때까지 재시도
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"  🔄 재시도 {attempt}/{max_retries-1}...")
            
            all_success, temp_results = run_single_test(map_size, obstacle_count, obstacle_size_range)
            
            if all_success:
                # 모두 성공하면 결과 저장하고 다음 맵으로
                results['Hybrid JPS']['times'].append(temp_results['Hybrid JPS']['time'])
                results['Hybrid JPS']['lengths'].append(temp_results['Hybrid JPS']['length'])
                results['JPS']['times'].append(temp_results['JPS']['time'])
                results['JPS']['lengths'].append(temp_results['JPS']['length'])
                results['Hybrid A*']['times'].append(temp_results['Hybrid A*']['time'])
                results['Hybrid A*']['lengths'].append(temp_results['Hybrid A*']['length'])
                print(f"  ✅ 모든 플래너 성공!")
                break
            else:
                # 실패한 플래너가 있으면 재시도
                failed_planners = [name for name, data in temp_results.items() if not data['success']]
                print(f"  ❌ 실패한 플래너: {', '.join(failed_planners)}")
                
                if attempt == max_retries - 1:
                    # 최대 재시도 횟수 도달
                    print(f"  ⚠️  최대 재시도 횟수 도달. None 값으로 저장합니다.")
                    results['Hybrid JPS']['times'].append(temp_results['Hybrid JPS']['time'])
                    results['Hybrid JPS']['lengths'].append(temp_results['Hybrid JPS']['length'])
                    results['JPS']['times'].append(temp_results['JPS']['time'])
                    results['JPS']['lengths'].append(temp_results['JPS']['length'])
                    results['Hybrid A*']['times'].append(temp_results['Hybrid A*']['time'])
                    results['Hybrid A*']['lengths'].append(temp_results['Hybrid A*']['length'])
    
    # 그래프 생성
    plot_results(obstacle_counts, results, 
                 xlabel='Number of Obstacles', 
                 title_time='Planning Time vs Obstacle Density (Map Size = 50x50)',
                 title_length='Path Length vs Obstacle Density (Map Size = 50x50)',
                 filename_prefix='obstacle_density')
    
    return results


def plot_results(x_values, results, xlabel, title_time, title_length, filename_prefix):
    """결과를 그래프로 시각화하고 저장"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {'Hybrid A*': 'blue', 'Hybrid JPS': 'green', 'JPS': 'red'}
    markers = {'Hybrid A*': 'o', 'Hybrid JPS': 's', 'JPS': '^'}
    
    # Planning Time 그래프
    for planner_name, data in results.items():
        # None 값 필터링
        valid_indices = [i for i, t in enumerate(data['times']) if t is not None]
        valid_x = [x_values[i] for i in valid_indices]
        valid_times = [data['times'][i] for i in valid_indices]
        
        if valid_times:
            ax1.plot(valid_x, valid_times, 
                    marker=markers[planner_name], 
                    label=planner_name, 
                    color=colors[planner_name],
                    linewidth=2,
                    markersize=8)
    
    ax1.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax1.set_ylabel('Planning Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title(title_time, fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # 로그 스케일
    
    # y축 눈금을 명시적으로 설정 (1, 0.1, 0.01, 0.001 등)
    from matplotlib.ticker import LogLocator, LogFormatter
    ax1.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    ax1.yaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=False))
    ax1.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=15))
    ax1.yaxis.set_minor_formatter(LogFormatter(base=10.0, labelOnlyBase=False, minor_thresholds=(2, 0.4)))
    
    # Path Length 그래프
    for planner_name, data in results.items():
        # None 값 필터링
        valid_indices = [i for i, l in enumerate(data['lengths']) if l is not None and l > 0]
        valid_x = [x_values[i] for i in valid_indices]
        valid_lengths = [data['lengths'][i] for i in valid_indices]
        
        if valid_lengths:
            ax2.plot(valid_x, valid_lengths, 
                    marker=markers[planner_name], 
                    label=planner_name, 
                    color=colors[planner_name],
                    linewidth=2,
                    markersize=8)
    
    ax2.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax2.set_ylabel('Path Length', fontsize=12, fontweight='bold')
    ax2.set_title(title_length, fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 그래프 저장
    filename = f'{filename_prefix}_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 그래프 저장됨: {filename}")
    
    plt.close()


def print_summary_table(x_values, results, experiment_name, x_label):
    """결과를 표로 출력"""
    print(f"\n{'=' * 100}")
    print(f"{experiment_name} - 요약 테이블")
    print(f"{'=' * 100}")
    
    # Planning Time 테이블
    print(f"\n[Planning Time (seconds)]")
    print(f"{x_label:>15} | {'Hybrid A*':>12} | {'Hybrid JPS':>12} | {'JPS':>12}")
    print("-" * 60)
    for i, x in enumerate(x_values):
        ha_time = f"{results['Hybrid A*']['times'][i]:.4f}" if results['Hybrid A*']['times'][i] else "Failed"
        hj_time = f"{results['Hybrid JPS']['times'][i]:.4f}" if results['Hybrid JPS']['times'][i] else "Failed"
        jps_time = f"{results['JPS']['times'][i]:.4f}" if results['JPS']['times'][i] else "Failed"
        print(f"{x:>15} | {ha_time:>12} | {hj_time:>12} | {jps_time:>12}")
    
    # Path Length 테이블
    print(f"\n[Path Length]")
    print(f"{x_label:>15} | {'Hybrid A*':>12} | {'Hybrid JPS':>12} | {'JPS':>12}")
    print("-" * 60)
    for i, x in enumerate(x_values):
        ha_len = f"{results['Hybrid A*']['lengths'][i]:.2f}" if results['Hybrid A*']['lengths'][i] else "Failed"
        hj_len = f"{results['Hybrid JPS']['lengths'][i]:.2f}" if results['Hybrid JPS']['lengths'][i] else "Failed"
        jps_len = f"{results['JPS']['lengths'][i]:.2f}" if results['JPS']['lengths'][i] else "Failed"
        print(f"{x:>15} | {ha_len:>12} | {hj_len:>12} | {jps_len:>12}")


def main():
    print("=" * 70)
    print("경로 계획 알고리즘 성능 분석 시작")
    print("=" * 70)
    
    # 실험 1: 맵 크기 변화
    map_size_results = test_map_size_variation()
    map_sizes = [10, 50, 100, 200, 300]
    print_summary_table(map_sizes, map_size_results, "실험 1: 맵 크기 변화", "Map Size")
    
    # 실험 2: 장애물 밀도 변화
    obstacle_results = test_obstacle_density_variation()
    obstacle_counts = [5, 10, 20, 30, 40]
    print_summary_table(obstacle_counts, obstacle_results, "실험 2: 장애물 밀도 변화", "Obstacles")
    
    print("\n" + "=" * 70)
    print("✅ 모든 실험 완료!")
    print("생성된 그래프 파일:")
    print("  📊 map_size_analysis.png")
    print("  📊 obstacle_density_analysis.png")
    print("=" * 70)


if __name__ == "__main__":
    main()