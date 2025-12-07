# path_planner/hybrid_astar/__init__.py
import sys
from pathlib import Path

# hybrid_astar 폴더를 sys.path에 추가 (같은 폴더 내 모듈들 import 가능하게)
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# ReedsSheppPath와 utils를 위한 경로 추가
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir / 'ReedsSheppPath'))
sys.path.insert(0, str(parent_dir / 'utils'))

# 프로젝트 루트 추가 (map, visualization 접근용)
project_root = parent_dir.parent
sys.path.insert(0, str(project_root))