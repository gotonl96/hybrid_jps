import pygame
import os
import datetime
import grid
import astar
import jps

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from map.map_generate import MapGenerator
from visualization.visualize import Visualizer

def main():

    MAP_WIDTH = 100
    MAP_HEIGHT = 100

    start = (10, 10)
    goal = (MAP_WIDTH - 10, MAP_HEIGHT - 10)

    map = MapGenerator(MAP_WIDTH, MAP_HEIGHT, 300, (1, 5))
    map.generate_obstacles()
    
    jps_result = jps.method(map.get_map(), start, goal, 2)
    path_xy = [(x, y) for y, x in jps_result[0]]
    
    # print jps path x, y
    for i in range(len(jps_result[0])):
        print(i, jps_result[0][i][0], jps_result[0][i][1])
        
    visualizer = Visualizer()
    visualizer.set_grid_map(map.get_map())
    visualizer.set_start_goal(start, goal)
    visualizer.set_path(path_xy, "JPS Path")
    visualizer.draw()
    
if __name__ == "__main__":
    main()
