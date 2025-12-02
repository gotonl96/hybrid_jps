import pygame
import os
import datetime
import grid
import astar
import jps


def main():

    MAP_WIDTH = 100
    MAP_HEIGHT = 100

    size = (MAP_WIDTH, MAP_HEIGHT)

    start = (10, 10)
    goal = (MAP_WIDTH - 10, MAP_HEIGHT - 10)

    

    jps.method(Grid.matrix, start, goal, 2)


if __name__ == "__main__":
    main()
