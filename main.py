import astar
import hybrid_astar
import jps
import hybrid_jps
import environment

# load parameter from config yaml 
def load_config(file_path):
    config = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            value = value.strip()
            try:
                config[key.strip()] = int(value)
            except ValueError:
                config[key.strip()] = value
    return config

if __name__ == "__main__":
    
    print("Hybrid JPS Benchmark Test")

    config = load_config('config.yaml')

    map_width = config['map_width']
    map_height = config['map_height']
    num_obstacles = config['num_obstacles']
    max_obstacle_size = config['max_obstacle_size']

    start = (config['start_x'], config['start_y'])
    goal = (config['goal_x'], config['goal_y'])

    env = environment.Env(map_width, map_height, num_obstacles, max_obstacle_size)
    
    print

    path1, t1 = jps.method(env.grid, start, goal, 1)
    path2, t2 = astar.method(env.grid, start, goal, 1)

    env.show(paths=[
        (path1, "red", f"jps ({t1}s)"),
        (path2, "blue", f"astar ({t2}s)")
    ])