
import random
import numpy as np
import matplotlib.pyplot as plt

class Env :
    def __init__(self, width, height, num_obstacles, max_obstacle_size):
        self.width = width
        self.height = height
        self.num_obstacles = num_obstacles
        self.max_obstacle_size = max_obstacle_size
        self.grid = np.zeros((height, width), dtype=int)

        self.generate_obstacles()

    def generate_obstacles(self):
        for _ in range(self.num_obstacles):
            w = random.randint(1, self.max_obstacle_size)
            h = random.randint(1, self.max_obstacle_size)
            x = random.randint(0, self.width - w - 1)
            y = random.randint(0, self.height - h - 1)

            self.grid[y:y+h, x:x+w] = 1  # Mark obstacle cells with 1

            grid[0:2, 0:2] = 1

    def is_free(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y, x] == 0
        return False
    
    def is_bound(self, x, y):
        if (0 <= x < self.width) and (0 <= y < self.height):
            return True
        return False
    
import matplotlib.pyplot as plt

class Env:
    def __init__(self, width, height, num_obstacles, max_obstacle_size):
        self.width = width
        self.height = height
        self.num_obstacles = num_obstacles
        self.max_obstacle_size = max_obstacle_size
        self.grid = np.zeros((height, width), dtype=int)

        self.generate_obstacles()

    def generate_obstacles(self):
        for _ in range(self.num_obstacles):
            w = random.randint(1, self.max_obstacle_size)
            h = random.randint(1, self.max_obstacle_size)
            x = random.randint(0, self.width - w - 1)
            y = random.randint(0, self.height - h - 1)

            self.grid[y:y+h, x:x+w] = 1  # Mark obstacle cells with 1

    def is_free(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y, x] == 0
        return False
    
    def is_bound(self, x, y):
        if (0 <= x < self.width) and (0 <= y < self.height):
            return True
        return False
        

    def show(self, paths=None, title=None):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(
            self.grid,
            cmap="Greys",
            origin="upper",
            interpolation="none",
            extent=[0, self.width, 0, self.height],
        )

        # 격자 표시
        ax.set_xticks(np.arange(0, self.width + 1, 1))
        ax.set_yticks(np.arange(0, self.height + 1, 1))
        ax.grid(which="both", color="lightgray", linewidth=0.5)
        ax.set_aspect("equal")

        # 축 눈금/숫자/테두리 제거
        ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # 여러 경로 표시
        if paths:
            for path, color, label in paths:
                if path:
                    xs = [x + 0.5 for (x, y) in path]
                    ys = [y + 0.5 for (x, y) in path]
                    ax.plot(xs, ys, color=color, linewidth=2, label=label)

        if title:
            ax.set_title(title, color="green")
        if paths and any(l for _, _, l in paths):
            ax.legend(loc="upper right", frameon=False)

        plt.tight_layout()
        plt.show()