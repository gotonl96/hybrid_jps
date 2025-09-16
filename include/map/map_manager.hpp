#pragma once

#include <vector>
#include <random>

class MapManager {

public :
    MapManager(int width=1000, int height=1000) {
        width = width;
        height = height;
        map_ = Map(width, height);
        updateCostmap();

    }

    Map getMap() { return map_; }
    

private:
    bool updateCostmap() {
        // random locate, size obstacle generation
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis_x(0, map_.width - 1);
        std::uniform_int_distribution<> dis_y(0, map_.height - 1);
        std::uniform_int_distribution<> dis_size(5, 50); // obstacle size between
        for (int i = 0; i < 20; ++i) { // generate 20 random obstacles
            int obs_x = dis_x(gen);
            int obs_y = dis_y(gen);
            int obs_size = dis_size(gen);
            for (int y = std::max(0, obs_y - obs_size / 2); y < std::min(map_.height, obs_y + obs_size / 2); ++y) {
                for (int x = std::max(0, obs_x - obs_size / 2); x < std::min(map_.width, obs_x + obs_size / 2); ++x) {
                    map_[y][x] = 1; // mark as obstacle
                }
            }
        }

        return true;
    }

private:
    Map map_;
};