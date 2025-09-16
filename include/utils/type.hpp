#pragma once

#include <vector>

struct State {
    double x;
    double y;
    double theta;
};

struct Goal {
    double x;
    double y;
};

struct Map {
    int width;
    int height;
    std::vector<std::vector<int>> data; // 0 for free space, 1 for obstacle

    Map(int w=1000, int h=1000) : width(w), height(h) {
        data.resize(h, std::vector<int>(w, 0)); // initialize with free space (0)
    }
};

struct PlanResult {
    std::vector<State> path;
};