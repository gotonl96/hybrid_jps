#include "motion_planner.hpp"

#include <chrono>
#include <iostream>

MotionPlanner::MotionPlanner() : IPathPlanner() {
    map_manager_ = MapManager(1000, 1000);
    hybrid_jps_ = HybridJPS();
}


PlanResult MotionPlanner::pathPlanning(State curr_state, Goal goal) {

    // execution time check
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    //* hybrid jps planning
    // update map
    auto map = map_manager_.getMap();

    // search path using hybrid jps
    PlanResult result;
    result = hybrid_jps_.search(curr_state, goal, map);

    // exectution time check
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "[HybridJPSPlanner] Execution Time = " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() 
              << " ms" << std::endl;

    return result;
}