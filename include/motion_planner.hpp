
#pragma once

#include "interface/path_planner.hpp"
#include "map/map_manager.hpp"
#include "search/hybrid_jps.hpp"

class MotionPlanner : public IPathPlanner {
   public:

    MotionPlanner();

    PlanResult pathPlanning(State curr_state, Goal goal);

private:
    MapManager map_manager_;
    HybridJPS hybrid_jps_;

};