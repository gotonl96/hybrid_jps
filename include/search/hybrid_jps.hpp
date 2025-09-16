#pragma once

#include "interface/path_planner.hpp"
#include "model/vehicle_model.hpp"

class HybridJPS
{
public:
    HybridJPS() {}
    ~HybridJPS() {}

public:

    PlanResult search(State curr_state, Goal goal, const Map &map);

    
};