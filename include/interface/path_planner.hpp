#pragma once

#include "utils/type.hpp"

class IPathPlanner {
   public:
    IPathPlanner() = default;
    virtual ~IPathPlanner() = default;

   public:
    virtual PlanResult pathPlanning(State curr_state, Goal goal) = 0;
};

