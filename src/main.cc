
#include <memory>

#include "interface/path_planner.hpp"
#include "motion_planner.hpp"

#include "utils/type.hpp"

int main(int argc, char** argv){
    
    std::shared_ptr<IPathPlanner> motion_planner = std::make_shared<MotionPlanner>();
        
    //* define map size
    int map_width = 1000;
    int map_height = 1000;

    //* define start, goal
    State init_state;
    init_state.x = 0; // m  
    init_state.y = 0; // m 
    init_state.theta = 0; // deg

    Goal goal;
    goal.x = map_width;
    goal.y = map_height;
    
    // Plan Result
    auto plan_result_astar = motion_planner->pathPlanning(init_state, goal);

    return 0;
}