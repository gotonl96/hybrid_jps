cmake_minimum_required(VERSION 3.10)
project(planner_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# src 안의 모든 cpp 파일 가져오기
file(GLOB_RECURSE SRC_FILES src/*.cc)

# 실행 파일 생성
add_executable(${PROJECT_NAME} ${SRC_FILES})

# include 경로 추가
target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)
