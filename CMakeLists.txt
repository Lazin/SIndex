project(SIndex)
cmake_minimum_required(VERSION 2.8)
add_definitions(-std=c++1y)

include_directories(CRoaring/include)
include_directories(CRoaring/cpp)
add_subdirectory(CRoaring)
add_executable(SIndex main.cpp)
target_link_libraries(SIndex roaring)

