project(BigAnn)

cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(OpenMP REQUIRED)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -O3 -pthread -fopenmp" )
include_directories(include)
#include_directories("util")
#include_directories("index")
#include_directories("flat")
#include_directories("ivf")
#include_directories("hnswlib")

set(bigann
    bigann.cpp
)

add_executable(test_bigann ${bigann})

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(scripts)
target_link_libraries(test_bigann BigAnn)


