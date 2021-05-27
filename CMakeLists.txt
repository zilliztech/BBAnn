project(BigAnn)

cmake_minimum_required(VERSION 3.14)

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -O3 -pthread -fopenmp" )

include_directories("util")
include_directories("flat")
include_directories("ivf")
include_directories("hnswlib")

set(bigann
    bigann.cpp
)

add_executable(BigAnn ${bigann})



