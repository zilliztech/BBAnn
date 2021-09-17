cmake_minimum_required(VERSION 3.14)
project(BigAnn)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP REQUIRED)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
find_package(Threads REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(BundledGTest)
include(Gbenchmark)
include(EnableAddressSanitizer)
include(EnableUndefinedSanitizer)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3 -fopenmp -mavx2 -mfma" )
# set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O0 -g -mavx2 -mfma" )

# Log with I/O statistics doing profiling
set(IOPERF 1)
if (IOPERF)
    add_compile_options(-DIOPERF=1)
    message(STATUS "Optional Flag DIOPERF is ON: to do IO Perf")
endif(IOPERF)

# Refine with Unix IO & O_DIRECT
set(DIRECTIO 1)
if (DIRECTIO)
    add_compile_options(-DDIRECTIO=1)
    message(STATUS "Optional Flag DIRECTIO is ON: to do DIRECT IO: O_DIRECT")
endif(DIRECTIO)

include_directories(include)

add_subdirectory(src)

add_executable(build_bigann build_bigann.cpp)
add_executable(search_bigann search_bigann.cpp)
add_executable(build_bbann build_bbann.cpp)
add_executable(search_bbann search_bbann.cpp)

add_subdirectory(test)
add_subdirectory(playground)
add_subdirectory(scripts)
target_link_libraries(build_bigann BigAnn)
target_link_libraries(search_bigann BigAnn)
target_link_libraries(build_bbann BBAnn)
target_link_libraries(search_bbann BBAnn)


