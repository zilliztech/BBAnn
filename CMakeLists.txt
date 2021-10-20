cmake_minimum_required(VERSION 3.10)
project(BigAnn)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP REQUIRED)
set (CMAKE_CXX_FLAGS "-g -O3 ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
find_package(Threads REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# -march=native: https://stackoverflow.com/a/54032969/10971650, https://stackoverflow.com/questions/3015306/what-exactly-does-march-native-do
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fPIC")
endif()

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
add_compile_options(-march=native)

add_subdirectory(src)

# add_subdirectory(test)
# add_subdirectory(scripts)

