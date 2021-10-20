set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(BBAnnLib2_SRCS bbannlib2.cpp)
add_library(BBAnnLib2_s STATIC ${BBAnnLib2_SRCS})

add_library(algo_s STATIC algo.cpp)
add_library(ivf_s STATIC ivf.cpp)

add_executable(wwwww test.cpp)
target_link_libraries(algo_s ivf_s)
target_link_libraries(wwwww BBAnnLib2_s algo_s ivf_s aio TimeRecorder)
