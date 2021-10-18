set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(BBAnnLib2_SRCS bbannlib2.cpp)
add_library(BBAnnLib2_s STATIC ${BBAnnLib2_SRCS})
