
set(BBAnnLib_SRCS bbannlib.cpp)
set(BBAnnLib2_SRCS bbannlib2.cpp)
add_library(BBAnnLib_s STATIC ${BBAnnLib_SRCS})
add_library(BBAnnLib2_s STATIC ${BBAnnLib2_SRCS})
