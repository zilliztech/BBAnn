set(SRCS diskann2.cpp TimeRecorder.cpp random.cpp)
add_library(${PROJECT_NAME} SHARED ${SRCS})
add_library(${PROJECT_NAME}_static STATIC ${SRCS})
