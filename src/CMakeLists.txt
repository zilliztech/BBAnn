set(SRCS build.cpp query.cpp TimeRecorder.cpp)
add_library(${PROJECT_NAME} SHARED ${SRCS})
add_library(${PROJECT_NAME}_static STATIC ${SRCS})
