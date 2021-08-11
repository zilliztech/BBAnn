set(SRCS diskann2.cpp TimeRecorder.cpp random.cpp)

set (depend_lib
    aio)
add_library(${PROJECT_NAME} SHARED ${SRCS})
add_library(${PROJECT_NAME}_static STATIC ${SRCS})
target_link_libraries(${PROJECT_NAME}  ${depend_lib})
