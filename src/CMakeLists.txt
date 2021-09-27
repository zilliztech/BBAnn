set(BigAnn_SRCS diskann2.cpp TimeRecorder.cpp random.cpp)
set(BBAnn_SRCS bbann.cpp TimeRecorder.cpp random.cpp)

add_library(BigAnn SHARED ${BigAnn_SRCS})
add_library(BigAnn_static STATIC ${BigAnn_SRCS})

add_library(BBAnn SHARED ${BBAnn_SRCS})
add_library(BBAnn_static STATIC ${BBAnn_SRCS})
