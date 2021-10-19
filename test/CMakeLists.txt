add_executable(test_statistics test_statistics.cpp)

add_executable(test_refine_id test_refine_id.cpp)
target_link_libraries(test_refine_id BBAnn)

add_executable(test_dist test_dist.cpp)
target_link_libraries(test_dist BBAnn)

add_executable(test_distance test_distance.cpp)
target_link_libraries(test_distance BBAnn)

add_executable(test_recursive_kmeans recursive_kmeans_test.cpp)
target_link_libraries(test_recursive_kmeans BBAnn)

add_executable(test_same_size_kmeans test_same_size_kmeans.cpp)
target_link_libraries(test_same_size_kmeans BBAnn)

add_executable(test_hnsw_range test_hnsw_range.cpp)
target_link_libraries(test_hnsw_range BBAnn)

add_executable(test_aio_reader test_aio_reader.cpp)
target_link_libraries(test_aio_reader aio TimeRecorder)

