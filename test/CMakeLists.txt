add_executable(test_statistics test_statistics.cpp)

add_executable(test_pq test_pq.cpp)
target_link_libraries(test_pq BigAnn)

add_executable(test_rpq test_pq_residual.cpp)
target_link_libraries(test_rpq BigAnn)

#add_executable(test_diskann1 test_diskann1.cpp)
#target_link_libraries(test_diskann1 BigAnn)

add_executable(test_refine_id test_refine_id.cpp)
target_link_libraries(test_refine_id BigAnn)

#add_executable(test_build_bigann build_bigann.cpp)
#target_link_libraries(test_build_bigann BigAnn)

#add_executable(test_search_bigann search_bigann.cpp)
#target_link_libraries(test_search_bigann BigAnn)

add_executable(test_dist test_dist.cpp)
target_link_libraries(test_dist BigAnn)

add_executable(test_distance test_distance.cpp)
target_link_libraries(test_distance BigAnn)

add_executable(test_recursive_kmeans recursive_kmeans_test.cpp)
target_link_libraries(test_recursive_kmeans BigAnn)

add_executable(test_same_size_kmeans test_same_size_kmeans.cpp)
target_link_libraries(test_same_size_kmeans BigAnn)

add_executable(test_hnsw_range test_hnsw_range.cpp)
target_link_libraries(test_hnsw_range BigAnn)

## Google Test
set(TEST_SOURCES
        # add your *.cpp files here
        Tester.cpp
        TestIOPerf.cpp)


add_executable(tester ${TEST_SOURCES})
target_link_libraries(tester PUBLIC
        BigAnn
        GTest::GTest)

add_executable(dummy_benchmark dummy_benchmark.cpp)
target_link_libraries(dummy_benchmark benchmark::benchmark)
