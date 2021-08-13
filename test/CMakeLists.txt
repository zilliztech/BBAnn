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

add_executable(test_distance test_distance.cpp)
target_link_libraries(test_distance BigAnn)

add_executable(helloworld_io_perf helloworld_io_perf.cpp)
target_link_libraries(helloworld_io_perf BigAnn)


## Google Test
set(TEST_SOURCES
        # add your *.cpp files here
        Tester.cpp
        TestIOPerf.cpp)


add_executable(tester ${TEST_SOURCES})
target_link_libraries(tester PUBLIC
        BigAnn
        GTest::GTest)
