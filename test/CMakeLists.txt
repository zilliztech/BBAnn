add_executable(test_pq EXCLUDE_FROM_ALL test_pq.cpp)

add_executable(test_diskann1 test_diskann1.cpp)
target_link_libraries(test_diskann1 BigAnn)

add_executable(test_refine_id test_refine_id.cpp)
target_link_libraries(test_refine_id BigAnn)

