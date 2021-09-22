add_executable(genenrate_sample genenrate_sample.cpp)

add_executable(gt_reader gt_reader.cpp)

add_executable(sequenital_slicer sequenital_slicer.cpp)

add_executable(equal_width_histogram equal_width_histogram.cpp)

add_executable(equal_height_histogram equal_height_histogram.cpp)

add_executable(norm_bias_histogram norm_bias_histogram.cpp)

add_executable(breakdown_bigann breakdown_bigann.cpp)
target_link_libraries(breakdown_bigann benchmark::benchmark BigAnn uring)
