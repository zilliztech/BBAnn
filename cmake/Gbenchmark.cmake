find_package(benchmark)

set(BENCHMARK_ENABLE_GTEST_TESTS OFF)

if (NOT benchmark_FOUND)
    message(STATUS "Adding bundled Google Benchmark From Local File.")
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/thirdparty/googlebenchmark)
    add_library(benchmark::benchmark ALIAS benchmark)
endif ()
