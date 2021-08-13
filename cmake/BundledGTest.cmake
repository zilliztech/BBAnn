find_package(GTest)

if (NOT GTEST_FOUND)
   message(STATUS "Adding bundled Google Test From Local File.")
   set(BUILD_GMOCK OFF CACHE BOOL INTERNAL)
   set(INSTALL_GTEST OFF CACHE BOOL INTERNAL)

   add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/thirdparty/googletest)

   add_library(GTest::GTest ALIAS gtest)
endif ()
