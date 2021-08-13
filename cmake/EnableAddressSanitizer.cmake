set(DISABLE_ASAN OFF CACHE BOOL "disable compilation with the address sanitizer for Debug builds")

include(CheckCXXSourceRuns)

if (NOT DISABLE_ASAN)
   set(CMAKE_REQUIRED_FLAGS "-Werror -fsanitize=address")
   check_cxx_source_runs("int main() { return 0; }" HAVE_FLAG_SANITIZE_ADDRESS)
   unset(CMAKE_REQUIRED_FLAGS)

   if (NOT HAVE_FLAG_SANITIZE_ADDRESS)
      message(WARNING "The address sanitizer is enabled but not supported on your system. Either disable the address sanitizer (-DDISABLE_ASAN=ON), choose a compiler that supports it, or fix your system (most likely by installing libasan).")
      return()
   endif ()

   set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()