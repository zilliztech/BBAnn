set(DISABLE_UBSAN OFF CACHE BOOL "disable compilation with the sanitizer for undefined behavior for Debug builds")

include(CheckCXXCompilerFlag)

if (NOT DISABLE_UBSAN)
   set(CMAKE_REQUIRED_FLAGS "-Werror -fsanitize=undefined")
   check_cxx_compiler_flag("-fsanitize=undefined" HAVE_FLAG_SANITIZE_UNDEFINED)
   unset(CMAKE_REQUIRED_FLAGS)

   if (NOT HAVE_FLAG_SANITIZE_UNDEFINED)
      message(WARNING "The undefined behavior sanitizer is enabled but not supported on your system. Please disable (-DDISABLE_UBSAN=ON) the undefined behavior sanitizer or choose a compiler that supports it.")
      return()
   endif ()

   set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=undefined")
endif ()