cmake_minimum_required (VERSION 3.8)
project(MiniML VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE Classes
     "src/*.h"
     "src/*.cpp"
     "src/*.hpp"
     "include/*.hpp"
)

find_package(OpenMP REQUIRED)

include(CTest)
enable_testing()

add_library(MiniML SHARED ${Classes})
target_compile_definitions(MiniML PRIVATE MINIML_API_EXPORTS)
target_link_libraries(MiniML PRIVATE OpenMP::OpenMP_CXX)