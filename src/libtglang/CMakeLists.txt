cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(tglang VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/fastText")

add_library(tglang SHARED tglang.cpp)
target_compile_options(tglang PRIVATE -Wall -Wextra -Werror -Wpedantic -Wno-error=sign-compare)
target_link_libraries(tglang PRIVATE fasttext-static_pic pthread)
