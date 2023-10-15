cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(tglang VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/fastText")

add_custom_target(
    generate_model_blob_header
    COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/create-header-from-blob.py" -f ../resources/fasttext-model.bin --header fasttext_model_blob
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)

add_library(tglang SHARED tglang.cpp)
add_dependencies(tglang generate_model_blob_header)
target_compile_options(tglang PRIVATE -Wall -Wextra -Werror -Wpedantic -Wno-error=sign-compare)
target_link_libraries(tglang PRIVATE fasttext-static_pic pthread)
