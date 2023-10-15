cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(tglang VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_RPATH "./resources")

add_compile_options(-Wall -Wextra -Werror)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)

add_custom_target(
    tfidf_mapping_header
    COMMAND cat tf_idf_data.json | python3 json-to-header.py > tfidf_mapping.h 
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(tglang SHARED tglang.cpp)

add_dependencies(tglang tfidf_mapping_header)

target_include_directories(tglang PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ort/include")
find_library(onnxrt_LIBRARY onnxruntime PATHS "${CMAKE_CURRENT_SOURCE_DIR}/ort/lib" NO_DEFAULT_PATH)

find_library(re2_LIBRARY NAMES re2 libre2 REQUIRED)
target_link_libraries(tglang PRIVATE "${onnxrt_LIBRARY}" ${re2_LIBRARY})