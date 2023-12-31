cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(tglang VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/deps/fastText")

# move iscode model bo binary to avoid additional resource load from disk
add_custom_target(
    generate_iscode_model_blob_obj_file
    COMMAND objcopy -I binary -O elf64-x86-64 -B i386:x86-64 fasttext-model-iscode.bin fasttext_model_iscode_blob.o
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_custom_target(
    copy_iscode_model_file
    COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/../resources/fasttext-model-iscode.bin fasttext-model-iscode.bin
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(generate_iscode_model_blob_obj_file copy_iscode_model_file)

# move iscode model bo binary to avoid additional resource load from disk
add_custom_target(
    generate_codetype_model_blob_obj_file
    COMMAND objcopy -I binary -O elf64-x86-64 -B i386:x86-64 fasttext-model-codetype.bin fasttext_model_codetype_blob.o
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_custom_target(
    copy_codetype_model_file
    COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/../resources/fasttext-model-codetype.bin fasttext-model-codetype.bin
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(generate_codetype_model_blob_obj_file copy_codetype_model_file)
# can be inspected with: `objdump -x fasttext_model_codetype_blob.o`
# inspired by `https://tratt.net/laurie/blog/2022/whats_the_most_portable_way_to_include_binary_blobs_in_an_executable.html`

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)

add_library(tglang SHARED src/tglang.cpp)
target_include_directories(tglang PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/deps/fastText/src")
target_compile_options(tglang PRIVATE -Wall -Wextra -Werror -Wpedantic -Wno-error=sign-compare)

find_library(re2_LIBRARY NAMES re2 libre2 REQUIRED)

add_dependencies(tglang generate_iscode_model_blob_obj_file generate_codetype_model_blob_obj_file)
target_link_libraries(tglang PUBLIC 
    fasttext-static_pic 
    pthread 
    ${re2_LIBRARY}
    ${CMAKE_CURRENT_BINARY_DIR}/fasttext_model_iscode_blob.o 
    ${CMAKE_CURRENT_BINARY_DIR}/fasttext_model_codetype_blob.o
)