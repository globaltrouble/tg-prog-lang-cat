cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(tglang VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/deps/cptrie")

add_custom_target(
    generate_tfidf_blob_obj_file
    COMMAND objcopy -I binary -O elf64-x86-64 -B i386:x86-64 cptrie-tfidf-dict.bin cptrie_tfidf_dict.o
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_custom_target(
    copy_tfidf_file
    COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/model/cptrie-tfidf-dict.bin cptrie-tfidf-dict.bin
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(generate_tfidf_blob_obj_file copy_tfidf_file)

add_custom_target(
    generate_id_blob_obj_file
    COMMAND objcopy -I binary -O elf64-x86-64 -B i386:x86-64 cptrie-id-dict.bin cptrie_id_dict.o
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_custom_target(
    copy_id_file
    COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/model/cptrie-id-dict.bin cptrie-id-dict.bin
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(generate_id_blob_obj_file copy_id_file)
# can be inspected with: `objdump -x cptrie_tfidf_dict.o`
# inspired by `https://tratt.net/laurie/blog/2022/whats_the_most_portable_way_to_include_binary_blobs_in_an_executable.html`

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)

add_library(tglang SHARED src/tglang.cpp)
target_include_directories(tglang PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_options(tglang PRIVATE -Wall -Wextra -Werror -Wpedantic -Wno-error=sign-compare)

find_library(re2_LIBRARY NAMES re2 libre2 REQUIRED)

add_dependencies(tglang generate_tfidf_blob_obj_file generate_id_blob_obj_file)
target_link_libraries(tglang PUBLIC cptrie ${re2_LIBRARY} ${CMAKE_CURRENT_BINARY_DIR}/cptrie_tfidf_dict.o ${CMAKE_CURRENT_BINARY_DIR}/cptrie_id_dict.o)