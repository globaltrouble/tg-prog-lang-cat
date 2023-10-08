cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(tglang-multitester VERSION 1.0.0 LANGUAGES C)

add_executable(tglang-multitester tglang-multitester.c)

find_library(tglang_LIBRARY tglang PATHS "${CMAKE_CURRENT_SOURCE_DIR}" NO_DEFAULT_PATH)
target_link_libraries(tglang-multitester PRIVATE "${tglang_LIBRARY}")

if (NOT WIN32)
  target_link_libraries(tglang-multitester PRIVATE m)
  if (UNIX AND NOT APPLE)
    target_link_libraries(tglang-multitester PRIVATE rt)
  endif()
else()
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()
