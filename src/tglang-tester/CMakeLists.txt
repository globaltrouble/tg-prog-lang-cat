cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(tglang-tester VERSION 1.0.0 LANGUAGES C)

add_executable(tglang-tester tglang-tester.c)

find_library(tglang_LIBRARY tglang PATHS "${CMAKE_CURRENT_SOURCE_DIR}" NO_DEFAULT_PATH)
target_link_libraries(tglang-tester PRIVATE "${tglang_LIBRARY}")

if (NOT WIN32)
  target_link_libraries(tglang-tester PRIVATE m)
  if (UNIX AND NOT APPLE)
    target_link_libraries(tglang-tester PRIVATE rt)
  endif()
else()
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()
