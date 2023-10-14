#include "tglang.h"

#include <cstdlib>
#include <cstring>
#include <atomic>
#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <memory>
#include <chrono>
#include <optional>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace {

struct ProfileIt {
  char const * const m_name = nullptr;
  std::chrono::steady_clock::time_point m_begin;
  
  ProfileIt(char const * const name) : m_name(name), m_begin(std::chrono::steady_clock::now()) {}

  ~ProfileIt() {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - m_begin);
    std::cerr << std::fixed << std::setprecision(6) << (elapsed.count() * 0.000001) << " sec \n";
  }
};

std::atomic<bool> wasInit = { false };
 
void init() {
  ProfileIt p("Init");

  bool expected = false;
  bool alreadyInit = !wasInit.compare_exchange_strong(expected, 
                                                      true,
                                                      std::memory_order_release,
                                                      std::memory_order_relaxed);
  if (alreadyInit) {
    return;
  }

  // TODO: make threadsafe
}

} // namespace

enum TglangLanguage tglang_detect_programming_language(const char *text) {
  ProfileIt inf("Inference");

  init();

  if (strstr(text, "std::") != NULL) {
    return TGLANG_LANGUAGE_CPLUSPLUS;
  }
  if (strstr(text, "let ") != NULL) {
    return TGLANG_LANGUAGE_JAVASCRIPT;
  }
  if (strstr(text, "int ") != NULL) {
    return TGLANG_LANGUAGE_C;
  }
  if (strstr(text, ";") == NULL) {
    return TGLANG_LANGUAGE_PYTHON;
  }
  return TGLANG_LANGUAGE_OTHER;
}
