#include "tglang.h"

#include "fastText/src/fasttext.h"

#include <atomic>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <cstring>

#define LABEL_PREFIX "__label__"

namespace {

// TODO: remove profiler
struct ProfileIt {
  char const * const m_name = nullptr;
  std::chrono::steady_clock::time_point m_begin;
  
  ProfileIt(char const * const name) : m_name(name), m_begin(std::chrono::steady_clock::now()) {}

  ~ProfileIt() {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - m_begin);
    std::cerr << std::fixed << std::setprecision(6) << (elapsed.count() * 0.000001) << " sec \n";
  }
};

} // namespace

static std::atomic<bool> wasInit = { false };
static fasttext::FastText model;
 
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

  model.loadModel("./resources/fasttext-model.bin");
}

enum TglangLanguage tglang_detect_programming_language(const char *text) {
  ProfileIt inf("Inference");

  init();
  
  std::stringstream ss(text);
  ss.seekg(0, ss.beg);

  constexpr int32_t kCount = 1;
  constexpr fasttext::real kProbThreshold = 0.4;

  std::vector<std::pair<fasttext::real, std::string>> result;
  try {
    model.predictLine(ss, result, kCount, kProbThreshold);
  } catch (...) {
    return TglangLanguage::TGLANG_LANGUAGE_OTHER;
  }

  if (result.empty()) {
    return TglangLanguage::TGLANG_LANGUAGE_OTHER;
  }

  auto const & res = result.front();

  // TODO: remove LABEL_PREFIX
  int converted = std::atoi(res.second.c_str() + std::strlen(LABEL_PREFIX));

  // TODO: remove logs
  std::cerr << "Fasttext, class=" << res.second << ",converted=" << converted << ",prob=" << res.first << '\n';
  
  assert(converted <= TglangLanguage::TGLANG_LANGUAGE_YAML);

  return static_cast<TglangLanguage>(converted);
}
