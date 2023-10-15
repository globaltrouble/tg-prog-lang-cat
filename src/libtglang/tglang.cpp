#include "tglang.h"
#include "symbols_to_replace.h"
#include "fasttext_model_blob.h"

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
#include <codecvt>
#include <unordered_set>
#include <iterator>

#define LABEL_PREFIX "__label__"

using UnicodeConverter = std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t>;

class FastText : public fasttext::FastText {
public:
  void loadModel(std::istream& in) {
    if (!fasttext::FastText::checkModel(in)) {
      throw std::invalid_argument("Stream has wrong format!");
    }
    fasttext::FastText::loadModel(in);
  }

};


struct ProfileIt {
#ifdef NDEBUG
  ProfileIt(char const * const) {};
#else
  char const * const m_name = nullptr;
  std::chrono::steady_clock::time_point m_begin;
  
  ProfileIt(char const * const name) : m_name(name), m_begin(std::chrono::steady_clock::now()) {}

  ~ProfileIt() {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - m_begin);
    std::cerr << m_name << std::fixed << std::setprecision(6) << (elapsed.count() * 0.000001) << " sec \n";
  }
#endif
};

struct LibResources {
  std::unordered_set<char32_t> to_replace;
  FastText model;
  UnicodeConverter u_converter;

  LibResources() {
    ProfileIt p("Init");
    auto replace_begin = std::cbegin(SYMBOLS_TO_REPLACE);
    auto replace_end = std::cend(SYMBOLS_TO_REPLACE);

    to_replace.reserve(replace_end - replace_begin);
    to_replace.insert(replace_begin, replace_end);

    size_t const blob_size = std::end(fasttext_model_blob) - std::begin(fasttext_model_blob);
    std::string const model_str(fasttext_model_blob, blob_size);
    std::istringstream model_blob(model_str, std::istringstream::in | std::istringstream::binary);
    model.loadModel(model_blob);
  }
};

LibResources lib_sources;

enum TglangLanguage tglang_detect_programming_language(const char *text) {
  std::stringstream ss;
  std::string preprocessed;
  {
    ProfileIt prep("Preprocessing");

    std::u32string unicode = lib_sources.u_converter.from_bytes(text, text + std::strlen(text));

    std::u32string replaced;
    replaced.reserve(unicode.size() * 2 + 1);
    for (size_t i = 0; i < unicode.size(); i++) {
      auto c = unicode[i];
      if (c == U'\n' && i > 0 && unicode[i-1] != U'\n') {
        replaced.append(U"!$");
      } else if (lib_sources.to_replace.find(c) == lib_sources.to_replace.end()) {
        replaced.push_back(c);
      }
    }

    preprocessed = lib_sources.u_converter.to_bytes(replaced.data(), replaced.data() + replaced.size());

    // std::cerr << "Processing << `" << preprocessed << "`\n";

    ss << preprocessed.c_str();

    ss.seekg(0, ss.beg);
  }

  constexpr int32_t kCount = 1;
  constexpr fasttext::real kProbThreshold = 0.3;

  int converted;
  std::vector<std::pair<fasttext::real, std::string>> result;
  {
    ProfileIt inf("Inference");
    try {
      lib_sources.model.predictLine(ss, result, kCount, kProbThreshold);
    } catch (...) {
#ifndef NDEBUG
      std::cerr << "EXCEPTION during predict" << "\n";
#endif
      return TglangLanguage::TGLANG_LANGUAGE_OTHER;
    }

    if (result.empty()) {
#ifndef NDEBUG
      std::cerr << "No results!" << "\n";
#endif
      return TglangLanguage::TGLANG_LANGUAGE_OTHER;
    }

    auto const & res = result.front();

    // TODO: remove LABEL_PREFIX
    converted = std::atoi(res.second.c_str() + std::strlen(LABEL_PREFIX));

#ifndef NDEBUG
    std::cerr << "Fasttext, class=" << res.second << ",converted=" << converted << ",prob=" << res.first << '\n';
#endif

    assert(converted <= TglangLanguage::TGLANG_LANGUAGE_YAML);
  }

  return static_cast<TglangLanguage>(converted);
}
