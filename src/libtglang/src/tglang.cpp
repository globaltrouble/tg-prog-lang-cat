#include "tglang.h"

#include <re2/re2.h>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <algorithm>
#include <cassert>

#include "cptrie.hpp"

// inspired by `https://tratt.net/laurie/blog/2022/whats_the_most_portable_way_to_include_binary_blobs_in_an_executable.html`
// must be linked with `fasttext_model_blob.o`
extern char _binary_cptrie_id_dict_bin_start;
extern char _binary_cptrie_id_dict_bin_end;
extern char _binary_cptrie_tfidf_dict_bin_start;
extern char _binary_cptrie_tfidf_dict_bin_end;

constexpr size_t kSvmInputSize = 324632;

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
  re2::RE2 spaces { R"(([.,;:\\\/{}\[\]\|!\"#\$%&\'\(\)\*\+\-\<\=\>\?@\^\`\)]))" };
  re2::RE2 to_find { R"((\b\w+\b|[.,;:\\\/{}\[\]\|!\"#\$%&\'\(\)\*\+\-\<\=\>\?@\^\`\~)]))" };

  std::pair<re2::RE2, char const *> bin { R"(0[bB]([01])+)", "0b0" };
  std::pair<re2::RE2, char const *> oct { R"(0[oO]([0-7])+)", "0o0" };
  std::pair<re2::RE2, char const *> hex { R"(0[xX]([0-9a-fA-F])+)", "0x0" };
  std::pair<re2::RE2, char const *> exp { R"(-?\d+[eE]-?\d+)", "0e0" };
  std::pair<re2::RE2, char const *> floating { R"(0[bB]([01])+)", "0f0" };
  std::pair<re2::RE2, char const *> integer { R"(-?\d+)", "0" };

  std::string buffer;
  std::string match;
  std::vector<double> svmInput;

  LibResources() {
    ProfileIt p("Init");

    buffer.reserve(65536);
    match.reserve(2048);
    svmInput.resize(kSvmInputSize);

  }
};

// init resources on library load
LibResources lib_sources;

void preprocess_text(char const * text, std::vector<double> & out) {
  size_t const textLen = std::strlen(text);
  
  lib_sources.buffer.clear();

  // leave only ascii
  std::copy_if(text, text + textLen, std::back_inserter(lib_sources.buffer),  [](char c) { return !(c>=0 && c < 127);});

  // add spaces
  RE2::GlobalReplace(&lib_sources.buffer, lib_sources.spaces, " \1 ");

  // change_nums_to_tokens
  RE2::GlobalReplace(&lib_sources.buffer, lib_sources.bin.first, lib_sources.bin.second);
  RE2::GlobalReplace(&lib_sources.buffer, lib_sources.oct.first, lib_sources.oct.second);
  RE2::GlobalReplace(&lib_sources.buffer, lib_sources.hex.first, lib_sources.hex.second);
  RE2::GlobalReplace(&lib_sources.buffer, lib_sources.exp.first, lib_sources.exp.second);
  RE2::GlobalReplace(&lib_sources.buffer, lib_sources.floating.first, lib_sources.floating.second);
  RE2::GlobalReplace(&lib_sources.buffer, lib_sources.integer.first, lib_sources.integer.second);

  // tokenize & vectorize
  re2::StringPiece input(lib_sources.buffer);

  assert(out.size() == kSvmInputSize);
  std::fill(out.begin(), out.end(), 0.0);

  lib_sources.match.clear();

  while (RE2::FindAndConsume(&input, lib_sources.to_find, &lib_sources.match)) {
    auto id = cptrie::get(lib_sources.match.c_str(), &_binary_cptrie_id_dict_bin_start);    
    auto tfidf = cptrie::get(lib_sources.match.c_str(), &_binary_cptrie_tfidf_dict_bin_start);
    
    assert(id.has_value() == tfidf.has_value());
    
    lib_sources.match.clear();
    
    if (!id.has_value()) {
      continue;
    }

    double tfidf_val = *reinterpret_cast<double const *>(&tfidf.value());

    out[id.value()] = tfidf_val;
  }
}

enum TglangLanguage tglang_detect_programming_language(const char *text) {
  
  {
    ProfileIt prep("Preprocessing");
    preprocess_text(text, lib_sources.svmInput);
  }

  return static_cast<TglangLanguage>(0);
}
