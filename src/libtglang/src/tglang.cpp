#include "tglang.h"

#include "fasttext.h"

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
#include <thread>

#define LABEL_PREFIX "__label__"

// inspired by `https://tratt.net/laurie/blog/2022/whats_the_most_portable_way_to_include_binary_blobs_in_an_executable.html`
// must be linked with `fasttext_model_iscode_blob.o`

extern char _binary_fasttext_model_iscode_bin_start;
extern char _binary_fasttext_model_iscode_bin_end;
extern char _binary_fasttext_model_codetype_bin_start;
extern char _binary_fasttext_model_codetype_bin_end;

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
  FastText iscode_model;
  FastText codetype_model;

  re2::RE2 add_spaces { R"(([.,;:\\\/{}\[\]\|!\"#\$%&\'\(\)\*\+\-\<\=\>\?@\^\`\)]))" };
  re2::RE2 newline { R"((\n)+)" };
  re2::RE2 multisep { R"((\s)+)" };
  re2::RE2 to_find { R"((\b\w+\b|[.,;:\\\/{}\[\]\|!\"#\$%&\'\(\)\*\+\-\<\=\>\?@\^\`\~)]))" };

  std::pair<re2::RE2, char const *> bin { R"(0[bB]([01])+)", "<num_binary>" };
  std::pair<re2::RE2, char const *> oct { R"(0[oO]([0-7])+)", "<num_octal>" };
  std::pair<re2::RE2, char const *> hex { R"(0[xX]([0-9a-fA-F])+)", "<num_hex>" };
  std::pair<re2::RE2, char const *> exp { R"(-?\d+[eE]-?\d+)", "<num_exp>" };
  std::pair<re2::RE2, char const *> floating { R"(0[bB]([01])+)", "<num_float>" };
  std::pair<re2::RE2, char const *> integer { R"(-?\d+)", "<num_int>" };

  std::string preprocessed;
  std::string match;
  std::vector<std::pair<fasttext::real, std::string>> isCodeResult;
  std::vector<std::pair<fasttext::real, std::string>> codeTypeResult;

  LibResources() {
    ProfileIt p("Init");

    std::thread iscode_loading([this]() {
        size_t iscode_blob_size = &_binary_fasttext_model_iscode_bin_end - &_binary_fasttext_model_iscode_bin_start;
        std::string const iscode_model_str(&_binary_fasttext_model_iscode_bin_start, iscode_blob_size);
        std::istringstream iscode_model_blob(iscode_model_str, std::istringstream::in | std::istringstream::binary);
        iscode_model.loadModel(iscode_model_blob);
    });

    std::thread codetype_loading([this]() {
        size_t codetype_blob_size = &_binary_fasttext_model_codetype_bin_end - &_binary_fasttext_model_codetype_bin_start;
        std::string const codetype_model_str(&_binary_fasttext_model_codetype_bin_start, codetype_blob_size);
        std::istringstream codetype_model_blob(codetype_model_str, std::istringstream::in | std::istringstream::binary);
        codetype_model.loadModel(codetype_model_blob);
    });

    preprocessed.reserve(65536);
    match.reserve(2048);
    isCodeResult.reserve(2048);
    codeTypeResult.reserve(2048);

    iscode_loading.join();
    codetype_loading.join();
  }

};

LibResources lib_sources;

void preprocess_text(char const * text, size_t const textLen) {
    ProfileIt prep("Preprocessing");
    lib_sources.preprocessed.clear();

    // leave only ascii
    std::copy_if(text, text + textLen, std::back_inserter(lib_sources.preprocessed),  [](char c) { return !(c>=0 && c < 127);});
    lib_sources.preprocessed.append(text, textLen);

    // add spaces
    RE2::GlobalReplace(&lib_sources.preprocessed, lib_sources.add_spaces, R"( \1 )");
    
    // change newline (single/multi) to a single special token
    RE2::GlobalReplace(&lib_sources.preprocessed, lib_sources.newline, R"(<newline>)");

    // change_nums_to_tokens
    RE2::GlobalReplace(&lib_sources.preprocessed, lib_sources.bin.first, lib_sources.bin.second);
    RE2::GlobalReplace(&lib_sources.preprocessed, lib_sources.oct.first, lib_sources.oct.second);
    RE2::GlobalReplace(&lib_sources.preprocessed, lib_sources.hex.first, lib_sources.hex.second);
    RE2::GlobalReplace(&lib_sources.preprocessed, lib_sources.exp.first, lib_sources.exp.second);
    RE2::GlobalReplace(&lib_sources.preprocessed, lib_sources.floating.first, lib_sources.floating.second);
    RE2::GlobalReplace(&lib_sources.preprocessed, lib_sources.integer.first, lib_sources.integer.second);

    // change newline (single/multi) to a single special token
    RE2::GlobalReplace(&lib_sources.preprocessed, lib_sources.multisep, R"( )");

#ifndef NDEBUG
    std::cerr << "Preprocessed: << `" << lib_sources.preprocessed << "`\n";
#endif
}

int predict(std::stringstream & ss, FastText const & model, std::vector<std::pair<fasttext::real, std::string>> & result, int defaultVal) {
  ProfileIt inf("Inference");

  ss.seekg(0, ss.beg);
  result.clear();

  constexpr int32_t kCount = 1;
  constexpr fasttext::real kProbThreshold = 0.3;

  int converted;

  try {
    model.predictLine(ss, result, kCount, kProbThreshold);
  } catch (...) {
#ifndef NDEBUG
    std::cerr << "EXCEPTION during predict" << "\n";
#endif
    return defaultVal;
  }

  if (result.empty()) {
#ifndef NDEBUG
    std::cerr << "No results!" << "\n";
#endif
    return defaultVal;
  }

  auto const & res = result.front();

  // TODO: remove LABEL_PREFIX
  converted = std::atoi(res.second.c_str() + std::strlen(LABEL_PREFIX));

#ifndef NDEBUG
  std::cerr << "Fasttext, class=" << res.second << ",converted=" << converted << ",prob=" << res.first << '\n';
#endif

  return converted;
}

enum TglangLanguage tglang_detect_programming_language(const char *text) {
  ProfileIt inf("Detecttotal");

  size_t const textLen = std::strlen(text);
  preprocess_text(text, textLen);
  std::stringstream ss(lib_sources.preprocessed.c_str());

  int is_code = predict(ss, lib_sources.iscode_model, lib_sources.isCodeResult, 0);
  assert(is_code >= 0 && is_code <= 1);
  if (!is_code) {
    return TglangLanguage::TGLANG_LANGUAGE_OTHER;
  }

  int code_type = predict(ss, lib_sources.codetype_model, lib_sources.codeTypeResult, TglangLanguage::TGLANG_LANGUAGE_OTHER);
  assert(code_type >= 0 && code_type <= TglangLanguage::TGLANG_LANGUAGE_XML);

  if (code_type == TglangLanguage::TGLANG_LANGUAGE_C 
    || code_type == TglangLanguage::TGLANG_LANGUAGE_CPLUSPLUS
    || code_type == TglangLanguage::TGLANG_LANGUAGE_OBJECTIVE_C) {
    // TODO: add C vs C++ vs ObjC
  }
  else if (code_type == TglangLanguage::TGLANG_LANGUAGE_JAVASCRIPT 
    || code_type == TglangLanguage::TGLANG_LANGUAGE_TYPESCRIPT) {
    // TODO: add JS vs TS
  }

  return static_cast<TglangLanguage>(code_type);
}