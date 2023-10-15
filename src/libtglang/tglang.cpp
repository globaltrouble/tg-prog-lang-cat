#include "tglang.h"
#include "tfidf_mapping.h"

#include "onnxruntime_cxx_api.h"

#include <re2/re2.h>

#include <cstdlib>
#include <cstring>
#include <atomic>
#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstring>
#include <memory>
#include <chrono>
#include <optional>
#include <algorithm>
#include <sstream>
#include <iomanip>

struct ProfileIt {
#ifdef NDEBUG
  ProfileIt(char const * const) {};
#else
  char const * const m_name = nullptr;
  std::chrono::steady_clock::time_point m_begin;
  
  ProfileIt(char const * const name) : m_name(name), m_begin(std::chrono::steady_clock::now()) {}

  ~ProfileIt() {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - m_begin);
    std::cerr << m_name << ": " << std::fixed << std::setprecision(6) << (elapsed.count() * 0.000001) << " sec \n";
  }
#endif
};

struct LibResources {
  const char* SPECIAL_SYMBOLS_REGEX = R"([.,;:\\\/{}\[\]\|!\"#\$%&\'\(\)\*\+\-\<\=\>\?@\^\`\~)])";
  const char* SPECIAL_SYMBOLS_REGEX_2 = R"(\b\w+\b|[.,;:\\\/{}\[\]\|!\"#\$%&\'\(\)\*\+\-\<\=\>\?@\^\`\~)])";

  RE2 to_find = SPECIAL_SYMBOLS_REGEX_2;
  RE2 spaces = SPECIAL_SYMBOLS_REGEX;
  RE2 newline = "\n";
  RE2 tab = "\t";
  
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<const char*> input_names_char;
  std::vector<const char*> output_names_char;
  std::vector<std::int64_t> input_shape;

  std::unordered_map<std::string, std::pair<float, int64_t>> tfidf_mapping;

  Ort::Env env;
  const char * model_file = "./resources/svm_model_best.onnx";
  Ort::SessionOptions session_options;
  std::optional<Ort::Session> session;

  LibResources() {
    ProfileIt p("Init");
    
    tfidf_mapping.reserve(std::end(raw_itfidf_mapping) - std::begin(raw_itfidf_mapping));
    tfidf_mapping.insert(std::begin(raw_itfidf_mapping), std::end(raw_itfidf_mapping));

    session = {Ort::Session(env, model_file, session_options)};
    // size_t const inputs_count = session->GetInputCount();
    // assert(inputs_count == 1);
    input_shape = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    // some models might have negative shape values to indicate dynamic shape, e.g., for variable batch size.
    for (auto& s : input_shape) {
      if (s < 0) {
        s = 1;
      }
    }

    Ort::AllocatorWithDefaultOptions allocator;
    input_names.emplace_back(session->GetInputNameAllocated(0, allocator).get());

    for (std::size_t i = 0; i < session->GetOutputCount(); i++) {
      output_names.emplace_back(session->GetOutputNameAllocated(i, allocator).get());
    }

    input_names_char.resize(input_names.size(), nullptr);
    output_names_char.resize(output_names.size(), nullptr);

    std::transform(std::begin(input_names), std::end(input_names), std::begin(input_names_char),
                  [&](const std::string& str) { return str.c_str(); });

    std::transform(std::begin(output_names), std::end(output_names), std::begin(output_names_char),
                  [&](const std::string& str) { return str.c_str(); });
  }
};

LibResources lib_sources;

template <typename T>
Ort::Value vec_to_tensor(std::vector<T> & data, std::vector<std::int64_t> const & shape) {
  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<T>(mem_info, data.data(), data.size(), shape.data(), shape.size());
  return tensor;
}

std::string print_shape(const std::vector<std::int64_t>& v) {
  if (v.empty()) return {};

  std::stringstream ss;
  for (std::size_t i = 0; i < v.size() - 1; i++) ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

void add_spaces(std::string & s) {
  RE2::GlobalReplace(&s, lib_sources.spaces, " \1 ");
}

std::vector<std::string> tokenize(std::string & s) {
  std::transform(s.begin(), s.end(), s.begin(),
    [](unsigned char c){ return std::tolower(c); });

    std::vector<std::string> result;

    re2::StringPiece input(s);
    std::string match;
    while (RE2::FindAndConsume(&input, lib_sources.to_find, &match)) {
        result.push_back(match);
    }

    return result;
}

// void add_special_symbols(std::string & s) {
//   RE2::GlobalReplace(&s, lib_sources.newline, " __newline__ ");
//   RE2::GlobalReplace(&s, lib_sources.tab, " __tab__ ");
// }

std::vector<float> generate_vector(std::vector<std::string> const & words) {
  std::vector<float> result(lib_sources.tfidf_mapping.size(), 0.0);
  for (auto const & word : words) {
    auto it = lib_sources.tfidf_mapping.find(word);
    if (it == lib_sources.tfidf_mapping.end()) {
      continue;
    }

    float tf_idf_val = it->second.first;
    int64_t idx = it->second.second;
    result[idx] += tf_idf_val;
  }

  return result;
}

enum TglangLanguage tglang_detect_programming_language(const char *text) {
  std::string s(text);
  add_spaces(s);
  // add_special_symbols(s);
  std::vector<std::string> input_string_list = tokenize(s);

  std::vector<float> input_tensor_values = generate_vector(input_string_list);
  // print name/shape of inputs
  // generate random numbers in the range [0, 255]
  // std::vector<float> input_tensor_values(total_number_elements);
  // std::generate(input_tensor_values.begin(), input_tensor_values.end(), [&] { return rand() % 255; });
  std::vector<Ort::Value> input_tensors;
  input_tensors.emplace_back(vec_to_tensor<float>(input_tensor_values, lib_sources.input_shape));

  std::cerr << "Running model..." << std::endl;
  try {
    auto output_tensors = lib_sources.session->Run(Ort::RunOptions{nullptr}, 
                                                  lib_sources.input_names_char.data(),
                                                  input_tensors.data(),
                                                  lib_sources.input_names_char.size(),
                                                  lib_sources.output_names_char.data(), 
                                                  lib_sources.output_names_char.size());
    std::cerr << "Done!" << std::endl;

    // double-check the dimensions of the output tensors
    // NOTE: the number of output tensors is equal to the number of output nodes specifed in the Run() call
    assert(output_tensors.size() == lib_sources.output_names.size() && output_tensors[0].IsTensor());
    // assert(output_tensors.size() == 2);
    auto & out1 = output_tensors[0];
    out1.
    // auto & out2 = output_tensors[1];
    // std::cerr << "Out1 has value: " << out1.HasValue() << std::endl;
    // std::cerr << "Out2 has value: " << out2.HasValue() << std::endl;
    // auto out1_shape_info = out1.GetTensorTypeAndShapeInfo();
    // auto out1_shape = out1_shape_info.GetShape();
    // size_t tag_len = out1.GetStringTensorElementLength(0);
    // std::cerr << "Out1 shape: " << print_shape(out1_shape) << ", elem length:" << tag_len << std::endl;
    // std::string result(tag_len, '\0');
    // out1.GetStringTensorElement(result.size(), 0, result.data());
    // std::cerr << "Out1 value: `" << result << "`" << std::endl;
  } catch (std::exception const & exception) {
#ifndef NDEBUG
      std::cerr << "ERROR running model inference: " << exception.what() << std::endl;
#endif
    return TGLANG_LANGUAGE_OTHER;
  }

  return TGLANG_LANGUAGE_OTHER;
}
