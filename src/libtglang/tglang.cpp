#include "tglang.h"

#include "onnxruntime_cxx_api.h"

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
std::vector<std::vector<float>> inputs;

std::vector<std::string> input_names;
std::vector<std::string> output_names;
std::vector<const char*> input_names_char;
std::vector<const char*> output_names_char;
std::vector<std::int64_t> input_shape;

const char * model_file = "/home/sun/Downloads/tg_challenge/mnb-1.onnx";
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "example-model-explorer");
Ort::SessionOptions session_options;
std::optional<Ort::Session> session;

// constexpr size_t rows = 200;
constexpr size_t rows = 10;
constexpr size_t cols = 764896;
 
void init() {
  bool expected = false;
  bool alreadyInit = !wasInit.compare_exchange_strong(expected, 
                                                      true,
                                                      std::memory_order_release,
                                                      std::memory_order_relaxed);
  if (alreadyInit) {
    return;
  }

  // TODO: wait till init for threadsafe use!

  const char * inptPath = "/home/sun/Downloads/tg_challenge/inputs.csv";
  inputs.reserve(rows);

  std::ifstream in(inptPath);
  std::cin.rdbuf(in.rdbuf());

  {
    auto profiler = ProfileIt("Parsing inputs");

    for (size_t r = 0; r < rows; r++) {
      std::vector<float> row(cols, 0);
      for (size_t c = 0; c < cols; c++) {
        std::cin >> row[c];
        
        char del;
        std::cin >> del;
        // char expectedDel = c < cols-1 ? ',' : '\n';
        const char expectedDel = ',';
        if (c < cols - 1 && del != expectedDel) {
          std::cerr << "Err wrong delimiter, expected: `" << expectedDel << "`, got: `" << del 
                    << "`, at row:" << r << ", col:" << c << std::endl;
          assert(false);
        }
      }
      inputs.push_back(std::move(row));
      if (r % 10 == 9) {
        std::cout << "Parsed " << r+1 <<" inputs, from" << rows << "\n";
      }
    }
  }
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

void predict(std::vector<float> & input_tensor_values) {
  // print name/shape of inputs
  // generate random numbers in the range [0, 255]
  // std::vector<float> input_tensor_values(total_number_elements);
  // std::generate(input_tensor_values.begin(), input_tensor_values.end(), [&] { return rand() % 255; });
  std::vector<Ort::Value> input_tensors;
  input_tensors.emplace_back(vec_to_tensor<float>(input_tensor_values, input_shape));

  std::cout << "Running model..." << std::endl;
  try {
    auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(),
                                      input_names_char.size(), output_names_char.data(), output_names_char.size());
    std::cout << "Done!" << std::endl;

    // double-check the dimensions of the output tensors
    // NOTE: the number of output tensors is equal to the number of output nodes specifed in the Run() call
    assert(output_tensors.size() == output_names.size() && output_tensors[0].IsTensor());
    assert(output_tensors.size() == 2);
    auto & out1 = output_tensors[0];
    auto & out2 = output_tensors[1];
    std::cout << "Out1 has value: " << out1.HasValue() << std::endl;
    std::cout << "Out2 has value: " << out2.HasValue() << std::endl;
    auto out1_shape_info = out1.GetTensorTypeAndShapeInfo();
    auto out1_shape = out1_shape_info.GetShape();
    size_t tag_len = out1.GetStringTensorElementLength(0);
    std::cout << "Out1 shape: " << print_shape(out1_shape) << ", elem length:" << tag_len << std::endl;
    std::string result(tag_len, '\0');
    out1.GetStringTensorElement(result.size(), 0, result.data());
    std::cout << "Out1 value: `" << result << "`" << std::endl;
  } catch (const Ort::Exception& exception) {
      std::cout << "ERROR running model inference: " << exception.what() << std::endl;
    exit(-1);
  }
}

}

enum TglangLanguage tglang_detect_programming_language(const char *text) {
  init();

  predict(inputs[0]);
  
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
