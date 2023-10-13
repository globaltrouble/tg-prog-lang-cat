#include "tglang.h"

#include "onnxruntime_cxx_api.h"

// Define these to print extra informational output and warnings.
#define MLPACK_PRINT_INFO
#define MLPACK_PRINT_WARN

#include <mlpack.hpp>

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
    std::cerr << m_name << ':' << std::fixed << std::setprecision(6) << (elapsed.count() * 0.000001) << " sec \n";
  }
};

std::atomic<bool> wasInit = { false };

// std::vector<std::vector<float>> inputs;

// std::vector<std::string> input_names;
// std::vector<std::string> output_names;
// std::vector<const char*> input_names_char;
// std::vector<const char*> output_names_char;
// std::vector<std::int64_t> input_shape;

// const char * model_file = "./resources/model.onnx";
// Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "example-model-explorer");
// Ort::SessionOptions session_options;
// std::optional<Ort::Session> session;

// // constexpr size_t rows = 200;
// constexpr size_t rows = 10;
// constexpr size_t cols = 764896;

// std::string print_shape(const std::vector<std::int64_t>& v) {
//   if (v.empty()) return {};

//   std::stringstream ss;
//   for (std::size_t i = 0; i < v.size() - 1; i++) ss << v[i] << "x";
//   ss << v[v.size() - 1];
//   return ss.str();
// }

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
  // {
    auto p = ProfileIt("Init session");

  //   session = {Ort::Session(env, model_file, session_options)};
  //   size_t const inputs_count = session->GetInputCount();
  //   assert(inputs_count == 1);
  //   input_shape = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  //   // some models might have negative shape values to indicate dynamic shape, e.g., for variable batch size.
  //   for (auto& s : input_shape) {
  //     if (s < 0) {
  //       s = 1;
  //     }
  //   }

  //   Ort::AllocatorWithDefaultOptions allocator;
  //   input_names.emplace_back(session->GetInputNameAllocated(0, allocator).get());

  //   for (std::size_t i = 0; i < session->GetOutputCount(); i++) {
  //     output_names.emplace_back(session->GetOutputNameAllocated(i, allocator).get());
  //   }

  //   input_names_char.resize(input_names.size(), nullptr);
  //   output_names_char.resize(output_names.size(), nullptr);

  //   std::transform(std::begin(input_names), std::end(input_names), std::begin(input_names_char),
  //                 [&](const std::string& str) { return str.c_str(); });

  //   std::transform(std::begin(output_names), std::end(output_names), std::begin(output_names_char),
  //                 [&](const std::string& str) { return str.c_str(); });
  // }

  // Ort::AllocatorWithDefaultOptions allocator;
  // for (std::size_t i = 0; i < session->GetInputCount(); i++) {
  //   auto name = session->GetInputNameAllocated(i, allocator).get();
  //   auto tp_info = session->GetInputTypeInfo(i);
  //   char name2[1024] = {};
  //   session->GetStringTensorElement(1024, i, name2);
  //   // std::cout << "\t" << name << " : " << print_shape(shape) << "\t" << std::endl;
  //   std::cout << "\t" << name << " : " << name2 << "\t" << std::endl;
  // }

  std::cout << "FINISHED!!!" << "\n";
}

// template <typename T>
// Ort::Value vec_to_tensor(std::vector<T> & data, std::vector<std::int64_t> const & shape) {
//   Ort::MemoryInfo mem_info =
//       Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
//   auto tensor = Ort::Value::CreateTensor<T>(mem_info, data.data(), data.size(), shape.data(), shape.size());
//   return tensor;
// }

// void predict(std::vector<float> & input_tensor_values) {
//   // print name/shape of inputs
//   // generate random numbers in the range [0, 255]
//   // std::vector<float> input_tensor_values(total_number_elements);
//   // std::generate(input_tensor_values.begin(), input_tensor_values.end(), [&] { return rand() % 255; });
//   std::vector<Ort::Value> input_tensors;
//   input_tensors.emplace_back(vec_to_tensor<float>(input_tensor_values, input_shape));
// 
//   std::cout << "Running model..." << std::endl;
//   try {
//     auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(),
//                                       input_names_char.size(), output_names_char.data(), output_names_char.size());
//     std::cout << "Done!" << std::endl;
// 
//     // double-check the dimensions of the output tensors
//     // NOTE: the number of output tensors is equal to the number of output nodes specifed in the Run() call
//     assert(output_tensors.size() == output_names.size() && output_tensors[0].IsTensor());
//     assert(output_tensors.size() == 2);
//     auto & out1 = output_tensors[0];
//     auto & out2 = output_tensors[1];
//     std::cout << "Out1 has value: " << out1.HasValue() << std::endl;
//     std::cout << "Out2 has value: " << out2.HasValue() << std::endl;
//     auto out1_shape_info = out1.GetTensorTypeAndShapeInfo();
//     auto out1_shape = out1_shape_info.GetShape();
//     size_t tag_len = out1.GetStringTensorElementLength(0);
//     std::cout << "Out1 shape: " << print_shape(out1_shape) << ", elem length:" << tag_len << std::endl;
//     std::string result(tag_len, '\0');
//     out1.GetStringTensorElement(result.size(), 0, result.data());
//     std::cout << "Out1 value: `" << result << "`" << std::endl;
//   } catch (const Ort::Exception& exception) {
//       std::cout << "ERROR running model inference: " << exception.what() << std::endl;
//     exit(-1);
//   }
// }

int test_mlpack() {
  // next dep packages are required:
  // apt install -y libarmadillo9 libarmadillo-dev libensmallen-dev libcereal-dev

  // before use install mlpack
  // git@github.com:mlpack/mlpack.git
  // cd mlpack && mkdir build && cd build && cmake -DDEBUG=ON .. && sudo cmake --build . --target install -j 10
  
  
  using namespace arma;
  using namespace mlpack;
  using namespace mlpack::tree;

  // The first step is about loading the dataset. Different dataset file formats are supported, but here we load a CSV dataset, and we assume the labels don't require normalization.
  // Note: make sure you update the path to your dataset file. For this sample, you can simply copy mlpack/tests/data/german.csv and paste into a new data folder in your project directory.
  mat dataset;
  bool loaded = mlpack::data::Load("/home/sun/Downloads/german.csv", dataset);
  if (!loaded) {
    return -1;
  }

  // Then we need to extract the labels from the last dimension of the dataset and remove the labels from the training set:
  Row<size_t> labels;
  labels = conv_to<Row<size_t>>::from(dataset.row(dataset.n_rows - 1));
  dataset.shed_row(dataset.n_rows - 1);


  const size_t numClasses = 2;
  const size_t minimumLeafSize = 5;
  const size_t numTrees = 10;

  // This app will use a Random Forest classifier. At first we define the classifier parameters and then we create the classifier to train it.
  RandomForest<GiniGain, RandomDimensionSelect> rf;

  {
    auto p = ProfileIt("Train RandomForest");
    rf = RandomForest<GiniGain, RandomDimensionSelect>(dataset, labels,
        numClasses, numTrees, minimumLeafSize);
  }
  // Now that the training is completed, we quickly compute the training accuracy:

  Row<size_t> predictions;
  {
    auto p = ProfileIt("Eval RandomForest");
    rf.Classify(dataset, predictions);
  }
  const size_t correct = arma::accu(predictions == labels);
  cout << "\nTraining Accuracy: " << (double(correct) / double(labels.n_elem));

  // Now that our model is trained and validated, we save it to a file so we can use it later. Here we save the model that was trained using the entire dataset. Alternatively, we could extract the model from the cross-validation stage by using cv.Model().
  bool fatal = false;
  {
    auto p = ProfileIt("Save RandomForest");
    mlpack::data::Save("mymodel.xml", "model", rf, fatal);
  }

  // In a real-life application, you may want to load a previously trained model to classify new samples. We load the model from a file using:
  {
    auto p = ProfileIt("Load RandomForest");
    mlpack::data::Load("mymodel.xml", "model", rf);
  }

  // Create a test sample containing only one point.  Because Armadillo is
  // column-major, this matrix has one column (one point) and the number of rows
  // is equal to the dimensionality of the point (23).
  {
    mat sample("2; 12; 2; 13; 1; 2; 2; 1; 3; 24; 3; 1; 1; 1; 1; 1; 0; 1; 0; 1; 0; 0; 0");
    Row<size_t> predictions;
    mat probabilities;
    {
      auto p = ProfileIt("Eval RandomForest single sample from loaded");
      rf.Classify(sample, predictions, probabilities);
    }
    u64 result = predictions.back();
    cout << "\nClassification result: " << result << " , Probabilities: " <<
        probabilities.at(0) << "/" << probabilities.at(1);
  }
  return 0;
}

}

enum TglangLanguage tglang_detect_programming_language(const char *text) {
  init();

  test_mlpack();

  // predict(inputs[0]);
  
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
