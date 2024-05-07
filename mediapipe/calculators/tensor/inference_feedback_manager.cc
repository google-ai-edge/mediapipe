#include "mediapipe/calculators/tensor/inference_feedback_manager.h"

#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/calculators/tensor/inference_io_mapper.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/util/tflite/tflite_signature_reader.h"
#include "mediapipe/util/tflite/utils.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"

namespace mediapipe {

namespace {

bool TfLiteTensorSpecEqual(const TfLiteTensor& a, const TfLiteTensor& b) {
  return a.type == b.type && TfLiteIntArrayEqual(a.dims, b.dims) &&
         a.params.scale == b.params.scale &&
         a.params.zero_point == b.params.zero_point &&
         a.allocation_type == b.allocation_type && a.bytes == b.bytes;
}

absl::flat_hash_map<std::string, int> CreateNameToIndexMap(
    const std::vector<std::string>& names) {
  absl::flat_hash_map<std::string, int> name_to_index_map;
  for (int i = 0; i < names.size(); ++i) {
    name_to_index_map[names[i]] = i;
  }
  return name_to_index_map;
}

}  // namespace

absl::Status InferenceFeedbackManager::Init(
    const InferenceCalculatorOptions::InputOutputConfig& io_config,
    const InputOutputTensorNames& input_output_tensor_names,
    tflite::Interpreter* interpreter) {
  interpreter_ = interpreter;
  MP_ASSIGN_OR_RETURN(feedback_tensor_indices_links_,
                      ConvertSignatureTensorNamesToModelIndices(
                          io_config, input_output_tensor_names));

  for (const auto& link : feedback_tensor_indices_links_) {
    const auto [output_unused_iter, output_was_inserted] =
        feedback_output_indices_.insert(link.from_idx);
    RET_CHECK(output_was_inserted) << "Feedback output tensors must be unique.";
    TfLiteTensor* from_tensor =
        interpreter_->tensor(interpreter->outputs()[link.from_idx]);
    RET_CHECK(!util::tflite::IsDynamicTensor(*from_tensor))
        << "Feedback output tensors must not be dynamic.";
    const auto [input_unused_iter, input_was_inserted] =
        feedback_input_indices_.insert(link.to_idx);
    RET_CHECK(input_was_inserted) << "Feedback input tensors must be unique.";
    TfLiteTensor* to_tensor =
        interpreter_->tensor(interpreter->inputs()[link.to_idx]);
    RET_CHECK(!util::tflite::IsDynamicTensor(*to_tensor))
        << "Feedback input tensors must not be dynamic.";
    RET_CHECK(TfLiteTensorSpecEqual(*from_tensor, *to_tensor))
        << "Feedback tensors must have the same spec.";
    // Since the TfLite API isn't specific about the initialization of newly
    // allocated Tensor memory, we initialize the input to_tensor tensor with
    // zeros.
    memset(to_tensor->data.raw, 0, to_tensor->bytes);
  }

  // Populate input_tensor_to_model_indices_ which maps InferenceRunner input
  // tensors indices to the model input indices.
  input_tensor_to_model_indices_.reserve(interpreter_->inputs().size());
  for (int i = 0; i < interpreter_->inputs().size(); ++i) {
    if (!feedback_input_indices_.contains(i)) {
      input_tensor_to_model_indices_.push_back(i);
    }
  }
  return absl::OkStatus();
}

void InferenceFeedbackManager::SwapFeedbackTensors() {
  for (const auto& link : feedback_tensor_indices_links_) {
    TfLiteTensor* from_tensor =
        interpreter_->tensor(interpreter_->outputs()[link.from_idx]);
    TfLiteTensor* to_tensor =
        interpreter_->tensor(interpreter_->inputs()[link.to_idx]);
    {
      using std::swap;
      // TODO b/338023494 - Use TfLite CustomAllocator to manage memory of
      // feedback tensors (replace std::swap)
      swap(*from_tensor, *to_tensor);
    }
  }
}

// static
absl::StatusOr<std::vector<InferenceFeedbackManager::TensorFeedbackIndicesLink>>
InferenceFeedbackManager::ConvertSignatureTensorNamesToModelIndices(
    const InferenceCalculatorOptions::InputOutputConfig& io_config,
    const InputOutputTensorNames& input_output_tensor_names_map) {
  std::vector<TensorFeedbackIndicesLink> indices_links;
  if (input_output_tensor_names_map.empty() ||
      input_output_tensor_names_map.size() > 1) {
    // Fail gracefully by returning an empty TensorFeedbackIndicesLink list if
    // SignatureDef is not available or not supported.
    ABSL_LOG(WARNING)
        << "Feedback manager requires a model with a single signature "
           "inference. Disabling support for feedback tensors.";
    return indices_links;
  }
  // Obtain reference to single-signature in input_output_tensor_names_map.
  const auto& input_output_tensor_names =
      input_output_tensor_names_map.begin()->second;

  const auto input_name_to_index_map =
      CreateNameToIndexMap(input_output_tensor_names.input_tensor_names);
  const auto output_name_to_index_map =
      CreateNameToIndexMap(input_output_tensor_names.output_tensor_names);

  // Create a set of all input/output tensor names used for InferenceCalculator
  // I/O mapping.
  absl::flat_hash_set<std::string> input_output_mapping_tensor_names;
  for (const auto& name : io_config.input_tensor_names_map().tensor_names()) {
    input_output_mapping_tensor_names.insert(name);
  }
  for (const auto& name : io_config.output_tensor_names_map().tensor_names()) {
    input_output_mapping_tensor_names.insert(name);
  }

  for (const auto& link : io_config.feedback_tensor_links()) {
    RET_CHECK(!input_output_mapping_tensor_names.contains(
        link.from_output_tensor_name()))
        << absl::StrFormat(
               "Feedback output tensor [%s] cannot be used for input/output "
               "mapping. Input/output mapping tensor names: [%s]",
               link.from_output_tensor_name(),
               absl::StrJoin(input_output_mapping_tensor_names, ", "));
    RET_CHECK(!input_output_mapping_tensor_names.contains(
        link.to_input_tensor_name()))
        << absl::StrFormat(
               "Feedback input tensor [%s] cannot be used for input/output "
               "mapping. Input/output mapping tensor names: [%s]",
               link.to_input_tensor_name(),
               absl::StrJoin(input_output_mapping_tensor_names, ", "));
    TensorFeedbackIndicesLink indices_link;
    auto from_it =
        output_name_to_index_map.find(link.from_output_tensor_name());
    RET_CHECK(from_it != output_name_to_index_map.end())
        << "Output tensor name not found: " << link.from_output_tensor_name();
    auto to_it = input_name_to_index_map.find(link.to_input_tensor_name());
    RET_CHECK(to_it != input_name_to_index_map.end())
        << "Input tensor name not found: " << link.to_input_tensor_name();
    indices_link.from_idx = from_it->second;
    indices_link.to_idx = to_it->second;
    indices_links.push_back(indices_link);
  }
  return indices_links;
}

bool InferenceFeedbackManager::IsFeedbackInputTensorAtIndex(int idx) const {
  return feedback_input_indices_.contains(idx);
}

bool InferenceFeedbackManager::IsFeedbackOutputTensorAtIndex(int idx) const {
  return feedback_output_indices_.contains(idx);
}

absl::StatusOr<int> InferenceFeedbackManager::MapInputTensorToModelIndex(
    int input_idx) const {
  RET_CHECK(input_idx >= 0 &&
            input_idx <= input_tensor_to_model_indices_.size())
      << "Invalid input tensor index: " << input_idx;
  return input_tensor_to_model_indices_[input_idx];
}

int InferenceFeedbackManager::GetNumberOfNonFeedbackInputTensors() const {
  return input_tensor_to_model_indices_.size();
}

int InferenceFeedbackManager::GetNumberOfFeedbackTensors() const {
  return feedback_tensor_indices_links_.size();
}
}  // namespace mediapipe
