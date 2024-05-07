// Copyright 2024 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/calculators/tensor/inference_io_mapper.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensor_span.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/util/tflite/tflite_signature_reader.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

namespace mediapipe {

namespace {

using ::tflite::FlatBufferModel;
using ::tflite::Interpreter;
using ::tflite::InterpreterBuilder;
using ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates;

// Checks for duplicate indices in a TensorIndicesMap.
absl::StatusOr<std::vector<int>> GenerateAndValidateTensorList(
    const InferenceCalculatorOptions::InputOutputConfig::TensorIndicesMap&
        tensor_indices_list) {
  absl::flat_hash_set<int> indices_set;
  std::vector<int> result;
  for (const int index : tensor_indices_list.model_tensor_indices()) {
    RET_CHECK(indices_set.insert(index).second)
        << "Indices in TensorIndicesMap are not unique.";
    result.push_back(index);
  }
  return result;
}

absl::StatusOr<absl::flat_hash_map<std::string, int>> CreateNameToIndexMap(
    const std::vector<std::string>& names) {
  absl::flat_hash_map<std::string, int> name_to_index_map;
  for (int i = 0; i < names.size(); ++i) {
    auto [unused_iter, was_inserted] = name_to_index_map.insert({names[i], i});
    RET_CHECK(was_inserted)
        << "Duplicate tensor names found in model signatures: "
        << absl::StrJoin(names, ", ");
  }
  return name_to_index_map;
}

template <typename T>
static bool ContainsDuplicates(const std::vector<T>& input) {
  absl::flat_hash_set<T> set;
  for (const auto& item : input) {
    if (!set.insert(item).second) {
      return true;
    }
  }
  return false;
}

static absl::StatusOr<std::vector<int>> MapTensorNamesToIndices(
    const std::vector<std::string>& signature_tensor_names,
    const InferenceCalculatorOptions::InputOutputConfig::TensorNamesMap&
        config_tensor_names) {
  std::vector<int> result;
  result.reserve(signature_tensor_names.size());
  MP_ASSIGN_OR_RETURN(const auto input_name_to_index_map,
                      CreateNameToIndexMap(signature_tensor_names));
  for (const auto& tensor_name : config_tensor_names.tensor_names()) {
    const auto it = input_name_to_index_map.find(tensor_name);
    RET_CHECK(it != input_name_to_index_map.end())
        << "Tensor name " << tensor_name
        << " not found in model signatures. Model tensor names: "
        << absl::StrJoin(signature_tensor_names, ", ");
    result.push_back(it->second);
  }
  RET_CHECK(!ContainsDuplicates(result))
      << "Duplicate tensor names found in TensorNamesMap: "
      << absl::StrJoin(config_tensor_names.tensor_names(), ", ");
  return result;
};

// Feedback tensors are excluded from the InferenceRunner input and output
// accordingly (since they are class-internally handled by the
// InferenceFeedbackManager). This means that the input and output Tensor orders
// of the InferenceRunner don't match the model I/O tensors anymore and
// therefore tensor I/O indices need to be adjusted accordingly.
absl::Status ExcludeFeedbackTensorsFromRemappingIndicesVector(
    const InferenceCalculatorOptions::InputOutputConfig& io_config,
    const std::vector<std::string>& model_tensor_names,
    std::vector<int>& remapping_tensor_indices) {
  // Create set of all feedback tensor names.
  absl::flat_hash_set<std::string> feedback_tensor_names;
  for (const auto& link : io_config.feedback_tensor_links()) {
    {
      // No need to check for name collisions. Inference feedback manager
      // confirms validity of feedback tensor names.
      feedback_tensor_names.insert(link.from_output_tensor_name());
      feedback_tensor_names.insert(link.to_input_tensor_name());
    }
  }
  // Built model index translation vector which maps InferenceRunner I/O tensor
  // indices to InferenceRunner I/O indices with excluded feedback tensors.
  std::vector<int> indices_translation(model_tensor_names.size(), -1);
  int model_output_idx = 0;
  for (int i = 0; i < model_tensor_names.size(); ++i) {
    if (!feedback_tensor_names.contains(model_tensor_names[i])) {
      indices_translation[i] = model_output_idx;
      ++model_output_idx;
    }
  }
  // Adjust remapping_tensor_indices.
  for (int i = 0; i < remapping_tensor_indices.size(); ++i) {
    const int model_index = remapping_tensor_indices[i];
    RET_CHECK(model_index >= 0 && model_index < indices_translation.size())
        << "Index " << model_index << " out of range.";
    remapping_tensor_indices[i] =
        indices_translation[remapping_tensor_indices[i]];
  }
  return absl::OkStatus();
}

}  // namespace

// static
absl::StatusOr<InputOutputTensorNames>
InferenceIoMapper::GetInputOutputTensorNamesFromInterpreter(
    const tflite::Interpreter& interpreter) {
  auto input_output_tensor_names =
      TfLiteSignatureReader::GetInputOutputTensorNamesFromAllTfliteSignatures(
          interpreter);
  if (!input_output_tensor_names.ok()) {
    // TODO b/336260063 - remove this warning once the bug is fixed.
    ABSL_LOG_FIRST_N(WARNING, 1)
        << "Unable to extract TfLite model's tensor names from "
           "TfliteSignature. Disabling tensor name-based I/O mapping.";
    return InputOutputTensorNames();
  }
  return *input_output_tensor_names;
}

// static
absl::StatusOr<InputOutputTensorNames>
InferenceIoMapper::GetInputOutputTensorNamesFromModel(
    const tflite::FlatBufferModel& flatbuffer,
    const tflite::OpResolver& op_resolver) {
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder interpreter_builder(flatbuffer, op_resolver);
  if (interpreter_builder(&interpreter) != kTfLiteOk || !interpreter) {
    ABSL_LOG_EVERY_N(WARNING, 1)
        << "Extracting input output tensor names from TfliteSignature failed: "
           "Unable to prepare interpreter. Ignoring tensor name-based I/O "
           "mapping.";
    return InputOutputTensorNames();
  }
  return GetInputOutputTensorNamesFromInterpreter(*interpreter);
}

absl::Status InferenceIoMapper::UpdateIoMap(
    const InferenceCalculatorOptions::InputOutputConfig& io_config,
    const InputOutputTensorNames& input_output_tensor_names) {
  num_feedback_tensors_ = io_config.feedback_tensor_links().size();

  if ((io_config.has_input_tensor_indices_map() ||
       io_config.has_output_tensor_indices_map()) &&
      num_feedback_tensors_ > 0) {
    // TODO b/336767692 - remove this check once indices-based feedback
    // tensors are supported.
    return absl::FailedPreconditionError(
        "Feedback tensors are not supported with tensor index-based I/O "
        "mapping.");
  }

  input_tensor_indices_.clear();
  output_tensor_indices_.clear();

  if (io_config.has_input_tensor_indices_map()) {
    input_tensor_indices_.reserve(
        io_config.input_tensor_indices_map().model_tensor_indices().size());
    MP_ASSIGN_OR_RETURN(
        input_tensor_indices_,
        GenerateAndValidateTensorList(io_config.input_tensor_indices_map()));
  }

  if (io_config.has_output_tensor_indices_map()) {
    output_tensor_indices_.reserve(
        io_config.output_tensor_indices_map().model_tensor_indices().size());
    MP_ASSIGN_OR_RETURN(
        output_tensor_indices_,
        GenerateAndValidateTensorList(io_config.output_tensor_indices_map()));
  }

  if (!io_config.has_input_tensor_names_map() &&
      !io_config.has_output_tensor_names_map()) {
    // No tensor name mapping is provided.
    return absl::OkStatus();
  }

  if (input_output_tensor_names.empty()) {
    return absl::FailedPreconditionError(
        "Tensor name-based mapping requires a model with one signature.");
  }

  if (input_output_tensor_names.size() > 1) {
    return absl::FailedPreconditionError(
        "Tensor name-based mapping is not supported with multi-signature "
        "models.");
  }

  // Use tensor names of default signature.
  const auto input_output_tensor_names_default_signature =
      input_output_tensor_names.begin()->second;

  if (io_config.has_input_tensor_names_map()) {
    // Read number of model inputs directly from the signature.
    const int num_model_input_tensors =
        input_output_tensor_names_default_signature.input_tensor_names.size();
    input_tensor_indices_.reserve(
        io_config.input_tensor_names_map().tensor_names().size());
    MP_ASSIGN_OR_RETURN(
        input_tensor_indices_,
        MapTensorNamesToIndices(
            input_output_tensor_names_default_signature.input_tensor_names,
            io_config.input_tensor_names_map()));
    if (num_feedback_tensors_ > 0) {
      MP_RETURN_IF_ERROR(ExcludeFeedbackTensorsFromRemappingIndicesVector(
          io_config,
          input_output_tensor_names_default_signature.input_tensor_names,
          input_tensor_indices_));
    }
    // Feedback tensors are excluded from the input_tensor_indices_.
    RET_CHECK_EQ(input_tensor_indices_.size() + num_feedback_tensors_,
                 num_model_input_tensors)
        << "Unexpected number of input tensors.";
  }

  if (io_config.has_output_tensor_names_map()) {
    const int num_model_output_tensors =
        input_output_tensor_names_default_signature.output_tensor_names.size();
    output_tensor_indices_.reserve(num_model_output_tensors);
    MP_ASSIGN_OR_RETURN(
        output_tensor_indices_,
        MapTensorNamesToIndices(
            input_output_tensor_names_default_signature.output_tensor_names,
            io_config.output_tensor_names_map()));
    if (num_feedback_tensors_ > 0) {
      MP_RETURN_IF_ERROR(ExcludeFeedbackTensorsFromRemappingIndicesVector(
          io_config,
          input_output_tensor_names_default_signature.output_tensor_names,
          output_tensor_indices_));
    }
    // Feedback tensors are excluded from the output_tensor_indices_.
    RET_CHECK_EQ(output_tensor_indices_.size() + num_feedback_tensors_,
                 num_model_output_tensors)
        << "Unexpected number of output tensors.";
  }
  return absl::OkStatus();
}

absl::StatusOr<TensorSpan> InferenceIoMapper::RemapInputTensors(
    const TensorSpan& unmapped_tensors) {
  if (input_tensor_indices_.empty()) {
    return unmapped_tensors;
  }
  RET_CHECK_EQ(unmapped_tensors.size(), input_tensor_indices_.size())
      << "Unexpected number of input tensors.";
  std::vector<const Tensor*> mapped_tensors(unmapped_tensors.size());
  for (int i = 0; i < unmapped_tensors.size(); ++i) {
    const int index = input_tensor_indices_[i];
    RET_CHECK(index < unmapped_tensors.size())
        << "Index " << index << " out of range"
        << ". Size of TensorIndicesMap: " << unmapped_tensors.size() << ".";
    mapped_tensors[index] = &unmapped_tensors[i];
  }
  return TensorSpan(std::move(mapped_tensors));
}

absl::StatusOr<std::vector<Tensor>> InferenceIoMapper::RemapOutputTensors(
    std::vector<Tensor>&& unmapped_tensors) {
  if (output_tensor_indices_.empty()) {
    return std::move(unmapped_tensors);
  }
  RET_CHECK_EQ(unmapped_tensors.size(), output_tensor_indices_.size())
      << "Unexpected number of output tensors.";
  std::vector<Tensor> mapped_tensors;
  mapped_tensors.reserve(unmapped_tensors.size());
  for (int i = 0; i < unmapped_tensors.size(); ++i) {
    const int index = output_tensor_indices_[i];
    RET_CHECK(index < unmapped_tensors.size())
        << "Index " << index << " out of range"
        << ". Size of TensorIndicesMap: " << unmapped_tensors.size() << ".";

    mapped_tensors.emplace_back(std::move(unmapped_tensors[index]));
  }
  return mapped_tensors;
}
}  // namespace mediapipe
