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

#include "mediapipe/calculators/tensor/inference_calculator_io_map.h"

#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensor_span.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"

namespace mediapipe {

namespace {

// Checks for duplicate model_tensor_indices in a TensorIndicesMap.
absl::Status ValidateTensorList(
    const InferenceCalculatorOptions::InputOutputConfig::TensorIndicesMap&
        io_map) {
  absl::flat_hash_set<int> indices_set;
  for (const int index : io_map.model_tensor_indices()) {
    RET_CHECK(!indices_set.contains(index))
        << "Indices in TensorIndicesMap are not unique.";
    indices_set.insert(index);
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status VerifyInputOutputConfig(
    const InferenceCalculatorOptions::InputOutputConfig& io_map) {
  MP_RETURN_IF_ERROR(ValidateTensorList(io_map.input_tensor_indices_map()));
  MP_RETURN_IF_ERROR(ValidateTensorList(io_map.output_tensor_indices_map()));
  return absl::OkStatus();
}

absl::StatusOr<TensorSpan> RemapInputTensors(
    const TensorSpan& unmapped_tensors,
    const InferenceCalculatorOptions::InputOutputConfig& io_map) {
  if (!io_map.has_input_tensor_indices_map()) {
    return unmapped_tensors;
  }
  const auto& tensor_map_indices =
      io_map.input_tensor_indices_map().model_tensor_indices();
  RET_CHECK_EQ(unmapped_tensors.size(), tensor_map_indices.size())
      << "Number of input tensors does not match the size of "
         "model_tensor_indices list in the provided mapping.";
  std::vector<const Tensor*> mapped_tensors(unmapped_tensors.size());
  for (int i = 0; i < unmapped_tensors.size(); ++i) {
    const int index = tensor_map_indices[i];
    RET_CHECK(index < unmapped_tensors.size())
        << "Index " << index << " out of range"
        << ". Size of TensorIndicesMap: " << unmapped_tensors.size() << ".";
    mapped_tensors[index] = &unmapped_tensors[i];
  }
  return TensorSpan(std::move(mapped_tensors));
}

absl::StatusOr<std::vector<Tensor>> RemapOutputTensors(
    std::vector<Tensor>&& unmapped_tensors,
    const InferenceCalculatorOptions::InputOutputConfig& io_map) {
  if (!io_map.has_output_tensor_indices_map()) {
    return std::move(unmapped_tensors);
  }
  const auto& tensor_map_indices =
      io_map.output_tensor_indices_map().model_tensor_indices();
  RET_CHECK_EQ(unmapped_tensors.size(), tensor_map_indices.size())
      << "Number of output tensors does not match the size of "
         "model_tensor_indices list in the provided mapping.";
  std::vector<Tensor> mapped_tensors;
  mapped_tensors.reserve(unmapped_tensors.size());
  for (int i = 0; i < unmapped_tensors.size(); ++i) {
    const int index = tensor_map_indices[i];
    RET_CHECK(index < unmapped_tensors.size())
        << "Index " << index << " out of range"
        << ". Size of TensorIndicesMap: " << unmapped_tensors.size() << ".";
    mapped_tensors.emplace_back(std::move(unmapped_tensors[index]));
  }
  return mapped_tensors;
}

}  // namespace mediapipe
