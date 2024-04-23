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

#ifndef MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_CALCULATOR_IO_MAP_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_CALCULATOR_IO_MAP_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensor_span.h"
#include "mediapipe/framework/formats/tensor.h"

namespace mediapipe {

// Verifies the correctness of the provided InferenceCalculatorIoMap. This
// verification should be applied before calling RemapInputTensors or
// RemapOutputTensors methods below.
absl::Status VerifyInputOutputConfig(
    const mediapipe::InferenceCalculatorOptions::InputOutputConfig& io_map);

// Reorders input tensors according to the provided mappings. The io_map should
// be verified using VerifyInputOutputConfig before calling this method.
absl::StatusOr<TensorSpan> RemapInputTensors(
    const TensorSpan& unmapped_tensors,
    const mediapipe::InferenceCalculatorOptions::InputOutputConfig& io_map);

// Reorders output tensors according to the provided mappings. The io_map should
// be verified using VerifyInputOutputConfig before calling this method.
absl::StatusOr<std::vector<Tensor>> RemapOutputTensors(
    std::vector<Tensor>&& unmapped_tensors,
    const mediapipe::InferenceCalculatorOptions::InputOutputConfig& io_map);

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_CALCULATOR_IO_MAP_H_
