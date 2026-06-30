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

#ifndef MEDIAPIPE_CALCULATORS_TENSOR_LITERT_LITERT_UTILS_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_LITERT_LITERT_UTILS_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/memory_manager.h"
namespace mediapipe {

// Returns absl::OkStatus() if the tensor specs are equal, otherwise returns
// an error with a message describing the difference.
absl::Status AreTensorSpecsEqual(
    const litert::RankedTensorType& litert_tensor_type,
    const mediapipe::Tensor& mp_tensor);

// Similar to AreTensorSpecsEqual but allows for dynamic dimensions.
// Returns OK if shapes are compatible (considering -1 as dynamic) and types
// match.
absl::Status AreTensorSpecsCompatible(
    const litert::RankedTensorType& litert_tensor_type,
    const mediapipe::Tensor& mp_tensor);

// Checks if the tensor shape is compatible with dynamic dimensions.
// Returns true if shapes are compatible (considering -1 as dynamic dimension).
// For example, model shape [1, -1, -1, 3] is compatible with input [1, 224,
// 224, 3].
bool IsShapeCompatibleWithDynamicDims(const std::vector<int>& model_shape,
                                      const std::vector<int>& input_shape);

// Creates a tensor with the given LiteRt ranked tensor type. The tensor is
// allocated using the given memory manager. The memory is aligned to the given
// alignment.
absl::StatusOr<Tensor> CreateTensorFromLiteRtRankedTensorType(
    const litert::RankedTensorType& litert_tensor_type,
    mediapipe::MemoryManager* memory_manager, int memory_allignment);

// Copies the data from a MP tensor to a LiteRT buffer.
absl::Status CopyMpTensorToLiteRtBuffer(const Tensor& mp_tensor,
                                        litert::TensorBuffer& litert_buffer);

// Copies the data from a LiteRT buffer to a MP output tensor.
absl::Status CopyLiteRtBufferToMpTensor(
    const litert::TensorBuffer& litert_buffer, Tensor& mp_tensor);

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_LITERT_LITERT_UTILS_H_
