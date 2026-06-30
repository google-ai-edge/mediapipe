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

#include "mediapipe/calculators/tensor/litert/litert_utils.h"

#include <cstddef>
#include <cstring>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

namespace {

// Helper function to get the tensor shape without leading ones.
template <typename T>
std::vector<int> GetTensorDimensionsWithoutLeadingOnes(const T& tensor_shape) {
  if (tensor_shape.size() <= 1) {
    // Skip if the tensor has no dimensions or only one dimension.
    return std::vector<int>(tensor_shape.begin(), tensor_shape.end());
  }
  auto itr = tensor_shape.begin();
  while (itr != tensor_shape.end() - 1 && *itr == 1) {
    ++itr;
  }
  return std::vector<int>(itr, tensor_shape.end());
}

bool operator==(Tensor::ElementType tensor_type,
                litert::ElementType litert_type) {
  switch (tensor_type) {
    case Tensor::ElementType::kFloat16:
      return litert_type == litert::ElementType::Float16;
    case Tensor::ElementType::kFloat32:
      return litert_type == litert::ElementType::Float32;
    case Tensor::ElementType::kUInt8:
      return litert_type == litert::ElementType::UInt8;
    case Tensor::ElementType::kInt8:
      return litert_type == litert::ElementType::Int8;
    case Tensor::ElementType::kInt32:
      return litert_type == litert::ElementType::Int32;
    case Tensor::ElementType::kInt64:
      return litert_type == litert::ElementType::Int64;
    case Tensor::ElementType::kBool:
      return litert_type == litert::ElementType::Bool;
    default:
      return false;
  }
}

absl::string_view GetTensorTypeString(const Tensor::ElementType& tensor_type) {
  return Tensor::ElementTypeName(tensor_type);
}

absl::string_view GetTensorTypeString(const litert::ElementType& tensor_type) {
  switch (tensor_type) {
    case litert::ElementType::None:
      return "None";
    case litert::ElementType::Float16:
      return "Float16";
    case litert::ElementType::Float32:
      return "Float32";
    case litert::ElementType::UInt8:
      return "UInt8";
    case litert::ElementType::Int8:
      return "Int8";
    case litert::ElementType::Int32:
      return "Int32";
    case litert::ElementType::Int64:
      return "Int64";
    case litert::ElementType::Bool:
      return "Bool";
    default:
      return "Unknown";
  }
}

}  // namespace

absl::Status AreTensorSpecsEqual(
    const litert::RankedTensorType& litert_tensor_type,
    const mediapipe::Tensor& mp_tensor) {
  const std::vector<int> litert_tensor_dims =
      GetTensorDimensionsWithoutLeadingOnes(
          litert_tensor_type.Layout().Dimensions());
  const std::vector<int> mp_tensor_dims =
      GetTensorDimensionsWithoutLeadingOnes(mp_tensor.shape().dims);
  const litert::ElementType litert_tensor_element_type =
      litert_tensor_type.ElementType();
  if (litert_tensor_dims.empty() && mp_tensor_dims.size() == 1 &&
      mp_tensor_dims[0] == 1) {
    // Scalar tensors have no dimensions in LiteRt, but have a single
    // dimension in MP tensors.
    return absl::OkStatus();
  }

  RET_CHECK(litert_tensor_dims == mp_tensor_dims) << absl::StrCat(
      "LiteRt [", absl::StrJoin(litert_tensor_dims, ","), "] and MediaPipe [",
      absl::StrJoin(mp_tensor_dims, ","), "] tensor shapes differ.");

  const Tensor::ElementType input_tensor_type = mp_tensor.element_type();
  RET_CHECK(input_tensor_type == litert_tensor_element_type)
      << absl::StrFormat("LiteRt and MediaPipe tensor types differ (%s vs %s)",
                         GetTensorTypeString(litert_tensor_element_type),
                         GetTensorTypeString(input_tensor_type));
  return absl::OkStatus();
}

bool IsShapeCompatibleWithDynamicDims(const std::vector<int>& model_shape,
                                      const std::vector<int>& input_shape) {
  // Shapes must have the same rank
  if (model_shape.size() != input_shape.size()) {
    return false;
  }

  // Check each dimension
  for (size_t i = 0; i < model_shape.size(); ++i) {
    // -1 indicates a dynamic dimension that can match any size
    if (model_shape[i] == -1) {
      continue;
    }
    // Static dimensions must match exactly
    if (model_shape[i] != input_shape[i]) {
      return false;
    }
  }
  return true;
}

absl::Status AreTensorSpecsCompatible(
    const litert::RankedTensorType& litert_tensor_type,
    const mediapipe::Tensor& mp_tensor) {
  const auto litert_tensor_dims = litert_tensor_type.Layout().Dimensions();
  const auto mp_tensor_dims = mp_tensor.shape().dims;
  const auto litert_tensor_element_type = litert_tensor_type.ElementType();

  // Handle scalar tensors
  if (litert_tensor_dims.empty() && mp_tensor_dims.size() == 1 &&
      mp_tensor_dims[0] == 1) {
    // Scalar tensors have no dimensions in LiteRt, but have a single
    // dimension in MP tensors.
    const auto input_tensor_type = mp_tensor.element_type();
    RET_CHECK(input_tensor_type == litert_tensor_element_type)
        << absl::StrFormat(
               "LiteRt and MediaPipe tensor types differ (%s vs %s)",
               GetTensorTypeString(litert_tensor_element_type),
               GetTensorTypeString(input_tensor_type));
    return absl::OkStatus();
  }

  // Check rank compatibility
  RET_CHECK_EQ(litert_tensor_dims.size(), mp_tensor_dims.size())
      << absl::StrFormat("LiteRt and MediaPipe tensor ranks differ (%d vs %d)",
                         litert_tensor_dims.size(), mp_tensor_dims.size());

  // Check shape compatibility with dynamic dimensions
  std::vector<int> litert_shape(litert_tensor_dims.begin(),
                                litert_tensor_dims.end());
  if (!IsShapeCompatibleWithDynamicDims(litert_shape, mp_tensor_dims)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Shape mismatch: model expects %s but got %s",
        absl::StrJoin(litert_shape, "x"), absl::StrJoin(mp_tensor_dims, "x")));
  }

  // Check element type
  const auto input_tensor_type = mp_tensor.element_type();
  RET_CHECK(input_tensor_type == litert_tensor_element_type)
      << absl::StrFormat("LiteRt and MediaPipe tensor types differ (%s vs %s)",
                         GetTensorTypeString(litert_tensor_element_type),
                         GetTensorTypeString(input_tensor_type));
  return absl::OkStatus();
}

absl::StatusOr<Tensor> CreateTensorFromLiteRtRankedTensorType(
    const litert::RankedTensorType& litert_tensor_type,
    MemoryManager* memory_manager, int memory_allignment) {
  const auto tensor_type = litert_tensor_type.ElementType();
  const auto tensor_dimensions = litert_tensor_type.Layout().Dimensions();
  // Scalar input in TensorFlow is described by an empty shape.
  // In MediaPipe, we represent it as a shape with a single dimension of size 1.
  const std::vector<int> tensor_shape_vector =
      tensor_dimensions.empty() ? std::vector<int>{1}
                                : std::vector<int>(tensor_dimensions.begin(),
                                                   tensor_dimensions.end());
  switch (tensor_type) {
    case litert::ElementType::Float16:
      return Tensor(Tensor::ElementType::kFloat16,
                    Tensor::Shape(tensor_shape_vector), memory_manager,
                    memory_allignment);
    case litert::ElementType::Float32:
      return Tensor(Tensor::ElementType::kFloat32,
                    Tensor::Shape(tensor_shape_vector), memory_manager,
                    memory_allignment);
    case litert::ElementType::UInt8:
      return Tensor(Tensor::ElementType::kUInt8,
                    Tensor::Shape(tensor_shape_vector), memory_manager,
                    memory_allignment);
    case litert::ElementType::Int8:
      return Tensor(Tensor::ElementType::kInt8,
                    Tensor::Shape(tensor_shape_vector), memory_manager,
                    memory_allignment);
    case litert::ElementType::Int32:
      return Tensor(Tensor::ElementType::kInt32,
                    Tensor::Shape(tensor_shape_vector), memory_manager,
                    memory_allignment);
    case litert::ElementType::Int64:
      return Tensor(Tensor::ElementType::kInt64,
                    Tensor::Shape(tensor_shape_vector), memory_manager,
                    memory_allignment);
    case litert::ElementType::Bool:
      return Tensor(Tensor::ElementType::kBool,
                    Tensor::Shape(tensor_shape_vector), memory_manager,
                    memory_allignment);
    default:
      break;
  }
  return absl::InvalidArgumentError("Unsupported tensor type.");
}

absl::Status CopyMpTensorToLiteRtBuffer(const Tensor& mp_tensor,
                                        litert::TensorBuffer& litert_buffer) {
  const size_t tensor_size_bytes = mp_tensor.bytes();
  LITERT_MP_ASSIGN_OR_RETURN(size_t litert_tensor_size_bytes,
                             litert_buffer.PackedSize());
  RET_CHECK_EQ(litert_tensor_size_bytes, tensor_size_bytes) << absl::StrFormat(
      "Tensor size differs between LiteRt and MP: %ld != %ld",
      litert_tensor_size_bytes, tensor_size_bytes);
  auto mp_tensor_view = mp_tensor.GetCpuReadView();
  LITERT_MP_ASSIGN_OR_RETURN(
      auto lock_and_addr,
      litert::TensorBufferScopedLock::Create(
          litert_buffer, litert::TensorBuffer::LockMode::kWrite));
  std::memcpy(lock_and_addr.second, mp_tensor_view.buffer<void>(),
              tensor_size_bytes);
  return absl::OkStatus();
}

absl::Status CopyLiteRtBufferToMpTensor(
    const litert::TensorBuffer& litert_buffer, Tensor& mp_tensor) {
  const size_t tensor_size_bytes = mp_tensor.bytes();
  LITERT_MP_ASSIGN_OR_RETURN(size_t litert_tensor_size_bytes,
                             litert_buffer.PackedSize());
  RET_CHECK_EQ(litert_tensor_size_bytes, tensor_size_bytes) << absl::StrFormat(
      "Tensor size differs between LiteRt and MP: %ld != %ld",
      litert_tensor_size_bytes, tensor_size_bytes);
  auto mp_tensor_view = mp_tensor.GetCpuWriteView();
  LITERT_MP_ASSIGN_OR_RETURN(
      auto lock_and_addr,
      litert::TensorBufferScopedLock::Create(
          litert_buffer, litert::TensorBuffer::LockMode::kRead));
  void* mp_tensor_data = mp_tensor_view.buffer<void>();
  std::memcpy(mp_tensor_data, lock_and_addr.second, tensor_size_bytes);
  return absl::OkStatus();
}

}  // namespace mediapipe
