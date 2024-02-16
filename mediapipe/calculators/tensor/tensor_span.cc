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

#include "mediapipe/calculators/tensor/tensor_span.h"

#include <utility>
#include <vector>

#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/input_stream_shard.h"

namespace mediapipe {

// Create reference pointer vector from a collection of input streams using the
// api2 framework. It is the caller's responsibility to check for empty inputs.
TensorSpan MakeTensorSpan(api2::internal::MultiplePortAccess<
                          Tensor, InputStreamShard, CalculatorContext>
                              tensor_streams) {
  std::vector<const Tensor*> refs;
  const int num_tensors = tensor_streams.Count();
  refs.reserve(num_tensors);
  for (int i = 0; i < num_tensors; ++i) {
    refs.push_back(&(*tensor_streams[i]));
  }
  return TensorSpan(std::move(refs));
}

// Create reference pointer vector from vector of tensors
TensorSpan MakeTensorSpan(const std::vector<Tensor>& tensors) {
  std::vector<const Tensor*> refs;
  refs.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    refs.push_back(&tensor);
  }
  return TensorSpan(std::move(refs));
}

TensorSpan::TensorSpan(std::vector<const Tensor*>&& tensor_refs)
    : tensor_refs_(tensor_refs) {}

int TensorSpan::size() const { return tensor_refs_.size(); }

const Tensor& TensorSpan::operator[](int index) const {
  return *(tensor_refs_[index]);
}

}  // namespace mediapipe
