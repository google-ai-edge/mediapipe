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

#include "mediapipe/framework/formats/tensor.h"

namespace mediapipe {

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
