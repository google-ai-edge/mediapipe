// Copyright 2022 The MediaPipe Authors.
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

#include <math.h>

#include <algorithm>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/tasks/cc/components/calculators/tensors_to_embeddings_calculator.pb.h"
#include "mediapipe/tasks/cc/components/containers/proto/embeddings.pb.h"
#include "mediapipe/tasks/cc/components/processors/proto/embedder_options.pb.h"

namespace mediapipe {
namespace api2 {

namespace {

using ::mediapipe::tasks::components::containers::proto::Embedding;
using ::mediapipe::tasks::components::containers::proto::EmbeddingResult;

// Computes the inverse L2 norm of the provided array of values. Returns 1.0 in
// case all values are 0.
float GetInverseL2Norm(const float* values, int size) {
  float squared_l2_norm = 0.0f;
  for (int i = 0; i < size; ++i) {
    squared_l2_norm += values[i] * values[i];
  }
  float inv_l2_norm = 1.0f;
  if (squared_l2_norm > 0.0f) {
    inv_l2_norm = 1.0f / std::sqrt(squared_l2_norm);
  }
  return inv_l2_norm;
}

}  // namespace

// Converts tensors into an EmbeddingResult object, performing optional
// L2-normalization and scalar-quantization on-the-fly if required through the
// options.
//
// Input:
//   TENSORS - std::vector<Tensor>
//     A vector of one or more Tensors of type kFloat32.
// Output:
//   EMBEDDINGS - EmbeddingResult
//     The contents of the input tensors converted into an EmbeddingResult
//     proto.
class TensorsToEmbeddingsCalculator : public Node {
 public:
  static constexpr Input<std::vector<Tensor>> kTensorsIn{"TENSORS"};
  static constexpr Output<EmbeddingResult> kEmbeddingsOut{"EMBEDDINGS"};
  MEDIAPIPE_NODE_CONTRACT(kTensorsIn, kEmbeddingsOut);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  bool l2_normalize_;
  bool quantize_;
  std::vector<std::string> head_names_;
  absl::flat_hash_set<std::string> ignored_head_names_;

  void FillFloatEmbedding(const Tensor& tensor, Embedding* embedding);
  void FillQuantizedEmbedding(const Tensor& tensor, Embedding* embedding);
};

absl::Status TensorsToEmbeddingsCalculator::Open(CalculatorContext* cc) {
  auto options = cc->Options<mediapipe::TensorsToEmbeddingsCalculatorOptions>();
  l2_normalize_ = options.embedder_options().l2_normalize();
  quantize_ = options.embedder_options().quantize();
  if (!options.head_names().empty()) {
    head_names_.assign(options.head_names().begin(),
                       options.head_names().end());
  }
  for (const absl::string_view head_name : options.ignored_head_names()) {
    ignored_head_names_.insert(std::string(head_name));
  }
  return absl::OkStatus();
}

absl::Status TensorsToEmbeddingsCalculator::Process(CalculatorContext* cc) {
  EmbeddingResult result;
  const auto& tensors = *kTensorsIn(cc);
  if (!head_names_.empty() && tensors.size() != head_names_.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Mismatch between number of provided head names (%d) and number "
        "of input tensors (%d).",
        head_names_.size(), tensors.size()));
  }
  for (int i = 0; i < tensors.size(); ++i) {
    if (!head_names_.empty() && ignored_head_names_.contains(head_names_[i])) {
      continue;
    }
    const auto& tensor = tensors[i];
    RET_CHECK(tensor.element_type() == Tensor::ElementType::kFloat32);
    auto* embedding = result.add_embeddings();
    embedding->set_head_index(i);
    if (!head_names_.empty()) {
      embedding->set_head_name(head_names_[i]);
    }
    if (quantize_) {
      FillQuantizedEmbedding(tensor, embedding);
    } else {
      FillFloatEmbedding(tensor, embedding);
    }
  }
  kEmbeddingsOut(cc).Send(result);
  return absl::OkStatus();
}

void TensorsToEmbeddingsCalculator::FillFloatEmbedding(const Tensor& tensor,
                                                       Embedding* embedding) {
  int size = tensor.shape().num_elements();
  auto tensor_view = tensor.GetCpuReadView();
  const float* tensor_buffer = tensor_view.buffer<float>();
  float inv_l2_norm =
      l2_normalize_ ? GetInverseL2Norm(tensor_buffer, size) : 1.0f;
  auto* float_embedding = embedding->mutable_float_embedding();
  for (int i = 0; i < size; ++i) {
    float_embedding->add_values(tensor_buffer[i] * inv_l2_norm);
  }
}

void TensorsToEmbeddingsCalculator::FillQuantizedEmbedding(
    const Tensor& tensor, Embedding* embedding) {
  int size = tensor.shape().num_elements();
  auto tensor_view = tensor.GetCpuReadView();
  const float* tensor_buffer = tensor_view.buffer<float>();
  float inv_l2_norm =
      l2_normalize_ ? GetInverseL2Norm(tensor_buffer, size) : 1.0f;
  auto* values = embedding->mutable_quantized_embedding()->mutable_values();
  values->resize(size);
  for (int i = 0; i < size; ++i) {
    // Normalize.
    float normalized = tensor_buffer[i] * inv_l2_norm;
    // Quantize.
    int unclamped_value = static_cast<int>(roundf(normalized * 128));
    // Clamp and assign.
    (*values)[i] =
        static_cast<char>(std::max(-128, std::min(unclamped_value, 127)));
  }
}

MEDIAPIPE_REGISTER_NODE(TensorsToEmbeddingsCalculator);

}  // namespace api2
}  // namespace mediapipe
