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

#ifndef MEDIAPIPE_TASKS_GENAI_INFERENCE_UTILS_XNN_UTILS_OPT_H_
#define MEDIAPIPE_TASKS_GENAI_INFERENCE_UTILS_XNN_UTILS_OPT_H_

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/graph_builder.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/opt_weights.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/xnn_tensor.h"

namespace mediapipe::tasks::genai {
namespace xnn_utils {
namespace opt {

class OptBuilder : public XnnGraphBuilder {
 public:
  using XnnGraphBuilder::XnnGraphBuilder;

  absl::StatusOr<std::shared_ptr<Tensor>> FeedForward(
      std::shared_ptr<Tensor> input, const FeedForwardWeights& weights);

  absl::StatusOr<std::shared_ptr<Tensor>> Attention(
      std::shared_ptr<Tensor> input, size_t num_heads,
      std::shared_ptr<Tensor> mask, std::shared_ptr<Tensor> k_cache,
      std::shared_ptr<Tensor> k_slice, std::shared_ptr<Tensor> v_cache,
      std::shared_ptr<Tensor> v_slice, const AttentionWeights& weights);
};

}  // namespace opt
}  // namespace xnn_utils
}  // namespace mediapipe::tasks::genai

#endif  // MEDIAPIPE_TASKS_GENAI_INFERENCE_UTILS_XNN_UTILS_OPT_H_
