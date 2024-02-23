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

#ifndef MEDIAPIPE_TASKS_GENAI_INFERENCE_UTILS_XNN_UTILS_OPT_WEIGHTS_H_
#define MEDIAPIPE_TASKS_GENAI_INFERENCE_UTILS_XNN_UTILS_OPT_WEIGHTS_H_

#include <memory>

#include "mediapipe/tasks/cc/genai/inference/proto/llm_params.pb.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/xnn_tensor.h"

namespace mediapipe::tasks::genai {
namespace xnn_utils {
namespace opt {

struct FeedForwardWeights {
  std::shared_ptr<Tensor> linear_1_weight;
  std::shared_ptr<Tensor> linear_1_bias;
  std::shared_ptr<Tensor> linear_2_weight;
  std::shared_ptr<Tensor> linear_2_bias;
};

struct AttentionWeights {
  std::shared_ptr<Tensor> query_weight;
  std::shared_ptr<Tensor> query_bias;
  std::shared_ptr<Tensor> key_weight;
  std::shared_ptr<Tensor> key_bias;
  std::shared_ptr<Tensor> value_weight;
  std::shared_ptr<Tensor> value_bias;
  std::shared_ptr<Tensor> output_weight;
  std::shared_ptr<Tensor> output_bias;
};

}  // namespace opt
}  // namespace xnn_utils
}  // namespace mediapipe::tasks::genai

#endif  // MEDIAPIPE_TASKS_GENAI_INFERENCE_UTILS_XNN_UTILS_OPT_WEIGHTS_H_
