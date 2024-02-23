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

#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/opt.h"

#include <cmath>
#include <cstddef>
#include <memory>

#include "absl/status/statusor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/opt_weights.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/xnn_tensor.h"

namespace mediapipe::tasks::genai {
namespace xnn_utils {
namespace opt {

absl::StatusOr<std::shared_ptr<Tensor>> OptBuilder::FeedForward(
    std::shared_ptr<Tensor> input, const FeedForwardWeights& weights) {
  MP_ASSIGN_OR_RETURN(auto linear1, FullConn(input, weights.linear_1_weight,
                                             weights.linear_1_bias));
  MP_ASSIGN_OR_RETURN(auto relu1, Relu(linear1));
  return FullConn(relu1, weights.linear_2_weight, weights.linear_2_bias);
}

absl::StatusOr<std::shared_ptr<Tensor>> OptBuilder::Attention(
    std::shared_ptr<Tensor> input, size_t num_heads,
    std::shared_ptr<Tensor> mask, std::shared_ptr<Tensor> k_cache,
    std::shared_ptr<Tensor> k_slice, std::shared_ptr<Tensor> v_cache,
    std::shared_ptr<Tensor> v_slice, const AttentionWeights& weights) {
  RET_CHECK_EQ(input->dims.size(), 3);
  const size_t batch_size = input->dims[0];
  const size_t sequence_length = input->dims[1];
  const size_t head_dim = weights.key_weight->dims[0] / num_heads;
  // B,T,D -> B,T,N,H
  MP_ASSIGN_OR_RETURN(auto q_proj,
                      SelfAttentionProj(input, weights.query_weight,
                                        weights.query_bias, num_heads));
  MP_ASSIGN_OR_RETURN(q_proj, ElementMul(q_proj, 1.0f / sqrt(head_dim)));
  MP_ASSIGN_OR_RETURN(auto k_proj,
                      SelfAttentionProj(input, weights.key_weight,
                                        weights.key_bias, num_heads));
  MP_ASSIGN_OR_RETURN(auto v_proj,
                      SelfAttentionProj(input, weights.value_weight,
                                        weights.value_bias, num_heads));

  MP_ASSIGN_OR_RETURN(auto k_permuted, Permute(k_proj, {0, 2, 1, 3}));
  MP_ASSIGN_OR_RETURN(auto q_permuted, Permute(q_proj, {0, 2, 1, 3}));
  // scores: B,N,T,T
  MP_ASSIGN_OR_RETURN(auto scores, QKVAttention(q_permuted, k_permuted,
                                                {0, k_permuted->dims.back()}));
  MP_ASSIGN_OR_RETURN(scores, ElementAdd(scores, mask));
  MP_ASSIGN_OR_RETURN(scores, Softmax(scores));

  // B,T,N,H -> B,N,H,T
  MP_ASSIGN_OR_RETURN(auto v_permuted, Permute(v_proj, {0, 2, 3, 1}));
  // output: B,N,T,H
  MP_ASSIGN_OR_RETURN(
      auto output, QKVAttention(scores, v_permuted, {v_permuted->dims[2], 0}));
  MP_ASSIGN_OR_RETURN(output, Permute(output, {0, 2, 1, 3}));
  // output: B,T,NH
  MP_ASSIGN_OR_RETURN(output, Reshape(output, {batch_size, sequence_length,
                                               num_heads * head_dim}));
  return FullConn(output, weights.output_weight, weights.output_bias);
}

}  // namespace opt
}  // namespace xnn_utils
}  // namespace mediapipe::tasks::genai
