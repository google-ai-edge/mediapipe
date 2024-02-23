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

#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/stablelm.h"

#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/llm.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/llm_weights.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/xnn_tensor.h"

namespace mediapipe::tasks::genai::xnn_utils {

absl::StatusOr<
    std::pair<std::shared_ptr<Tensor>, Stablelm4E1T3BBuilder::InputResource>>
Stablelm4E1T3BBuilder::PreProcess(std::shared_ptr<Tensor> token_embedding,
                                  bool is_prefix) {
  InputResource resource;
  // size of partial rotary positional embedding per Stablelm3B4E1T config.
  const size_t rope_size = 20;
  if (is_prefix) {
    MP_ASSIGN_OR_RETURN(
        resource.atten_mask,
        NewInput({llm_params_.seq_size_T, llm_params_.seq_size_T}));
    resource.segment_pos = std::make_shared<Tensor>(
        Tensor::DimsType({llm_params_.seq_size_T, rope_size}));
    MP_RETURN_IF_ERROR(
        InitSegmentPos(0, llm_params_.seq_size_T, *resource.segment_pos));
  } else {
    MP_ASSIGN_OR_RETURN(resource.atten_mask,
                        NewInput({1, llm_params_.seq_size_T}));
    resource.segment_pos =
        std::make_shared<Tensor>(Tensor::DimsType{1, rope_size});
    MP_RETURN_IF_ERROR(InitSegmentPos(0, 1, *resource.segment_pos));
  }
  return std::make_pair(token_embedding, resource);
};

absl::StatusOr<std::shared_ptr<Tensor>>
Stablelm4E1T3BBuilder::SelfAttentionExcludeNorm(
    std::shared_ptr<Tensor> input, InputResource resource,
    const LlmWeights::SelfAttentionWeights& sa_weights) {
  // [B, 1|T, N, H]
  MP_ASSIGN_OR_RETURN(auto k_proj,
                      SelfAttentionProj(input, sa_weights.k_weight));
  MP_ASSIGN_OR_RETURN(auto q_proj,
                      SelfAttentionProj(input, sa_weights.q_weight));
  MP_ASSIGN_OR_RETURN(auto v_proj,
                      SelfAttentionProj(input, sa_weights.v_weight));

  MP_ASSIGN_OR_RETURN(auto query_proj_after_rope,
                      PartialRope(q_proj, /*idx=*/20, resource.segment_pos));
  MP_ASSIGN_OR_RETURN(auto key_proj_after_rope,
                      PartialRope(k_proj, /*idx=*/20, resource.segment_pos));

  MP_RETURN_IF_ERROR(BuildKVCache(key_proj_after_rope, v_proj, resource));

  // [B, 1|T, N, H]
  MP_ASSIGN_OR_RETURN(
      auto kqv_merged,
      DotAttention(query_proj_after_rope, key_proj_after_rope, v_proj,
                   resource.atten_mask, sa_weights.per_dim_scale));

  const size_t B = kqv_merged->dims[0];
  const size_t NH = kqv_merged->dims[2] * kqv_merged->dims[3];
  MP_ASSIGN_OR_RETURN(auto outcome_reshaped, Reshape(kqv_merged, {B, 0, NH}));
  return MatMul(outcome_reshaped, sa_weights.post_proj_weight);
}

absl::StatusOr<std::shared_ptr<Tensor>> Stablelm4E1T3BBuilder::DotAttention(
    std::shared_ptr<Tensor> query_proj, std::shared_ptr<Tensor> key_proj,
    std::shared_ptr<Tensor> value_proj, std::shared_ptr<Tensor> atten_mask,
    std::shared_ptr<Tensor> per_dim_scale) {
  // BTNH -> BNTH
  MP_ASSIGN_OR_RETURN(auto query_permuted, Permute(query_proj, {0, 2, 1, 3}));
  // BSNH -> BNSH
  MP_ASSIGN_OR_RETURN(auto key_permuted, Permute(key_proj, {0, 2, 1, 3}));
  // BNTH.BNSH -> BNTS
  MP_ASSIGN_OR_RETURN(auto logits, QKVAttention(query_permuted, key_permuted,
                                                {0, llm_params_.head_dim_H}));
  float scale = 1.0f / sqrt(query_proj->dims[3]);
  MP_ASSIGN_OR_RETURN(auto scaled_logits, ElementMul(logits, scale));
  // mask
  MP_ASSIGN_OR_RETURN(auto padded_logits,
                      ElementAdd(atten_mask, scaled_logits));
  MP_ASSIGN_OR_RETURN(auto probs, Softmax(padded_logits));
  MP_ASSIGN_OR_RETURN(auto value_permuted, Permute(value_proj, {0, 2, 3, 1}));
  // Outcome
  // BNTS.BNHS -> BNTH
  MP_ASSIGN_OR_RETURN(
      auto outcome_before_permute,
      QKVAttention(probs, value_permuted, {llm_params_.head_dim_H, 0}));
  // BNTH -> BTNH
  return Permute(outcome_before_permute, {0, 2, 1, 3});
}

}  // namespace mediapipe::tasks::genai::xnn_utils
