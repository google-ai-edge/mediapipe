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

#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/falcon.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/llm.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/llm_weights.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/xnn_tensor.h"

namespace mediapipe::tasks::genai::xnn_utils {

absl::StatusOr<
    std::pair<std::shared_ptr<Tensor>, FalconRW1BBuilder::InputResource>>
FalconRW1BBuilder::PreProcess(std::shared_ptr<Tensor> token_embedding,
                              bool is_prefix) {
  InputResource resource;
  if (is_prefix) {
    // Fused attention mask includes AliBi
    MP_ASSIGN_OR_RETURN(resource.atten_mask,
                        NewInput({llm_params_.seq_size_T, llm_params_.n_heads_N,
                                  llm_params_.seq_size_T}));
  } else {
    MP_ASSIGN_OR_RETURN(
        resource.atten_mask,
        NewInput({1, llm_params_.n_heads_N, llm_params_.seq_size_T}));
  }
  return std::make_pair(token_embedding, resource);
};

absl::StatusOr<std::shared_ptr<Tensor>>
FalconRW1BBuilder::SelfAttentionExcludeNorm(
    std::shared_ptr<Tensor> input, InputResource resource,
    const LlmWeights::SelfAttentionWeights& sa_weights) {
  // [B, 1|T, N, H]
  MP_ASSIGN_OR_RETURN(auto k_proj, SelfAttentionProj(input, sa_weights.k_weight,
                                                     sa_weights.k_bias));
  MP_ASSIGN_OR_RETURN(auto q_proj, SelfAttentionProj(input, sa_weights.q_weight,
                                                     sa_weights.q_bias));
  MP_ASSIGN_OR_RETURN(auto v_proj, SelfAttentionProj(input, sa_weights.v_weight,
                                                     sa_weights.v_bias));

  MP_RETURN_IF_ERROR(BuildKVCache(k_proj, v_proj, resource));

  // [B, 1|T, N, H]
  MP_ASSIGN_OR_RETURN(auto kqv_merged, DotAttention(q_proj, k_proj, v_proj,
                                                    resource.atten_mask));

  const size_t B = kqv_merged->dims[0];
  const size_t NH = kqv_merged->dims[2] * kqv_merged->dims[3];
  MP_ASSIGN_OR_RETURN(auto outcome_reshaped, Reshape(kqv_merged, {B, 0, NH}));
  return FullConn(outcome_reshaped, sa_weights.post_proj_weight,
                  sa_weights.post_proj_bias);
}

absl::StatusOr<std::shared_ptr<Tensor>>
FalconRW1BBuilder::FeedForwardExcludeNorm(
    std::shared_ptr<Tensor> input,
    const LlmWeights::FeedForwardWeights& ff_weights) {
  MP_ASSIGN_OR_RETURN(auto linear1, FullConn(input, ff_weights.layer_1_weight,
                                             ff_weights.layer_1_bias));
  MP_ASSIGN_OR_RETURN(auto gelu1, Gelu(linear1));
  return FullConn(gelu1, ff_weights.layer_2_weight, ff_weights.layer_2_bias);
}

absl::StatusOr<std::shared_ptr<Tensor>> FalconRW1BBuilder::DotAttention(
    std::shared_ptr<Tensor> query_proj, std::shared_ptr<Tensor> key_proj,
    std::shared_ptr<Tensor> value_proj, std::shared_ptr<Tensor> atten_mask) {
  // BTNH -> BNTH
  MP_ASSIGN_OR_RETURN(auto query_permuted, Permute(query_proj, {0, 2, 1, 3}));
  // BSNH -> BNSH
  MP_ASSIGN_OR_RETURN(auto key_permuted, Permute(key_proj, {0, 2, 1, 3}));
  // BNTH.BNSH -> BNTS
  MP_ASSIGN_OR_RETURN(auto logits, QKVAttention(query_permuted, key_permuted,
                                                {0, llm_params_.head_dim_H}));
  const float scale = 1.0f / sqrt(query_proj->dims[3]);
  MP_ASSIGN_OR_RETURN(auto scaled_logits, ElementMul(logits, scale));
  // mask
  MP_ASSIGN_OR_RETURN(auto mask_permuted, Permute(atten_mask, {1, 0, 2}));
  MP_ASSIGN_OR_RETURN(auto padded_logits,
                      ElementAdd(mask_permuted, scaled_logits));
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

absl::Status FalconRW1BBuilder::InitAttentionMask(size_t current_seq_len,
                                                  size_t process_seq_len,
                                                  bool is_prefix,
                                                  Tensor& out_attn_mask) {
  if (!attention_mask_values_) {
    MP_RETURN_IF_ERROR(InitAlibiAttentionMaskValues());
  }

  if (llm_params_.enable_dynamic_shape) {
    if (!is_prefix) {
      out_attn_mask.Resize(Tensor::DimsType{1, llm_params_.n_heads_N,
                                            current_seq_len + process_seq_len});
      for (size_t n = 0; n < llm_params_.n_heads_N; ++n) {
        auto slice = out_attn_mask.Slice(1, n);
        MP_RETURN_IF_ERROR(slice->LoadFromBuffer(
            attention_mask_values_->data() +
            (current_seq_len * llm_params_.n_heads_N + n) *
                llm_params_.seq_size_T));
      }
    } else {
      out_attn_mask.Resize(Tensor::DimsType{process_seq_len,
                                            llm_params_.n_heads_N,
                                            current_seq_len + process_seq_len});
      for (size_t r = 0; r < out_attn_mask.dims[0]; ++r) {
        for (size_t n = 0; n < llm_params_.n_heads_N; ++n) {
          auto slice = out_attn_mask.Slice(0, r)->Slice(1, n);
          MP_RETURN_IF_ERROR(slice->LoadFromBuffer(
              attention_mask_values_->data() +
              ((current_seq_len + r) * llm_params_.n_heads_N + n) *
                  llm_params_.seq_size_T));
        }
      }
    }
  } else {
    if (!is_prefix) {
      RET_CHECK_EQ(out_attn_mask.num_elements,
                   llm_params_.n_heads_N * llm_params_.seq_size_T);
      out_attn_mask.flat_data = std::shared_ptr<char>(
          attention_mask_values_,
          reinterpret_cast<char*>(attention_mask_values_->data() +
                                  llm_params_.n_heads_N *
                                      llm_params_.seq_size_T *
                                      current_seq_len));
    } else {
      RET_CHECK_EQ(out_attn_mask.num_elements, llm_params_.seq_size_T *
                                                   llm_params_.n_heads_N *
                                                   llm_params_.seq_size_T);
      out_attn_mask.flat_data = std::shared_ptr<char>(
          attention_mask_values_,
          reinterpret_cast<char*>(attention_mask_values_->data()));
    }
  }

  return absl::OkStatus();
}

absl::Status FalconRW1BBuilder::InitAlibiAttentionMaskValues() {
  RET_CHECK_EQ(llm_params_.n_heads_N, 32)
      << "Hardcoded base only works with 32 Heads.";
  const float base = 1 / sqrt(sqrt(2));
  const float scale = 1.0f / sqrt(llm_params_.head_dim_H);

  attention_mask_values_ = std::make_shared<std::vector<float>>(
      llm_params_.n_heads_N * llm_params_.seq_size_T * llm_params_.seq_size_T,
      0.8 * std::numeric_limits<float>::lowest());

  // mask: T,N,T
  // Note: Since the mask has different values across the heads, we use an
  // alternative mask shape to allow tensor slicing. The mask gets transposed
  // before being added to the attention scores.
  for (int i = 0; i < llm_params_.seq_size_T; ++i) {
    float alibi = 1.0f;
    for (int j = 0; j < llm_params_.n_heads_N; ++j) {
      alibi *= base;
      for (int k = 0; k < llm_params_.seq_size_T; ++k) {
        if (k > i) {
          break;
        }
        int idx = i * llm_params_.n_heads_N * llm_params_.seq_size_T +
                  j * llm_params_.seq_size_T + k;
        (*attention_mask_values_)[idx] = k * alibi * scale;
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace mediapipe::tasks::genai::xnn_utils
