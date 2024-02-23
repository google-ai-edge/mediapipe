#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/phi.h"

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

absl::StatusOr<std::pair<std::shared_ptr<Tensor>, Phi2Builder::InputResource>>
Phi2Builder::PreProcess(std::shared_ptr<Tensor> token_embedding,
                        bool is_prefix) {
  InputResource resource;
  // size of partial rotary positional embedding per Phi2 config.
  const size_t rope_size = 32;
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

absl::StatusOr<std::shared_ptr<Tensor>> Phi2Builder::OneStackTransformer(
    int layer_index, std::shared_ptr<Tensor> input,
    Phi2Builder::InputResource resource,
    const LlmWeights::SelfAttentionWeights& sa_weights,
    const LlmWeights::FeedForwardWeights& ff_weights, bool is_prefix) {
  MP_ASSIGN_OR_RETURN(auto normalized_input,
                      ApplyNorm(input, sa_weights.pre_norm_weight,
                                llm_params_.sa_params.pre_norm));
  MP_ASSIGN_OR_RETURN(
      auto sa_output,
      SelfAttentionExcludeNorm(normalized_input, resource, sa_weights));
  if (is_prefix && internal_llm_params_.stop_at_last_kv_cache &&
      (layer_index == llm_params_.num_transformer_M - 1)) {
    return sa_output;
  }
  MP_ASSIGN_OR_RETURN(auto ff_output,
                      FeedForwardExcludeNorm(normalized_input, ff_weights));
  MP_ASSIGN_OR_RETURN(auto output, ElementAdd(ff_output, sa_output));
  return ElementAdd(input, output);
}

absl::StatusOr<std::shared_ptr<Tensor>> Phi2Builder::SelfAttentionExcludeNorm(
    std::shared_ptr<Tensor> input, InputResource resource,
    const LlmWeights::SelfAttentionWeights& sa_weights) {
  // [B, 1|T, N, H]
  MP_ASSIGN_OR_RETURN(auto k_proj, SelfAttentionProj(input, sa_weights.k_weight,
                                                     sa_weights.k_bias));
  MP_ASSIGN_OR_RETURN(auto q_proj, SelfAttentionProj(input, sa_weights.q_weight,
                                                     sa_weights.q_bias));
  MP_ASSIGN_OR_RETURN(auto v_proj, SelfAttentionProj(input, sa_weights.v_weight,
                                                     sa_weights.v_bias));

  MP_ASSIGN_OR_RETURN(auto query_proj_after_rope,
                      PartialRope(q_proj, /*idx=*/32, resource.segment_pos));
  MP_ASSIGN_OR_RETURN(auto key_proj_after_rope,
                      PartialRope(k_proj, /*idx=*/32, resource.segment_pos));

  MP_RETURN_IF_ERROR(BuildKVCache(key_proj_after_rope, v_proj, resource));

  // [B, 1|T, N, H]
  MP_ASSIGN_OR_RETURN(auto kqv_merged,
                      DotAttention(query_proj_after_rope, key_proj_after_rope,
                                   v_proj, resource.atten_mask));

  const size_t B = kqv_merged->dims[0];
  const size_t NH = kqv_merged->dims[2] * kqv_merged->dims[3];
  MP_ASSIGN_OR_RETURN(auto outcome_reshaped, Reshape(kqv_merged, {B, 0, NH}));
  return FullConn(outcome_reshaped, sa_weights.post_proj_weight,
                  sa_weights.post_proj_bias);
}

absl::StatusOr<std::shared_ptr<Tensor>> Phi2Builder::FeedForwardExcludeNorm(
    std::shared_ptr<Tensor> input,
    const LlmWeights::FeedForwardWeights& ff_weights) {
  MP_ASSIGN_OR_RETURN(auto linear1, FullConn(input, ff_weights.layer_1_weight,
                                             ff_weights.layer_1_bias));
  MP_ASSIGN_OR_RETURN(auto gelu1, Gelu(linear1));
  return FullConn(gelu1, ff_weights.layer_2_weight, ff_weights.layer_2_bias);
}

absl::StatusOr<std::shared_ptr<Tensor>> Phi2Builder::DotAttention(
    std::shared_ptr<Tensor> query_proj, std::shared_ptr<Tensor> key_proj,
    std::shared_ptr<Tensor> value_proj, std::shared_ptr<Tensor> atten_mask) {
  // BTNH -> BNTH
  MP_ASSIGN_OR_RETURN(auto query_permuted, Permute(query_proj, {0, 2, 1, 3}));
  // BSNH -> BNSH
  MP_ASSIGN_OR_RETURN(auto key_permuted, Permute(key_proj, {0, 2, 1, 3}));
  // BNTH.BNSH -> BNTS
  MP_ASSIGN_OR_RETURN(auto logits, QKVAttention(query_permuted, key_permuted,
                                                {0, llm_params_.head_dim_H}));
  float scale = 1.0f / sqrt(llm_params_.head_dim_H);
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
