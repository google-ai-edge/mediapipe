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

#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/well_known_models.h"

#include "mediapipe/tasks/cc/genai/inference/proto/llm_params.pb.h"
#include "mediapipe/tasks/cc/genai/inference/proto/transformer_params.pb.h"

namespace mediapipe::tasks::genai::llm_utils {
namespace {
using LlmModelType = odml::infra::proto::LlmModelType;
using LlmParameters = odml::infra::proto::LlmParameters;
using TransformerParameters = odml::infra::proto::TransformerParameters;

constexpr int kBatchSize = 1;
}  // namespace

LlmParameters GetGemma2BParams() {
  LlmParameters llm_params;
  llm_params.set_start_token_id(2);
  llm_params.add_stop_tokens("<eos>");
  llm_params.set_vocab_size(256000);

  TransformerParameters& transformer_params =
      *llm_params.mutable_transformer_parameters();
  transformer_params.set_batch_size(kBatchSize);
  transformer_params.set_embedding_dim(2048);
  transformer_params.set_hidden_dimension(16384);
  transformer_params.set_head_dimension(256);
  transformer_params.set_num_heads(8);
  transformer_params.set_num_stacks(18);
  // MQA
  transformer_params.set_num_kv_heads(1);
  transformer_params.set_pre_norm(TransformerParameters::RMS_NORM);
  transformer_params.set_post_norm(TransformerParameters::NO_NORM);
  transformer_params.set_final_norm(TransformerParameters::RMS_NORM);
  transformer_params.set_skip_absolute_positional_embeddings(true);

  TransformerParameters::SelfAttentionParameters& sa_params =
      *transformer_params.mutable_self_attention_parameters();
  sa_params.set_attention_mask_type(TransformerParameters::CAUSAL);
  sa_params.set_qkv_no_bias(true);
  sa_params.set_post_proj_no_bias(true);
  sa_params.set_attention_scale_type(
      TransformerParameters::SCALE_TYPE_INV_SQRT_HEAD_DIM);
  // Disable soft cap.
  sa_params.set_soft_cap_value(0.0f);

  TransformerParameters::FeedForwardParameters& ff_params =
      *transformer_params.mutable_feed_forward_parameters();
  ff_params.set_no_bias(true);
  ff_params.set_activation(TransformerParameters::GELU);
  ff_params.set_pre_norm(TransformerParameters::RMS_NORM);
  ff_params.set_post_norm(TransformerParameters::NO_NORM);

  TransformerParameters::FinalProjectParameters& fp_params =
      *transformer_params.mutable_final_project_parameters();
  fp_params.set_no_bias(true);
  // Disable soft cap.
  fp_params.set_soft_cap_value(0.0f);

  return llm_params;
}

LlmParameters GetGemma7BParams() {
  LlmParameters llm_params;
  llm_params.set_start_token_id(2);
  llm_params.add_stop_tokens("<eos>");
  llm_params.set_vocab_size(256000);

  TransformerParameters& transformer_params =
      *llm_params.mutable_transformer_parameters();
  transformer_params.set_batch_size(kBatchSize);
  transformer_params.set_embedding_dim(3072);
  transformer_params.set_hidden_dimension(8 * 3072);
  transformer_params.set_head_dimension(256);
  transformer_params.set_num_heads(16);
  transformer_params.set_num_stacks(28);
  // MHA
  transformer_params.set_num_kv_heads(0);
  transformer_params.set_pre_norm(TransformerParameters::RMS_NORM);
  transformer_params.set_post_norm(TransformerParameters::NO_NORM);
  transformer_params.set_final_norm(TransformerParameters::RMS_NORM);
  transformer_params.set_skip_absolute_positional_embeddings(true);

  TransformerParameters::SelfAttentionParameters& sa_params =
      *transformer_params.mutable_self_attention_parameters();
  sa_params.set_attention_mask_type(TransformerParameters::CAUSAL);
  sa_params.set_qkv_no_bias(true);
  sa_params.set_post_proj_no_bias(true);
  sa_params.set_attention_scale_type(
      TransformerParameters::SCALE_TYPE_INV_SQRT_HEAD_DIM);
  // Disable soft cap.
  sa_params.set_soft_cap_value(0.0f);

  TransformerParameters::FeedForwardParameters& ff_params =
      *transformer_params.mutable_feed_forward_parameters();
  ff_params.set_no_bias(true);
  ff_params.set_activation(TransformerParameters::GELU);
  ff_params.set_pre_norm(TransformerParameters::RMS_NORM);
  ff_params.set_post_norm(TransformerParameters::NO_NORM);

  TransformerParameters::FinalProjectParameters& fp_params =
      *transformer_params.mutable_final_project_parameters();
  fp_params.set_no_bias(true);
  // Disable soft cap.
  fp_params.set_soft_cap_value(0.0f);

  return llm_params;
}

LlmParameters GetGemma2_2BParams() {
  LlmParameters llm_params;
  llm_params.set_start_token_id(2);
  llm_params.add_stop_tokens("<eos>");
  llm_params.set_vocab_size(256000);

  TransformerParameters& transformer_params =
      *llm_params.mutable_transformer_parameters();
  transformer_params.set_batch_size(kBatchSize);
  transformer_params.set_embedding_dim(2304);
  transformer_params.set_hidden_dimension(9216);
  transformer_params.set_head_dimension(256);
  transformer_params.set_num_heads(8);
  transformer_params.set_num_stacks(26);
  // GQA, num_groups=2
  transformer_params.set_num_kv_heads(4);
  transformer_params.set_pre_norm(TransformerParameters::RMS_NORM);
  transformer_params.set_post_norm(TransformerParameters::RMS_NORM);
  transformer_params.set_final_norm(TransformerParameters::RMS_NORM);
  transformer_params.set_skip_absolute_positional_embeddings(true);
  // Alternating L-G-L-G-...
  // Commenting out for now, since without hybrid cache or runtime sliding
  // window size, this wouldn't have any effect. TODO: Fix.
  // transformer_params.set_num_local_layers_per_global(1);

  TransformerParameters::SelfAttentionParameters& sa_params =
      *transformer_params.mutable_self_attention_parameters();
  sa_params.set_attention_mask_type(TransformerParameters::CAUSAL);
  sa_params.set_qkv_no_bias(true);
  sa_params.set_post_proj_no_bias(true);
  sa_params.set_attention_scale_type(
      TransformerParameters::SCALE_TYPE_INV_SQRT_HEAD_DIM);
  sa_params.set_soft_cap_value(50.0f);

  // This should be a runtime parameter, since it doesn't make sense to have a
  // sliding window size of 4096 when our global context size is often smaller
  // than that. TODO: Fix.
  // sa_params.set_sliding_window_size(4096);

  TransformerParameters::FeedForwardParameters& ff_params =
      *transformer_params.mutable_feed_forward_parameters();
  ff_params.set_no_bias(true);
  ff_params.set_activation(TransformerParameters::GELU);
  ff_params.set_pre_norm(TransformerParameters::RMS_NORM);
  ff_params.set_post_norm(TransformerParameters::RMS_NORM);

  TransformerParameters::FinalProjectParameters& fp_params =
      *transformer_params.mutable_final_project_parameters();
  fp_params.set_no_bias(true);
  fp_params.set_soft_cap_value(30.0f);

  return llm_params;
}

LlmParameters GetGemma3_1BParams() {
  LlmParameters llm_params;
  llm_params.set_start_token_id(2);
  llm_params.add_stop_tokens("<eos>");
  llm_params.add_stop_tokens("<end_of_turn>");

  // New tokenizer
  llm_params.set_vocab_size(262144);

  // We don't use gemma3_bfloat16_fix for the 1B variant because the maximum
  // activations so far seem to stay below the f16 max cap.

  TransformerParameters& transformer_params =
      *llm_params.mutable_transformer_parameters();
  transformer_params.set_batch_size(kBatchSize);
  transformer_params.set_embedding_dim(1152);
  transformer_params.set_hidden_dimension(6 * 1152);
  transformer_params.set_head_dimension(256);
  transformer_params.set_num_heads(4);
  transformer_params.set_num_stacks(26);
  transformer_params.set_num_kv_heads(1);

  transformer_params.set_pre_norm(TransformerParameters::RMS_NORM);
  transformer_params.set_post_norm(TransformerParameters::RMS_NORM);
  transformer_params.set_final_norm(TransformerParameters::RMS_NORM);
  transformer_params.set_skip_absolute_positional_embeddings(true);
  // LLLLLGLLLLLG...
  transformer_params.set_num_local_layers_per_global(5);
  transformer_params.set_global_rope_wavelength(1000000.0f);

  TransformerParameters::SelfAttentionParameters& sa_params =
      *transformer_params.mutable_self_attention_parameters();
  sa_params.set_attention_mask_type(TransformerParameters::CAUSAL);
  sa_params.set_qkv_no_bias(true);
  sa_params.set_post_proj_no_bias(true);
  sa_params.set_attention_scale_type(
      TransformerParameters::SCALE_TYPE_INV_SQRT_HEAD_DIM);
  // Disable softcap
  sa_params.set_soft_cap_value(0.0f);
  // Enable qk norms
  sa_params.set_qk_norm(true);
  sa_params.set_sliding_window_size(512);

  TransformerParameters::FeedForwardParameters& ff_params =
      *transformer_params.mutable_feed_forward_parameters();
  ff_params.set_no_bias(true);
  ff_params.set_activation(TransformerParameters::GELU);
  ff_params.set_pre_norm(TransformerParameters::RMS_NORM);
  ff_params.set_post_norm(TransformerParameters::RMS_NORM);

  TransformerParameters::FinalProjectParameters& fp_params =
      *transformer_params.mutable_final_project_parameters();
  fp_params.set_no_bias(true);
  // Disable soft cap.
  fp_params.set_soft_cap_value(0.0f);

  return llm_params;
}

LlmParameters GetGemma3_4BParams() {
  LlmParameters llm_params;
  llm_params.set_start_token_id(2);
  llm_params.add_stop_tokens("<eos>");
  llm_params.add_stop_tokens("<end_of_turn>");
  // Vocab is 262144, but with MM tokens, our embedding tensors are size 262208
  llm_params.set_vocab_size(262208);

  TransformerParameters& transformer_params =
      *llm_params.mutable_transformer_parameters();
  transformer_params.set_batch_size(kBatchSize);
  transformer_params.set_embedding_dim(2560);
  transformer_params.set_hidden_dimension(2560 * 4);
  transformer_params.set_head_dimension(256);
  transformer_params.set_num_heads(8);
  transformer_params.set_num_stacks(34);
  transformer_params.set_num_kv_heads(4);

  transformer_params.set_pre_norm(TransformerParameters::RMS_NORM);
  transformer_params.set_post_norm(TransformerParameters::RMS_NORM);
  transformer_params.set_final_norm(TransformerParameters::RMS_NORM);
  transformer_params.set_skip_absolute_positional_embeddings(true);
  // LLLLLGLLLLLG...
  transformer_params.set_num_local_layers_per_global(5);
  transformer_params.set_global_rope_wavelength(1000000.0f);
  transformer_params.set_global_rope_scaling(8.0f);

  // To allow fp16 usage when bf16 trained
  transformer_params.set_gemma3_bfloat16_fix(true);

  TransformerParameters::SelfAttentionParameters& sa_params =
      *transformer_params.mutable_self_attention_parameters();
  sa_params.set_attention_mask_type(TransformerParameters::CAUSAL);
  sa_params.set_qkv_no_bias(true);
  sa_params.set_post_proj_no_bias(true);
  sa_params.set_attention_scale_type(
      TransformerParameters::SCALE_TYPE_INV_SQRT_HEAD_DIM);
  // Disable softcap
  sa_params.set_soft_cap_value(0.0f);
  // Enable qk norms
  sa_params.set_qk_norm(true);
  sa_params.set_sliding_window_size(1024);

  TransformerParameters::FeedForwardParameters& ff_params =
      *transformer_params.mutable_feed_forward_parameters();
  ff_params.set_no_bias(true);
  ff_params.set_activation(TransformerParameters::GELU);
  ff_params.set_pre_norm(TransformerParameters::RMS_NORM);
  ff_params.set_post_norm(TransformerParameters::RMS_NORM);

  TransformerParameters::FinalProjectParameters& fp_params =
      *transformer_params.mutable_final_project_parameters();
  fp_params.set_no_bias(true);
  // Disable soft cap.
  fp_params.set_soft_cap_value(0.0f);
  return llm_params;
}

LlmParameters GetGemma3_12BParams() {
  LlmParameters llm_params;
  llm_params.set_start_token_id(2);
  llm_params.add_stop_tokens("<eos>");
  llm_params.add_stop_tokens("<end_of_turn>");
  // Vocab is 262144, but with MM tokens, our embedding tensors are size 262208
  llm_params.set_vocab_size(262208);

  TransformerParameters& transformer_params =
      *llm_params.mutable_transformer_parameters();
  transformer_params.set_batch_size(kBatchSize);
  transformer_params.set_embedding_dim(3840);
  transformer_params.set_hidden_dimension(15360);  // 3840 * 4
  transformer_params.set_head_dimension(256);
  transformer_params.set_num_heads(16);
  transformer_params.set_num_stacks(48);
  transformer_params.set_num_kv_heads(8);

  transformer_params.set_pre_norm(TransformerParameters::RMS_NORM);
  transformer_params.set_post_norm(TransformerParameters::RMS_NORM);
  transformer_params.set_final_norm(TransformerParameters::RMS_NORM);
  transformer_params.set_skip_absolute_positional_embeddings(true);
  // LLLLLGLLLLLG...
  transformer_params.set_num_local_layers_per_global(5);
  transformer_params.set_global_rope_wavelength(1000000.0f);
  transformer_params.set_global_rope_scaling(8.0f);

  // To allow fp16 usage when bf16 trained
  transformer_params.set_gemma3_bfloat16_fix(true);

  TransformerParameters::SelfAttentionParameters& sa_params =
      *transformer_params.mutable_self_attention_parameters();
  sa_params.set_attention_mask_type(TransformerParameters::CAUSAL);
  sa_params.set_qkv_no_bias(true);
  sa_params.set_post_proj_no_bias(true);
  sa_params.set_attention_scale_type(
      TransformerParameters::SCALE_TYPE_INV_SQRT_HEAD_DIM);
  // Disable softcap
  sa_params.set_soft_cap_value(0.0f);
  // Enable qk norms
  sa_params.set_qk_norm(true);
  sa_params.set_sliding_window_size(1024);

  TransformerParameters::FeedForwardParameters& ff_params =
      *transformer_params.mutable_feed_forward_parameters();
  ff_params.set_no_bias(true);
  ff_params.set_activation(TransformerParameters::GELU);
  ff_params.set_pre_norm(TransformerParameters::RMS_NORM);
  ff_params.set_post_norm(TransformerParameters::RMS_NORM);

  TransformerParameters::FinalProjectParameters& fp_params =
      *transformer_params.mutable_final_project_parameters();
  fp_params.set_no_bias(true);
  // Disable soft cap.
  fp_params.set_soft_cap_value(0.0f);
  return llm_params;
}

LlmParameters GetGemma3_27BParams() {
  LlmParameters llm_params;
  llm_params.set_start_token_id(2);
  llm_params.add_stop_tokens("<eos>");
  llm_params.add_stop_tokens("<end_of_turn>");
  // Vocab is 262144, but with MM tokens, our embedding tensors are size 262208
  llm_params.set_vocab_size(262208);

  TransformerParameters& transformer_params =
      *llm_params.mutable_transformer_parameters();
  transformer_params.set_batch_size(kBatchSize);
  transformer_params.set_embedding_dim(5376);
  transformer_params.set_hidden_dimension(21504);  // 5376 * 4
  transformer_params.set_head_dimension(128);      // Not 256!
  transformer_params.set_num_heads(32);
  transformer_params.set_num_stacks(62);
  transformer_params.set_num_kv_heads(16);

  transformer_params.set_pre_norm(TransformerParameters::RMS_NORM);
  transformer_params.set_post_norm(TransformerParameters::RMS_NORM);
  transformer_params.set_final_norm(TransformerParameters::RMS_NORM);
  transformer_params.set_skip_absolute_positional_embeddings(true);
  // LLLLLGLLLLLG...
  transformer_params.set_num_local_layers_per_global(5);
  transformer_params.set_global_rope_wavelength(1000000.0f);
  transformer_params.set_global_rope_scaling(8.0f);

  // To allow fp16 usage when bf16 trained
  transformer_params.set_gemma3_bfloat16_fix(true);

  TransformerParameters::SelfAttentionParameters& sa_params =
      *transformer_params.mutable_self_attention_parameters();
  sa_params.set_attention_mask_type(TransformerParameters::CAUSAL);
  sa_params.set_qkv_no_bias(true);
  sa_params.set_post_proj_no_bias(true);
  // NOTE: This is different from previous Gemma3 models!
  // It corresponds to `query_pre_attn_scalar = 168`, since 5376/32 = 168.
  sa_params.set_attention_scale_type(
      TransformerParameters::SCALE_TYPE_INV_SQRT_D_MODEL_DIV_NUM_HEADS);
  // Disable softcap
  sa_params.set_soft_cap_value(0.0f);
  // Enable qk norms
  sa_params.set_qk_norm(true);
  sa_params.set_sliding_window_size(1024);

  TransformerParameters::FeedForwardParameters& ff_params =
      *transformer_params.mutable_feed_forward_parameters();
  ff_params.set_no_bias(true);
  ff_params.set_activation(TransformerParameters::GELU);
  ff_params.set_pre_norm(TransformerParameters::RMS_NORM);
  ff_params.set_post_norm(TransformerParameters::RMS_NORM);

  TransformerParameters::FinalProjectParameters& fp_params =
      *transformer_params.mutable_final_project_parameters();
  fp_params.set_no_bias(true);
  // Disable soft cap.
  fp_params.set_soft_cap_value(0.0f);
  return llm_params;
}

LlmParameters GetFalconRW1BParams() {
  LlmParameters llm_params;
  llm_params.set_start_token_id(1);
  llm_params.add_stop_tokens("<|endoftext|>");
  llm_params.set_vocab_size(50304);

  TransformerParameters& transformer_params =
      *llm_params.mutable_transformer_parameters();
  transformer_params.set_batch_size(kBatchSize);
  transformer_params.set_embedding_dim(2048);
  transformer_params.set_hidden_dimension(4 * 2048);
  transformer_params.set_head_dimension(64);
  transformer_params.set_num_heads(32);
  // `num_kv_heads` is same as `num_heads` in MHA.
  transformer_params.set_num_kv_heads(32);
  transformer_params.set_num_stacks(24);
  transformer_params.set_pre_norm(TransformerParameters::LAYER_NORM);
  transformer_params.set_post_norm(TransformerParameters::NO_NORM);
  transformer_params.set_final_norm(TransformerParameters::LAYER_NORM);
  transformer_params.set_skip_absolute_positional_embeddings(true);

  TransformerParameters::SelfAttentionParameters& sa_params =
      *transformer_params.mutable_self_attention_parameters();
  sa_params.set_attention_mask_type(TransformerParameters::CAUSAL);
  sa_params.set_qkv_no_bias(false);
  sa_params.set_post_proj_no_bias(false);
  sa_params.set_attention_scale_type(
      TransformerParameters::SCALE_TYPE_INV_SQRT_HEAD_DIM);
  // Disable soft cap.
  sa_params.set_soft_cap_value(0.0f);

  TransformerParameters::FeedForwardParameters& ff_params =
      *transformer_params.mutable_feed_forward_parameters();
  ff_params.set_no_bias(false);
  ff_params.set_activation(TransformerParameters::GELU);
  ff_params.set_pre_norm(TransformerParameters::LAYER_NORM);
  ff_params.set_post_norm(TransformerParameters::NO_NORM);

  TransformerParameters::FinalProjectParameters& fp_params =
      *transformer_params.mutable_final_project_parameters();
  fp_params.set_no_bias(true);
  // Disable soft cap.
  fp_params.set_soft_cap_value(0.0f);

  return llm_params;
}

LlmParameters GetStablelm4E1T3BParams() {
  LlmParameters llm_params;
  llm_params.set_start_token_id(0);
  llm_params.add_stop_tokens("<|endoftext|>");
  llm_params.set_vocab_size(50304);

  TransformerParameters& transformer_params =
      *llm_params.mutable_transformer_parameters();
  transformer_params.set_batch_size(kBatchSize);
  transformer_params.set_embedding_dim(2560);
  transformer_params.set_hidden_dimension(6912);
  transformer_params.set_head_dimension(80);
  transformer_params.set_num_heads(32);
  // MHA.
  transformer_params.set_num_kv_heads(0);
  transformer_params.set_num_stacks(32);
  transformer_params.set_pre_norm(TransformerParameters::LAYER_NORM);
  transformer_params.set_post_norm(TransformerParameters::NO_NORM);
  transformer_params.set_final_norm(TransformerParameters::LAYER_NORM);
  transformer_params.set_skip_absolute_positional_embeddings(true);

  TransformerParameters::SelfAttentionParameters& sa_params =
      *transformer_params.mutable_self_attention_parameters();
  sa_params.set_attention_mask_type(TransformerParameters::CAUSAL);
  sa_params.set_qkv_no_bias(true);
  sa_params.set_post_proj_no_bias(true);
  sa_params.set_attention_scale_type(
      TransformerParameters::SCALE_TYPE_INV_SQRT_HEAD_DIM);
  // Disable soft cap.
  sa_params.set_soft_cap_value(0.0f);

  TransformerParameters::FeedForwardParameters& ff_params =
      *transformer_params.mutable_feed_forward_parameters();
  ff_params.set_no_bias(true);
  ff_params.set_activation(TransformerParameters::SILU);
  ff_params.set_pre_norm(TransformerParameters::LAYER_NORM);
  ff_params.set_post_norm(TransformerParameters::NO_NORM);

  TransformerParameters::FinalProjectParameters& fp_params =
      *transformer_params.mutable_final_project_parameters();
  fp_params.set_no_bias(true);
  // Disable soft cap.
  fp_params.set_soft_cap_value(0.0f);

  return llm_params;
}

LlmParameters GetPhi2Params() {
  LlmParameters llm_params;
  llm_params.set_start_token_id(50256);
  llm_params.add_stop_tokens("<|endoftext|>");
  llm_params.set_vocab_size(51200);

  TransformerParameters& transformer_params =
      *llm_params.mutable_transformer_parameters();
  transformer_params.set_batch_size(kBatchSize);
  transformer_params.set_embedding_dim(2560);
  transformer_params.set_hidden_dimension(10240);
  transformer_params.set_head_dimension(80);
  transformer_params.set_num_heads(32);
  // MHA.
  transformer_params.set_num_kv_heads(0);
  transformer_params.set_num_stacks(32);
  transformer_params.set_pre_norm(TransformerParameters::LAYER_NORM);
  transformer_params.set_post_norm(TransformerParameters::NO_NORM);
  transformer_params.set_final_norm(TransformerParameters::LAYER_NORM);
  transformer_params.set_skip_absolute_positional_embeddings(true);

  TransformerParameters::SelfAttentionParameters& sa_params =
      *transformer_params.mutable_self_attention_parameters();
  sa_params.set_qkv_no_bias(false);
  sa_params.set_post_proj_no_bias(false);
  sa_params.set_attention_mask_type(TransformerParameters::CAUSAL);
  sa_params.set_attention_scale_type(
      TransformerParameters::SCALE_TYPE_INV_SQRT_HEAD_DIM);
  // Disable soft cap.
  sa_params.set_soft_cap_value(0.0f);

  TransformerParameters::FeedForwardParameters& ff_params =
      *transformer_params.mutable_feed_forward_parameters();
  ff_params.set_no_bias(false);
  ff_params.set_activation(TransformerParameters::GELU);
  ff_params.set_pre_norm(TransformerParameters::NO_NORM);
  ff_params.set_post_norm(TransformerParameters::NO_NORM);

  TransformerParameters::FinalProjectParameters& fp_params =
      *transformer_params.mutable_final_project_parameters();
  fp_params.set_no_bias(false);
  // Disable soft cap.
  fp_params.set_soft_cap_value(0.0f);

  return llm_params;
}

}  // namespace mediapipe::tasks::genai::llm_utils
