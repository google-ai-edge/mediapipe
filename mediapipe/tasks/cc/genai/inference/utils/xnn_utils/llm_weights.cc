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

#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/llm_weights.h"

#include <sys/stat.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/genai/inference/proto/transformer_params.pb.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/pack_weights_cache.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/tflite_weight_accessor.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/utils.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/xnn_tensor.h"

namespace mediapipe::tasks::genai::xnn_utils {

namespace {

using FeedForwardWeights = LlmWeights::FeedForwardWeights;
using SelfAttentionWeights = LlmWeights::SelfAttentionWeights;
using TransformerParameters = odml::infra::proto::TransformerParameters;

LlmParams::Norm TransformerParametersProtoNormTypeToLlmParamsNormType(
    TransformerParameters::Norm norm_type) {
  switch (norm_type) {
    case TransformerParameters::NORM_UNSPECIFIED:
      ABSL_LOG(DFATAL) << "Unspecified norm type.";
      return LlmParams::Norm::UNSPECIFIED;
    case TransformerParameters::NO_NORM:
      return LlmParams::Norm::NO_NORM;
    case TransformerParameters::RMS_NORM:
      return LlmParams::Norm::RMS_NORM;
    case TransformerParameters::LAYER_NORM:
      return LlmParams::Norm::LAYER_NORM;
    default:
      ABSL_LOG(DFATAL) << "Unknown norm type: " << norm_type;
  }
  return LlmParams::Norm::UNSPECIFIED;
}

// According to norm_type, load necessary weights with given basename.
absl::StatusOr<std::optional<LlmWeights::NormWeights>> LoadNormWeights(
    LlmParams::Norm norm_type, const LlmParams& params,
    absl::string_view basename, WeightAccessor& weight_accessor) {
  switch (norm_type) {
    case LlmParams::Norm::UNSPECIFIED:
      break;
    case LlmParams::Norm::NO_NORM:
      break;
    case LlmParams::Norm::RMS_NORM: {
      auto rms_norm_weights = RMSNormWeights();
      MP_ASSIGN_OR_RETURN(
          rms_norm_weights.norm_weight,
          weight_accessor.LoadWeight(absl::StrCat(basename, ".scale"),
                                     {params.model_dim_D}));
      return rms_norm_weights;
    }
    case LlmParams::Norm::LAYER_NORM: {
      auto layer_norm_weights = LayerNormWeights();
      MP_ASSIGN_OR_RETURN(
          layer_norm_weights.beta,
          weight_accessor.LoadWeight(absl::StrCat(basename, ".bias"),
                                     {1, 1, params.model_dim_D}));
      MP_ASSIGN_OR_RETURN(
          layer_norm_weights.gamma,
          weight_accessor.LoadWeight(absl::StrCat(basename, ".scale"),
                                     {1, 1, params.model_dim_D}));
      return layer_norm_weights;
    }
    default:
      break;
  }
  return std::nullopt;
}

}  // namespace

LlmParams LlmParams::FromLLMParametersProto(
    const odml::infra::proto::LlmParameters& llm_params) {
  const auto& transformer_params = llm_params.transformer_parameters();
  LlmParams params = {
      .num_transformer_M = static_cast<size_t>(transformer_params.num_stacks()),
      .batch_size_B = static_cast<size_t>(transformer_params.batch_size()),
      .seq_size_T = static_cast<size_t>(transformer_params.max_seq_length()),
      .model_dim_D = static_cast<size_t>(transformer_params.embedding_dim()),
      .hidden_dim_HD =
          static_cast<size_t>(transformer_params.hidden_dimension()),
      .head_dim_H = static_cast<size_t>(transformer_params.head_dimension()),
      .n_heads_N = static_cast<size_t>(transformer_params.num_heads()),
      .voc_size_V = static_cast<size_t>(llm_params.vocab_size()),
      .query_rescale_factor = transformer_params.query_rescale_factor(),

      .num_kv_heads =
          static_cast<size_t>(transformer_params.num_kv_heads() == 0
                                  ? transformer_params.num_heads()
                                  : transformer_params.num_kv_heads()),
      .enable_kv_cache = true,
      .enable_dynamic_shape = true};
  if (llm_params.has_num_draft_tokens()) {
    params.draft_size_G = llm_params.num_draft_tokens();
  }
  switch (
      transformer_params.self_attention_parameters().attention_mask_type()) {
    case TransformerParameters::UNSPECIFIED:
      ABSL_LOG(DFATAL) << "Unspecified attention_mask_type, assuming causal";
      params.model_type = LlmParams::ModelType::UNSPECIFIED;
      break;
    case TransformerParameters::CAUSAL:
      params.model_type = LlmParams::ModelType::CAUSAL;
      break;
    case TransformerParameters::PREFIX:
      params.model_type = LlmParams::ModelType::PREFIX;
      break;
    default:
      ABSL_LOG(DFATAL) << "Unknown attention_mask_type: "
                       << transformer_params.self_attention_parameters()
                              .attention_mask_type();
  }
  params.ff_params = LlmParams::FeedForwardParams{
      .no_bias = transformer_params.feed_forward_parameters().no_bias(),
  };
  params.final_proj_params = LlmParams::FinalProjectParams{
      .no_bias = transformer_params.final_project_parameters().no_bias(),
      .soft_cap_value =
          transformer_params.final_project_parameters().soft_cap_value(),
  };
  switch (transformer_params.feed_forward_parameters().activation()) {
    case TransformerParameters::ACTIVATION_UNSPECIFIED:
      ABSL_LOG(DFATAL) << "Unspecified feed_forward_parameters.activation.";
      params.ff_params.activation = LlmParams::Activation::UNSPECIFIED;
      break;
    case TransformerParameters::GELU:
      params.ff_params.activation = LlmParams::Activation::GELU;
      break;
    case TransformerParameters::SILU:
      params.ff_params.activation = LlmParams::Activation::SILU;
      break;
    case TransformerParameters::RELU:
      params.ff_params.activation = LlmParams::Activation::RELU;
      break;
    case TransformerParameters::RELU1P5:
      params.ff_params.activation = LlmParams::Activation::RELU1P5;
      break;
    default:
      ABSL_LOG(DFATAL)
          << "Unknown feed_forward_parameters.activation: "
          << transformer_params.feed_forward_parameters().activation();
  }
  params.sa_params.qkv_no_bias =
      transformer_params.self_attention_parameters().qkv_no_bias();
  params.sa_params.post_proj_no_bias =
      transformer_params.self_attention_parameters().post_proj_no_bias();
  params.sa_params.pre_norm =
      TransformerParametersProtoNormTypeToLlmParamsNormType(
          transformer_params.pre_norm());
  params.sa_params.post_norm =
      TransformerParametersProtoNormTypeToLlmParamsNormType(
          transformer_params.post_norm());
  params.sa_params.soft_cap_value =
      transformer_params.self_attention_parameters().soft_cap_value();
  params.ff_params.pre_norm =
      TransformerParametersProtoNormTypeToLlmParamsNormType(
          transformer_params.feed_forward_parameters().pre_norm());
  params.ff_params.post_norm =
      TransformerParametersProtoNormTypeToLlmParamsNormType(
          transformer_params.feed_forward_parameters().post_norm());
  params.final_norm = TransformerParametersProtoNormTypeToLlmParamsNormType(
      transformer_params.final_norm());
  params.skip_absolute_positional_embeddings =
      transformer_params.skip_absolute_positional_embeddings();
  if (transformer_params.self_attention_parameters()
          .has_attention_scale_type()) {
    switch (
        transformer_params.self_attention_parameters().attention_scale_type()) {
      case TransformerParameters::SCALE_TYPE_UNSPECIFIED:
        ABSL_LOG(DFATAL) << "Unspecified attention_scale_type.";
        params.sa_params.attention_scale_type =
            LlmParams::AttentionScaleType::UNSPECIFIED;
        break;
      case TransformerParameters::SCALE_TYPE_PER_DIM_SCALE:
        params.sa_params.attention_scale_type =
            LlmParams::AttentionScaleType::PER_DIM_SCALE;
        break;
      case TransformerParameters::SCALE_TYPE_INV_SQRT_HEAD_DIM:
        params.sa_params.attention_scale_type =
            LlmParams::AttentionScaleType::INV_SQRT_HEAD_DIM;
        break;
      default:
        ABSL_LOG(DFATAL) << "Unknown attention_scale_type: "
                         << transformer_params.self_attention_parameters()
                                .attention_scale_type();
    }
  } else {
    if (transformer_params.num_kv_heads() == 0 ||
        transformer_params.num_heads() == transformer_params.num_kv_heads()) {
      // If MHA, PER_DIM_SCALE is used.
      params.sa_params.attention_scale_type =
          LlmParams::AttentionScaleType::PER_DIM_SCALE;
    } else {
      // If MQA or GQA, INV_SQRT_HEAD_DIM is used.
      params.sa_params.attention_scale_type =
          LlmParams::AttentionScaleType::INV_SQRT_HEAD_DIM;
    }
  }

  return params;
}

absl::StatusOr<std::shared_ptr<Tensor>>
LlmWeightsLoader::TryCacheThenLoadSelfAttention(
    absl::string_view filename_prefix, absl::string_view alt_filename_prefix,
    bool is_query) {
  std::shared_ptr<Tensor> r;
  if (!is_query) {
    MP_ASSIGN_OR_RETURN(
        r, weight_accessor_->LoadTransposedWeight(
               filename_prefix,
               {params_.model_dim_D, params_.num_kv_heads * params_.head_dim_H},
               1));
    if (!r) {
      MP_ASSIGN_OR_RETURN(r, weight_accessor_->LoadTransposedWeight(
                                 alt_filename_prefix,
                                 {params_.model_dim_D,
                                  params_.num_kv_heads * params_.head_dim_H},
                                 1));
    }
    RET_CHECK(r) << "Could not load " << filename_prefix << " (or "
                 << alt_filename_prefix << ")";
    r->SetMetadata(xnn_utils::kKeySelfAttentionReshapedWeight,
                   params_.num_kv_heads);
  } else {
    MP_ASSIGN_OR_RETURN(
        r,
        weight_accessor_->LoadTransposedWeight(
            filename_prefix,
            {params_.model_dim_D, params_.n_heads_N * params_.head_dim_H}, 1));
    if (!r) {
      MP_ASSIGN_OR_RETURN(
          r, weight_accessor_->LoadTransposedWeight(
                 alt_filename_prefix,
                 {params_.model_dim_D, params_.n_heads_N * params_.head_dim_H},
                 1));
    }
    RET_CHECK(r) << "Could not load " << filename_prefix << " (or "
                 << alt_filename_prefix << ")";
    r->SetMetadata(xnn_utils::kKeySelfAttentionReshapedWeight,
                   params_.n_heads_N);
  }
  r->SetMetadata(kKeyInDimLastInWeight, 1);
  return r;
}

absl::StatusOr<FeedForwardWeights> LlmWeightsLoader::LoadFeedForward(
    int layer_id) {
  const auto& params = params_;
  auto ff_file_prefix =
      absl::StrCat(kTransformerWeightPrefix, layer_id, ".ff_layer.");
  FeedForwardWeights feed_forward;

  MP_ASSIGN_OR_RETURN(
      feed_forward.pre_norm_weight,
      LoadNormWeights(params.ff_params.pre_norm, params,
                      absl::StrCat(ff_file_prefix, "pre_layer_norm"),
                      *weight_accessor_));

  MP_ASSIGN_OR_RETURN(
      feed_forward.post_norm_weight,
      LoadNormWeights(params.ff_params.post_norm, params,
                      absl::StrCat(ff_file_prefix, "post_layer_norm"),
                      *weight_accessor_));

  MP_ASSIGN_OR_RETURN(feed_forward.layer_1_weight,
                      weight_accessor_->LoadTransposedWeight(
                          absl::StrCat(ff_file_prefix, "ffn_layer1.w"),
                          {params.model_dim_D, params.hidden_dim_HD},
                          /*original_dim_scale=*/1));
  if (!feed_forward.layer_1_weight) {
    MP_ASSIGN_OR_RETURN(feed_forward.layer_1_weight,
                        weight_accessor_->LoadTransposedWeight(
                            absl::StrCat(ff_file_prefix, "ffn_layer1.linear.w"),
                            {params.model_dim_D, params.hidden_dim_HD},
                            /*original_dim_scale=*/1));
  }
  MP_ASSIGN_OR_RETURN(feed_forward.layer_1_gate_weight,
                      weight_accessor_->LoadTransposedWeight(
                          absl::StrCat(ff_file_prefix, "ffn_layer1_gate.w"),
                          {params.model_dim_D, params.hidden_dim_HD},
                          /*original_dim_scale=*/1));
  if (!feed_forward.layer_1_gate_weight) {
    MP_ASSIGN_OR_RETURN(
        feed_forward.layer_1_gate_weight,
        weight_accessor_->LoadTransposedWeight(
            absl::StrCat(ff_file_prefix, "ffn_layer1_gate.linear.w"),
            {params.model_dim_D, params.hidden_dim_HD},
            /*original_dim_scale=*/1));
  }
  MP_ASSIGN_OR_RETURN(
      feed_forward.layer_2_weight,
      weight_accessor_->LoadTransposedWeight(
          absl::StrCat(ff_file_prefix, "ffn_layer2.w"),
          Tensor::DimsType{params.hidden_dim_HD, params.model_dim_D},
          /*original_dim_scale=*/1));
  if (!feed_forward.layer_2_weight) {
    MP_ASSIGN_OR_RETURN(
        feed_forward.layer_2_weight,
        weight_accessor_->LoadTransposedWeight(
            absl::StrCat(ff_file_prefix, "ffn_layer2.linear.w"),
            Tensor::DimsType{params.hidden_dim_HD, params.model_dim_D},
            /*original_dim_scale=*/1));
  }

  if (!params.ff_params.no_bias) {
    MP_ASSIGN_OR_RETURN(feed_forward.layer_1_bias,
                        weight_accessor_->LoadWeight(
                            absl::StrCat(ff_file_prefix, "ffn_layer1.bias.b"),
                            {params.hidden_dim_HD}));
    MP_ASSIGN_OR_RETURN(
        feed_forward.layer_1_gate_bias,
        weight_accessor_->LoadWeight(
            absl::StrCat(ff_file_prefix, "ffn_layer1_gate.bias.b"),
            {params.hidden_dim_HD}));
    MP_ASSIGN_OR_RETURN(feed_forward.layer_2_bias,
                        weight_accessor_->LoadWeight(
                            absl::StrCat(ff_file_prefix, "ffn_layer2.bias.b"),
                            {params.model_dim_D}));
  }

  return feed_forward;
}

absl::StatusOr<SelfAttentionWeights> LlmWeightsLoader::LoadSelfAttention(
    int layer_id) {
  const auto& params = params_;
  SelfAttentionWeights self_attention;

  auto sa_file_prefix = absl::StrCat(kTransformerWeightPrefix, layer_id);

  MP_ASSIGN_OR_RETURN(
      self_attention.pre_norm_weight,
      LoadNormWeights(params.sa_params.pre_norm, params,
                      absl::StrCat(sa_file_prefix, ".pre_layer_norm"),
                      *weight_accessor_));
  MP_ASSIGN_OR_RETURN(
      self_attention.post_norm_weight,
      LoadNormWeights(params.sa_params.post_norm, params,
                      absl::StrCat(sa_file_prefix, ".post_layer_norm"),
                      *weight_accessor_));

  absl::StrAppend(&sa_file_prefix, ".self_attention.");

  MP_ASSIGN_OR_RETURN(
      self_attention.k_weight,
      TryCacheThenLoadSelfAttention(absl::StrCat(sa_file_prefix, "k.w"),
                                    absl::StrCat(sa_file_prefix, "k.linear.w"),
                                    /*is_query=*/false));
  MP_ASSIGN_OR_RETURN(
      self_attention.q_weight,
      TryCacheThenLoadSelfAttention(absl::StrCat(sa_file_prefix, "q.w"),
                                    absl::StrCat(sa_file_prefix, "q.linear.w"),
                                    /*is_query=*/true));
  MP_ASSIGN_OR_RETURN(
      self_attention.v_weight,
      TryCacheThenLoadSelfAttention(absl::StrCat(sa_file_prefix, "v.w"),
                                    absl::StrCat(sa_file_prefix, "v.linear.w"),
                                    /*is_query=*/false));

  if (!params.sa_params.qkv_no_bias) {
    MP_ASSIGN_OR_RETURN(
        self_attention.q_bias,
        weight_accessor_->LoadWeight(absl::StrCat(sa_file_prefix, "q.bias.b"),
                                     {params.n_heads_N * params.head_dim_H}));
    MP_ASSIGN_OR_RETURN(
        self_attention.k_bias,
        weight_accessor_->LoadWeight(absl::StrCat(sa_file_prefix, "k.bias.b"),
                                     {params.n_heads_N * params.head_dim_H}));
    MP_ASSIGN_OR_RETURN(
        self_attention.v_bias,
        weight_accessor_->LoadWeight(absl::StrCat(sa_file_prefix, "v.bias.b"),
                                     {params.n_heads_N * params.head_dim_H}));
  }

  if (params.sa_params.attention_scale_type ==
      LlmParams::AttentionScaleType::PER_DIM_SCALE) {
    MP_ASSIGN_OR_RETURN(
        self_attention.per_dim_scale,
        weight_accessor_->LoadWeight(
            absl::StrCat(sa_file_prefix, "per_dim_scale.per_dim_scale"),
            {params.head_dim_H}));
  }
  MP_ASSIGN_OR_RETURN(
      self_attention.post_proj_weight,
      weight_accessor_->LoadWeight(
          absl::StrCat(sa_file_prefix, "post.w"),
          {params.model_dim_D, params.n_heads_N * params.head_dim_H},
          /*dim_scale_if_any=*/0));
  if (!self_attention.post_proj_weight) {
    MP_ASSIGN_OR_RETURN(
        self_attention.post_proj_weight,
        weight_accessor_->LoadWeight(
            absl::StrCat(sa_file_prefix, "post.linear.w"),
            {params.model_dim_D, params.n_heads_N * params.head_dim_H},
            /*dim_scale_if_any=*/0));
  }
  if (!params.sa_params.post_proj_no_bias) {
    MP_ASSIGN_OR_RETURN(
        self_attention.post_proj_bias,
        weight_accessor_->LoadWeight(
            absl::StrCat(sa_file_prefix, "post.bias.b"), {params.model_dim_D}));
  }

  return self_attention;
}

absl::StatusOr<LlmWeights> LlmWeightsLoader::LoadWeights() {
  RET_CHECK(weight_accessor_);

  LlmWeights result;

  for (int layer_id = 0; layer_id < params_.num_transformer_M; ++layer_id) {
    MP_ASSIGN_OR_RETURN(auto ff, LoadFeedForward(layer_id));
    result.ffs.push_back(std::move(ff));
    MP_ASSIGN_OR_RETURN(auto sa, LoadSelfAttention(layer_id));
    result.sas.push_back(std::move(sa));
  }

  MP_ASSIGN_OR_RETURN(result.final_norm_weight,
                      LoadNormWeights(params_.final_norm, params_,
                                      "params.lm.final_ln", *weight_accessor_));

  MP_ASSIGN_OR_RETURN(
      result.softmax_linear,
      weight_accessor_->LoadTransposedWeight(
          absl::StrReplaceAll(kLogitsFfnWeightFilename, {{".linear.", "."}}),
          {params_.model_dim_D, params_.voc_size_V}, 1));
  if (!result.softmax_linear) {
    MP_ASSIGN_OR_RETURN(result.softmax_linear,
                        weight_accessor_->LoadTransposedWeight(
                            kLogitsFfnWeightFilename,
                            {params_.model_dim_D, params_.voc_size_V}, 1));
  }
  if (!params_.final_proj_params.no_bias) {
    MP_ASSIGN_OR_RETURN(result.softmax_bias,
                        weight_accessor_->LoadWeight(kLogitsFfnBiasFilename,
                                                     {params_.voc_size_V}));
  }
  RET_CHECK(result.softmax_linear) << kLogitsFfnWeightFilename;

  MP_ASSIGN_OR_RETURN(
      result.token_embedding,
      weight_accessor_->LoadWeight(kTokenEmbedding,
                                   {params_.voc_size_V, params_.model_dim_D},
                                   /*dim_scale_if_any=*/0));

  return result;
}

DefaultLlmWeightsLoader::DefaultLlmWeightsLoader(absl::string_view weight_path,
                                                 const LlmParams& params)
    : LlmWeightsLoader(nullptr, params) {
  xnn_weights_cache_ = std::make_shared<PackWeightsCache>(
      params.cache_dir.empty()
          ? absl::StrCat(weight_path, ".cache")
          : mediapipe::file::JoinPath(
                params.cache_dir,
                absl::StrCat(mediapipe::file::Basename(weight_path),
                             ".cache")));
  ABSL_CHECK_OK(xnn_weights_cache_->Initialize());
  weight_accessor_ = std::make_unique<WeightAccessorCompositeWithCache>(
      std::make_shared<TfLiteWeightAccessor>(weight_path),
      xnn_weights_cache_.get());
}

}  // namespace mediapipe::tasks::genai::xnn_utils
