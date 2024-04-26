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

#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/llm.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/graph_builder.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/llm_weights.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/sampling.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/utils.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/xnn_tensor.h"
#include "xnnpack.h"  // from @XNNPACK

namespace mediapipe::tasks::genai {
namespace xnn_utils {
namespace {

using FeedForwardWeights = LlmWeights::FeedForwardWeights;
using SelfAttentionWeights = LlmWeights::SelfAttentionWeights;

}  // namespace

class PrefixDecodeLlm : public Llm {
 public:
  PrefixDecodeLlm(std::unique_ptr<Llm> prefix_llm, XnnGraph&& other)
      : Llm(std::move(other)), prefix_llm_(std::move(prefix_llm)) {}
  ~PrefixDecodeLlm() override = default;

  absl::Status InitInputTokens(
      absl::Span<const std::vector<int>> batch_input_ids) override {
    MP_RETURN_IF_ERROR(Reset());

    MP_RETURN_IF_ERROR(prefix_llm_->InitInputTokens(batch_input_ids));
    for (size_t batch_size = 0; batch_size < llm_params_.batch_size_B;
         ++batch_size) {
      auto& prev_ids = prev_ids_[batch_size];
      const auto& input_ids = batch_input_ids[batch_size];
      prev_ids.insert(prev_ids.end(), input_ids.begin(), input_ids.end());
    }
    computed_tokens_count_ = batch_input_ids[0].size();
    // prev_id.size - 1 is the output.
    return prefix_llm_->Run();
  }

  absl::Status AddInputTokens(
      absl::Span<const std::vector<int>> batch_input_ids) override {
    RET_CHECK_EQ(batch_input_ids.size(), llm_params_.batch_size_B);
    RET_CHECK_EQ(prev_ids_.size(), llm_params_.batch_size_B);

    // This check is required because we hardcode transformer_input_'s sequence
    // size to be 1.
    // TODO - b/325325100: make transformer_input_ dynamic shape.
    RET_CHECK_EQ(computed_tokens_count_, prev_ids_[0].size())
        << "Call ComputeLogits() or GetNextToken() before continuous "
           "AddInputTokens()";
    if (prev_ids_[0].empty()) {
      return PrefixDecodeLlm::InitInputTokens(batch_input_ids);
    }

    const size_t decode_step = prev_ids_[0].size() - 1;

    for (size_t batch_size = 0; batch_size < llm_params_.batch_size_B;
         ++batch_size) {
      auto& prev_ids = prev_ids_[batch_size];
      const auto& input_ids = batch_input_ids[batch_size];
      auto slice = prefix_llm_->input_pivot_->Slice(0, batch_size)
                       ->Slice(1, decode_step + 1);
      MP_RETURN_IF_ERROR(GetTokenEmbedding(input_ids, slice->DataAs<float>()));
      prev_ids.insert(prev_ids.end(), input_ids.begin(), input_ids.end());
    }
    return absl::OkStatus();
  }

  absl::Status GetNextToken(std::vector<int>* output_ids) override {
    MP_ASSIGN_OR_RETURN(auto logits_output, ComputeLogits());

    MP_ASSIGN_OR_RETURN(*output_ids, builder_->Sample(*logits_output));

    // output_ids->size() is batch_size, while AddInputTokens() expects [batch,
    // seq=1], thus transform.
    std::vector<std::vector<int>> batch_input_ids(llm_params_.batch_size_B,
                                                  std::vector<int>(1));
    for (size_t batch_size = 0; batch_size < llm_params_.batch_size_B;
         ++batch_size) {
      batch_input_ids[batch_size][0] = (output_ids->at(batch_size));
    }
    return PrefixDecodeLlm::AddInputTokens(batch_input_ids);
  }

  absl::StatusOr<std::shared_ptr<Tensor>> ComputeLogits() override {
    const size_t decode_step = prev_ids_[0].size() - 1;
    VLOG(2) << "Decode step " << decode_step;

    RET_CHECK(decode_step < llm_params_.seq_size_T - 1)
        .SetCode(absl::StatusCode::kOutOfRange);

    // TODO - b/329445989: This copy could be performance bottleneck. Consider
    // [T, B, D] shape.
    for (size_t batch = 0; batch < llm_params_.batch_size_B; ++batch) {
      MP_RETURN_IF_ERROR(transformer_input_->Slice(0, batch)->LoadFromBuffer(
          prefix_llm_->input_pivot_->Slice(0, batch)
              ->Slice(1, decode_step)
              ->Data()));
    }
    // Let builder re-populate the values of these tensors.
    MP_RETURN_IF_ERROR(builder_->InitAttentionMask(
        decode_step, 1, /*is_prefix=*/false, *atten_masks_));
    if (!llm_params_.skip_absolute_positional_embeddings) {
      MP_RETURN_IF_ERROR(
          builder_->InitPosEmbedding(decode_step, 1, *pos_embedding_));
    }
    if (segment_pos_) {
      MP_RETURN_IF_ERROR(
          builder_->InitSegmentPos(decode_step, 1, *segment_pos_));
    }

    if (llm_params_.enable_dynamic_shape) {
      MP_RETURN_IF_ERROR(ReshapeInputResource());
      for (auto& kv_cache : kv_cache_) {
        auto key = kv_cache.k_cache;
        auto value = kv_cache.v_cache;
        key->Resize({llm_params_.batch_size_B, decode_step + 1,
                     llm_params_.num_kv_heads, llm_params_.head_dim_H});
        value->Resize({llm_params_.batch_size_B, decode_step + 1,
                       llm_params_.num_kv_heads, llm_params_.head_dim_H});
        RET_CHECK_EQ(xnn_status_success,
                     xnn_reshape_external_value(
                         runtime_.get(), key->tensor_id(owned_subgraph_.get()),
                         key->dims.size(), key->dims.data()));
        RET_CHECK_EQ(
            xnn_status_success,
            xnn_reshape_external_value(runtime_.get(),
                                       value->tensor_id(owned_subgraph_.get()),
                                       value->dims.size(), value->dims.data()));
      }
      RET_CHECK_EQ(xnn_status_success, xnn_reshape_runtime(runtime_.get()));
    }

    for (auto& kv_cache : kv_cache_) {
      ABSL_DCHECK(kv_cache.k_slice);
      ABSL_DCHECK(kv_cache.v_slice);
      kv_cache.k_slice->Borrow(kv_cache.k_cache->Slice(1, decode_step));
      kv_cache.v_slice->Borrow(kv_cache.v_cache->Slice(1, decode_step));
    }

    MP_RETURN_IF_ERROR(SetupRuntime());
    MP_RETURN_IF_ERROR(Run());

    RET_CHECK(logits_output_);
    ABSL_DCHECK_EQ(logits_output_->num_elements,
                   llm_params_.batch_size_B * llm_params_.voc_size_V);

    computed_tokens_count_ = decode_step + 1;
    return logits_output_;
  }

 private:
  // Tracks the offset of computed tokens, such that we can remind user to call
  // ComputeLogits() after each AddInputTokens();
  size_t computed_tokens_count_ = 0;
  std::unique_ptr<Llm> prefix_llm_;
};

absl::StatusOr<std::unique_ptr<Llm>> Llm::CreatePrefixDecodeLlm(
    std::unique_ptr<LlmWeightsLoader> weight_loader,
    std::shared_ptr<LlmBuilder> builder) {
  RET_CHECK_EQ(builder->runtime_configs_->weights_cache, nullptr);
  std::shared_ptr<xnn_utils::XnnWeightsCache> weights_cache =
      weight_loader->GetXnnWeightsCache();
  if (!weights_cache) {
    MP_ASSIGN_OR_RETURN(weights_cache, xnn_utils::CreateWeightsCache());
  }
  builder->runtime_configs_->weights_cache = weights_cache;
  MP_ASSIGN_OR_RETURN(auto weights, weight_loader->LoadWeights());

  LlmParams decoding_llm_params = builder->llm_params_;
  RET_CHECK_GT(decoding_llm_params.seq_size_T, 0);

  RET_CHECK(decoding_llm_params.enable_kv_cache)
      << "CreatePrefixDecodeLlm function should only be executed when "
         "enable_kv_cache is true.";
  // For prefix model, there is no need to execute the final projection layer
  // which maps the internal embedding back to the vocab id space as the main
  // goal of the prefix model is to run through the stacked Transformers to
  // populate the kv-cache values.
  builder->llm_params_.final_norm = LlmParams::Norm::NO_NORM;
  builder->internal_llm_params_.stop_at_last_kv_cache = true;

  MP_ASSIGN_OR_RETURN(auto prefix_llm, CreatePrefixOnlyLlm(weights, builder));
  // Revert any overwrite above.
  builder->llm_params_ = decoding_llm_params;

  MP_ASSIGN_OR_RETURN(auto input,
                      builder->NewInput({decoding_llm_params.batch_size_B, 1,
                                         decoding_llm_params.model_dim_D},
                                        "decoder_input"));

  std::vector<KVCache>& kv_cache = prefix_llm->kv_cache_;
  RET_CHECK_EQ(kv_cache.size(), decoding_llm_params.num_transformer_M);

  MP_ASSIGN_OR_RETURN(auto preprocess_out,
                      builder->PreProcess(input, /*is_prefix=*/false));
  auto& inter_layer = preprocess_out.first;
  auto& resource = preprocess_out.second;

  for (int i = 0; i < decoding_llm_params.num_transformer_M; ++i) {
    resource.cache = &kv_cache[i];
    const auto& sa = weights.sas[i];
    const auto& ff = weights.ffs[i];
    MP_ASSIGN_OR_RETURN(inter_layer, builder->OneStackTransformer(
                                         i, inter_layer, resource, sa, ff,
                                         /*is_prefix=*/false));
    {
      resource.cache->k_cache->source += absl::StrCat("#", i);
      resource.cache->v_cache->source += absl::StrCat("#", i);
      resource.cache->k_slice->source += absl::StrCat("#", i);
      resource.cache->v_slice->source += absl::StrCat("#", i);
    }
  }

  MP_ASSIGN_OR_RETURN(auto logits_output,
                      builder->PostProcess(inter_layer, weights));
  logits_output->MarkOutput();

  MP_ASSIGN_OR_RETURN(auto graph, builder->Build());
  MP_RETURN_IF_ERROR(weights_cache->Finalize());
  VLOG(2) << "Xnn weights cache finalized.";
  auto result = std::make_unique<PrefixDecodeLlm>(std::move(prefix_llm),
                                                  std::move(*graph));
  result->prev_ids_.resize(decoding_llm_params.batch_size_B);

  result->transformer_input_ = input;
  result->logits_output_ = logits_output;

  result->pos_embedding_ = resource.pos_embedding;
  result->segment_pos_ = resource.segment_pos;
  result->atten_masks_ = resource.atten_mask;
  result->kv_cache_ = std::move(kv_cache);

  result->weights_ = std::move(weights);
  result->llm_params_ = decoding_llm_params;
  result->builder_ = std::move(builder);

  return result;
}

absl::StatusOr<std::unique_ptr<Llm>> Llm::CreateLlm(
    absl::string_view weights_folder, const LlmParams& llm_params,
    std::unique_ptr<xnn_utils::RuntimeConfigs> runtime_configs) {
  auto weight_loader =
      std::make_unique<DefaultLlmWeightsLoader>(weights_folder, llm_params);
  return CreateLlm(std::move(weight_loader), std::move(runtime_configs));
}

absl::StatusOr<std::unique_ptr<Llm>> Llm::CreateLlm(
    std::unique_ptr<LlmWeightsLoader> weight_loader,
    std::unique_ptr<xnn_utils::RuntimeConfigs> runtime_configs) {
  const auto& llm_params = weight_loader->llm_params();
  return CreateLlm(
      std::move(weight_loader),
      std::make_unique<LlmBuilder>(llm_params, std::move(runtime_configs)));
}

absl::StatusOr<std::unique_ptr<Llm>> Llm::CreateLlm(
    std::unique_ptr<LlmWeightsLoader> weight_loader,
    std::unique_ptr<LlmBuilder> builder) {
  const auto& llm_params = weight_loader->llm_params();
  if (llm_params.enable_kv_cache) {
    return CreatePrefixDecodeLlm(std::move(weight_loader), std::move(builder));
  }
  if (llm_params.enable_dynamic_shape) {
    return absl::InvalidArgumentError(
        "Dynamic shape should be enabled together with KV cache.");
  }
  MP_ASSIGN_OR_RETURN(auto weights, weight_loader->LoadWeights());
  return CreatePrefixOnlyLlm(weights, std::move(builder));
}

absl::StatusOr<std::unique_ptr<Llm>> Llm::CreatePrefixOnlyLlm(
    LlmWeights weights, std::shared_ptr<LlmBuilder> builder) {
  RET_CHECK(builder);
  const LlmParams& llm_params = builder->llm_params_;
  RET_CHECK_NE(llm_params.batch_size_B, 0);

  MP_ASSIGN_OR_RETURN(auto input, builder->NewInput({llm_params.batch_size_B,
                                                     llm_params.seq_size_T,
                                                     llm_params.model_dim_D},
                                                    "prefix_input"));

  MP_ASSIGN_OR_RETURN(auto preprocess_out,
                      builder->PreProcess(input, /*is_prefix=*/true));

  auto& inter_layer = preprocess_out.first;
  auto& resource = preprocess_out.second;

  std::vector<KVCache> kv_cache;
  std::shared_ptr<Tensor> logits_output;

  for (int i = 0; i < llm_params.num_transformer_M; ++i) {
    KVCache* cache = nullptr;
    if (llm_params.enable_kv_cache) {
      kv_cache.push_back(KVCache{});
      cache = &kv_cache.back();
    }
    resource.cache = cache;
    const auto& sa = weights.sas[i];
    const auto& ff = weights.ffs[i];
    MP_ASSIGN_OR_RETURN(inter_layer, builder->OneStackTransformer(
                                         i, inter_layer, resource, sa, ff,
                                         /*is_prefix=*/true));
  }

  if (builder->internal_llm_params_.stop_at_last_kv_cache) {
    logits_output = inter_layer;
  } else {
    MP_ASSIGN_OR_RETURN(logits_output,
                        builder->PostProcess(inter_layer, weights));
  }
  logits_output->MarkOutput();

  MP_ASSIGN_OR_RETURN(auto graph, builder->Build());
  auto llm = std::make_unique<Llm>(std::move(*graph));
  llm->prev_ids_.resize(llm_params.batch_size_B);

  llm->transformer_input_ = input;
  llm->logits_output_ = logits_output;

  llm->pos_embedding_ = resource.pos_embedding;
  llm->segment_pos_ = resource.segment_pos;
  llm->atten_masks_ = resource.atten_mask;
  llm->kv_cache_ = std::move(kv_cache);

  llm->weights_ = std::move(weights);
  llm->llm_params_ = llm_params;
  llm->builder_ = builder;

  return llm;
}

size_t Llm::TotalTokenSize() const {
  ABSL_CHECK(!prev_ids_.empty());
  // prev_ids_ is of length llm_params.batch_size_B, and we assume each batch
  // decode simultaneously, thus prev_ids[i] have the same size, which is total
  // token size.
  return prev_ids_[0].size();
}

absl::Status Llm::ReshapeInputResource() {
  if (llm_params_.enable_dynamic_shape) {
    RET_CHECK_EQ(
        xnn_status_success,
        xnn_reshape_external_value(
            runtime_.get(), atten_masks_->tensor_id(owned_subgraph_.get()),
            atten_masks_->dims.size(), atten_masks_->dims.data()));
    if (!llm_params_.skip_absolute_positional_embeddings) {
      RET_CHECK_EQ(
          xnn_status_success,
          xnn_reshape_external_value(
              runtime_.get(), pos_embedding_->tensor_id(owned_subgraph_.get()),
              pos_embedding_->dims.size(), pos_embedding_->dims.data()));
    }
    if (segment_pos_) {
      RET_CHECK_EQ(
          xnn_status_success,
          xnn_reshape_external_value(
              runtime_.get(), segment_pos_->tensor_id(owned_subgraph_.get()),
              segment_pos_->dims.size(), segment_pos_->dims.data()));
    }
  }
  return absl::OkStatus();
}

absl::Status Llm::Reset() {
  RET_CHECK_EQ(prev_ids_.size(), llm_params_.batch_size_B);
  for (auto& prev_id : prev_ids_) {
    // TODO - b/325325100: avoid clear().
    prev_id.clear();
  }
  builder_->attention_mask_values_.reset();
  builder_->position_embedding_values_.reset();
  builder_->segment_pos_values_.reset();

  return absl::OkStatus();
}

absl::Status Llm::InitInputTokens(const std::vector<int>& input_ids) {
  RET_CHECK_EQ(llm_params_.batch_size_B, 1)
      << "For batch inference, use the batch version API";
  auto span = absl::Span<const std::vector<int>>(&input_ids, 1);
  return InitInputTokens(span);
}

absl::Status Llm::AddInputTokens(
    absl::Span<const std::vector<int>> batch_input_ids) {
  return absl::UnimplementedError("Need to turn on kv cache");
}

absl::Status Llm::InitInputTokens(
    absl::Span<const std::vector<int>> batch_input_ids) {
  MP_RETURN_IF_ERROR(Reset());

  RET_CHECK_EQ(batch_input_ids.size(), llm_params_.batch_size_B);
  const size_t input_seq_len = batch_input_ids.at(0).size();
  for (auto it = batch_input_ids.begin() + 1; it != batch_input_ids.end();
       ++it) {
    RET_CHECK_EQ(it->size(), input_seq_len);
  }

  RET_CHECK(!prev_ids_.empty());
  const size_t current_seq_len = prev_ids_[0].size();

  // Initialize the attention mask.
  MP_RETURN_IF_ERROR(builder_->InitAttentionMask(
      current_seq_len, input_seq_len, /*is_prefix=*/true, *atten_masks_));
  if (!llm_params_.skip_absolute_positional_embeddings) {
    // Initialize the positional embedding data.
    MP_RETURN_IF_ERROR(builder_->InitPosEmbedding(
        current_seq_len, input_seq_len, *pos_embedding_));
  }
  if (segment_pos_) {
    // Initialize the segment pos.
    MP_RETURN_IF_ERROR(builder_->InitSegmentPos(current_seq_len, input_seq_len,
                                                *segment_pos_));
  }

  if (!input_pivot_) {
    input_pivot_ = std::make_shared<Tensor>(transformer_input_->dims,
                                            transformer_input_->datatype);
    input_pivot_->flat_data = transformer_input_->flat_data;
  }
  for (size_t batch = 0; batch < llm_params_.batch_size_B; ++batch) {
    auto slice = input_pivot_->Slice(0, batch)->Slice(1, current_seq_len);
    MP_RETURN_IF_ERROR(
        GetTokenEmbedding(batch_input_ids[batch], slice->DataAs<float>()));
  }

  if (llm_params_.enable_dynamic_shape) {
    MP_RETURN_IF_ERROR(ReshapeInputResource());

    transformer_input_->Borrow(input_pivot_,
                               current_seq_len * llm_params_.model_dim_D);
    transformer_input_->Resize(Tensor::DimsType{
        llm_params_.batch_size_B, input_seq_len, llm_params_.model_dim_D});
    RET_CHECK_EQ(
        xnn_status_success,
        xnn_reshape_external_value(
            runtime_.get(),
            transformer_input_->tensor_id(owned_subgraph_.get()),
            transformer_input_->dims.size(), transformer_input_->dims.data()));

    RET_CHECK_EQ(xnn_status_success, xnn_reshape_runtime(runtime_.get()));
  }

  for (size_t batch = 0; batch < llm_params_.batch_size_B; ++batch) {
    auto& prev_ids = prev_ids_[batch];
    const auto& input_ids = batch_input_ids[batch];
    prev_ids.insert(prev_ids.end(), input_ids.begin(), input_ids.end());
  }
  return SetupRuntime();
}

absl::Status Llm::GetNextToken(std::vector<int>* output_ids) {
  MP_ASSIGN_OR_RETURN(auto logits, ComputeLogits());
  const size_t decode_step = prev_ids_[0].size() - 1;

  MP_ASSIGN_OR_RETURN(*output_ids, builder_->Sample(*logits));
  RET_CHECK_EQ(output_ids->size(), llm_params_.batch_size_B);

  for (size_t batch = 0; batch < llm_params_.batch_size_B; ++batch) {
    auto slice = transformer_input_->Slice(0, batch)->Slice(1, decode_step + 1);
    MP_RETURN_IF_ERROR(GetTokenEmbedding(*output_ids, slice->DataAs<float>()));
    prev_ids_[batch].push_back(output_ids->at(batch));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<Tensor>> Llm::ComputeLogits() {
  const size_t decode_step = prev_ids_[0].size() - 1;
  VLOG(2) << "Decode step " << decode_step;

  if (decode_step + 1 >= llm_params_.seq_size_T) {
    return absl::OutOfRangeError(
        absl::StrCat("Hit max sequence length ", llm_params_.seq_size_T));
  }

  MP_RETURN_IF_ERROR(Run());

  RET_CHECK(logits_output_);
  std::shared_ptr<Tensor> logits = logits_output_->Slice({0, decode_step, 0});
  ABSL_DCHECK_EQ(logits->num_elements,
                 llm_params_.batch_size_B * llm_params_.voc_size_V);
  MP_ASSIGN_OR_RETURN(auto logits_tensor, logits->ConvertToMediapipeTensor());

  return logits;
}

absl::Status Llm::GetTokenEmbedding(const std::vector<int>& ids,
                                    float* embedding) {
  RET_CHECK_LE(ids.size(), llm_params_.seq_size_T);
  auto token_embedding = weights_.token_embedding ? weights_.token_embedding
                                                  : weights_.softmax_linear;
  RET_CHECK(token_embedding);
  RET_CHECK(token_embedding->dims[0] == llm_params_.voc_size_V)
      << "shape must be [vocab_size, _], such that following Slice() makes "
         "sense.";
  for (int id : ids) {
    MP_ASSIGN_OR_RETURN(auto embedding_slice,
                        token_embedding->Slice(0, id)->ConvertToF32());
    memcpy(embedding, embedding_slice->Data(),
           llm_params_.model_dim_D * sizeof(float));
    embedding += llm_params_.model_dim_D;
  }
  return absl::OkStatus();
}

absl::StatusOr<std::pair<std::shared_ptr<Tensor>, LlmBuilder::InputResource>>
LlmBuilder::PreProcess(std::shared_ptr<Tensor> token_embedding,
                       bool is_prefix) {
  InputResource resource;
  constexpr absl::string_view kAttnMaskSource = "atten_mask";
  constexpr absl::string_view kPosEmbeddingSource = "pos_embedding";
  constexpr absl::string_view kSegmentPosSource = "segment_pos";
  if (is_prefix) {
    MP_ASSIGN_OR_RETURN(resource.atten_mask, NewInput({llm_params_.seq_size_T,
                                                       llm_params_.seq_size_T},
                                                      kAttnMaskSource));
    MP_ASSIGN_OR_RETURN(resource.segment_pos, NewInput({llm_params_.seq_size_T,
                                                        llm_params_.head_dim_H},
                                                       kSegmentPosSource));
    MP_RETURN_IF_ERROR(
        InitSegmentPos(0, llm_params_.seq_size_T, *resource.segment_pos));
    MP_ASSIGN_OR_RETURN(
        resource.pos_embedding,
        NewInput({llm_params_.seq_size_T, llm_params_.model_dim_D},
                 kPosEmbeddingSource));
  } else {
    MP_ASSIGN_OR_RETURN(
        resource.pos_embedding,
        NewInput({1, llm_params_.model_dim_D}, kPosEmbeddingSource));
    MP_ASSIGN_OR_RETURN(resource.atten_mask,
                        NewInput({1, llm_params_.seq_size_T}, kAttnMaskSource));
    MP_ASSIGN_OR_RETURN(
        resource.segment_pos,
        NewInput({1, llm_params_.head_dim_H}, kSegmentPosSource));
    MP_RETURN_IF_ERROR(InitSegmentPos(0, 1, *resource.segment_pos));
  }
  const float dim_scale = std::sqrt(llm_params_.model_dim_D);
  MP_ASSIGN_OR_RETURN(auto scaled_embedding,
                      ElementMul(token_embedding, dim_scale));
  return std::make_pair(scaled_embedding, resource);
}

absl::StatusOr<std::shared_ptr<Tensor>> LlmBuilder::OneStackTransformer(
    int layer_index, std::shared_ptr<Tensor> input,
    LlmBuilder::InputResource resource,
    const LlmWeights::SelfAttentionWeights& sa_weights,
    const LlmWeights::FeedForwardWeights& ff_weights, bool is_prefix) {
  std::shared_ptr<Tensor> output;
  if (is_prefix) {
    MP_ASSIGN_OR_RETURN(
        output, SelfAttentionIncludeResidual(input, resource, sa_weights));
    if (internal_llm_params_.stop_at_last_kv_cache &&
        (layer_index == llm_params_.num_transformer_M - 1)) {
      return output;
    }
    MP_ASSIGN_OR_RETURN(output, FeedForwardIncludeResidual(output, ff_weights));
  } else {
    MP_ASSIGN_OR_RETURN(
        output, SelfAttentionIncludeResidual(input, resource, sa_weights));
    MP_ASSIGN_OR_RETURN(output, FeedForwardIncludeResidual(output, ff_weights));
  }
  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> LlmBuilder::SelfAttentionExcludeNorm(
    std::shared_ptr<Tensor> input, InputResource resource,
    const SelfAttentionWeights& sa_weights) {
  // [B, 1|T, N, H]
  MP_ASSIGN_OR_RETURN(auto k_proj,
                      SelfAttentionProj(input, sa_weights.k_weight));
  MP_ASSIGN_OR_RETURN(auto q_proj,
                      SelfAttentionProj(input, sa_weights.q_weight));
  MP_ASSIGN_OR_RETURN(auto v_proj,
                      SelfAttentionProj(input, sa_weights.v_weight));

  MP_ASSIGN_OR_RETURN(auto query_proj_after_rope,
                      Rope(q_proj, resource.segment_pos));
  MP_ASSIGN_OR_RETURN(auto key_proj_after_rope,
                      Rope(k_proj, resource.segment_pos));

  MP_RETURN_IF_ERROR(BuildKVCache(key_proj_after_rope, v_proj, resource));

  // encoded, [B, 1|T, N, H]
  MP_ASSIGN_OR_RETURN(auto kqv_merged,
                      DotAttention(query_proj_after_rope, key_proj_after_rope,
                                   v_proj, resource.atten_mask, sa_weights));

  const size_t B = kqv_merged->dims[0];
  const size_t NH = kqv_merged->dims[2] * kqv_merged->dims[3];
  MP_ASSIGN_OR_RETURN(auto outcome_reshaped, Reshape(kqv_merged, {B, 0, NH}));

  return MatMul(outcome_reshaped, sa_weights.post_proj_weight);
}

absl::StatusOr<std::shared_ptr<Tensor>>
LlmBuilder::SelfAttentionIncludeResidual(
    std::shared_ptr<Tensor> input, InputResource resource,
    const SelfAttentionWeights& sa_weights) {
  MP_ASSIGN_OR_RETURN(auto pre_attention,
                      ApplyNorm(input, sa_weights.pre_norm_weight,
                                llm_params_.sa_params.pre_norm));

  MP_ASSIGN_OR_RETURN(
      auto post_attention,
      SelfAttentionExcludeNorm(pre_attention, std::move(resource), sa_weights));

  MP_ASSIGN_OR_RETURN(auto post_norm,
                      ApplyNorm(post_attention, sa_weights.post_norm_weight,
                                llm_params_.sa_params.post_norm));

  return ElementAdd(input, post_norm);
}

absl::StatusOr<std::shared_ptr<Tensor>> LlmBuilder::FeedForwardExcludeNorm(
    std::shared_ptr<Tensor> input, const FeedForwardWeights& ff_weights) {
  MP_ASSIGN_OR_RETURN(auto layer_1, FullConn(input, ff_weights.layer_1_weight,
                                             ff_weights.layer_1_bias));

  MP_ASSIGN_OR_RETURN(auto layer_1_gate_before_activation,
                      FullConn(input, ff_weights.layer_1_gate_weight,
                               ff_weights.layer_1_gate_bias));
  std::shared_ptr<Tensor> layer_1_gate;
  switch (llm_params_.ff_params.activation) {
    case LlmParams::Activation::UNSPECIFIED:
      layer_1_gate = layer_1_gate_before_activation;
      break;
    case LlmParams::Activation::GELU: {
      MP_ASSIGN_OR_RETURN(layer_1_gate, Gelu(layer_1_gate_before_activation));
      break;
    }
    case LlmParams::Activation::SILU: {
      MP_ASSIGN_OR_RETURN(layer_1_gate, Silu(layer_1_gate_before_activation));
      break;
    }
    case LlmParams::Activation::RELU: {
      MP_ASSIGN_OR_RETURN(layer_1_gate, Relu(layer_1_gate_before_activation));
      break;
    }
    default: {
      break;
    }
  }

  MP_ASSIGN_OR_RETURN(auto layer_1_and_gate, ElementMul(layer_1, layer_1_gate));
  MP_ASSIGN_OR_RETURN(auto layer_2,
                      FullConn(layer_1_and_gate, ff_weights.layer_2_weight,
                               ff_weights.layer_2_bias));

  return layer_2;
}

absl::StatusOr<std::shared_ptr<Tensor>> LlmBuilder::FeedForwardIncludeResidual(
    std::shared_ptr<Tensor> input, const FeedForwardWeights& ff_weights) {
  MP_ASSIGN_OR_RETURN(auto pre_ff, ApplyNorm(input, ff_weights.pre_norm_weight,
                                             llm_params_.ff_params.pre_norm));

  MP_ASSIGN_OR_RETURN(auto pre_norm,
                      FeedForwardExcludeNorm(pre_ff, ff_weights));

  MP_ASSIGN_OR_RETURN(auto post_norm,
                      ApplyNorm(pre_norm, ff_weights.post_norm_weight,
                                llm_params_.ff_params.post_norm));

  return ElementAdd(post_norm, input);
}

absl::StatusOr<std::shared_ptr<Tensor>> LlmBuilder::PostProcess(
    std::shared_ptr<Tensor> transformer_out, const LlmWeights& weights) {
  MP_ASSIGN_OR_RETURN(transformer_out,
                      ApplyNorm(transformer_out, weights.final_norm_weight,
                                llm_params_.final_norm));
  RET_CHECK(weights.softmax_linear);
  MP_ASSIGN_OR_RETURN(
      auto logits_output,
      FullConn(transformer_out, weights.softmax_linear, weights.softmax_bias));
  return logits_output;
}

absl::Status LlmBuilder::InitAttentionMask(size_t current_seq_len,
                                           size_t process_seq_len,
                                           bool is_prefix,
                                           Tensor& out_attn_mask) {
  if (!attention_mask_values_) {
    MP_RETURN_IF_ERROR(InitAttentionMaskValues(process_seq_len));
  }

  if (llm_params_.enable_dynamic_shape) {
    if (!is_prefix) {
      out_attn_mask.Resize(
          Tensor::DimsType{1, current_seq_len + process_seq_len});
      out_attn_mask.flat_data = std::shared_ptr<char>(
          attention_mask_values_,
          reinterpret_cast<char*>(attention_mask_values_->data() +
                                  llm_params_.seq_size_T * current_seq_len));
    } else {
      out_attn_mask.Resize(
          Tensor::DimsType{process_seq_len, current_seq_len + process_seq_len});
      for (size_t r = 0; r < out_attn_mask.dims[0]; ++r) {
        auto slice = out_attn_mask.Slice(0, r);
        MP_RETURN_IF_ERROR(slice->LoadFromBuffer(
            attention_mask_values_->data() +
            (r + current_seq_len) * llm_params_.seq_size_T));
      }
    }
  } else {
    if (!is_prefix) {
      RET_CHECK_EQ(out_attn_mask.num_elements, llm_params_.seq_size_T);
      out_attn_mask.flat_data = std::shared_ptr<char>(
          attention_mask_values_,
          reinterpret_cast<char*>(attention_mask_values_->data() +
                                  llm_params_.seq_size_T * current_seq_len));
    } else {
      RET_CHECK_EQ(out_attn_mask.num_elements,
                   llm_params_.seq_size_T * llm_params_.seq_size_T);
      out_attn_mask.flat_data = std::shared_ptr<char>(
          attention_mask_values_,
          reinterpret_cast<char*>(attention_mask_values_->data()));
    }
  }

  return absl::OkStatus();
}

absl::Status LlmBuilder::InitAttentionMaskValues(size_t process_seq_len) {
  const auto& seq_size = llm_params_.seq_size_T;
  constexpr float neg_value = 0.8 * std::numeric_limits<float>::lowest();
  attention_mask_values_ =
      std::make_shared<std::vector<float>>(seq_size * seq_size, neg_value);
  switch (llm_params_.model_type) {
    case LlmParams::ModelType::PREFIX: {
      RET_CHECK_LE(process_seq_len, seq_size);
      // Prefix full attention for all tokens within input ids size(input),
      // and causal attention mask for all following tokens.
      for (int i = 0; i < seq_size; ++i) {
        for (int j = 0; j < seq_size; ++j) {
          if (j <= i || std::max(j, i) < process_seq_len) {
            (*attention_mask_values_)[seq_size * i + j] = 0;
          } else {
            break;
          }
        }
      }
      break;
    }
    case LlmParams::ModelType::CAUSAL: {
      for (int i = 0; i < seq_size; ++i) {
        for (int j = 0; j < seq_size; ++j) {
          if (j <= i) {
            (*attention_mask_values_)[seq_size * i + j] = 0;
          } else {
            break;
          }
        }
      }
      break;
    }
    default: {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported model type: ", llm_params_.model_type));
    }
  }
  return absl::OkStatus();
}

absl::Status LlmBuilder::InitPosEmbeddingValues(size_t process_seq_len) {
  return absl::OkStatus();
}

absl::Status LlmBuilder::InitPosEmbedding(size_t current_seq_len,
                                          size_t process_seq_len,
                                          Tensor& out_pos_embedding) {
  if (!position_embedding_values_) {
    MP_RETURN_IF_ERROR(InitPosEmbeddingValues(process_seq_len));
  }

  RET_CHECK_EQ(out_pos_embedding.dims.size(), 2);
  if (out_pos_embedding.dims[0] == 1) {
    RET_CHECK_EQ(out_pos_embedding.num_elements, llm_params_.model_dim_D);
    MP_RETURN_IF_ERROR(out_pos_embedding.LoadFromBuffer(
        position_embedding_values_->data() +
        llm_params_.model_dim_D * current_seq_len));
  } else {
    out_pos_embedding.Resize(
        Tensor::DimsType{process_seq_len, llm_params_.model_dim_D});
    MP_RETURN_IF_ERROR(out_pos_embedding.LoadFromBuffer(
        position_embedding_values_->data() +
        llm_params_.model_dim_D * current_seq_len));
  }
  return absl::OkStatus();
}

absl::Status LlmBuilder::InitSegmentPosValues(size_t rope_size) {
  segment_pos_values_ = std::make_shared<std::vector<float>>();
  *segment_pos_values_ = FillXnnRoPEWeights(llm_params_.seq_size_T, rope_size);
  return absl::OkStatus();
}

absl::Status LlmBuilder::InitSegmentPos(size_t current_seq_len,
                                        size_t process_seq_len,
                                        Tensor& out_segment_pos) {
  RET_CHECK_EQ(out_segment_pos.dims.size(), 2);
  const size_t rope_size = out_segment_pos.dims[1];
  if (!segment_pos_values_) {
    MP_RETURN_IF_ERROR(InitSegmentPosValues(rope_size));
  }

  out_segment_pos.Resize(Tensor::DimsType{process_seq_len, rope_size});
  MP_RETURN_IF_ERROR(out_segment_pos.LoadFromBuffer(
      segment_pos_values_->data() + rope_size * current_seq_len));
  return absl::OkStatus();
}

absl::StatusOr<std::vector<int>> LlmBuilder::Sample(const Tensor& logits) {
  if (sampler_ == nullptr) {
    MP_ASSIGN_OR_RETURN(
        sampler_,
        Sampler::Create(Sampler::Type::kGreedy, /*top_k=*/0, /*top_p=*/0.0,
                        /*top_temperature=*/0.0, /*seed=*/0));
  }
  return sampler_->Sample(logits);
}

absl::StatusOr<std::shared_ptr<Tensor>> LlmBuilder::DotAttention(
    std::shared_ptr<Tensor> query_proj, std::shared_ptr<Tensor> key_proj,
    std::shared_ptr<Tensor> value_proj, std::shared_ptr<Tensor> atten_mask,
    const SelfAttentionWeights& sa_weights) {
  // BTNH
  std::shared_ptr<Tensor> query_after_scale;
  switch (llm_params_.sa_params.attention_scale_type) {
    case LlmParams::AttentionScaleType::PER_DIM_SCALE: {
      MP_ASSIGN_OR_RETURN(query_after_scale,
                          PerDimScale(query_proj, sa_weights.per_dim_scale));
      break;
    }
    case LlmParams::AttentionScaleType::INV_SQRT_HEAD_DIM: {
      // Scale the query values by multiplying 1 / sqrt(dim_per_head).
      float scale = 1.0f / sqrt(llm_params_.head_dim_H);
      MP_ASSIGN_OR_RETURN(query_after_scale, ElementMul(query_proj, scale));
      break;
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported attention scale type: ",
                       llm_params_.sa_params.attention_scale_type));
  }

  // Dot similarity
  // BTNH -> BNTH
  MP_ASSIGN_OR_RETURN(auto query_permuted,
                      Permute(query_after_scale, {0, 2, 1, 3}));
  // BSN'H -> BN'SH
  MP_ASSIGN_OR_RETURN(auto key_permuted, Permute(key_proj, {0, 2, 1, 3}));
  // einsum(BNTH.BN'SH -> BNTS)
  MP_ASSIGN_OR_RETURN(auto logits, QKVAttention(query_permuted, key_permuted,
                                                {0, llm_params_.head_dim_H}));

  // Cap, mask
  if (llm_params_.sa_params.soft_cap_value > 0.0f) {
    MP_ASSIGN_OR_RETURN(logits,
                        CapTanh(logits, llm_params_.sa_params.soft_cap_value));
  }
  MP_ASSIGN_OR_RETURN(auto padded_logits, ElementAdd(atten_mask, logits));
  MP_ASSIGN_OR_RETURN(auto probs, Softmax(padded_logits));
  MP_ASSIGN_OR_RETURN(auto value_permuted, Permute(value_proj, {0, 2, 3, 1}));

  // Outcome
  // einsum(BNTS.BN'HS) -> BNTH
  MP_ASSIGN_OR_RETURN(
      auto outcome_before_permute,
      QKVAttention(probs, value_permuted, {llm_params_.head_dim_H, 0}));
  // [B, N, T, H] -> BTNH
  return Permute(outcome_before_permute, {0, 2, 1, 3});
}

absl::StatusOr<std::shared_ptr<Tensor>> LlmBuilder::ApplyNorm(
    std::shared_ptr<Tensor> input,
    std::optional<LlmWeights::NormWeights> weights, LlmParams::Norm norm_type) {
  std::shared_ptr<Tensor> output = input;
  switch (norm_type) {
    case LlmParams::Norm::NO_NORM:
      break;
    case LlmParams::Norm::RMS_NORM: {
      MP_ASSIGN_OR_RETURN(
          output,
          RmsNorm(input,
                  std::get<RMSNormWeights>(weights.value()).norm_weight));
      break;
    }
    case LlmParams::Norm::LAYER_NORM: {
      const auto& layer_norm_weights =
          std::get<LayerNormWeights>(weights.value());
      MP_ASSIGN_OR_RETURN(
          output, LayerNorm(input, layer_norm_weights.epsilon,
                            layer_norm_weights.gamma, layer_norm_weights.beta));
      break;
    }
    default:
      return absl::NotFoundError("No norm specified.");
  }
  return output;
}

absl::Status LlmBuilder::BuildKVCache(std::shared_ptr<Tensor>& key,
                                      std::shared_ptr<Tensor>& value,
                                      InputResource& resource) {
  if (resource.cache) {
    RET_CHECK_EQ(key->dims.size(), 4);
    RET_CHECK_EQ(key->dims[0], llm_params_.batch_size_B);
    // When cache is provided, there are 2 cases:
    if (key->dims[1] != 1) {
      // Building a normal graph, which is used to initialize cache.
      key->source = "prefix_k_cache";
      (resource.cache->k_cache = key)->MarkOutput();
      value->source = "prefix_v_cache";
      (resource.cache->v_cache = value)->MarkOutput();
    } else {
      RET_CHECK(resource.cache->k_cache);
      RET_CHECK(resource.cache->v_cache);
      // Building a one-token graph, which consumes initialized cache.
      key->source = "decode_k_slice";
      (resource.cache->k_slice = key)->MarkOutput();
      value->source = "decode_v_slice";
      (resource.cache->v_slice = value)->MarkOutput();

      MP_RETURN_IF_ERROR(MarkInput(key = resource.cache->k_cache));
      MP_RETURN_IF_ERROR(MarkInput(value = resource.cache->v_cache));
    }
  }

  return absl::OkStatus();
}

}  // namespace xnn_utils
}  // namespace mediapipe::tasks::genai
