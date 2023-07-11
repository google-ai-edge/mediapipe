#include "mediapipe/tasks/cc/text/utils/xnn_utils/ulm.h"

#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/text/text_generator/calculators/preprocessor_util.h"
#include "mediapipe/tasks/cc/text/text_generator/calculators/sampler_util.h"
#include "mediapipe/tasks/cc/text/utils/xnn_utils/graph_builder.h"
#include "mediapipe/tasks/cc/text/utils/xnn_utils/ulm_weights.h"
#include "mediapipe/tasks/cc/text/utils/xnn_utils/utils.h"
#include "mediapipe/tasks/cc/text/utils/xnn_utils/xnn_tensor.h"
#include "util/gtl/stl_logging.h"

namespace mediapipe {
namespace xnn_utils {
namespace {

absl::StatusOr<std::shared_ptr<Tensor>> ApplyFinalProj(
    std::shared_ptr<Tensor> inter_layer, const UlmWeights& weights,
    XnnGraphBuilder& builder) {
  return builder.FullConn(inter_layer, weights.softmax_linear,
                          weights.softmax_bias);
}

}  // namespace

class OneTokenUlm : public Ulm {
 public:
  OneTokenUlm(std::unique_ptr<Ulm> full_ulm, XnnGraph&& other)
      : Ulm(std::move(other)), full_ulm_(std::move(full_ulm)) {}
  ~OneTokenUlm() override = default;

  absl::Status InitInputTokens(const std::vector<int>& input_ids) override {
    prev_ids_ = input_ids;
    MP_RETURN_IF_ERROR(full_ulm_->InitInputTokens(input_ids));
    // prev_id.size - 1 is the output.
    return full_ulm_->Run();
  }

  absl::Status GetNextToken(std::vector<int>* output_ids) override {
    size_t decode_step = prev_ids_.size() - 1;
    VLOG(2) << "Decode step " << decode_step;

    if (decode_step == ulm_params_.seq_size_T - 1) {
      return absl::OutOfRangeError(
          absl::StrCat("Hit max sequence length ", ulm_params_.seq_size_T));
    }

    transformer_input_->Borrow(
        full_ulm_->transformer_input_->Slice(1, decode_step));
    atten_masks_->Borrow(full_ulm_->atten_masks_->Slice(0, decode_step));
    MP_RETURN_IF_ERROR(segment_pos_->LoadFromBuffer(
        full_ulm_->segment_pos_->Slice(0, decode_step)->Data()));
    for (auto& kv_cache : kv_cache_) {
      DCHECK(kv_cache.k_slice);
      DCHECK(kv_cache.v_slice);
      kv_cache.k_slice->Borrow(kv_cache.k_cache->Slice(1, decode_step));
      kv_cache.v_slice->Borrow(kv_cache.v_cache->Slice(1, decode_step));
    }

    MP_RETURN_IF_ERROR(SetupRuntime());
    MP_RETURN_IF_ERROR(Run());

    RET_CHECK(logits_output_);
    DCHECK_EQ(logits_output_->num_elements, ulm_params_.voc_size_V);

    ASSIGN_OR_RETURN(*output_ids,
                     mediapipe::SampleNextToken(
                         logits_output_->DataAs<float>(),
                         /*batch_size=*/1,
                         /*vocab_size=*/ulm_params_.voc_size_V, /*top_k=*/10,
                         /*top_p=*/1, /*temperature=*/-1));
    RET_CHECK_EQ(output_ids->size(), 1);
    prev_ids_.push_back(output_ids->at(0));

    return GetTokenEmbedding(
        *output_ids,
        pos_embedding_data_->Slice({decode_step + 1, 0})->DataAs<float>(),
        full_ulm_->transformer_input_->Slice({0, decode_step + 1, 0})
            ->DataAs<float>());
  }

 private:
  std::unique_ptr<Ulm> full_ulm_;
};

absl::StatusOr<std::shared_ptr<Tensor>> UlmBuilder::SelfAttentionExcludeNorm(
    std::shared_ptr<Tensor> input, SelfAttentionArgs args,
    const SelfAttentionWeights& sa_weights, absl::SourceLocation loc) {
  // [B, 1|T, N, H]
  ASSIGN_OR_RETURN(auto k_proj, SelfAttentionProj(input, sa_weights.k_weight));
  ASSIGN_OR_RETURN(auto q_proj, SelfAttentionProj(input, sa_weights.q_weight));
  ASSIGN_OR_RETURN(auto v_proj, SelfAttentionProj(input, sa_weights.v_weight));

  ASSIGN_OR_RETURN(auto query_proj_after_rope, Rope(q_proj, args.segment_pos));
  ASSIGN_OR_RETURN(auto key_proj_after_rope, Rope(k_proj, args.segment_pos));

  if (args.cache) {
    RET_CHECK(args.cache->k_cache);
    RET_CHECK(args.cache->v_cache);
    // When cache is provided, there are 2 cases:
    if (*(input->dims.end() - 2) != 1) {
      // Building a normal graph, which is used to initialize cache.
      key_proj_after_rope->Borrow(args.cache->k_cache).MarkOutput();
      v_proj->Borrow(args.cache->v_cache).MarkOutput();
    } else {
      // Building a one-token graph, which consumes initialized cache.
      key_proj_after_rope->MarkOutput();
      args.cache->k_slice = key_proj_after_rope;
      v_proj->MarkOutput();
      args.cache->v_slice = v_proj;

      ASSIGN_OR_RETURN(key_proj_after_rope,
                       NewInput(args.cache->k_cache->dims));
      key_proj_after_rope->Borrow(args.cache->k_cache);
      ASSIGN_OR_RETURN(v_proj, NewInput(args.cache->v_cache->dims));
      v_proj->Borrow(args.cache->v_cache);
    }
  }

  // encoded, [B, 1|T, N, H]
  ASSIGN_OR_RETURN(
      auto kqv_merged,
      DotAttention(query_proj_after_rope, key_proj_after_rope, v_proj,
                   args.atten_mask, sa_weights.per_dim_scale));

  const size_t B = kqv_merged->dims[0];
  const size_t T_or_1 = kqv_merged->dims[1];
  const size_t NH = kqv_merged->num_elements / (B * T_or_1);
  ASSIGN_OR_RETURN(auto outcome_reshaped, Reshape(kqv_merged, {B, T_or_1, NH}));

  return MatMul(outcome_reshaped, sa_weights.post_proj_weight,
                {.transpose = false});
}

absl::StatusOr<std::shared_ptr<Tensor>>
UlmBuilder::SelfAttentionIncludeResidual(std::shared_ptr<Tensor> input,
                                         SelfAttentionArgs args,
                                         const SelfAttentionWeights& params,
                                         absl::SourceLocation loc) {
  ASSIGN_OR_RETURN(auto pre_attention, RmsNorm(input, params.pre_norm));

  ASSIGN_OR_RETURN(
      auto post_attention,
      SelfAttentionExcludeNorm(pre_attention, std::move(args), params));

  ASSIGN_OR_RETURN(auto post_norm, RmsNorm(post_attention, params.post_norm));

  return ElementAdd(input, post_norm);
}

absl::StatusOr<std::shared_ptr<Tensor>> UlmBuilder::FeedForwardExcludeResidual(
    std::shared_ptr<Tensor> input, const FeedForwardWeights& params,
    absl::SourceLocation loc) {
  ASSIGN_OR_RETURN(auto first_rms_norm, RmsNorm(input, params.pre_norm));

  ASSIGN_OR_RETURN(auto layer_1, FullConn(first_rms_norm, params.layer_1_weight,
                                          params.layer_1_bias));

  ASSIGN_OR_RETURN(auto layer_1_gate_before_gelu,
                   FullConn(first_rms_norm, params.layer_1_gate_weight,
                            params.layer_1_gate_bias));
  ASSIGN_OR_RETURN(auto layer_1_gate, Gelu(layer_1_gate_before_gelu));

  ASSIGN_OR_RETURN(auto layer_1_and_gate, ElementMul(layer_1, layer_1_gate));
  if (params.opt_padding) {
    // activations *= 1.0 - paddings
    ASSIGN_OR_RETURN(auto tmp, ElementMul(params.opt_padding, -1.0f));
    ASSIGN_OR_RETURN(tmp, ElementMul(layer_1_and_gate, tmp));
    ASSIGN_OR_RETURN(layer_1_and_gate, ElementAdd(tmp, layer_1_and_gate));
  }
  ASSIGN_OR_RETURN(
      auto layer_2,
      FullConn(layer_1_and_gate, params.layer_2_weight, params.layer_2_bias));
  if (params.opt_padding) {
    // activations *= 1.0 - paddings
    ASSIGN_OR_RETURN(auto tmp, ElementMul(params.opt_padding, -1.0f));
    ASSIGN_OR_RETURN(tmp, ElementMul(layer_2, tmp));
    ASSIGN_OR_RETURN(layer_2, ElementAdd(tmp, layer_2));
  }

  return RmsNorm(layer_2, params.post_norm);
}

absl::StatusOr<std::shared_ptr<Tensor>> UlmBuilder::FeedForwardIncludeResidual(
    std::shared_ptr<Tensor> input, const FeedForwardWeights& params,
    absl::SourceLocation loc) {
  ASSIGN_OR_RETURN(auto before_residual,
                   FeedForwardExcludeResidual(input, params));
  return ElementAdd(before_residual, input);
}

absl::StatusOr<std::unique_ptr<Ulm>> Ulm::CreateUlm(
    absl::string_view weights_folder, const UlmParams& ulm_params,
    std::unique_ptr<RuntimeConfigs> runtime_configs) {
  auto weight_loader =
      std::make_unique<DefaultUlmWeightsLoader>(weights_folder, ulm_params);
  return CreateUlm(std::move(weight_loader), std::move(runtime_configs));
}

absl::StatusOr<std::unique_ptr<Ulm>> Ulm::CreateOneTokenUlm(
    std::unique_ptr<UlmWeightsLoader> weight_loader,
    std::unique_ptr<RuntimeConfigs> runtime_configs) {
  UlmBuilder builder;
  // TODO: might be memory waste here, benchmark.
  weight_loader->SetBuilder(builder);
  ASSIGN_OR_RETURN(auto weights, weight_loader->LoadWeights());

  UlmParams ulm_params = weight_loader->ulm_params();
  ulm_params.enable_kv_cache = true;

  weight_loader->ulm_params().enable_kv_cache = true;
  weight_loader->ulm_params().final_norm = false;
  weight_loader->ulm_params().final_project = false;
  ASSIGN_OR_RETURN(auto full_ulm, CreateUlm(std::move(weight_loader)));

  ASSIGN_OR_RETURN(auto input, builder.NewInput({ulm_params.batch_size_B, 1,
                                                 ulm_params.model_dim_D}));
  ASSIGN_OR_RETURN(auto atten_masks,
                   builder.NewInput({1, ulm_params.seq_size_T}));
  ASSIGN_OR_RETURN(auto segment_pos,
                   builder.NewWeight({1, ulm_params.head_dim_H}));
  // To allocate buffer before creating runtime.
  MP_RETURN_IF_ERROR(segment_pos->LoadFromVec({}, /*exact_match=*/false));

  std::vector<KVCache>& kv_cache = full_ulm->kv_cache_;
  RET_CHECK_EQ(kv_cache.size(), ulm_params.num_transformer_M);

  auto inter_layer = input;
  for (int i = 0; i < ulm_params.num_transformer_M; ++i) {
    const auto& sa = weights.sas[i];
    ASSIGN_OR_RETURN(auto tmp, builder.SelfAttentionIncludeResidual(
                                   inter_layer,
                                   {.atten_mask = atten_masks,
                                    .segment_pos = segment_pos,
                                    .cache = &kv_cache[i]},
                                   sa));

    auto& ff = weights.ffs[i];
    // ff.opt_padding = paddings;
    ASSIGN_OR_RETURN(inter_layer, builder.FeedForwardIncludeResidual(tmp, ff));
  }

  std::shared_ptr<Tensor> logits_output, transformer_output, normed_output;

  if (ulm_params.final_norm) {
    ASSIGN_OR_RETURN(inter_layer,
                     builder.RmsNorm(inter_layer, weights.final_ln_scale));
    normed_output = inter_layer;
    normed_output->MarkOutput();
  }
  if (ulm_params.final_project) {
    RET_CHECK(weights.softmax_linear);
    ASSIGN_OR_RETURN(logits_output,
                     ApplyFinalProj(inter_layer, weights, builder));
    logits_output->MarkOutput();
  }

  ASSIGN_OR_RETURN(auto graph, builder.Build(std::move(runtime_configs)));
  Ulm* full_ulm_p = full_ulm.get();
  auto result =
      std::make_unique<OneTokenUlm>(std::move(full_ulm), std::move(*graph));
  {
    Tensor::DimsType dims{ulm_params.seq_size_T, ulm_params.model_dim_D};
    result->pos_embedding_data_ =
        std::make_shared<Tensor>(std::move(dims), xnn_datatype_fp32);
    result->pos_embedding_data_->Borrow(full_ulm_p->pos_embedding_data_);
  }
  result->transformer_input_ = input;
  result->transformer_output_ = transformer_output;
  result->normed_output_ = normed_output;
  result->logits_output_ = logits_output;
  result->segment_pos_ = segment_pos;
  result->atten_masks_ = atten_masks;
  if (ulm_params.use_padding) {
    // result->paddings_ = paddings;
  }
  result->kv_cache_ = std::move(kv_cache);

  result->weights_ = std::move(weights);
  result->ulm_params_ = ulm_params;

  return result;
}

absl::StatusOr<std::unique_ptr<Ulm>> Ulm::CreateUlm(
    std::unique_ptr<UlmWeightsLoader> weight_loader,
    std::unique_ptr<RuntimeConfigs> runtime_configs) {
  UlmBuilder builder;
  weight_loader->SetBuilder(builder);
  const auto& ulm_params = weight_loader->ulm_params();
  RET_CHECK_NE(ulm_params.batch_size_B, 0);

  ASSIGN_OR_RETURN(auto input, builder.NewInput({ulm_params.batch_size_B,
                                                 ulm_params.seq_size_T,
                                                 ulm_params.model_dim_D}));
  ASSIGN_OR_RETURN(auto atten_masks, builder.NewInput({ulm_params.seq_size_T,
                                                       ulm_params.seq_size_T}));
  VLOG(1) << "atten mask id " << atten_masks->tensor_id;
  ASSIGN_OR_RETURN(
      auto segment_pos,
      builder.NewWeight({ulm_params.seq_size_T, ulm_params.head_dim_H}));
  MP_RETURN_IF_ERROR(FillXnnRoPEWeights(*segment_pos));
  VLOG(1) << "segment pos id " << segment_pos->tensor_id;
  std::shared_ptr<Tensor> paddings;
  if (ulm_params.use_padding) {
    ASSIGN_OR_RETURN(paddings, builder.NewInput({ulm_params.batch_size_B,
                                                 ulm_params.seq_size_T, 1}));
    VLOG(1) << "paddings id " << paddings->tensor_id;
  }

  ASSIGN_OR_RETURN(auto weights, weight_loader->LoadWeights());
  std::vector<KVCache> kv_cache;

  auto inter_layer = input;
  for (int i = 0; i < ulm_params.num_transformer_M; ++i) {
    const auto& sa = weights.sas[i];
    KVCache* cache = nullptr;
    if (ulm_params.enable_kv_cache) {
      auto k_cache = std::make_shared<Tensor>(
          Tensor::DimsType{ulm_params.batch_size_B, ulm_params.seq_size_T,
                           ulm_params.n_heads_N, ulm_params.head_dim_H});
      MP_RETURN_IF_ERROR(k_cache->LoadFromVec({}, /*exact_match=*/false));
      auto v_cache = std::make_shared<Tensor>(
          Tensor::DimsType{ulm_params.batch_size_B, ulm_params.seq_size_T,
                           ulm_params.n_heads_N, ulm_params.head_dim_H});
      MP_RETURN_IF_ERROR(v_cache->LoadFromVec({}, /*exact_match=*/false));
      kv_cache.push_back(KVCache{.k_cache = k_cache, .v_cache = v_cache});
      cache = &kv_cache.back();
    }
    ASSIGN_OR_RETURN(auto tmp, builder.SelfAttentionIncludeResidual(
                                   inter_layer,
                                   {.atten_mask = atten_masks,
                                    .segment_pos = segment_pos,
                                    .cache = cache},
                                   sa));

    auto& ff = weights.ffs[i];
    ff.opt_padding = paddings;
    ASSIGN_OR_RETURN(inter_layer, builder.FeedForwardIncludeResidual(tmp, ff));
  }

  std::shared_ptr<Tensor> logits_output, transformer_output, normed_output;

  if (!ulm_params.final_norm && !ulm_params.final_project) {
    transformer_output = inter_layer;
    transformer_output->MarkOutput();
  }

  if (ulm_params.final_norm) {
    ASSIGN_OR_RETURN(inter_layer,
                     builder.RmsNorm(inter_layer, weights.final_ln_scale));
    normed_output = inter_layer;
    normed_output->MarkOutput();
  }

  if (ulm_params.final_project) {
    RET_CHECK(weights.softmax_linear);
    ASSIGN_OR_RETURN(logits_output,
                     ApplyFinalProj(inter_layer, weights, builder));
    logits_output->MarkOutput();
  }

  ASSIGN_OR_RETURN(auto graph, builder.Build(std::move(runtime_configs)));
  auto ulm = std::make_unique<Ulm>(std::move(*graph));
  {
    ASSIGN_OR_RETURN(auto pos_embedding_data,
                     mediapipe::PositionEmbedding(ulm_params.seq_size_T,
                                                  ulm_params.model_dim_D));
    Tensor::DimsType dims{ulm_params.seq_size_T, ulm_params.model_dim_D};
    ulm->pos_embedding_data_ =
        std::make_shared<Tensor>(std::move(dims), xnn_datatype_fp32);
    MP_RETURN_IF_ERROR(
        ulm->pos_embedding_data_->LoadFromVec(pos_embedding_data));
  }
  ulm->transformer_input_ = input;
  ulm->transformer_output_ = transformer_output;
  ulm->normed_output_ = normed_output;
  ulm->logits_output_ = logits_output;
  ulm->segment_pos_ = segment_pos;
  ulm->atten_masks_ = atten_masks;
  if (ulm_params.use_padding) {
    ulm->paddings_ = paddings;
  }
  ulm->kv_cache_ = std::move(kv_cache);

  ulm->weights_ = std::move(weights);
  ulm->ulm_params_ = ulm_params;

  return ulm;
}

absl::Status Ulm::InitInputTokens(const std::vector<int>& input_ids) {
  prev_ids_ = input_ids;

  constexpr float neg_value = 0.7 * std::numeric_limits<float>::lowest();
  const auto& seq_size = ulm_params_.seq_size_T;
  std::vector<float> attention_array(seq_size * seq_size, neg_value);
  for (int i = 0; i < seq_size; ++i) {
    for (int j = 0; j < seq_size; ++j) {
      if (i < input_ids.size() && j < input_ids.size()) {
        attention_array[seq_size * i + j] = 0;
      } else if (i >= seq_size && j <= i) {
        attention_array[seq_size * i + j] = 0;
      } else {
        break;
      }
    }
  }

  MP_RETURN_IF_ERROR(atten_masks_->LoadFromVec(attention_array));

  MP_RETURN_IF_ERROR(GetTokenEmbedding(input_ids,
                                       pos_embedding_data_->DataAs<float>(),
                                       transformer_input_->DataAs<float>()));
  return SetupRuntime();
}

absl::Status Ulm::GetNextToken(std::vector<int>* output_ids) {
  VLOG(2) << "Decode step " << prev_ids_.size() - 1;

  MP_RETURN_IF_ERROR(Run());

  RET_CHECK(logits_output_);
  std::shared_ptr<Tensor> logits =
      logits_output_->Slice({0, prev_ids_.size() - 1, 0});
  DCHECK_EQ(logits->num_elements, ulm_params_.voc_size_V);

  ASSIGN_OR_RETURN(*output_ids,
                   mediapipe::SampleNextToken(
                       logits->DataAs<float>(),
                       /*batch_size=*/1,
                       /*vocab_size=*/ulm_params_.voc_size_V, /*top_k=*/10,
                       /*top_p=*/1, /*temperature=*/-1));
  RET_CHECK_EQ(output_ids->size(), 1);
  prev_ids_.push_back(output_ids->at(0));

  return GetTokenEmbedding(
      *output_ids,
      pos_embedding_data_->Slice({prev_ids_.size() - 1, 0})->DataAs<float>(),
      transformer_input_->Slice({0, prev_ids_.size() - 1, 0})->DataAs<float>());
}

absl::Status Ulm::GetTokenEmbedding(const std::vector<int>& ids,
                                    const float* pos_embedding_data,
                                    float* embedding) {
  auto token_embedding = weights_.token_embedding ? weights_.token_embedding
                                                  : weights_.softmax_linear;
  RET_CHECK(token_embedding->dims[0] == ulm_params_.voc_size_V)
      << "shape must be [vocab_size, _], such that following Slice() makes "
         "sense.";
  for (size_t id : ids) {
    memcpy(embedding, token_embedding->Slice(0, id)->Data(),
           ulm_params_.model_dim_D * sizeof(float));
    for (size_t i = 0; i < ulm_params_.model_dim_D; ++i) {
      embedding[i] += pos_embedding_data[i];
    }
    pos_embedding_data += ulm_params_.model_dim_D;
    embedding += ulm_params_.model_dim_D;
  }
  return absl::OkStatus();
}

}  // namespace xnn_utils
}  // namespace mediapipe
