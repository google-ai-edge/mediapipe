#include "mediapipe/tasks/cc/text/utils/xnn_utils/ulm_weights.h"

#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "file/base/filesystem.h"
#include "file/base/options.h"
#include "file/base/path.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/text/utils/xnn_utils/graph_builder.h"
#include "mediapipe/tasks/cc/text/utils/xnn_utils/xnn_tensor.h"
#include "third_party/XNNPACK/include/xnnpack.h"

namespace mediapipe {
namespace xnn_utils {

namespace {

absl::StatusOr<std::shared_ptr<Tensor>> LoadFromAbsPathPrefixHelper(
    XnnGraphBuilder& builder, absl::string_view prefix,
    const Tensor::DimsType& dims, size_t dim_scale_if_any) {
  RET_CHECK(!prefix.empty() && prefix.back() != '.');
  std::vector<std::string> filenames;
  auto s = file::Match(absl::StrCat(prefix, "*"), &filenames, file::Defaults());
  if (!s.ok()) {
    LOG(WARNING) << s;
    return nullptr;
  } else if (filenames.empty()) {
    return nullptr;
  }

  if (filenames.size() == 1) {
    RET_CHECK_EQ(filenames[0], prefix);
    return builder.NewWeight(filenames[0], dims);
  }

  bool is_quantized_tensor = false;
  for (const auto& filename : filenames) {
    if (absl::StrContains(filename, kQuantizedScaleSuffix)) {
      is_quantized_tensor = true;
      continue;
    }
  }

  RET_CHECK(is_quantized_tensor)
      << "At least one of {" << filenames << "} must be quantize scale file.";

  std::shared_ptr<Tensor> result;
  result = std::make_shared<QCTensor>(dims, dim_scale_if_any);

  MP_RETURN_IF_ERROR(result->LoadFromFile(prefix));
  builder.NewWeight(result);

  return result;
}

absl::Status TransposeSelfAttentionWeight(
    const UlmWeightsLoader& loader, std::shared_ptr<Tensor>& original_weight,
    absl::string_view cache_file_prefix) {
  const auto& ulm_param = loader.ulm_params();
  RET_CHECK(original_weight);

  std::optional<int> from_cache =
      original_weight->GetMetadata(UlmWeights::kKeyLoadedFromCache);
  if (from_cache && *from_cache) {
    return absl::OkStatus();
  }

  if (auto s = original_weight->DumpToFile(cache_file_prefix); !s.ok()) {
    LOG(WARNING) << s;
  } else {
    MP_RETURN_IF_ERROR(original_weight->LoadFromFile(cache_file_prefix));
  }
  loader.builder().NewWeight(original_weight);
  original_weight->SetMetadata(XnnGraphBuilder::kKeySelfAttentionReshapedWeight,
                               ulm_param.n_heads_N);
  return absl::OkStatus();
}

}  // namespace

absl::Status PrepareTokenEmbeddingDecorator::Decorate(
    const UlmWeightsLoader& loader, UlmWeights& weight) {
  if (weight.token_embedding) {
    return absl::OkStatus();
  }

  const auto& ulm_params = loader.ulm_params();
  absl::string_view cache_path = loader.ulm_params().weight_cache_path;
  std::string token_embedding_cache_path =
      cache_path.empty() ? "" : file::JoinPath(cache_path, "token_embedding.w");
  // 1. try cache
  if (!token_embedding_cache_path.empty()) {
    auto token_embedding =
        Tensor::FromFile(token_embedding_cache_path,
                         {ulm_params.voc_size_V, ulm_params.model_dim_D});
    if (token_embedding.ok()) {
      weight.token_embedding = *token_embedding;
      return absl::OkStatus();
    }
  }

  // 2. fill embedding from softmax_linear
  auto& softmax_linear = *weight.softmax_linear;
  RET_CHECK(softmax_linear.dims[0] == ulm_params.voc_size_V) << softmax_linear;
  if (softmax_linear.datatype == xnn_datatype_fp32) {
    weight.token_embedding = softmax_linear.View();
  } else if (softmax_linear.datatype == xnn_datatype_qcint8) {
    ASSIGN_OR_RETURN(weight.token_embedding, softmax_linear.ConvertToF32());
  }

  float* embedding_data = weight.token_embedding->DataAs<float>();
  for (size_t i = 0; i < softmax_linear.num_elements; ++i) {
    embedding_data[i] *= std::sqrt(loader.ulm_params().model_dim_D);
  }

  // 3. save cache
  if (!token_embedding_cache_path.empty()) {
    MP_RETURN_IF_ERROR(
        weight.token_embedding->DumpToFile(token_embedding_cache_path));
    return weight.token_embedding->LoadFromFile(token_embedding_cache_path);
  }

  return absl::OkStatus();
}

absl::Status TransposeSelfAttentionWeightDecorator::Decorate(
    const UlmWeightsLoader& loader, UlmWeights& weight) {
  absl::string_view cache_path = loader.ulm_params().weight_cache_path;
  if (cache_path.empty()) {
    return absl::OkStatus();
  }

  for (size_t i = 0; i < weight.sas.size(); ++i) {
    auto& sa = weight.sas[i];
    auto prefix = absl::StrCat(UlmWeightsLoader::kTransformerWeightPrefix, i,
                               ".self_attention.");
    MP_RETURN_IF_ERROR(TransposeSelfAttentionWeight(
        loader, sa.k_weight,
        file::JoinPath(cache_path, absl::StrCat(prefix, "k.w"))));
    MP_RETURN_IF_ERROR(TransposeSelfAttentionWeight(
        loader, sa.q_weight,
        file::JoinPath(cache_path, absl::StrCat(prefix, "q.w"))));
    MP_RETURN_IF_ERROR(TransposeSelfAttentionWeight(
        loader, sa.v_weight,
        file::JoinPath(cache_path, absl::StrCat(prefix, "v.w"))));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<Tensor>> UlmWeightsLoader::LoadFromAbsPathPrefix(
    absl::string_view prefix, const Tensor::DimsType& dims,
    size_t dim_scale_if_any) const {
  return LoadFromAbsPathPrefixHelper(*builder_, prefix, dims, dim_scale_if_any);
}

absl::StatusOr<std::shared_ptr<Tensor>>
UlmWeightsLoader::TryCacheThenLoadSelfAttention(
    absl::string_view filename_prefix) const {
  ASSIGN_OR_RETURN(
      auto r,
      TryCacheThenLoadWeightTranspose(
          filename_prefix,
          {params_.model_dim_D, params_.n_heads_N * params_.head_dim_H}, 1));
  r->SetMetadata(XnnGraphBuilder::kKeySelfAttentionReshapedWeight,
                 params_.n_heads_N);
  return r;
}

absl::StatusOr<std::shared_ptr<Tensor>>
UlmWeightsLoader::TryCacheThenLoadFeedForward(
    absl::string_view filename_prefix,
    std::optional<Tensor::DimsType> dims) const {
  if (!dims) {
    dims = {params_.model_dim_D, params_.hidden_dim_HD};
  }
  return TryCacheThenLoadWeightTranspose(filename_prefix, *dims, 1);
}

absl::StatusOr<std::shared_ptr<Tensor>>
UlmWeightsLoader::TryCacheThenLoadWeightTranspose(
    absl::string_view filename_prefix, Tensor::DimsType original_dims,
    size_t original_dim_cale) const {
  if (!params_.weight_cache_path.empty()) {
    auto cache_full_prefix =
        file::JoinPath(params_.weight_cache_path, filename_prefix);
    Tensor::DimsType cache_dim{original_dims.rbegin(), original_dims.rend()};
    ASSIGN_OR_RETURN(auto r, LoadFromAbsPathPrefix(
                                 cache_full_prefix, std::move(cache_dim),
                                 /*dim_scale_if_any=*/1 - original_dim_cale));
    if (r) {
      r->SetMetadata(UlmWeights::kKeyLoadedFromCache, 1);
      return r;
    }
  }

  ASSIGN_OR_RETURN(auto r, LoadFromAbsPathPrefix(
                               file::JoinPath(weight_path_, filename_prefix),
                               std::move(original_dims),
                               /*dim_scale_if_any=*/original_dim_cale));
  RET_CHECK(r) << file::JoinPath(weight_path_, filename_prefix);
  r = r->Transpose();
  builder_->NewWeight(r);
  return r;
}

absl::StatusOr<FeedForwardWeights> UlmWeightsLoader::LoadFeedForward(
    int layer_id) {
  absl::string_view weights_folder = weight_path_;
  const auto& params = params_;
  auto ff_file_prefix =
      absl::StrCat(kTransformerWeightPrefix, layer_id, ".ff_layer.");
  auto ff_prefix = file::JoinPath(weights_folder, ff_file_prefix);
  FeedForwardWeights feed_forward;

  ASSIGN_OR_RETURN(
      feed_forward.pre_norm,
      LoadFromAbsPathPrefix(absl::StrCat(ff_prefix, "pre_layer_norm.scale"),
                            {params.model_dim_D}));
  ASSIGN_OR_RETURN(
      feed_forward.post_norm,
      LoadFromAbsPathPrefix(absl::StrCat(ff_prefix, "post_layer_norm.scale"),
                            {params.model_dim_D}));
  ASSIGN_OR_RETURN(
      feed_forward.layer_1_bias,
      LoadFromAbsPathPrefix(absl::StrCat(ff_prefix, "ffn_layer1.bias.b"),
                            {params.hidden_dim_HD}));
  ASSIGN_OR_RETURN(feed_forward.layer_1_weight,
                   TryCacheThenLoadFeedForward(
                       absl::StrCat(ff_file_prefix, "ffn_layer1.linear.w")));
  ASSIGN_OR_RETURN(
      feed_forward.layer_1_gate_bias,
      LoadFromAbsPathPrefix(absl::StrCat(ff_prefix, "ffn_layer1_gate.bias.b"),
                            {params.hidden_dim_HD}));
  ASSIGN_OR_RETURN(feed_forward.layer_1_gate_weight,
                   TryCacheThenLoadFeedForward(absl::StrCat(
                       ff_file_prefix, "ffn_layer1_gate.linear.w")));
  ASSIGN_OR_RETURN(
      feed_forward.layer_2_bias,
      LoadFromAbsPathPrefix(absl::StrCat(ff_prefix, "ffn_layer2.bias.b"),
                            {params.model_dim_D}, /*dim_scale_if_any=*/0));
  ASSIGN_OR_RETURN(
      feed_forward.layer_2_weight,
      TryCacheThenLoadFeedForward(
          absl::StrCat(ff_file_prefix, "ffn_layer2.linear.w"),
          Tensor::DimsType{params.hidden_dim_HD, params.model_dim_D}));

  return feed_forward;
}

absl::StatusOr<SelfAttentionWeights> UlmWeightsLoader::LoadSelfAttention(
    int layer_id) {
  absl::string_view weights_folder = weight_path_;
  const auto& params = params_;
  SelfAttentionWeights self_attention;

  auto sa_file_prefix = absl::StrCat(kTransformerWeightPrefix, layer_id);
  auto sa_prefix = file::JoinPath(weights_folder, sa_file_prefix);
  ASSIGN_OR_RETURN(
      self_attention.pre_norm,
      LoadFromAbsPathPrefix(absl::StrCat(sa_prefix, ".pre_layer_norm.scale"),
                            {params.model_dim_D}));
  ASSIGN_OR_RETURN(
      self_attention.post_norm,
      LoadFromAbsPathPrefix(absl::StrCat(sa_prefix, ".post_layer_norm.scale"),
                            {params.model_dim_D}));

  absl::StrAppend(&sa_file_prefix, ".self_attention.");

  ASSIGN_OR_RETURN(
      self_attention.k_weight,
      TryCacheThenLoadSelfAttention(absl::StrCat(sa_file_prefix, "k.w")));
  ASSIGN_OR_RETURN(
      self_attention.q_weight,
      TryCacheThenLoadSelfAttention(absl::StrCat(sa_file_prefix, "q.w")));
  ASSIGN_OR_RETURN(
      self_attention.v_weight,
      TryCacheThenLoadSelfAttention(absl::StrCat(sa_file_prefix, "v.w")));

  sa_prefix = file::JoinPath(weights_folder, sa_file_prefix);
  ASSIGN_OR_RETURN(self_attention.per_dim_scale,
                   LoadFromAbsPathPrefix(
                       absl::StrCat(sa_prefix, "per_dim_scale.per_dim_scale"),
                       {params.head_dim_H}));
  ASSIGN_OR_RETURN(self_attention.post_proj_weight,
                   LoadFromAbsPathPrefix(absl::StrCat(sa_prefix, "post.w"),
                                         {params.model_dim_D,
                                          params.n_heads_N * params.head_dim_H},
                                         /*dim_scale_if_any=*/0));

  return self_attention;
}

absl::StatusOr<UlmWeights> UlmWeightsLoader::LoadWeights() {
  absl::string_view weights_folder = weight_path_;
  const auto& params = params_;
  UlmWeights result;

  for (int layer_id = 0; layer_id < params.num_transformer_M; ++layer_id) {
    ASSIGN_OR_RETURN(auto ff, LoadFeedForward(layer_id));
    result.ffs.push_back(std::move(ff));
    ASSIGN_OR_RETURN(auto sa, LoadSelfAttention(layer_id));
    result.sas.push_back(std::move(sa));
  }
  if (params.final_norm) {
    ASSIGN_OR_RETURN(result.final_ln_scale,
                     LoadFromAbsPathPrefix(
                         file::JoinPath(weights_folder, kFinalScaleFilename),
                         {params.model_dim_D}));
  }
  ASSIGN_OR_RETURN(result.softmax_bias,
                   LoadFromAbsPathPrefix(
                       file::JoinPath(weights_folder, kLogitsFfnBiasFilename),
                       {params.voc_size_V}));
  ASSIGN_OR_RETURN(result.softmax_linear,
                   TryCacheThenLoadWeightTranspose(
                       kLogitsFfnWeightFilename,
                       {params.model_dim_D, params.voc_size_V}, 1));

  return result;
}

BenchmarkUlmWeightsLoader::BenchmarkUlmWeightsLoader(const UlmParams& params,
                                                     xnn_datatype data_type)
    : DefaultUlmWeightsLoader("", params), data_type_(data_type) {
  params_.weight_cache_path.clear();
}

absl::StatusOr<std::shared_ptr<Tensor>>
BenchmarkUlmWeightsLoader::TryCacheThenLoadWeightTranspose(
    absl::string_view filename_prefix, Tensor::DimsType original_dims,
    size_t original_dim_cale) const {
  auto result = std::make_shared<QCTensor>(
      Tensor::DimsType{original_dims.rbegin(), original_dims.rend()},
      1 - original_dim_cale);
  auto real_data = std::make_shared<std::string>(result->num_elements, 0xA5);
  result->flat_data = std::shared_ptr<char>(real_data, real_data->data());
  auto real_scale = std::make_shared<std::vector<float>>(
      original_dims[original_dim_cale], 1.0f);
  result->scale_data = std::shared_ptr<float>(real_scale, real_scale->data());
  builder_->NewWeight(result);
  return result;
}

absl::StatusOr<std::shared_ptr<Tensor>>
BenchmarkUlmWeightsLoader::LoadFromAbsPathPrefix(
    absl::string_view prefix, const Tensor::DimsType& dims,
    size_t dim_scale_if_any) const {
  // If loader calls this function directly, it's always non-quantized weights.
  auto result = std::make_shared<Tensor>(dims);
  MP_RETURN_IF_ERROR(result->LoadFromVec({}, /*exact_match=*/false));
  builder_->NewWeight(result);
  return result;
}

}  // namespace xnn_utils
}  // namespace mediapipe
