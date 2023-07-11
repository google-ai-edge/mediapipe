#ifndef MEDIAPIPE_TASKS_CC_TEXT_UTILS_XNN_UTILS_ULM_WEIGHTS_H_
#define MEDIAPIPE_TASKS_CC_TEXT_UTILS_XNN_UTILS_ULM_WEIGHTS_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/text/utils/xnn_utils/graph_builder.h"
#include "mediapipe/tasks/cc/text/utils/xnn_utils/xnn_tensor.h"
#include "third_party/XNNPACK/include/xnnpack.h"

namespace mediapipe {
namespace xnn_utils {

struct UlmParams {
  size_t num_transformer_M = 18;
  size_t batch_size_B = 1;
  size_t seq_size_T = 16;
  size_t model_dim_D = 1536;
  size_t hidden_dim_HD = 8 * 1536;
  size_t head_dim_H = 128;
  size_t n_heads_N = 12;
  size_t voc_size_V = 32000;

  bool use_padding = true;
  bool final_norm = true;
  bool final_project = true;

  bool enable_kv_cache = false;
  // Path to store reshaped weights as cache. Set empty to disable caching.
  std::string weight_cache_path;
};

struct SelfAttentionWeights {
  std::shared_ptr<Tensor> pre_norm;

  std::shared_ptr<Tensor> k_weight;
  std::shared_ptr<Tensor> q_weight;
  std::shared_ptr<Tensor> v_weight;
  std::shared_ptr<Tensor> per_dim_scale;
  std::shared_ptr<Tensor> post_proj_weight;

  std::shared_ptr<Tensor> post_norm;
};

struct FeedForwardWeights {
  std::shared_ptr<Tensor> pre_norm;
  std::shared_ptr<Tensor> layer_1_weight;
  std::shared_ptr<Tensor> layer_1_bias;
  std::shared_ptr<Tensor> layer_1_gate_weight;
  std::shared_ptr<Tensor> layer_1_gate_bias;
  std::shared_ptr<Tensor> layer_2_weight;
  std::shared_ptr<Tensor> layer_2_bias;
  std::shared_ptr<Tensor> post_norm;

  std::shared_ptr<Tensor> opt_padding;
};

struct UlmWeights {
  std::vector<FeedForwardWeights> ffs;
  std::vector<SelfAttentionWeights> sas;
  std::shared_ptr<Tensor> final_ln_scale;
  std::shared_ptr<Tensor> softmax_linear;
  std::shared_ptr<Tensor> softmax_bias;

  // Optional. Usually softmax_linear can be used as embedding, but sometimes we
  // need to scale/transpose it.
  std::shared_ptr<Tensor> token_embedding;

  static constexpr absl::string_view kKeyLoadedFromCache{"loaded_from_cache"};
};

class UlmWeightsLoader {
 public:
  constexpr static absl::string_view kTransformerWeightPrefix{
      "params.lm.transformer.x_layers_"};
  constexpr static absl::string_view kFinalScaleFilename{
      "params.lm.final_ln.scale"};
  constexpr static absl::string_view kLogitsFfnBiasFilename{
      "params.lm.softmax.logits_ffn.bias.b"};
  constexpr static absl::string_view kLogitsFfnWeightFilename{
      "params.lm.softmax.logits_ffn.linear.w"};

  UlmWeightsLoader(absl::string_view weight_path, const UlmParams& params)
      : weight_path_(weight_path), params_(params) {}
  virtual ~UlmWeightsLoader() = default;

  void SetBuilder(XnnGraphBuilder& builder) { builder_ = &builder; }

  virtual absl::StatusOr<UlmWeights> LoadWeights();

  virtual absl::StatusOr<SelfAttentionWeights> LoadSelfAttention(int layer_id);
  virtual absl::StatusOr<FeedForwardWeights> LoadFeedForward(int layer_id);

  UlmParams& ulm_params() { return params_; }
  const UlmParams& ulm_params() const { return params_; }
  XnnGraphBuilder& builder() const { return *builder_; }

 protected:
  // Find the files that matches prefix, then read from file.
  virtual absl::StatusOr<std::shared_ptr<Tensor>> LoadFromAbsPathPrefix(
      absl::string_view prefix, const Tensor::DimsType& dims,
      size_t dim_scale_if_any) const;
  absl::StatusOr<std::shared_ptr<Tensor>> LoadFromAbsPathPrefix(
      absl::string_view prefix, const Tensor::DimsType& dims) const {
    return LoadFromAbsPathPrefix(prefix, dims, 0);
  }

  absl::StatusOr<std::shared_ptr<Tensor>> TryCacheThenLoadSelfAttention(
      absl::string_view filename_prefix) const;
  absl::StatusOr<std::shared_ptr<Tensor>> TryCacheThenLoadFeedForward(
      absl::string_view filename_prefix,
      std::optional<Tensor::DimsType> dims = std::nullopt) const;
  virtual absl::StatusOr<std::shared_ptr<Tensor>>
  TryCacheThenLoadWeightTranspose(absl::string_view filename_prefix,
                                  Tensor::DimsType original_dims,
                                  size_t original_dim_cale) const;

  std::string weight_path_;
  UlmParams params_;
  XnnGraphBuilder* builder_ = nullptr;
};

// Try: 1. load token embedding from cache; 2. fill token embedding by transpose
// softmax linear then scale; 3. dump token embedding to cache.
struct PrepareTokenEmbeddingDecorator {
  static absl::Status Decorate(const UlmWeightsLoader&, UlmWeights&);
};
struct TransposeSoftmaxWeightDecorator {
  static absl::Status Decorate(const UlmWeightsLoader&, UlmWeights&);
};
struct TransposeSelfAttentionWeightDecorator {
  // If KQV weight are reshaped, ignore.
  // If KQV weight are not properly shaped, load from cache if any, or build.
  // If KQV weight are missing, try loading from cache path, or fail if missing.
  static absl::Status Decorate(const UlmWeightsLoader&, UlmWeights&);
};

// Apply some decoration (in order) to the weights loaded by base class.
template <class... Decorators>
class UlmWeightsLoaderWith : public UlmWeightsLoader {
 public:
  UlmWeightsLoaderWith(absl::string_view weight_path, const UlmParams& params)
      : UlmWeightsLoader(weight_path, params),
        decorators_{Decorators::Decorate...} {}

  absl::StatusOr<UlmWeights> LoadWeights() override {
    ASSIGN_OR_RETURN(auto result, UlmWeightsLoader::LoadWeights());
    for (const auto& decorator : decorators_) {
      MP_RETURN_IF_ERROR(decorator(*this, result));
    }
    return result;
  }

 protected:
  std::vector<std::function<absl::Status(const UlmWeightsLoader&, UlmWeights&)>>
      decorators_;
};

using DefaultUlmWeightsLoader =
    UlmWeightsLoaderWith<TransposeSelfAttentionWeightDecorator,
                         PrepareTokenEmbeddingDecorator>;

// Generate weights with some random value.
class BenchmarkUlmWeightsLoader : public DefaultUlmWeightsLoader {
 public:
  explicit BenchmarkUlmWeightsLoader(
      const UlmParams& params, xnn_datatype data_type = xnn_datatype_fp32);

  absl::StatusOr<std::shared_ptr<Tensor>> TryCacheThenLoadWeightTranspose(
      absl::string_view filename_prefix, Tensor::DimsType original_dims,
      size_t original_dim_cale) const override;

  absl::StatusOr<std::shared_ptr<Tensor>> LoadFromAbsPathPrefix(
      absl::string_view prefix, const Tensor::DimsType& dims,
      size_t dim_scale_if_any) const override;

 private:
  xnn_datatype data_type_;
  std::shared_ptr<Tensor> random_value_buffer_;
};

}  // namespace xnn_utils
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_TEXT_UTILS_XNN_UTILS_ULM_WEIGHTS_H_
