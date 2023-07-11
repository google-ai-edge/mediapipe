#ifndef MEDIAPIPE_TASKS_CC_TEXT_UTILS_XNN_UTILS_ULM_H_
#define MEDIAPIPE_TASKS_CC_TEXT_UTILS_XNN_UTILS_ULM_H_

#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/tasks/cc/text/utils/xnn_utils/graph_builder.h"
#include "mediapipe/tasks/cc/text/utils/xnn_utils/ulm_weights.h"
#include "mediapipe/tasks/cc/text/utils/xnn_utils/xnn_tensor.h"

namespace mediapipe {
namespace xnn_utils {

class Ulm : public XnnGraph {
 public:
  using UlmParams = UlmParams;

  explicit Ulm(XnnGraph&& other) : XnnGraph(std::move(other)) {}
  ~Ulm() override = default;

  // Creating ULM graph with default params. The default param corresponds to
  // ULM1B 256k model.
  static absl::StatusOr<std::unique_ptr<Ulm>> CreateUlm(
      absl::string_view weights_folder,
      const UlmParams& ulm_params =
          UlmParams{
              .num_transformer_M = 18,
              .batch_size_B = 1,
              .seq_size_T = 16,
              .model_dim_D = 1536,
              .hidden_dim_HD = 8 * 1536,
              .head_dim_H = 128,
              .n_heads_N = 12,
              .voc_size_V = 256128,
          },
      std::unique_ptr<RuntimeConfigs> runtime_configs = nullptr);
  static absl::StatusOr<std::unique_ptr<Ulm>> CreateUlm(
      std::unique_ptr<UlmWeightsLoader> weight_loader,
      std::unique_ptr<RuntimeConfigs> runtime_configs = nullptr);
  // Build the graph for one-token inference.
  static absl::StatusOr<std::unique_ptr<Ulm>> CreateOneTokenUlm(
      std::unique_ptr<UlmWeightsLoader> weight_loader,
      std::unique_ptr<RuntimeConfigs> runtime_configs = nullptr);

  // (Re)Initialize with input token ids. This will reset the cache, mask etc.
  virtual absl::Status InitInputTokens(const std::vector<int>& input_ids);

  // Get the next token id.
  virtual absl::Status GetNextToken(std::vector<int>* output_ids);

 protected:
  friend class OneTokenUlm;
  friend class UlmTest;
  friend class UlmBuilder;

  // Enable if enable_kv_cache
  struct KVCache {
    std::shared_ptr<Tensor> k_cache;
    std::shared_ptr<Tensor> v_cache;
    std::shared_ptr<Tensor> k_slice;
    std::shared_ptr<Tensor> v_slice;
  };

  absl::Status GetTokenEmbedding(const std::vector<int>& ids,
                                 const float* pos_embedding_data,
                                 float* embedding);

  UlmWeights weights_;
  UlmParams ulm_params_;

  std::shared_ptr<Tensor> pos_embedding_data_;
  std::shared_ptr<Tensor> atten_masks_;
  std::shared_ptr<Tensor> segment_pos_;
  std::shared_ptr<Tensor> paddings_;

  std::shared_ptr<Tensor> transformer_input_;
  std::shared_ptr<Tensor> transformer_output_;
  std::shared_ptr<Tensor> normed_output_;
  std::shared_ptr<Tensor> logits_output_;

  // Previous ids, including prompt.
  std::vector<int> prev_ids_;
  // If enable_kv_cache, expect a mask of [0, ... 0, 1, 0, 0...], size 1 x T.
  std::shared_ptr<Tensor> decode_step_mask_;
  // [1, 1, ..., 1, 0, 0...], applied on cache
  std::shared_ptr<Tensor> decode_step_mask_for_cache_;
  std::vector<KVCache> kv_cache_;
};

class UlmBuilder : public XnnGraphBuilder {
 public:
  struct SelfAttentionArgs {
    std::shared_ptr<Tensor> atten_mask;
    std::shared_ptr<Tensor> segment_pos;

    Ulm::KVCache* cache = nullptr;
  };

  absl::StatusOr<std::shared_ptr<Tensor>> SelfAttentionExcludeNorm(
      std::shared_ptr<Tensor> input, SelfAttentionArgs args,
      const SelfAttentionWeights& sa_weights,
      absl::SourceLocation loc = absl::SourceLocation::current());

  absl::StatusOr<std::shared_ptr<Tensor>> SelfAttentionIncludeResidual(
      std::shared_ptr<Tensor> input, SelfAttentionArgs args,
      const SelfAttentionWeights& params,
      absl::SourceLocation loc = absl::SourceLocation::current());

  absl::StatusOr<std::shared_ptr<Tensor>> FeedForwardExcludeResidual(
      std::shared_ptr<Tensor> input, const FeedForwardWeights& params,
      absl::SourceLocation loc = absl::SourceLocation::current());
  absl::StatusOr<std::shared_ptr<Tensor>> FeedForwardIncludeResidual(
      std::shared_ptr<Tensor> input, const FeedForwardWeights& params,
      absl::SourceLocation loc = absl::SourceLocation::current());
};

}  // namespace xnn_utils
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_TEXT_UTILS_XNN_UTILS_ULM_H_
