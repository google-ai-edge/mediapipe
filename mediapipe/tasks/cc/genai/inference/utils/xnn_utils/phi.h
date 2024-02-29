#ifndef MEDIAPIPE_TASKS_GENAI_INFERENCE_UTILS_XNN_UTILS_PHI_H_
#define MEDIAPIPE_TASKS_GENAI_INFERENCE_UTILS_XNN_UTILS_PHI_H_

#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/llm.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/llm_weights.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/xnn_tensor.h"

namespace mediapipe::tasks::genai::xnn_utils {

class Phi2Builder : public LlmBuilder {
 public:
  using LlmBuilder::LlmBuilder;
  ~Phi2Builder() override = default;

 protected:
  // Overrides the default PreProcess with the following changes:
  // * Initialized `resource.segment_pos` with partial rope dimmensions.
  // * Creates dummy `resource.pos_embedding` that is not used.
  // * Skips token embedding scaling.
  absl::StatusOr<std::pair<std::shared_ptr<Tensor>, InputResource>> PreProcess(
      std::shared_ptr<Tensor> token_embedding, bool is_prefix) override;

  // Overrides the default DecoderLayer with the following changes:
  // * Support parallel decoder connectivity.
  absl::StatusOr<std::shared_ptr<Tensor>> OneStackTransformer(
      int layer_index, std::shared_ptr<Tensor> input, InputResource resource,
      const LlmWeights::SelfAttentionWeights& sa_weights,
      const LlmWeights::FeedForwardWeights& ff_weights,
      bool is_prefix) override;

  // Defines an alternative to `SelfAttentionExcludeNorm` defined in the base
  // class with the following changes:
  // * Replaces `Rope` with `PartialRope`.
  absl::StatusOr<std::shared_ptr<Tensor>> SelfAttentionExcludeNorm(
      std::shared_ptr<Tensor> input, InputResource resource,
      const LlmWeights::SelfAttentionWeights& sa_weights) override;

  // Defines an alternative to `FeedForwardExcludeNorm` defined in the base
  // class with the following changes:
  // * Vanilla feed forward network with sequential structure (as opposed to
  //   gated FFNs.)
  absl::StatusOr<std::shared_ptr<Tensor>> FeedForwardExcludeNorm(
      std::shared_ptr<Tensor> input,
      const LlmWeights::FeedForwardWeights& ff_weights) override;
};

}  // namespace mediapipe::tasks::genai::xnn_utils

#endif  // MEDIAPIPE_TASKS_GENAI_INFERENCE_UTILS_XNN_UTILS_PHI_H_
