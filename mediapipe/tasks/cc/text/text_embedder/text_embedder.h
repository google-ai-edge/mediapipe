/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MEDIAPIPE_TASKS_CC_TEXT_TEXT_EMBEDDER_TEXT_EMBEDDER_H_
#define MEDIAPIPE_TASKS_CC_TEXT_TEXT_EMBEDDER_TEXT_EMBEDDER_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "mediapipe/tasks/cc/components/processors/embedder_options.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/core/base_task_api.h"

namespace mediapipe::tasks::text::text_embedder {

// Alias the shared EmbeddingResult struct as result typo.
using TextEmbedderResult =
    ::mediapipe::tasks::components::containers::EmbeddingResult;

// Options for configuring a MediaPipe text embedder task.
struct TextEmbedderOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  tasks::core::BaseOptions base_options;

  // Options for configuring the embedder behavior, such as L2-normalization or
  // scalar-quantization.
  components::processors::EmbedderOptions embedder_options;
};

// Performs embedding extraction on text.
//
// This API expects a TFLite model with TFLite Model Metadata that contains the
// mandatory (described below) input tensors and output tensors. Metadata should
// contain the input process unit for the model's Tokenizer as well as input /
// output tensor metadata.
//
// TODO: Support Universal Sentence Encoder.
// Input tensors:
//   (kTfLiteInt32)
//    - 3 input tensors of size `[batch_size x bert_max_seq_len]` with names
//    "ids", "mask", and "segment_ids" representing the input ids, mask ids, and
//    segment ids respectively
//    - or 1 input tensor of size `[batch_size x max_seq_len]` representing the
//      input ids
//
// At least one output tensor with:
//   (kTfLiteFloat32)
//    - `N` components corresponding to the `N` dimensions of the returned
//      feature vector for this output layer.
//    - Either 2 or 4 dimensions, i.e. `[1 x N]` or `[1 x 1 x 1 x N]`.
class TextEmbedder : core::BaseTaskApi {
 public:
  using BaseTaskApi::BaseTaskApi;

  // Creates a TextEmbedder from the provided `options`. A non-default
  // OpResolver can be specified in the BaseOptions in order to support custom
  // Ops or specify a subset of built-in Ops.
  static absl::StatusOr<std::unique_ptr<TextEmbedder>> Create(
      std::unique_ptr<TextEmbedderOptions> options);

  // Performs embedding extraction on the input `text`.
  absl::StatusOr<TextEmbedderResult> Embed(absl::string_view text);

  // Shuts down the TextEmbedder when all the work is done.
  absl::Status Close() { return runner_->Close(); }

  // Utility function to compute cosine similarity [1] between two embeddings.
  // May return an InvalidArgumentError if e.g. the embeddings are of different
  // types (quantized vs. float), have different sizes, or have a an L2-norm of
  // 0.
  //
  // [1]: https://en.wikipedia.org/wiki/Cosine_similarity
  static absl::StatusOr<double> CosineSimilarity(
      const components::containers::Embedding& u,
      const components::containers::Embedding& v);
};

}  // namespace mediapipe::tasks::text::text_embedder

#endif  // MEDIAPIPE_TASKS_CC_TEXT_TEXT_EMBEDDER_TEXT_EMBEDDER_H_
