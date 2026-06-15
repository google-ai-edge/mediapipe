/* Copyright 2022 The MediaPipe Authors.

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
#include <optional>
#include <string>

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

// The embedding task type, used to format input text.
enum class EmbeddingType {
  // Embed text for retrieval query.
  RETRIEVAL_QUERY = 1,
  // Embed text for retrieval document.
  RETRIEVAL_DOCUMENT = 2,
  // Embed text for semantic similarity.
  SEMANTIC_SIMILARITY = 3,
  // Embed text for classification.
  CLASSIFICATION = 4,
  // Embed text for clustering.
  CLUSTERING = 5,
  // Embed text for question answering.
  QUESTION_ANSWERING = 6,
  // Embed text for fact verification.
  FACT_CHECKING = 7,
  // Embed text for code retrieval.
  CODE_RETRIEVAL = 8,
};

// The role of the text in the context of the embedding task.
enum class TextRole {
  // The embedding is extracted to perform a query.
  kQuery = 1,
  // The embedding is extracted to store a document.
  kDocument = 2
};

// Encapsulates formatting instructions for models that require it (like Gecko).
struct TextFormatContext {
  EmbeddingType task_type = EmbeddingType::RETRIEVAL_QUERY;
  std::optional<std::string> title = std::nullopt;
  TextRole role = TextRole::kQuery;
};

// Performs embedding extraction on text.
//
// This API expects a TFLite model with TFLite Model Metadata that contains the
// mandatory (described below) input tensors and output tensors.
//
// 1. BERT-based model
//    - 3 input tensors of size `[batch_size x bert_max_seq_len]` and type
//      kTfLiteInt32 with names "ids", "mask", and "segment_ids" representing
//      the input ids, mask ids, and segment ids respectively
//    - at least one output tensor (all of type kTfLiteFloat32) with `N`
//      components corresponding to the `N` dimensions of the returned
//      feature vector for this output layer and with either 2 or 4 dimensions,
//      i.e. `[1 x N]` or `[1 x 1 x 1 x N]`
//    - input process units for a BertTokenizer or SentencePieceTokenizer
// 2. Regex-based model
//    - 1 input tensor of size `[batch_size x max_seq_len]` and type
//      kTfLiteInt32 representing the input ids
//    - at least one output tensor (all of type kTfLiteFloat32) with `N`
//      components corresponding to the `N` dimensions of the returned
//      feature vector for this output layer and with either 2 or 4 dimensions,
//      i.e. `[1 x N]` or `[1 x 1 x 1 x N]`
//    - input process units for a RegexTokenizer
// 3. UniversalSentenceEncoder-based model
//    - 3 input tensors with names "inp_text", "res_context" and "res_text"
//    - 2 output tensors with names "query_encoding" and "response_encoding" of
//      type kTfLiteFloat32. The "query_encoding" is filtered and only the other
//      output tensor is used for the embedding.
// 4. Gecko-based model (e.g. gecko-1b-en-cpu)
//    - 1 input tensor of size `[1 x max_seq_len]` and type kTfLiteInt32
//      representing the input ids from SentencePiece Tokenizer with BOS/EOS
//      and padding.
//    - Input text formatting follows Gecko's instructions. The exact
//      format depends on EmbeddingType provided in `Embed` method.
//    - 1 output tensor of type kTfLiteFloat32 with `N` components corresponding
//      to the `N` dimensions of the returned feature vector and with shape
//      `[1 x N]`.
//    - input process units for a SentencePieceTokenizer in model metadata.
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

  // Performs embedding extraction on the input `text` with formatting options.
  absl::StatusOr<TextEmbedderResult> Embed(
      absl::string_view text, const TextFormatContext& format_context);

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
