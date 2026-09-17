/* Copyright 2026 The MediaPipe Authors. All Rights Reserved.

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

#ifndef MEDIAPIPE_TASKS_CC_RETRIEVAL_UNIVERSAL_EMBEDDER_UNIVERSAL_EMBEDDER_H_
#define MEDIAPIPE_TASKS_CC_RETRIEVAL_UNIVERSAL_EMBEDDER_UNIVERSAL_EMBEDDER_H_

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/core/embedding_provider.h"
#include "mediapipe/tasks/cc/core/logging/tasks_logger.h"
#include "runtime/engine/embedding_engine.h"  // from @litert_lm
#include "runtime/util/memory_mapped_file.h"  // from @litert_lm

namespace mediapipe::tasks::retrieval::universal_embedder {

// Activation data type for model execution.
enum class ActivationDataType {
  FLOAT32 = 1,
  FLOAT16 = 2,
  INT16 = 3,
  INT8 = 4,
};

// Options for configuring a MediaPipe Universal Embedder task.
struct UniversalEmbedderOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, etc.
  tasks::core::BaseOptions base_options;

  // Whether to L2-normalize the output embedding vector.
  bool l2_normalize = true;

  // Optional maximum sequence length (in tokens) for text encoder signatures.
  std::optional<int> max_input_length;

  // Optional vision tokens per image for vision encoder signatures.
  std::optional<int> vision_tokens_per_image;

  // Optional activation data type for model execution.
  std::optional<ActivationDataType> activation_data_type;

  // Optional directory path for storing compiled model cache artifacts.
  std::optional<std::string> cache_dir;
};

// Performs multimodal embedding extraction on text, image, audio, or custom
// content blocks.
//
// This C++ API wraps the native LiteRT-LM EmbeddingEngine directly,
// providing real on-device multimodal embeddings.
class UniversalEmbedder {
 public:
  // Creates a UniversalEmbedder from the provided options.
  static absl::StatusOr<std::unique_ptr<UniversalEmbedder>> Create(
      std::unique_ptr<UniversalEmbedderOptions> options);

  ~UniversalEmbedder();

  // Performs embedding extraction on the input text.
  absl::StatusOr<components::containers::EmbeddingResult> EmbedText(
      absl::string_view text);

  // Performs embedding extraction on the input image bytes (e.g. JPEG, PNG
  // contents).
  absl::StatusOr<components::containers::EmbeddingResult> EmbedImage(
      absl::string_view image_bytes);

  // Performs embedding extraction on the input mediapipe Image object.
  absl::StatusOr<components::containers::EmbeddingResult> EmbedImage(
      const mediapipe::Image& image);

  // Performs embedding extraction on the input audio float samples.
  absl::StatusOr<components::containers::EmbeddingResult> EmbedAudio(
      const std::vector<float>& audio_data);

  // Parts representing the contents of a multi-modal block.
  struct TextPart {
    std::string text;
  };

  struct ImagePart {
    std::string image_bytes;
  };

  struct AudioPart {
    std::vector<float> audio_data;
  };

  using Part = std::variant<TextPart, ImagePart, AudioPart>;

  // Performs embedding extraction on multi-modal parts of a content block.
  absl::StatusOr<components::containers::EmbeddingResult> EmbedContent(
      const std::vector<Part>& content);

  // Returns an EmbeddingProvider using this UniversalEmbedder as the backend.
  std::unique_ptr<core::EmbeddingProvider> GetProvider();

  // Shuts down the UniversalEmbedder when all the work is done.
  absl::Status Close();

  // Utility function to compute cosine similarity between two embeddings.
  static absl::StatusOr<double> CosineSimilarity(
      const components::containers::Embedding& u,
      const components::containers::Embedding& v);

 private:
  explicit UniversalEmbedder(
      std::shared_ptr<::litert::lm::MemoryMappedFile> shared_mmap,
      std::unique_ptr<::litert::lm::EmbeddingEngine> engine, bool l2_normalize,
      std::unique_ptr<tasks::core::logging::TasksLogger> tasks_logger);

  absl::StatusOr<components::containers::EmbeddingResult> ExecuteInference(
      const std::vector<::litert::lm::InputData>& contents,
      absl::string_view error_message);

  std::shared_ptr<::litert::lm::MemoryMappedFile> shared_mmap_;
  std::unique_ptr<::litert::lm::EmbeddingEngine> engine_;
  bool l2_normalize_;

  std::unique_ptr<tasks::core::logging::TasksLogger> tasks_logger_;
  uint64_t tasks_logger_timestamp_ = 0;
};

}  // namespace mediapipe::tasks::retrieval::universal_embedder

#endif  // MEDIAPIPE_TASKS_CC_RETRIEVAL_UNIVERSAL_EMBEDDER_UNIVERSAL_EMBEDDER_H_
