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

#ifndef MEDIAPIPE_TASKS_CC_CORE_EMBEDDING_PROVIDER_H_
#define MEDIAPIPE_TASKS_CC_CORE_EMBEDDING_PROVIDER_H_

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"

namespace mediapipe::tasks::core {

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

using TaskPart = std::variant<TextPart, ImagePart, AudioPart>;

// A package-neutral interface for generating embeddings across different
// modalities using content parts. Distributed via mediapipe:tasks-core.
class EmbeddingProvider {
 public:
  virtual ~EmbeddingProvider() = default;

  // Generates a high-dimensional vector embedding for the given list of
  // multi-modal content parts.
  // Returns std::nullopt if the input content parts type is not supported
  // by this embedder.
  virtual absl::StatusOr<std::optional<std::vector<float>>> EmbedContent(
      const std::vector<TaskPart>& content) = 0;
};

}  // namespace mediapipe::tasks::core

#endif  // MEDIAPIPE_TASKS_CC_CORE_EMBEDDING_PROVIDER_H_
