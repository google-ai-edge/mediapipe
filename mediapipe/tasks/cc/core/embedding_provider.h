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
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"

namespace mediapipe::tasks::core {

// Parts representing the contents of a multi-modal block.
struct TextPart {
  std::string text;
};

// Represents an image part of a multi-modal content block.
struct ImagePart {
  // Path or URI to the image file. If image_bytes is empty, the file is loaded
  // from this path on-demand during database insertion.
  std::string file_path;

  // Raw encoded image bytes (e.g., JPEG, PNG file content). If empty, the image
  // is loaded from the file_path instead.
  std::string image_bytes;

  ImagePart() = default;

  // Constructor for raw in-memory image bytes.
  explicit ImagePart(std::string bytes)
      : file_path(""), image_bytes(std::move(bytes)) {}

  // Constructor for specifying both a file path and optional in-memory image
  // bytes.
  ImagePart(std::string path, std::string bytes)
      : file_path(std::move(path)), image_bytes(std::move(bytes)) {}
};

// Represents an audio part of a multi-modal content block.
struct AudioPart {
  // Path or URI to the 16-bit PCM mono WAV audio file. If audio_data is empty,
  // the file is loaded and decoded from this path on-demand during database
  // insertion.
  std::string file_path;

  // Raw pre-decoded mono PCM float audio samples. If empty, the audio is loaded
  // and decoded from the file_path instead.
  std::vector<float> audio_data;

  AudioPart() = default;

  // Constructor for raw pre-decoded mono PCM float audio samples.
  explicit AudioPart(std::vector<float> data)
      : file_path(""), audio_data(std::move(data)) {}

  // Constructor for specifying both a file path and optional in-memory PCM
  // audio samples.
  AudioPart(std::string path, std::vector<float> data)
      : file_path(std::move(path)), audio_data(std::move(data)) {}
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
