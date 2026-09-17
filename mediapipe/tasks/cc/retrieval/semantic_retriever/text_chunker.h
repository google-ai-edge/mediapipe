/* Copyright 2026 The MediaPipe Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
you may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MEDIAPIPE_TASKS_CC_RETRIEVAL_SEMANTIC_RETRIEVER_TEXT_CHUNKER_H_
#define MEDIAPIPE_TASKS_CC_RETRIEVAL_SEMANTIC_RETRIEVER_TEXT_CHUNKER_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace mediapipe::tasks::retrieval {

enum class ChunkingMode {
  kCharacter = 0,
  kWord = 1,
};

class TextChunker {
 public:
  virtual ~TextChunker() = default;
  virtual absl::StatusOr<std::vector<std::string>> Chunk(
      absl::string_view text) = 0;
};

class DefaultTextChunker : public TextChunker {
 public:
  DefaultTextChunker(int chunk_size, int chunk_overlap, ChunkingMode mode);

  absl::StatusOr<std::vector<std::string>> Chunk(
      absl::string_view text) override;

 private:
  int chunk_size_;
  int chunk_overlap_;
  ChunkingMode mode_;
};

}  // namespace mediapipe::tasks::retrieval

#endif  // MEDIAPIPE_TASKS_CC_RETRIEVAL_SEMANTIC_RETRIEVER_TEXT_CHUNKER_H_
