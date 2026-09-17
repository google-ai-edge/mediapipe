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

#include "mediapipe/tasks/cc/retrieval/semantic_retriever/text_chunker.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/tasks/cc/text/utils/genai_utils.h"

namespace mediapipe::tasks::retrieval {

DefaultTextChunker::DefaultTextChunker(int chunk_size, int chunk_overlap,
                                       ChunkingMode mode)
    : chunk_size_(chunk_size), chunk_overlap_(chunk_overlap), mode_(mode) {}

absl::StatusOr<std::vector<std::string>> DefaultTextChunker::Chunk(
    absl::string_view text) {
  if (text.empty()) {
    return std::vector<std::string>();
  }
  std::string text_str(text);

  if (mode_ == ChunkingMode::kCharacter) {
    return ::mediapipe::tasks::text::utils::ChunkTextByCountingCharacters(
        text_str, chunk_size_, chunk_overlap_);
  } else if (mode_ == ChunkingMode::kWord) {
    int word_chunk_threshold = chunk_size_;
    int word_chunk_size = chunk_size_ - chunk_overlap_;
    std::vector<::mediapipe::tasks::text::utils::TextChunk> chunks =
        ::mediapipe::tasks::text::utils::ChunkText(
            text_str, word_chunk_threshold, word_chunk_size);
    std::vector<std::string> results;
    results.reserve(chunks.size());
    for (const auto& chunk : chunks) {
      results.push_back(std::string(chunk.text));
    }
    return results;
  }

  return absl::InvalidArgumentError("Unsupported chunking mode");
}

}  // namespace mediapipe::tasks::retrieval
