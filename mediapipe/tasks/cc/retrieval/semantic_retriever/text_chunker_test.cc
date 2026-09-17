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

#include "mediapipe/framework/port/gtest.h"  // NOLINT
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::tasks::retrieval {
namespace {

TEST(DefaultTextChunkerTest, TestCharacterModeChunking) {
  DefaultTextChunker chunker(/*chunk_size=*/4, /*chunk_overlap=*/2,
                             ChunkingMode::kCharacter);
  MP_ASSERT_OK_AND_ASSIGN(std::vector<std::string> chunks,
                          chunker.Chunk("abcdefgh"));
  // Character sliding window with size 4 and overlap 2 should produce:
  // "abcd", "cdef", "efgh"
  ASSERT_EQ(chunks.size(), 3);
  EXPECT_EQ(chunks[0], "abcd");
  EXPECT_EQ(chunks[1], "cdef");
  EXPECT_EQ(chunks[2], "efgh");
}

TEST(DefaultTextChunkerTest, TestWordModeChunking) {
  DefaultTextChunker chunker(/*chunk_size=*/2, /*chunk_overlap=*/0,
                             ChunkingMode::kWord);
  MP_ASSERT_OK_AND_ASSIGN(std::vector<std::string> chunks,
                          chunker.Chunk("apple banana cherry date"));
  ASSERT_EQ(chunks.size(), 2);
  EXPECT_EQ(chunks[0], "apple banana");
  EXPECT_EQ(chunks[1], "cherry date");
}

TEST(DefaultTextChunkerTest, TestEmptyText) {
  DefaultTextChunker chunker(/*chunk_size=*/4, /*chunk_overlap=*/2,
                             ChunkingMode::kCharacter);
  MP_ASSERT_OK_AND_ASSIGN(std::vector<std::string> chunks, chunker.Chunk(""));
  EXPECT_TRUE(chunks.empty());
}

}  // namespace
}  // namespace mediapipe::tasks::retrieval
