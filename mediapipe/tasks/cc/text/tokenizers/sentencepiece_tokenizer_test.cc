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

#include "mediapipe/tasks/cc/text/tokenizers/sentencepiece_tokenizer.h"

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/cc/core/utils.h"

namespace mediapipe {
namespace tasks {
namespace text {
namespace tokenizers {

using ::mediapipe::tasks::core::LoadBinaryContent;
using ::testing::ElementsAre;

namespace {
constexpr char kTestSPModelPath[] =
    "mediapipe/tasks/testdata/text/30k-clean.model";
}  // namespace

std::unique_ptr<SentencePieceTokenizer> CreateSentencePieceTokenizer(
    absl::string_view model_path) {
  // We are using `LoadBinaryContent()` instead of loading the model directly
  // via `SentencePieceTokenizer` so that the file can be located on Windows
  std::string buffer = LoadBinaryContent(kTestSPModelPath);
  return absl::make_unique<SentencePieceTokenizer>(buffer.data(),
                                                   buffer.size());
}

TEST(SentencePieceTokenizerTest, TestTokenize) {
  auto tokenizer = CreateSentencePieceTokenizer(kTestSPModelPath);
  auto results = tokenizer->Tokenize("good morning, i'm your teacher.\n");
  EXPECT_THAT(results.subwords, ElementsAre("▁good", "▁morning", ",", "▁i", "'",
                                            "m", "▁your", "▁teacher", "."));
}

TEST(SentencePieceTokenizerTest, TestTokenizeFromFileBuffer) {
  auto tokenizer = CreateSentencePieceTokenizer(kTestSPModelPath);
  EXPECT_THAT(tokenizer->Tokenize("good morning, i'm your teacher.\n").subwords,
              ElementsAre("▁good", "▁morning", ",", "▁i", "'", "m", "▁your",
                          "▁teacher", "."));
}

TEST(SentencePieceTokenizerTest, TestLookupId) {
  auto tokenizer = CreateSentencePieceTokenizer(kTestSPModelPath);
  std::vector<std::string> subwords = {"▁good", "▁morning", ",", "▁i", "'", "m",
                                       "▁your", "▁teacher", "."};
  std::vector<int> true_ids = {254, 959, 15, 31, 22, 79, 154, 2197, 9};
  int id;
  for (int i = 0; i < subwords.size(); i++) {
    tokenizer->LookupId(subwords[i], &id);
    ASSERT_EQ(id, true_ids[i]);
  }
}

TEST(SentencePieceTokenizerTest, TestLookupWord) {
  auto tokenizer = CreateSentencePieceTokenizer(kTestSPModelPath);
  std::vector<int> ids = {254, 959, 15, 31, 22, 79, 154, 2197, 9};
  std::vector<std::string> subwords = {"▁good", "▁morning", ",", "▁i", "'", "m",
                                       "▁your", "▁teacher", "."};
  absl::string_view result;
  for (int i = 0; i < ids.size(); i++) {
    tokenizer->LookupWord(ids[i], &result);
    ASSERT_EQ(result, subwords[i]);
  }
}

}  // namespace tokenizers
}  // namespace text
}  // namespace tasks
}  // namespace mediapipe
