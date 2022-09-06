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

#include "mediapipe/tasks/cc/components/tokenizers/sentencepiece_tokenizer.h"

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/cc/core/utils.h"

namespace mediapipe {
namespace tasks {
namespace tokenizer {

using ::mediapipe::tasks::core::LoadBinaryContent;
using ::testing::ElementsAre;

namespace {
constexpr char kTestSPModelPath[] =
    "mediapipe/tasks/testdata/text/30k-clean.model";
}  // namespace

TEST(SentencePieceTokenizerTest, TestTokenize) {
  auto tokenizer = absl::make_unique<SentencePieceTokenizer>(kTestSPModelPath);
  auto results = tokenizer->Tokenize("good morning, i'm your teacher.\n");
  EXPECT_THAT(results.subwords, ElementsAre("▁good", "▁morning", ",", "▁i", "'",
                                            "m", "▁your", "▁teacher", "."));
}

TEST(SentencePieceTokenizerTest, TestTokenizeFromFileBuffer) {
  std::string buffer = LoadBinaryContent(kTestSPModelPath);
  auto tokenizer =
      absl::make_unique<SentencePieceTokenizer>(buffer.data(), buffer.size());
  EXPECT_THAT(tokenizer->Tokenize("good morning, i'm your teacher.\n").subwords,
              ElementsAre("▁good", "▁morning", ",", "▁i", "'", "m", "▁your",
                          "▁teacher", "."));
}

TEST(SentencePieceTokenizerTest, TestLookupId) {
  auto tokenizer = absl::make_unique<SentencePieceTokenizer>(kTestSPModelPath);
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
  auto tokenizer = absl::make_unique<SentencePieceTokenizer>(kTestSPModelPath);
  std::vector<int> ids = {254, 959, 15, 31, 22, 79, 154, 2197, 9};
  std::vector<std::string> subwords = {"▁good", "▁morning", ",", "▁i", "'", "m",
                                       "▁your", "▁teacher", "."};
  absl::string_view result;
  for (int i = 0; i < ids.size(); i++) {
    tokenizer->LookupWord(ids[i], &result);
    ASSERT_EQ(result, subwords[i]);
  }
}

}  // namespace tokenizer
}  // namespace tasks
}  // namespace mediapipe
