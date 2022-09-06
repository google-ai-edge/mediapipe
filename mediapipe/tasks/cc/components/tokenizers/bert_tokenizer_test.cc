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

#include "mediapipe/tasks/cc/components/tokenizers/bert_tokenizer.h"

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/cc/core/utils.h"

namespace mediapipe {
namespace tasks {
namespace tokenizer {

using ::mediapipe::tasks::core::LoadBinaryContent;
using ::testing::ElementsAre;

namespace {
constexpr char kTestVocabPath[] =
    "mediapipe/tasks/testdata/text/mobilebert_vocab.txt";
}  // namespace

void AssertTokenizerResults(std::unique_ptr<BertTokenizer> tokenizer) {
  auto results = tokenizer->TokenizeWordpiece("i'm question");

  EXPECT_THAT(results.subwords, ElementsAre("i", "'", "m", "question"));
  EXPECT_THAT(results.wp_begin_offset, ElementsAre(0, 1, 2, 4));
  EXPECT_THAT(results.wp_end_offset, ElementsAre(1, 2, 3, 12));
  EXPECT_THAT(results.row_lengths, ElementsAre(1, 1, 1, 1));
}

TEST(TokenizerTest, TestTokenizerCreationFromBuffer) {
  std::string buffer = LoadBinaryContent(kTestVocabPath);
  auto tokenizer =
      absl::make_unique<BertTokenizer>(buffer.data(), buffer.size());
  AssertTokenizerResults(std::move(tokenizer));
}

TEST(TokenizerTest, TestTokenizerCreationFromFile) {
  auto tokenizer = absl::make_unique<BertTokenizer>(kTestVocabPath);

  AssertTokenizerResults(std::move(tokenizer));
}

TEST(TokenizerTest, TestTokenizerCreationFromVector) {
  std::vector<std::string> vocab;
  vocab.emplace_back("i");
  vocab.emplace_back("'");
  vocab.emplace_back("m");
  vocab.emplace_back("question");
  auto tokenizer = absl::make_unique<BertTokenizer>(vocab);

  AssertTokenizerResults(std::move(tokenizer));
}

TEST(TokenizerTest, TestTokenizerMultipleRows) {
  auto tokenizer = absl::make_unique<BertTokenizer>(kTestVocabPath);

  auto results = tokenizer->TokenizeWordpiece("i'm questionansweraskask");

  EXPECT_THAT(results.subwords, ElementsAre("i", "'", "m", "question", "##ans",
                                            "##wer", "##ask", "##ask"));
  EXPECT_THAT(results.wp_begin_offset, ElementsAre(0, 1, 2, 4, 12, 15, 18, 21));
  EXPECT_THAT(results.wp_end_offset, ElementsAre(1, 2, 3, 12, 15, 18, 21, 24));
  EXPECT_THAT(results.row_lengths, ElementsAre(1, 1, 1, 5));
}

TEST(TokenizerTest, TestTokenizerUnknownTokens) {
  std::vector<std::string> vocab;
  vocab.emplace_back("i");
  vocab.emplace_back("'");
  vocab.emplace_back("m");
  vocab.emplace_back("question");
  auto tokenizer = absl::make_unique<BertTokenizer>(vocab);

  auto results = tokenizer->TokenizeWordpiece("i'm questionansweraskask");

  EXPECT_THAT(results.subwords,
              ElementsAre("i", "'", "m", kDefaultUnknownToken));
  EXPECT_THAT(results.wp_begin_offset, ElementsAre(0, 1, 2, 4));
  EXPECT_THAT(results.wp_end_offset, ElementsAre(1, 2, 3, 24));
  EXPECT_THAT(results.row_lengths, ElementsAre(1, 1, 1, 1));
}

TEST(TokenizerTest, TestLookupId) {
  std::vector<std::string> vocab;
  vocab.emplace_back("i");
  vocab.emplace_back("'");
  vocab.emplace_back("m");
  vocab.emplace_back("question");
  auto tokenizer = absl::make_unique<BertTokenizer>(vocab);

  int i;
  ASSERT_FALSE(tokenizer->LookupId("iDontExist", &i));

  ASSERT_TRUE(tokenizer->LookupId("i", &i));
  ASSERT_EQ(i, 0);
  ASSERT_TRUE(tokenizer->LookupId("'", &i));
  ASSERT_EQ(i, 1);
  ASSERT_TRUE(tokenizer->LookupId("m", &i));
  ASSERT_EQ(i, 2);
  ASSERT_TRUE(tokenizer->LookupId("question", &i));
  ASSERT_EQ(i, 3);
}

TEST(TokenizerTest, TestLookupWord) {
  std::vector<std::string> vocab;
  vocab.emplace_back("i");
  vocab.emplace_back("'");
  vocab.emplace_back("m");
  vocab.emplace_back("question");
  auto tokenizer = absl::make_unique<BertTokenizer>(vocab);

  absl::string_view result;
  ASSERT_FALSE(tokenizer->LookupWord(6, &result));

  ASSERT_TRUE(tokenizer->LookupWord(0, &result));
  ASSERT_EQ(result, "i");
  ASSERT_TRUE(tokenizer->LookupWord(1, &result));
  ASSERT_EQ(result, "'");
  ASSERT_TRUE(tokenizer->LookupWord(2, &result));
  ASSERT_EQ(result, "m");
  ASSERT_TRUE(tokenizer->LookupWord(3, &result));
  ASSERT_EQ(result, "question");
}

TEST(TokenizerTest, TestContains) {
  std::vector<std::string> vocab;
  vocab.emplace_back("i");
  vocab.emplace_back("'");
  vocab.emplace_back("m");
  vocab.emplace_back("question");
  auto tokenizer = absl::make_unique<BertTokenizer>(vocab);

  bool result;
  tokenizer->Contains("iDontExist", &result);
  ASSERT_FALSE(result);

  tokenizer->Contains("i", &result);
  ASSERT_TRUE(result);
  tokenizer->Contains("'", &result);
  ASSERT_TRUE(result);
  tokenizer->Contains("m", &result);
  ASSERT_TRUE(result);
  tokenizer->Contains("question", &result);
  ASSERT_TRUE(result);
}

TEST(TokenizerTest, TestLVocabularySize) {
  std::vector<std::string> vocab;
  vocab.emplace_back("i");
  vocab.emplace_back("'");
  vocab.emplace_back("m");
  vocab.emplace_back("question");
  auto tokenizer = absl::make_unique<BertTokenizer>(vocab);

  ASSERT_EQ(tokenizer->VocabularySize(), 4);
}

}  // namespace tokenizer
}  // namespace tasks
}  // namespace mediapipe
