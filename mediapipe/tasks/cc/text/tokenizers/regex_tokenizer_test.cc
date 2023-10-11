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

#include "mediapipe/tasks/cc/text/tokenizers/regex_tokenizer.h"

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
constexpr char kTestRegexVocabPath[] =
    "mediapipe/tasks/testdata/text/"
    "vocab_for_regex_tokenizer.txt";

constexpr char kTestRegexEmptyVocabPath[] =
    "mediapipe/tasks/testdata/text/"
    "empty_vocab_for_regex_tokenizer.txt";

constexpr char kRegex[] = "[^\\w\\']+";

std::unique_ptr<RegexTokenizer> CreateRegexTokenizer(
    const std::string& regexp_pattern, const std::string& vocab_path) {
  std::string buffer = LoadBinaryContent(vocab_path.c_str());
  return absl::make_unique<RegexTokenizer>(regexp_pattern, buffer.data(),
                                           buffer.size());
}

TEST(RegexTokenizerTest, TestTokenize) {
  auto tokenizer = CreateRegexTokenizer(kRegex, kTestRegexVocabPath);
  auto results = tokenizer->Tokenize("good    morning, i'm your teacher.\n");
  EXPECT_THAT(results.subwords,
              ElementsAre("good", "morning", "i'm", "your", "teacher"));
}

TEST(RegexTokenizerTest, TestTokenizeFromFileBuffer) {
  std::string buffer = LoadBinaryContent(kTestRegexVocabPath);
  auto tokenizer = CreateRegexTokenizer(kRegex, kTestRegexVocabPath);
  auto results = tokenizer->Tokenize("good    morning, i'm your teacher.\n");
  EXPECT_THAT(results.subwords,
              ElementsAre("good", "morning", "i'm", "your", "teacher"));
}

TEST(RegexTokenizerTest, TestLookupId) {
  std::string buffer = LoadBinaryContent(kTestRegexVocabPath);
  auto tokenizer = CreateRegexTokenizer(kRegex, kTestRegexVocabPath);
  std::vector<std::string> subwords = {"good", "morning", "i'm", "your",
                                       "teacher"};
  std::vector<int> true_ids = {52, 1972, 146, 129, 1750};
  int id;
  for (int i = 0; i < subwords.size(); i++) {
    ASSERT_TRUE(tokenizer->LookupId(subwords[i], &id));
    ASSERT_EQ(id, true_ids[i]);
  }
}

TEST(RegexTokenizerTest, TestLookupWord) {
  std::string buffer = LoadBinaryContent(kTestRegexVocabPath);
  auto tokenizer = CreateRegexTokenizer(kRegex, kTestRegexVocabPath);
  std::vector<int> ids = {52, 1972, 146, 129, 1750};
  std::vector<std::string> subwords = {"good", "morning", "i'm", "your",
                                       "teacher"};
  absl::string_view result;
  for (int i = 0; i < ids.size(); i++) {
    ASSERT_TRUE(tokenizer->LookupWord(ids[i], &result));
    ASSERT_EQ(result, subwords[i]);
  }
}

TEST(RegexTokenizerTest, TestGetSpecialTokens) {
  // The vocab the following tokens:
  // <PAD> 0
  // <START> 1
  // <UNKNOWN> 2
  std::string buffer = LoadBinaryContent(kTestRegexVocabPath);
  auto tokenizer = CreateRegexTokenizer(kRegex, kTestRegexVocabPath);

  int start_token;
  ASSERT_TRUE(tokenizer->GetStartToken(&start_token));
  ASSERT_EQ(start_token, 1);

  int pad_token;
  ASSERT_TRUE(tokenizer->GetPadToken(&pad_token));
  ASSERT_EQ(pad_token, 0);

  int unknown_token;
  ASSERT_TRUE(tokenizer->GetUnknownToken(&unknown_token));
  ASSERT_EQ(unknown_token, 2);
}

TEST(RegexTokenizerTest, TestGetSpecialTokensFailure) {
  auto tokenizer =
      absl::make_unique<RegexTokenizer>(kRegex, kTestRegexEmptyVocabPath);

  int start_token;
  ASSERT_FALSE(tokenizer->GetStartToken(&start_token));

  int pad_token;
  ASSERT_FALSE(tokenizer->GetPadToken(&pad_token));

  int unknown_token;
  ASSERT_FALSE(tokenizer->GetUnknownToken(&unknown_token));
}

}  // namespace

}  // namespace tokenizers
}  // namespace text
}  // namespace tasks
}  // namespace mediapipe
