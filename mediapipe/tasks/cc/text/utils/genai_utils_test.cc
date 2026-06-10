/* Copyright 2026 The MediaPipe Authors.

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

#include "mediapipe/tasks/cc/text/utils/genai_utils.h"

#include <string>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe::tasks::text::utils {
namespace {

struct HasRepeatingSuffixTestCase {
  std::string test_name;
  int min_loop_length;
  int suffix_count;
  std::string input;
  bool expected_result;
};

class HasRepeatingSuffixTest
    : public ::testing::TestWithParam<HasRepeatingSuffixTestCase> {};

TEST_P(HasRepeatingSuffixTest, ReturnsExpectedResult) {
  const HasRepeatingSuffixTestCase& test_case = GetParam();
  EXPECT_EQ(HasRepeatingSuffix(test_case.min_loop_length,
                               test_case.suffix_count, test_case.input),
            test_case.expected_result);
}

INSTANTIATE_TEST_SUITE_P(
    GenaiUtilsTest, HasRepeatingSuffixTest,
    ::testing::ValuesIn<HasRepeatingSuffixTestCase>({
        // No repeats
        {"NoRepeats", 3, 3, "abcdef", false},
        {"OnlyTwoRepeats", 3, 3, "abcabc", false},
        {"NotRepeated", 3, 3, "abcabcabd", false},
        {"TwoRepeats", 2, 2, "abcd", false},

        // 3 repeats of "abc"
        {"ThreeRepeats", 3, 3, "abcabcabc", true},
        // 3 repeats of "abc" with prefix
        {"ThreeRepeatsWithPrefix", 3, 3, "helloabcabcabc", true},

        // Min chars check
        // "abc" is 3 chars, min is 4.
        {"MinCharsCheckSmaller", 4, 3, "abcabcabc", false},
        {"MinCharsCheckExact", 3, 3, "abcabcabc", true},

        // Larger min chars
        {"LargerMinChars", 3, 2, "abcabc", true},
        // "abcabc" (6 chars) repeats 2 times.
        {"LargerMinCharsSix", 6, 2, "abcabcabcabc", true},

        // Edge cases
        {"ZeroSuffixCount", 3, 0, "abcabcabc", false},
        {"NegativeSuffixCount", 3, -1, "abcabcabc", false},
        // 1 repeat is just the suffix itself (if size >= min).
        {"OneRepeatSizeGreaterEqualMin", 3, 1, "abc", true},
        {"OneRepeatSizeLessMin", 5, 1, "abc", false},

        // 4 repeats of "abc" (still has 3 repeats)
        {"FourRepeats", 3, 3, "abcabcabcabc", true},

        // Loop detected with phase-shifted repeat and incomplete last repeat.
        {"LoopDetected", 15, 3,
         "Prefix info. I am a student. I am a student. I am a student. "
         "I am a stu",
         true},
    }),
    [](const ::testing::TestParamInfo<HasRepeatingSuffixTest::ParamType>&
           info) { return info.param.test_name; });

int WordCount(absl::string_view text) {
  int count = 0;
  bool in_word = false;
  for (char c : text) {
    if (absl::ascii_isspace(c)) {
      if (in_word) {
        ++count;
        in_word = false;
      }
    } else {
      in_word = true;
    }
  }
  if (in_word) {
    ++count;
  }
  return count;
}

TEST(GenaiUtilsTest, ChunkTextShortInput) {
  std::string input = "This is a short input.";
  auto result = ChunkText(input);
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0].text, "This is a short input.");
  EXPECT_EQ(result[0].trailing_separator, "");
}

TEST(GenaiUtilsTest, ChunkTextLongInputNoSpecialPunctuation) {
  // Construct 550 words without periods or newlines.
  std::string input;
  for (int i = 0; i < 550; ++i) {
    input += "word ";
  }
  auto result = ChunkText(input);
  // Should force split at 500 words.
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(WordCount(result[0].text), 500);
  EXPECT_EQ(result[0].trailing_separator, " ");
  EXPECT_EQ(WordCount(result[1].text), 50);
  EXPECT_EQ(result[1].trailing_separator, " ");
}

TEST(GenaiUtilsTest, ChunkTextLongInputWithPeriod) {
  // Construct 550 words with a period at 480th word.
  std::string input;
  for (int i = 0; i < 479; ++i) {
    input += "word ";
  }
  input += "word. ";  // 480th word
  for (int i = 0; i < 70; ++i) {
    input += "word ";
  }

  auto result = ChunkText(input);
  // Should split at the period (480 words).
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(WordCount(result[0].text), 480);
  EXPECT_EQ(result[0].trailing_separator, " ");
  EXPECT_EQ(WordCount(result[1].text), 70);
  EXPECT_EQ(result[1].trailing_separator, " ");
}

TEST(GenaiUtilsTest, ChunkTextLongInputWithNewline) {
  // Construct 550 words with a newline at 460th word.
  std::string input;
  for (int i = 0; i < 459; ++i) {
    input += "word ";
  }
  input += "word\n";  // 460th word ends with newline
  for (int i = 0; i < 90; ++i) {
    input += "word ";
  }

  auto result = ChunkText(input);
  // Should split at the newline (460 words).
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(WordCount(result[0].text), 460);
  EXPECT_EQ(result[0].trailing_separator, "\n");
  EXPECT_EQ(WordCount(result[1].text), 90);
  EXPECT_EQ(result[1].trailing_separator, " ");
}

TEST(GenaiUtilsTest, ChunkTextLayoutPreservation) {
  std::string input = "Hello\n\nworld!   This is  a test.   \n";
  auto result = ChunkText(input);

  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0].text, "Hello");
  EXPECT_EQ(result[0].trailing_separator, "\n\n");
  EXPECT_EQ(result[1].text, "world!   This is  a test.");
  EXPECT_EQ(result[1].trailing_separator, "   \n");

  std::string reconstructed;
  for (const auto& chunk : result) {
    absl::StrAppend(&reconstructed, chunk.text, chunk.trailing_separator);
  }
  EXPECT_EQ(reconstructed, input);
}

}  // namespace
}  // namespace mediapipe::tasks::text::utils
