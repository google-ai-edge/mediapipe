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

}  // namespace
}  // namespace mediapipe::tasks::text::utils
