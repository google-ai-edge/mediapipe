/* Copyright 2023 The MediaPipe Authors.

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

#include "mediapipe/tasks/cc/text/language_detector/custom_ops/utils/ngram_hash_ops_utils.h"

#include <string>

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe::tasks::text::language_detector::custom_ops {

namespace {

using ::testing::Values;

std::string ReconstructStringFromTokens(TokenizedOutput output) {
  std::string reconstructed_str;
  for (int i = 0; i < output.tokens.size(); i++) {
    reconstructed_str.append(
        output.str.c_str() + output.tokens[i].first,
        output.str.c_str() + output.tokens[i].first + output.tokens[i].second);
  }
  return reconstructed_str;
}

struct TokenizeTestParams {
  std::string input_str;
  size_t max_tokens;
  bool exclude_nonalphaspace_tokens;
  std::string expected_output_str;
};

class TokenizeParameterizedTest
    : public ::testing::Test,
      public testing::WithParamInterface<TokenizeTestParams> {};

TEST_P(TokenizeParameterizedTest, Tokenize) {
  // Checks that the Tokenize method returns the expected value.
  const TokenizeTestParams params = TokenizeParameterizedTest::GetParam();
  const TokenizedOutput output = Tokenize(
      /*input_str=*/params.input_str.c_str(),
      /*len=*/params.input_str.size(),
      /*max_tokens=*/params.max_tokens,
      /*exclude_nonalphaspace_tokens=*/params.exclude_nonalphaspace_tokens);

  // The output string should have the necessary prefixes, and the "!" token
  // should have been replaced with a " ".
  EXPECT_EQ(output.str, params.expected_output_str);
  EXPECT_EQ(ReconstructStringFromTokens(output), params.expected_output_str);
}

INSTANTIATE_TEST_SUITE_P(
    TokenizeParameterizedTests, TokenizeParameterizedTest,
    Values(
        // Test including non-alphanumeric characters.
        TokenizeTestParams({/*input_str=*/"hi!", /*max_tokens=*/100,
                            /*exclude_alphanonspace=*/false,
                            /*expected_output_str=*/"^hi!$"}),
        // Test not including non-alphanumeric characters.
        TokenizeTestParams({/*input_str=*/"hi!", /*max_tokens=*/100,
                            /*exclude_alphanonspace=*/true,
                            /*expected_output_str=*/"^hi $"}),
        // Test with a maximum of 3 tokens.
        TokenizeTestParams({/*input_str=*/"hi!", /*max_tokens=*/3,
                            /*exclude_alphanonspace=*/true,
                            /*expected_output_str=*/"^h$"}),
        // Test with non-latin characters.
        TokenizeTestParams({/*input_str=*/"ありがと", /*max_tokens=*/100,
                            /*exclude_alphanonspace=*/true,
                            /*expected_output_str=*/"^ありがと$"})));

TEST(LowercaseUnicodeTest, TestLowercaseUnicode) {
  {
    // Check that the method is a no-op when the string is lowercase.
    std::string input_str = "hello";
    std::string output_str;
    LowercaseUnicodeStr(
        /*input_str=*/input_str.c_str(),
        /*len=*/input_str.size(),
        /*output_str=*/&output_str);

    EXPECT_EQ(output_str, "hello");
  }
  {
    // Check that the method has uppercase characters.
    std::string input_str = "hElLo";
    std::string output_str;
    LowercaseUnicodeStr(
        /*input_str=*/input_str.c_str(),
        /*len=*/input_str.size(),
        /*output_str=*/&output_str);

    EXPECT_EQ(output_str, "hello");
  }
  {
    // Check that the method works with non-latin scripts.
    // Cyrillic has the concept of cases, so it should change the input.
    std::string input_str = "БЙп";
    std::string output_str;
    LowercaseUnicodeStr(
        /*input_str=*/input_str.c_str(),
        /*len=*/input_str.size(),
        /*output_str=*/&output_str);

    EXPECT_EQ(output_str, "бйп");
  }
  {
    // Check that the method works with non-latin scripts.
    // Japanese doesn't have the concept of cases, so it should not change.
    std::string input_str = "ありがと";
    std::string output_str;
    LowercaseUnicodeStr(
        /*input_str=*/input_str.c_str(),
        /*len=*/input_str.size(),
        /*output_str=*/&output_str);

    EXPECT_EQ(output_str, "ありがと");
  }
}

}  // namespace
}  // namespace mediapipe::tasks::text::language_detector::custom_ops
