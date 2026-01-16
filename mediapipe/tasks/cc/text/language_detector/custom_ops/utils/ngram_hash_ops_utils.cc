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
#include <utility>
#include <vector>

#include "mediapipe/tasks/cc/text/language_detector/custom_ops/utils/utf/utf.h"

namespace mediapipe::tasks::text::language_detector::custom_ops {

TokenizedOutput Tokenize(const char* input_str, int len, int max_tokens,
                         bool exclude_nonalphaspace_tokens) {
  const std::string kPrefix = "^";
  const std::string kSuffix = "$";
  const std::string kReplacementToken = " ";

  TokenizedOutput output;

  size_t token_start = 0;
  output.str.reserve(len + 2);
  output.tokens.reserve(len + 2);

  output.str.append(kPrefix);
  output.tokens.push_back(std::make_pair(token_start, kPrefix.size()));
  token_start += kPrefix.size();

  Rune token;
  for (int i = 0; i < len && output.tokens.size() + 1 < max_tokens;) {
    // Use the standard UTF-8 library to find the next token.
    size_t bytes_read = utf_charntorune(&token, input_str + i, len - i);

    // Stop processing, if we can't read any more tokens, or we have reached
    // maximum allowed tokens, allocating one token for the suffix.
    if (bytes_read == 0) {
      break;
    }

    // If `exclude_nonalphaspace_tokens` is set to true, and the token is not
    // alphanumeric, replace it with a replacement token.
    if (exclude_nonalphaspace_tokens && !utf_isalpharune(token)) {
      output.str.append(kReplacementToken);
      output.tokens.push_back(
          std::make_pair(token_start, kReplacementToken.size()));
      token_start += kReplacementToken.size();
      i += bytes_read;
      continue;
    }

    // Append the token in the output string, and note its position and the
    // number of bytes that token consumed.
    output.str.append(input_str + i, bytes_read);
    output.tokens.push_back(std::make_pair(token_start, bytes_read));
    token_start += bytes_read;
    i += bytes_read;
  }
  output.str.append(kSuffix);
  output.tokens.push_back(std::make_pair(token_start, kSuffix.size()));
  token_start += kSuffix.size();

  return output;
}

void LowercaseUnicodeStr(const char* input_str, int len,
                         std::string* output_str) {
  for (int i = 0; i < len;) {
    Rune token;

    // Tokenize the given string, and get the appropriate lowercase token.
    size_t bytes_read = utf_charntorune(&token, input_str + i, len - i);
    token = utf_isalpharune(token) ? utf_tolowerrune(token) : token;

    // Write back the token to the output string.
    char token_buf[UTFmax];
    size_t bytes_to_write = utf_runetochar(token_buf, &token);
    output_str->append(token_buf, bytes_to_write);

    i += bytes_read;
  }
}

}  // namespace mediapipe::tasks::text::language_detector::custom_ops
