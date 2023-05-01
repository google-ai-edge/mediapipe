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

#include <iostream>

#include "absl/strings/substitute.h"
#include "mediapipe/tasks/cc/text/utils/vocab_utils.h"

namespace mediapipe {
namespace tasks {
namespace text {
namespace tokenizers {

namespace {

using ::mediapipe::tasks::text::LoadVocabAndIndexFromBuffer;
using ::mediapipe::tasks::text::LoadVocabAndIndexFromFile;

constexpr char kStart[] = "<START>";
constexpr char kPad[] = "<PAD>";
constexpr char kUnknown[] = "<UNKNOWN>";

void buildIndexTokenMap(
    const absl::node_hash_map<std::string, int>& token_index_map,
    absl::node_hash_map<int, absl::string_view>* index_token_map) {
  for (const auto& token : token_index_map) {
    (*index_token_map)[token.second] = token.first;
  }
}

}  // namespace

// RE2::FindAndConsume requires the delim_re_ to have a matching group in order
// to capture the matched delimiter length. Surround the regex with a
// parenthesis to create a matching group, it's fine if the regex is already
// surrounded by parenthesis.
RegexTokenizer::RegexTokenizer(const std::string& regex_pattern,
                               const std::string& path_to_vocab)
    : delim_re_{absl::Substitute("($0)", regex_pattern)},
      token_index_map_{LoadVocabAndIndexFromFile(path_to_vocab)} {
  buildIndexTokenMap(token_index_map_, &index_token_map_);
}

RegexTokenizer::RegexTokenizer(const std::string& regex_pattern,
                               const char* vocab_buffer_data,
                               size_t vocab_buffer_size)
    : delim_re_{absl::Substitute("($0)", regex_pattern)},
      token_index_map_{
          LoadVocabAndIndexFromBuffer(vocab_buffer_data, vocab_buffer_size)} {
  buildIndexTokenMap(token_index_map_, &index_token_map_);
}

TokenizerResult RegexTokenizer::Tokenize(const std::string& input) {
  absl::string_view leftover(input.data());
  absl::string_view last_end = leftover;

  TokenizerResult result;

  // Keep looking for split points until we have reached the end of the input.
  absl::string_view extracted_delim_token;
  while (RE2::FindAndConsume(&leftover, delim_re_, &extracted_delim_token)) {
    absl::string_view token(last_end.data(),
                            extracted_delim_token.data() - last_end.data());
    bool has_non_empty_token = token.length() > 0;

    last_end = leftover;

    // Mark the end of the previous token, only if there was something.
    if (has_non_empty_token) {
      result.subwords.push_back(std::string(token));
    }
  }

  // Close the last token.
  if (!leftover.empty()) {
    result.subwords.push_back(std::string(leftover));
  }

  return result;
}

bool RegexTokenizer::LookupId(absl::string_view key, int* result) const {
  auto it = token_index_map_.find(key);
  if (it == token_index_map_.end()) {
    return false;
  }
  *result = it->second;
  return true;
}

bool RegexTokenizer::LookupWord(int vocab_id, absl::string_view* result) const {
  auto it = index_token_map_.find(vocab_id);
  if (it == index_token_map_.end()) {
    return false;
  }
  *result = it->second;
  return true;
}

bool RegexTokenizer::GetStartToken(int* start_token) {
  return LookupId(kStart, start_token);
}

bool RegexTokenizer::GetPadToken(int* pad_token) {
  return LookupId(kPad, pad_token);
}

bool RegexTokenizer::GetUnknownToken(int* unknown_token) {
  return LookupId(kUnknown, unknown_token);
}

}  // namespace tokenizers
}  // namespace text
}  // namespace tasks
}  // namespace mediapipe
