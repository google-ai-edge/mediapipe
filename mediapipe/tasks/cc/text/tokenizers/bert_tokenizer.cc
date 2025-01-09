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

#include "mediapipe/tasks/cc/text/tokenizers/bert_tokenizer.h"

#include <cstdint>

#include "tensorflow_text/core/kernels/regex_split.h"

namespace mediapipe {
namespace tasks {
namespace text {
namespace tokenizers {

FlatHashMapBackedWordpiece::FlatHashMapBackedWordpiece(
    const std::vector<std::string>& vocab)
    : vocab_{vocab} {
  for (int i = 0; i < vocab_.size(); ++i) {
    index_map_[vocab_[i]] = i;
  }
}

tensorflow::text::LookupStatus FlatHashMapBackedWordpiece::Contains(
    absl::string_view key, bool* value) const {
  *value = index_map_.contains(key);
  return tensorflow::text::LookupStatus();
}

bool FlatHashMapBackedWordpiece::LookupId(const absl::string_view key,
                                          int* result) const {
  auto it = index_map_.find(key);
  if (it == index_map_.end()) {
    return false;
  }
  *result = it->second;
  return true;
}

bool FlatHashMapBackedWordpiece::LookupWord(int vocab_id,
                                            absl::string_view* result) const {
  if (vocab_id >= vocab_.size() || vocab_id < 0) {
    return false;
  }
  *result = vocab_[vocab_id];
  return true;
}

TokenizerResult BertTokenizer::Tokenize(const std::string& input) {
  return TokenizeWordpiece(input);
}

WordpieceTokenizerResult BertTokenizer::TokenizeWordpiece(
    const std::string& input) const {
  WordpieceTokenizerResult result;
  std::vector<std::string>& subwords = result.subwords;
  std::vector<int>& wp_absolute_begin_offset = result.wp_begin_offset;
  std::vector<int>& wp_absolute_end_offset = result.wp_end_offset;

  std::vector<absl::string_view> tokens;
  std::vector<long long> begin_offsets;
  std::vector<long long> end_offsets;

  // Run through tokenize function
  tensorflow::text::RegexSplit(input, delim_re_, true, include_delim_re_,
                               &tokens, &begin_offsets, &end_offsets);

  for (int token_index = 0; token_index < tokens.size(); token_index++) {
    auto& token = tokens[token_index];
    int num_word_pieces = 0;
    tensorflow::text::LookupStatus status = WordpieceTokenize(
        token, options_.max_bytes_per_token, options_.max_chars_per_subtoken,
        options_.suffix_indicator, options_.use_unknown_token,
        options_.unknown_token, options_.split_unknown_chars, &vocab_,
        &subwords, &wp_absolute_begin_offset, &wp_absolute_end_offset,
        &num_word_pieces);

    result.row_lengths.emplace_back(num_word_pieces);
    // for the last num_word_pieces added into wp_absolute_begin_offset and
    // wp_absolute_end_offset, offset them with begin_offsets[token_index]
    int absolute_offset_size = wp_absolute_begin_offset.size();
    for (int i = num_word_pieces; i > 0; i--) {
      wp_absolute_begin_offset[absolute_offset_size - i] +=
          begin_offsets[token_index];
      wp_absolute_end_offset[absolute_offset_size - i] +=
          begin_offsets[token_index];
    }
    if (!status.success) {
      return result;
    }
  }

  return result;
}

}  // namespace tokenizers
}  // namespace text
}  // namespace tasks
}  // namespace mediapipe
