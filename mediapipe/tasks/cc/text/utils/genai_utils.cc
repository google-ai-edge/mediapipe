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

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "nlohmann/json.hpp"

namespace mediapipe::tasks::text::utils {

namespace {

// Finds the position to split the remaining text when a paragraph is longer
// than word_chunk_threshold.
//
// NOTE: The caller (ChunkText) is responsible for chunking by paragraph
// (Priority 1).
//
// The split position is determined based on the following priorities:
// Priority 2: Chunk by period. Look for the last period between
//             word_chunk_size and word_chunk_threshold words.
// Priority 3: Chunk by space if no period is found within the chunking range.
//             Split at the word_chunk_threshold word boundary.
size_t FindSplitPosition(absl::string_view remaining, int word_chunk_threshold,
                         int word_chunk_size) {
  int word_count = 0;
  bool in_word = false;

  size_t soft_limit_pos = absl::string_view::npos;
  size_t hard_limit_pos = absl::string_view::npos;
  size_t last_period_pos = absl::string_view::npos;

  for (size_t i = 0; i < remaining.size(); ++i) {
    if (absl::ascii_isspace(remaining[i])) {
      in_word = false;
    } else {
      if (!in_word) {
        ++word_count;
        in_word = true;

        if (word_count == word_chunk_size) {
          // Start of the word that marks the soft limit.
          soft_limit_pos = i;
        }
        // We check for `word_chunk_threshold + 1` because we want to include up
        // to `word_chunk_threshold` words in the chunk. When we see the start
        // of the (word_chunk_threshold + 1)-th word, the previous character
        // before it marks the maximum position where we can split to stay
        // within limits.
        if (word_count == word_chunk_threshold + 1) {
          // Start of the word that exceeds the threshold.
          hard_limit_pos = i;
          break;
        }
      }
    }

    // Look for sentence boundaries (periods).
    if (remaining[i] == '.') {
      // A period counts as a split boundary if it's at the end of the text
      // or followed by a space.
      if (i + 1 == remaining.size() || absl::ascii_isspace(remaining[i + 1])) {
        // Only consider periods after we've reached the minimum chunk size.
        if (soft_limit_pos != absl::string_view::npos) {
          last_period_pos = i + 1;
        }
      }
    }
  }

  // If the entire text fits within the threshold, no split is needed.
  if (word_count <= word_chunk_threshold) {
    return remaining.size();
  }

  // Priority 2: Split at the last period found within the target range.
  if (last_period_pos != absl::string_view::npos &&
      (hard_limit_pos == absl::string_view::npos ||
       last_period_pos <= hard_limit_pos)) {
    return last_period_pos;
  }

  // Priority 3: Split at the hard limit (word boundary).
  if (hard_limit_pos != absl::string_view::npos) {
    return hard_limit_pos;
  }

  return remaining.size();
}

}  // namespace

std::string CreateConversationTextUserMessage(
    absl::string_view text, std::optional<absl::string_view> mode) {
  nlohmann::json content = nlohmann::json::array();
  if (mode.has_value()) {
    content.push_back({{"type", "text"}, {"mode", *mode}, {"text", text}});
  } else {
    content.push_back({{"type", "text"}, {"text", text}});
  }
  nlohmann::json message_json = {{"role", "user"}, {"content", content}};
  return message_json.dump();
}

absl::StatusOr<std::string> ExtractConversationTextResponse(
    absl::string_view json_response) {
  auto conversation_content =
      nlohmann::json::parse(json_response, nullptr, false);
  if (conversation_content.is_discarded()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse JSON response: ", json_response));
  }

  if (conversation_content.contains("content") &&
      conversation_content["content"].is_array() &&
      !conversation_content["content"].empty()) {
    auto& content_item = conversation_content["content"][0];
    if (content_item.contains("text") && content_item["text"].is_string()) {
      return content_item["text"];
    }
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Cannot extract text content from JSON response: ", json_response));
}

bool HasRepeatingSuffix(int min_chars, int suffix_count,
                        absl::string_view text) {
  if (suffix_count <= 0) {
    return false;
  }
  size_t cur_size = min_chars;
  while (text.size() >= cur_size * suffix_count) {
    size_t index = text.size() - cur_size;
    bool is_repeating = true;
    for (int i = 1; i < suffix_count; i++) {
      if (text.substr(index, cur_size) !=
          text.substr(index - cur_size * i, cur_size)) {
        is_repeating = false;
        break;
      }
    }
    if (is_repeating) {
      return true;
    }
    ++cur_size;
  }
  return false;
}

std::string GetTrailingPunctuation(absl::string_view original_text,
                                   absl::string_view response_text) {
  std::string appended_punct = "";
  if (!original_text.empty() && !response_text.empty()) {
    char last_orig = original_text.back();
    if (last_orig == '.' || last_orig == '!' || last_orig == '"' ||
        last_orig == '\'') {
      if (absl::ascii_isalnum(response_text.back())) {
        appended_punct.push_back(last_orig);
      }
    }
  }
  return appended_punct;
}

std::vector<TextChunk> ChunkText(absl::string_view text,
                                 int word_chunk_threshold,
                                 int word_chunk_size) {
  std::vector<TextChunk> chunks;
  size_t pos = 0;

  // Helper lambda to add a chunk and retrospectively update the trailing
  // separator of the previous chunk. This ensures all whitespace/layout
  // between chunks is preserved.
  auto add_chunk = [&](absl::string_view raw_chunk) {
    if (!chunks.empty()) {
      // The separator starts right after the previous chunk ends, and ends
      // where the current chunk starts.
      size_t sep_start =
          chunks.back().text.data() + chunks.back().text.size() - text.data();
      size_t sep_end = raw_chunk.data() - text.data();
      chunks.back().trailing_separator =
          text.substr(sep_start, sep_end - sep_start);
    }
    chunks.push_back(TextChunk{.text = raw_chunk});
  };

  // Priority 1: Chunk by paragraph (split along newline boundaries).
  while (pos < text.size()) {
    size_t next_newline = text.find('\n', pos);
    absl::string_view segment;
    size_t next_pos;
    if (next_newline == absl::string_view::npos) {
      segment = text.substr(pos);
      next_pos = text.size();
    } else {
      segment = text.substr(pos, next_newline - pos);
      next_pos = next_newline + 1;
    }

    absl::string_view clean_segment = absl::StripAsciiWhitespace(segment);
    if (!clean_segment.empty()) {
      absl::string_view remaining = clean_segment;
      // Process the paragraph.
      // FindSplitPosition will count the words. If the paragraph is within
      // `word_chunk_threshold`, it returns the full length (kept as a single
      // chunk). If it exceeds the threshold, it breaks it down further using
      // Priority 2 (by period) or Priority 3 (by space).
      while (!remaining.empty()) {
        remaining = absl::StripAsciiWhitespace(remaining);
        if (remaining.empty()) break;

        size_t split_pos =
            FindSplitPosition(remaining, word_chunk_threshold, word_chunk_size);

        absl::string_view chunk =
            absl::StripAsciiWhitespace(remaining.substr(0, split_pos));
        if (!chunk.empty()) {
          add_chunk(chunk);
        }

        // If split_pos covers the rest of the text, we are done with this
        // segment.
        if (split_pos >= remaining.size()) {
          break;
        }
        remaining = remaining.substr(split_pos);
      }
    }
    pos = next_pos;
  }

  // Handle the trailing separator for the final chunk.
  if (!chunks.empty()) {
    size_t sep_start =
        chunks.back().text.data() + chunks.back().text.size() - text.data();
    chunks.back().trailing_separator = text.substr(sep_start);
  }

  return chunks;
}

std::vector<std::string> ChunkTextByCountingCharacters(absl::string_view text,
                                                       int chunk_size,
                                                       int chunk_overlap) {
  if (chunk_size <= 0) {
    return {std::string(text)};
  }
  if (text.length() <= static_cast<size_t>(chunk_size)) {
    return {std::string(text)};
  }

  int effective_overlap = std::max(0, std::min(chunk_overlap, chunk_size - 1));
  int step = chunk_size - effective_overlap;

  std::vector<std::string> chunks;
  for (size_t i = 0; i < text.length(); i += step) {
    size_t end = std::min(i + chunk_size, text.length());
    chunks.push_back(std::string(text.substr(i, end - i)));
    if (end == text.length()) break;
  }
  return chunks;
}
}  // namespace mediapipe::tasks::text::utils
