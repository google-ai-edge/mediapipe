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

#ifndef MEDIAPIPE_TASKS_CC_TEXT_UTILS_GENAI_UTILS_H_
#define MEDIAPIPE_TASKS_CC_TEXT_UTILS_GENAI_UTILS_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace mediapipe::tasks::text::utils {

// Creates a JSON text message string for the user role.
//
// Follows the LiteRtLm's text message format:
// With mode:
// JsonMessage {
//   {"role", "user"},
//   {"content", [{"type", "text"}, {"mode", mode}, {"text", text}]}
// }
// Without mode:
// JsonMessage {
//   {"role", "user"},
//   {"content", [{"type", "text"}, {"text", text}]}
// }
//
// The basic template can refer to
// https://huggingface.co/docs/transformers/main/en/chat_templating
std::string CreateConversationTextUserMessage(
    absl::string_view text,
    std::optional<absl::string_view> mode = std::nullopt);

// Extracts the text content from a LiteRtLm conversation JSON response.
//
// JsonMessage {
//   {"content", [{"type", "text"}, {"text", text_content}]}
// }
//
// Returns the extracted text content, or an error status if parsing fails or
// the expected fields are not found.
absl::StatusOr<std::string> ExtractConversationTextResponse(
    absl::string_view json_response);

// Checks if the text has a repeating suffix of at least `min_chars`
// characters that repeats `suffix_count` times consecutively at the very end of
// the text to catch echo loops issues shared by LLMs.
//
// The function iterates through possible pattern sizes starting from
// `min_chars` up to `text.size() / suffix_count`. For each size, it checks if
// the last `size * suffix_count` characters consist of the same string
// repeated `suffix_count` times.
//
// Note that the time complexity of this function is O(|text|^2 / suffix_count).
// This implies non-trivial post-processing time if called per token for long
// texts.
//
// Examples:
// - HasRepeatingSuffix(3, 2, "helloabcabc") -> true
//   (The suffix "abcabc" is "abc" repeated 2 times, size 3 >= 3)
// - HasRepeatingSuffix(2, 3, "abcxyxyx") -> false
//   (The suffix doesn't contain a pattern repeated 3 times)
bool HasRepeatingSuffix(int min_chars, int suffix_count,
                        absl::string_view text);

struct TextChunk {
  absl::string_view text;
  absl::string_view trailing_separator;
};

// Returns trailing punctuation from the original text if the original text ends
// with a punctuation mark ('.', '!', '"', '\'') and the response text ends with
// an alphanumeric character. The caller can append the returned punctuation to
// the response text.
//
// Returns a string containing the trailing punctuation, which can be empty.
std::string GetTrailingPunctuation(absl::string_view original_text,
                                   absl::string_view response_text);

// Splits the input text into chunks while preserving layout.
//
// The text is first split into segments along paragraph boundaries (newlines).
// - If a paragraph is at most `word_chunk_threshold` words, it is kept as a
//   single chunk (this includes paragraphs shorter than `word_chunk_size`).
// - If a paragraph exceeds `word_chunk_threshold` words, it is broken down into
//   smaller chunks. The algorithm attempts to split at sentence boundaries
//   (periods) within the range of `word_chunk_size` to `word_chunk_threshold`
//   words. If no sentence boundary is found in this range, it forces a split at
//   `word_chunk_threshold` words.
//
// The spacing/newlines that split the chunks are consolidated into the
// contiguous `trailing_separator` field of each TextChunk, allowing callers
// to cleanly reconstruct the exact layout by appending it back after the
// model's output.
std::vector<TextChunk> ChunkText(absl::string_view text,
                                 int word_chunk_threshold = 500,
                                 int word_chunk_size = 450);

// Chunks text by character count with a sliding window overlap.
std::vector<std::string> ChunkTextByCountingCharacters(absl::string_view text,
                                                       int chunk_size,
                                                       int chunk_overlap);
}  // namespace mediapipe::tasks::text::utils

#endif  // MEDIAPIPE_TASKS_CC_TEXT_UTILS_GENAI_UTILS_H_
