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

#ifndef MEDIAPIPE_TASKS_CC_TEXT_TEXT_SUMMARIZER_TEXT_SUMMARIZER_H_
#define MEDIAPIPE_TASKS_CC_TEXT_TEXT_SUMMARIZER_TEXT_SUMMARIZER_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "c/conversation.h"  // from @litert_lm
#include "c/engine.h"        // from @litert_lm
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/core/logging/tasks_logger.h"

namespace mediapipe::tasks::text::text_summarizer {

namespace internal_text_summarizer {

// A custom deleter for LiteRtLmEngine to be used with std::unique_ptr.
struct LiteRtLmEngineDeleter {
  void operator()(LiteRtLmEngine* engine) const {
    if (engine) {
      litert_lm_engine_delete(engine);
    }
  }
};

// A custom deleter for LiteRtLmConversation to be used with std::unique_ptr.
struct LiteRtLmConversationDeleter {
  void operator()(LiteRtLmConversation* conversation) const {
    if (conversation) {
      litert_lm_conversation_delete(conversation);
    }
  }
};

}  // namespace internal_text_summarizer

// The result of a summarization operation.
struct TextSummarizerResult {
  // The generated summary text.
  std::string summary;
};

// The result of a streaming summarization operation.
struct TextSummarizerStreamResult {
  // The generated summary text.
  std::string chunk;
  // Whether the summarization is done.
  bool done = false;
};

// The options for configuring a MediaPipe text summarizer task.
struct TextSummarizerOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file, delegate options, op resolver, etc.
  tasks::core::BaseOptions base_options;

  // The mode of the text summarizer.
  enum class Mode {
    // Generate a TLDR summary.
    TLDR,
    // Generate Key Points.
    KEYPOINTS,
  };

  // The mode of the text summarizer.
  Mode mode = Mode::KEYPOINTS;

  // The maximum number of tokens for summarization tasks. If set, the
  // summarization will be truncated if the input and output exceed this value.
  //
  // If not set, then the default max_num_tokens is based on the model's
  // capacity.
  int max_num_tokens = 0;

  // Cache directory for the model weight cache and program cache.
  // By default, it will be set by LiteRT-LM engine based on the model file
  // path.
  //
  // To disable caching, set this to ":nocache".
  std::string cache_dir = "";

  // Whether to load model files in parallel.
  // Set to false to disable parallel file section loading for wasm target.
  bool parallel_file_section_loading = true;

  // Minimum logging severity (default 4 = ERROR).
  int min_log_severity = 4;
};

// Performs summarization on text.
// This API expects a LiteRT-LM model asset bundle.
//
// Input:
//   text: The input text to be summarized.
//
// Output:
//   TextSummarizerResult: The generated summary.
class TextSummarizer {
 public:
  // Creates a TextSummarizer from the provided `options`.
  static absl::StatusOr<std::unique_ptr<TextSummarizer>> Create(
      std::unique_ptr<TextSummarizerOptions> options);

  ~TextSummarizer();

  // Performs summarization on the input `text`.
  // Returns the complete summary result.
  absl::StatusOr<TextSummarizerResult> Summarize(absl::string_view text);

  // Performs summarization on the input `text` and returns the result
  // incrementally via the callback.
  absl::Status SummarizeStreaming(
      absl::string_view text,
      absl::AnyInvocable<void(absl::StatusOr<TextSummarizerStreamResult>)>
          callback);

  // Shuts down the TextSummarizer when all the work is done.
  absl::Status Close();

 private:
  friend class TextSummarizerLoggerTest;

  static absl::StatusOr<std::unique_ptr<TextSummarizer>> Create(
      std::unique_ptr<TextSummarizerOptions> options,
      std::unique_ptr<tasks::core::logging::TasksLogger> tasks_logger);

  using LiteRtLmEnginePtr =
      std::unique_ptr<LiteRtLmEngine,
                      internal_text_summarizer::LiteRtLmEngineDeleter>;
  using LiteRtLmConversationPtr =
      std::unique_ptr<LiteRtLmConversation,
                      internal_text_summarizer::LiteRtLmConversationDeleter>;

  TextSummarizer(
      LiteRtLmEnginePtr engine, TextSummarizerOptions::Mode mode,
      int max_num_tokens, std::string cache_dir,
      std::unique_ptr<tasks::core::logging::TasksLogger> tasks_logger);

  // Creates a conversation and checks input validity.
  absl::StatusOr<LiteRtLmConversationPtr> CreateConversation(
      absl::string_view text);

  // Resets the current conversation and creates a new one under a lock.
  // Returns a raw pointer to the created conversation.
  absl::StatusOr<LiteRtLmConversation*> ResetAndCreateConversation(
      absl::string_view text) ABSL_LOCKS_EXCLUDED(mutex_);

  // The engine for the text summarizer.
  LiteRtLmEnginePtr engine_;
  const TextSummarizerOptions::Mode mode_;
  int max_num_tokens_ = 0;
  std::string cache_dir_;

  // Mutex to protect the active status of the engine.
  absl::Mutex mutex_;
  LiteRtLmConversationPtr conversation_ ABSL_GUARDED_BY(mutex_);
  bool is_active_ ABSL_GUARDED_BY(mutex_) = false;

  std::unique_ptr<tasks::core::logging::TasksLogger> tasks_logger_;
  uint64_t tasks_logger_timestamp_ = 0;
};

}  // namespace mediapipe::tasks::text::text_summarizer

#endif  // MEDIAPIPE_TASKS_CC_TEXT_TEXT_SUMMARIZER_TEXT_SUMMARIZER_H_
