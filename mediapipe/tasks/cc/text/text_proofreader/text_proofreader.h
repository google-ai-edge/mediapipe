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

#ifndef MEDIAPIPE_TASKS_CC_TEXT_TEXT_PROOFREADER_TEXT_PROOFREADER_H_
#define MEDIAPIPE_TASKS_CC_TEXT_TEXT_PROOFREADER_TEXT_PROOFREADER_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/port/threadpool.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/core/logging/tasks_logger.h"

namespace mediapipe::tasks::text::text_proofreader {

// The type of correction.
enum class CorrectionType {
  // The text is the same in the original and proofread text.
  kSame,
  // The text was inserted in the proofread text.
  kInsertion,
  // The text was deleted from the original text.
  kDeletion,
};

// A single correction.
struct Correction {
  // The type of correction.
  CorrectionType type;
  // The text associated with the correction.
  std::string text;
};

// The partial result of a streaming proofread operation.
struct TextProofreaderStreamResult {
  // The partial proofread text generated from the stream.
  std::string chunk;
  // The list of corrections to transform the input text to the proofread text.
  std::optional<std::vector<Correction>> corrections;
  // Whether the proofread stream is done.
  bool done = false;
};

// The result of a proofread operation.
struct TextProofreaderResult {
  // The generated proofread text.
  std::string proofread_text;
  // The list of corrections to transform the input text to the proofread text.
  std::vector<Correction> corrections;
};

// The options for configuring a MediaPipe text proofreader task.
struct TextProofreaderOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file, delegate options, op resolver, etc.
  tasks::core::BaseOptions base_options;

  // The maximum number of tokens for proofread tasks. If set, the
  // result will be truncated if the input and output exceed this value.
  //
  // If not set, then the default max_num_tokens is due to the model's capacity.
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

class TextProofreaderRunner;

// Performs proofreading on text.
// This API expects a LiteRT-LM model asset bundle.
//
// Input:
//   text: The input text to be proofread.
//
// Output:
//   TextProofreaderResult: The generated proofread text.
class TextProofreader {
 public:
  // Creates a TextProofreader from the provided `options`.
  static absl::StatusOr<std::unique_ptr<TextProofreader>> Create(
      std::unique_ptr<TextProofreaderOptions> options);

  ~TextProofreader();

  // Performs proofreading on the input `text`.
  // Returns the complete proofread result.
  absl::StatusOr<TextProofreaderResult> Proofread(absl::string_view text);

  // Performs proofreading on the input `text` and returns the result
  // incrementally via the result_callback.
  absl::Status ProofreadStreaming(
      absl::string_view text,
      absl::AnyInvocable<void(absl::StatusOr<TextProofreaderStreamResult>)>
          result_callback) ABSL_LOCKS_EXCLUDED(mutex_);

  // Cancels the active proofreading streaming session, if any.
  void Cancel() ABSL_LOCKS_EXCLUDED(mutex_);

  // Shuts down the TextProofreader when all the work is done.
  absl::Status Close();

 private:
  friend class TextProofreaderTestPeer;

  TextProofreader(
      std::unique_ptr<TextProofreaderRunner> runner,
      std::unique_ptr<tasks::core::logging::TasksLogger> tasks_logger,
      int word_chunk_threshold, int word_chunk_size);

  std::unique_ptr<TextProofreaderRunner> runner_;

  absl::Mutex mutex_;
  bool is_active_ ABSL_GUARDED_BY(mutex_) = false;
  bool cancelled_ ABSL_GUARDED_BY(mutex_) = false;

  std::unique_ptr<tasks::core::logging::TasksLogger> tasks_logger_;
  std::atomic<uint64_t> tasks_logger_timestamp_{0};

  int word_chunk_threshold_;
  int word_chunk_size_;

  // Only used when running on a thread pool.
  std::unique_ptr<::mediapipe::ThreadPool> thread_pool_;
};

}  // namespace mediapipe::tasks::text::text_proofreader

#endif  // MEDIAPIPE_TASKS_CC_TEXT_TEXT_PROOFREADER_TEXT_PROOFREADER_H_
