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

#include "mediapipe/tasks/cc/text/text_proofreader/text_proofreader.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/functional/any_invocable.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "editscript.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/threadpool.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/core/logging/factory/logging_factory.h"
#include "mediapipe/tasks/cc/core/logging/tasks_logger.h"
#include "mediapipe/tasks/cc/core/running_mode.h"
#include "mediapipe/tasks/cc/text/text_proofreader/text_proofreader_runner.h"
#include "mediapipe/tasks/cc/text/utils/genai_utils.h"

namespace mediapipe::tasks::text::text_proofreader {
namespace {

using ::mediapipe::tasks::text::utils::ChunkText;

constexpr int kDefaultWordChunkThreshold = 500;
constexpr int kDefaultWordChunkSize = 450;

std::vector<Correction> ComputeCorrections(absl::string_view original,
                                           absl::string_view proofread) {
  std::vector<Correction> corrections;
  auto edits = third_party::diff::GetTokenDiffs(
      original.begin(), original.end(), proofread.begin(), proofread.end());

  for (const auto& edit : edits) {
    switch (edit.operation) {
      case third_party::diff::Operation::EQUALS:
        corrections.push_back(
            {CorrectionType::kSame,
             std::string(original.substr(edit.start, edit.end - edit.start))});
        break;
      case third_party::diff::Operation::DELETE:
        corrections.push_back(
            {CorrectionType::kDeletion,
             std::string(original.substr(edit.start, edit.end - edit.start))});
        break;
      case third_party::diff::Operation::INSERT:
        corrections.push_back(
            {CorrectionType::kInsertion,
             std::string(proofread.substr(edit.start, edit.end - edit.start))});
        break;
    }
  }
  return corrections;
}

absl::Status TryLockEngine(absl::Mutex& mutex, bool* is_active) {
  absl::MutexLock lock(mutex);
  if (*is_active) {
    return absl::UnavailableError(
        "Another conversation is active for this engine.");
  }
  *is_active = true;
  return absl::OkStatus();
}

void UnlockEngine(absl::Mutex& mutex, bool* is_active) {
  absl::MutexLock lock(mutex);
  *is_active = false;
}

class StreamingSession : public std::enable_shared_from_this<StreamingSession> {
 public:
  static std::shared_ptr<StreamingSession> Create(
      absl::string_view text, int threshold, int size,
      absl::AnyInvocable<void(absl::StatusOr<TextProofreaderStreamResult>)>
          callback,
      tasks::core::logging::TasksLogger* logger, mediapipe::Timestamp ts,
      TextProofreaderRunner* runner, mediapipe::ThreadPool* pool,
      absl::AnyInvocable<bool() const> check_cancelled,
      absl::AnyInvocable<void()> on_done) {
    return std::shared_ptr<StreamingSession>(new StreamingSession(
        text, threshold, size, std::move(callback), logger, ts, runner, pool,
        std::move(check_cancelled), std::move(on_done)));
  }

  void Start() {
    if (tasks_logger_) {
      tasks_logger_->RecordCpuInputArrival(task_logger_ts_);
    }
    if (thread_pool_ == nullptr) {
      RunChunksOnCallingThread();
    } else {
      ScheduleNextChunk();
    }
  }

 private:
  explicit StreamingSession(
      absl::string_view text, int threshold, int size,
      absl::AnyInvocable<void(absl::StatusOr<TextProofreaderStreamResult>)>
          callback,
      tasks::core::logging::TasksLogger* logger, mediapipe::Timestamp ts,
      TextProofreaderRunner* runner, mediapipe::ThreadPool* pool,
      absl::AnyInvocable<bool() const> check_cancelled,
      absl::AnyInvocable<void()> on_done)
      : stable_text_(text),
        user_callback_(std::move(callback)),
        tasks_logger_(logger),
        task_logger_ts_(ts),
        runner_(runner),
        thread_pool_(pool),
        check_cancelled_(std::move(check_cancelled)),
        on_done_(std::move(on_done)) {
    chunks_ = ChunkText(stable_text_, threshold, size);
  }

  // Processes all chunks on the calling thread. This is used when no thread
  // pool is available.
  void RunChunksOnCallingThread() {
    // Keeps the session alive until all chunks have been processed.
    auto self = shared_from_this();
    while (has_more_chunks_) {
      has_more_chunks_ = false;
      RunNextChunk();
    }
  }

  void ScheduleNextChunk() {
    auto self = shared_from_this();
    thread_pool_->Schedule([self]() { self->RunNextChunk(); });
  }

  void RunNextChunk() {
    if (check_cancelled_()) {
      Finish(absl::CancelledError("Cancelled."));
      return;
    }

    // All chunks have been processed, deliver the final result.
    if (current_chunk_index_ >= chunks_.size()) {
      auto corrections = ComputeCorrections(stable_text_, full_response_);
      Finish(TextProofreaderStreamResult{
          .chunk = "", .corrections = std::move(corrections), .done = true});
      return;
    }

    size_t i = current_chunk_index_;
    const auto& chunk = chunks_[i];

    auto self = shared_from_this();
    absl::Status status = runner_->RunStreaming(
        chunk.text,
        [self, sep = chunk.trailing_separator](
            absl::StatusOr<absl::string_view> result, bool is_final) {
          if (!result.ok()) {
            self->Finish(result.status());
            return;
          }

          if (!result->empty()) {
            absl::StrAppend(&self->full_response_, *result);
            self->DeliverResult(TextProofreaderStreamResult{
                .chunk = std::string(*result), .done = false});
          }

          if (is_final) {
            if (!sep.empty()) {
              absl::StrAppend(&self->full_response_, sep);
              self->DeliverResult(TextProofreaderStreamResult{
                  .chunk = std::string(sep), .done = false});
            }
            ++self->current_chunk_index_;
            self->has_more_chunks_ = true;
            if (self->thread_pool_ != nullptr) {
              self->ScheduleNextChunk();
            }
          }
        });
    if (!status.ok()) {
      Finish(status);
    }
  }

  void Finish(absl::StatusOr<TextProofreaderStreamResult> result) {
    if (tasks_logger_) {
      tasks_logger_->RecordInvocationEnd(task_logger_ts_);
    }
    auto cb = std::move(user_callback_);
    auto cleanup = std::move(on_done_);
    if (cleanup) {
      cleanup();
    }
    if (cb) {
      cb(std::move(result));
    }
  }

  void DeliverResult(TextProofreaderStreamResult result) {
    if (result.done) {
      Finish(std::move(result));
    } else {
      if (user_callback_) {
        user_callback_(std::move(result));
      }
    }
  }

  std::string stable_text_;
  std::vector<utils::TextChunk> chunks_;
  size_t current_chunk_index_ = 0;
  std::string full_response_;
  absl::AnyInvocable<void(absl::StatusOr<TextProofreaderStreamResult>)>
      user_callback_;
  tasks::core::logging::TasksLogger* tasks_logger_ = nullptr;
  mediapipe::Timestamp task_logger_ts_;

  TextProofreaderRunner* runner_ = nullptr;
  mediapipe::ThreadPool* thread_pool_ = nullptr;
  bool has_more_chunks_ = true;
  absl::AnyInvocable<bool() const> check_cancelled_;
  absl::AnyInvocable<void()> on_done_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<TextProofreader>> TextProofreader::Create(
    std::unique_ptr<TextProofreaderOptions> options) {
  if (!options) {
    return absl::InvalidArgumentError("Options must not be null.");
  }

  auto tasks_logger = tasks::core::logging::CreateTasksLogger(
      {.task_name = "TextProofreader",
       .task_running_mode = tasks::core::RunningMode::kUnspecified,
       .host_environment = options->base_options.host_environment,
       .host_system = options->base_options.host_system,
       .host_version = options->base_options.host_version,
       .app_id = options->base_options.app_id,
       .app_version = options->base_options.app_version,
       .ca_bundle_path = options->base_options.ca_bundle_path});
  tasks_logger->LogSessionStart();

  ABSL_ASSIGN_OR_RETURN(auto runner,
                        TextProofreaderRunner::Create(std::move(options)));

  return absl::WrapUnique(
      new TextProofreader(std::move(runner), std::move(tasks_logger),
                          kDefaultWordChunkThreshold, kDefaultWordChunkSize));
}

TextProofreader::TextProofreader(
    std::unique_ptr<TextProofreaderRunner> runner,
    std::unique_ptr<tasks::core::logging::TasksLogger> tasks_logger,
    int word_chunk_threshold, int word_chunk_size)
    : runner_(std::move(runner)),
      tasks_logger_(std::move(tasks_logger)),
      word_chunk_threshold_(word_chunk_threshold),
      word_chunk_size_(word_chunk_size) {
#ifndef __EMSCRIPTEN__
  thread_pool_ =
      std::make_unique<::mediapipe::ThreadPool>("proofreader_pool", 1);
  thread_pool_->StartWorkers();
#endif  // __EMSCRIPTEN__
}

TextProofreader::~TextProofreader() { Close().IgnoreError(); }

absl::StatusOr<TextProofreaderResult> TextProofreader::Proofread(
    absl::string_view text) {
  if (text.empty()) {
    return TextProofreaderResult();
  }

  ABSL_RETURN_IF_ERROR(TryLockEngine(mutex_, /*is_active=*/&is_active_));
  {
    absl::MutexLock lock(mutex_);
    cancelled_ = false;
  }
  absl::Cleanup failure_cleanup = [this] { UnlockEngine(mutex_, &is_active_); };

  mediapipe::Timestamp task_logger_ts(tasks_logger_timestamp_++);
  tasks_logger_->RecordCpuInputArrival(task_logger_ts);
  absl::Cleanup log_invocation_end = [this, task_logger_ts] {
    tasks_logger_->RecordInvocationEnd(task_logger_ts);
  };

  std::vector<utils::TextChunk> chunks =
      ChunkText(text, word_chunk_threshold_, word_chunk_size_);
  TextProofreaderResult combined_result;

  for (size_t i = 0; i < chunks.size(); ++i) {
    const auto& chunk = chunks[i];
    std::string proofread_clean_chunk;
    absl::Status error_status;
    absl::Notification done;

    ABSL_RETURN_IF_ERROR(runner_->RunStreaming(
        chunk.text,
        [&](absl::StatusOr<absl::string_view> result, bool is_final) {
          if (!result.ok()) {
            error_status = result.status();
          } else {
            absl::StrAppend(&proofread_clean_chunk, *result);
          }
          if (is_final) {
            done.Notify();
          }
        }));
    done.WaitForNotification();
    ABSL_RETURN_IF_ERROR(error_status);

    absl::StrAppend(&combined_result.proofread_text, proofread_clean_chunk,
                    chunk.trailing_separator);
  }

  combined_result.corrections =
      ComputeCorrections(text, combined_result.proofread_text);
  return combined_result;
}

absl::Status TextProofreader::ProofreadStreaming(
    absl::string_view text,
    absl::AnyInvocable<void(absl::StatusOr<TextProofreaderStreamResult>)>
        result_callback) {
  if (text.empty()) {
    result_callback(TextProofreaderStreamResult{
        .chunk = "", .corrections = std::vector<Correction>(), .done = true});
    return absl::OkStatus();
  }

  ABSL_RETURN_IF_ERROR(TryLockEngine(mutex_, /*is_active=*/&is_active_));
  {
    absl::MutexLock lock(mutex_);
    cancelled_ = false;
  }

  auto session = StreamingSession::Create(
      text, word_chunk_threshold_, word_chunk_size_, std::move(result_callback),
      tasks_logger_.get(), mediapipe::Timestamp(tasks_logger_timestamp_++),
      runner_.get(), thread_pool_.get(),
      /*check_cancelled=*/
      [this]() {
        absl::MutexLock lock(mutex_);
        return cancelled_;
      },
      /*on_done=*/[this]() { UnlockEngine(mutex_, &is_active_); });

  session->Start();

  return absl::OkStatus();
}

void TextProofreader::Cancel() {
  absl::MutexLock lock(mutex_);
  cancelled_ = true;
  if (is_active_ && runner_) {
    runner_->Cancel();
  }
}

absl::Status TextProofreader::Close() {
  {
    absl::MutexLock lock(mutex_);
    if (!runner_) {
      return absl::OkStatus();
    }
    mutex_.Await(
        absl::Condition(+[](bool* active) { return !*active; }, &is_active_));
  }
  if (tasks_logger_) {
    tasks_logger_->LogSessionEnd();
  }
  if (runner_) {
    ABSL_RETURN_IF_ERROR(runner_->Close());
  }
  return absl::OkStatus();
}

}  // namespace mediapipe::tasks::text::text_proofreader
