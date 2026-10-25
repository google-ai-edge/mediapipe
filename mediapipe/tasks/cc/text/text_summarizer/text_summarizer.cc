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

#include "mediapipe/tasks/cc/text/text_summarizer/text_summarizer.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/absl_log.h"
#include "absl/log/globals.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/mutex.h"
#include "c/conversation.h"    // from @litert_lm
#include "c/engine.h"          // from @litert_lm
#include "c/error_reporter.h"  // from @litert_lm
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/core/logging/factory/logging_factory.h"
#include "mediapipe/tasks/cc/core/logging/tasks_logger.h"
#include "mediapipe/tasks/cc/core/running_mode.h"
#include "mediapipe/tasks/cc/text/utils/genai_utils.h"

namespace mediapipe::tasks::text::text_summarizer {
namespace {

using ::mediapipe::tasks::core::BaseOptions;
using ::mediapipe::tasks::text::utils::CreateConversationTextUserMessage;
using ::mediapipe::tasks::text::utils::ExtractConversationTextResponse;

constexpr int kDefaultMaxNumTokens = 8192;
constexpr int kPrefillChunkSize = 64;

absl::string_view ModeToString(TextSummarizerOptions::Mode mode) {
  switch (mode) {
    case TextSummarizerOptions::Mode::TLDR:
      return "TLDR";
    case TextSummarizerOptions::Mode::KEYPOINTS:
      return "Keypoints";
    default:
      return "";
  }
}

// Post processes the response text from the LiteRT LM engine. If
// `trim_whitespace` is true, leading whitespace is trimmed from the decoded
// string. Returns the processed string or an error status if parsing fails.
absl::StatusOr<std::string> PostProcessResponse(absl::string_view response_text,
                                                bool trim_whitespace = true) {
  ABSL_ASSIGN_OR_RETURN(std::string decoded,
                        ExtractConversationTextResponse(response_text));
  if (trim_whitespace && !decoded.empty()) {
    // Trim leading whitespace from the decoded string.
    absl::StripLeadingAsciiWhitespace(&decoded);
  }
  return decoded;
}

// Updates the active status to true.
// Returns UnavailableError if already active.
absl::Status TryLockEngine(absl::Mutex* mutex, bool* is_active) {
  absl::MutexLock lock(*mutex);
  if (*is_active) {
    return absl::UnavailableError(
        "Another conversation is working on the engine.");
  }
  *is_active = true;
  return absl::OkStatus();
}

// Updates the active status to false.
void UnlockEngine(absl::Mutex* mutex, bool* is_active) {
  absl::MutexLock lock(*mutex);
  *is_active = false;
}

struct StreamContext {
  absl::AnyInvocable<void(absl::StatusOr<TextSummarizerStreamResult>)> callback;
  std::string input_msg;
  bool trim_leading_whitespace = true;
  absl::Mutex* mutex = nullptr;
  bool* is_active = nullptr;
  tasks::core::logging::TasksLogger* tasks_logger = nullptr;
  mediapipe::Timestamp task_logger_ts;
};

// Helper to finish the stream.
// - Move the callback out of the context
// - Delete the context and release the engine.
// - Invoke the callback.
void FinishStream(StreamContext* context,
                  absl::StatusOr<TextSummarizerStreamResult> result) {
  context->tasks_logger->RecordInvocationEnd(context->task_logger_ts);
  auto callback = std::move(context->callback);
  // Capture mutex and active flag pointers before deleting context.
  absl::Mutex* mutex = context->mutex;
  bool* is_active = context->is_active;

  UnlockEngine(mutex, is_active);

  // Delete context.
  delete context;
  callback(result);
}

// Internal callback for the engine.
void StaticStreamCallback(void* callback_data,
                          const LiteRtLmStreamChunk* chunk) {
  if (!callback_data) {
    ABSL_LOG(ERROR)
        << "TextSummarizer failed: The callback_data should not be null.";
    return;
  }
  auto* context = reinterpret_cast<StreamContext*>(callback_data);

  const char* error_msg = nullptr;
  if (litert_lm_stream_chunk_get_error(chunk, &error_msg) !=
      kLiteRtLmStatusOk) {
    FinishStream(context,
                 absl::InternalError("Failed to read stream chunk error."));
    return;
  }
  if (error_msg) {
    FinishStream(context, absl::InternalError(error_msg));
    return;
  }

  std::string output_chunk;
  const char* text = nullptr;
  if (litert_lm_stream_chunk_get_text(chunk, &text) != kLiteRtLmStatusOk) {
    FinishStream(context,
                 absl::InternalError("Failed to read stream chunk text."));
    return;
  }
  if (text) {
    auto processed_chunk =
        PostProcessResponse(text, context->trim_leading_whitespace);
    if (!processed_chunk.ok()) {
      FinishStream(context, processed_chunk.status());
      return;
    }
    output_chunk = *processed_chunk;
    // After reaching the first non-empty chunk, stop trimming leading
    // whitespace from the decoded string.
    if (!output_chunk.empty()) {
      context->trim_leading_whitespace = false;
    }
  }

  bool is_final = false;
  if (litert_lm_stream_chunk_is_final(chunk, &is_final) != kLiteRtLmStatusOk) {
    FinishStream(context,
                 absl::InternalError("Failed to read stream chunk state."));
    return;
  }
  if (is_final) {
    FinishStream(context, TextSummarizerStreamResult{
                              .chunk = std::move(output_chunk), .done = true});
  } else {
    context->callback(TextSummarizerStreamResult{
        .chunk = std::move(output_chunk), .done = false});
  }
}

}  // namespace

absl::StatusOr<std::unique_ptr<TextSummarizer>> TextSummarizer::Create(
    std::unique_ptr<TextSummarizerOptions> options) {
  return Create(std::move(options), /*tasks_logger=*/nullptr);
}

absl::StatusOr<std::unique_ptr<TextSummarizer>> TextSummarizer::Create(
    std::unique_ptr<TextSummarizerOptions> options,
    std::unique_ptr<tasks::core::logging::TasksLogger> tasks_logger) {
  if (!options) {
    return absl::InvalidArgumentError("Options must not be null.");
  }
  // Throw error if both the model path is empty and the file descriptor is
  // invalid
  if (options->base_options.model_asset_path.empty() &&
      options->base_options.model_asset_descriptor_meta.fd < 0) {
    return absl::InvalidArgumentError(
        "Model path or valid file descriptor must be provided.");
  }

  if (options->base_options.delegate != BaseOptions::Delegate::CPU) {
    return absl::InvalidArgumentError("Only CPU delegate is supported.");
  }
  auto previous_absl_level = absl::MinLogLevel();
  auto previous_stderr_threshold = absl::StderrThreshold();
  if (options->min_log_severity > 0) {
    litert_lm_set_min_log_level(
        static_cast<LiteRtLmLogSeverity>(options->min_log_severity));
  }
  const char* backend = "cpu";

  LiteRtLmEngineSettings* settings = nullptr;
  if (options->base_options.model_asset_descriptor_meta.fd >= 0) {
    litert_lm_engine_settings_create_from_raw_file_descriptor(
        options->base_options.model_asset_descriptor_meta.fd, backend, nullptr,
        nullptr, &settings);
  } else {
    litert_lm_engine_settings_create(
        options->base_options.model_asset_path.c_str(), backend, nullptr,
        nullptr, &settings);
  }

  if (!settings) {
    return absl::InternalError("Failed to create engine settings.");
  }

  // Set the prefill chunk size for dynamic model to lower the peak ram usage.
  litert_lm_engine_settings_set_prefill_chunk_size(settings, kPrefillChunkSize);

  litert_lm_engine_settings_set_parallel_file_section_loading(
      settings, options->parallel_file_section_loading);

#ifdef __EMSCRIPTEN__
  litert_lm_engine_settings_set_single_threaded_execution(settings, true);
#endif

  // Set the max num tokens to the required buffer size.
  if (options->max_num_tokens > 0) {
    if (options->max_num_tokens > kDefaultMaxNumTokens) {
      litert_lm_engine_settings_delete(settings);
      return absl::InvalidArgumentError(
          absl::Substitute("Max num tokens exceeds the model's capacity around "
                           "$0 num tokens.",
                           kDefaultMaxNumTokens));
    }
    litert_lm_engine_settings_set_max_num_tokens(settings,
                                                 options->max_num_tokens);
  }

  if (!options->cache_dir.empty()) {
    litert_lm_engine_settings_set_cache_dir(settings,
                                            options->cache_dir.c_str());
  }

  LiteRtLmEngine* engine = nullptr;
  litert_lm_engine_create(settings, &engine);
  litert_lm_engine_settings_delete(settings);
  if (options->min_log_severity > 0) {
    absl::SetMinLogLevel(previous_absl_level);
    absl::SetStderrThreshold(previous_stderr_threshold);
  }

  if (!engine) {
    return absl::InternalError("Failed to create engine.");
  }

  if (tasks_logger == nullptr) {
    tasks_logger = tasks::core::logging::CreateTasksLogger(
        {.task_name = "TextSummarizer",
         .task_running_mode = tasks::core::RunningMode::kUnspecified,
         .host_environment = options->base_options.host_environment,
         .host_system = options->base_options.host_system,
         .host_version = options->base_options.host_version,
         .app_id = options->base_options.app_id,
         .app_version = options->base_options.app_version,
         .ca_bundle_path = options->base_options.ca_bundle_path});
  }
  tasks_logger->LogSessionStart();

  return absl::WrapUnique(new TextSummarizer(
      TextSummarizer::LiteRtLmEnginePtr(engine), options->mode,
      options->max_num_tokens, options->cache_dir, std::move(tasks_logger)));
}

TextSummarizer::TextSummarizer(
    TextSummarizer::LiteRtLmEnginePtr engine, TextSummarizerOptions::Mode mode,
    int max_num_tokens, std::string cache_dir,
    std::unique_ptr<tasks::core::logging::TasksLogger> tasks_logger)
    : engine_(std::move(engine)),
      mode_(mode),
      max_num_tokens_(max_num_tokens),
      cache_dir_(cache_dir),
      tasks_logger_(std::move(tasks_logger)) {}

TextSummarizer::~TextSummarizer() { Close().IgnoreError(); }

absl::StatusOr<TextSummarizerResult> TextSummarizer::Summarize(
    absl::string_view text) {
  ABSL_RETURN_IF_ERROR(TryLockEngine(&mutex_, /*is_active=*/&is_active_));
  absl::Cleanup failure_cleanup = [this] {
    UnlockEngine(&mutex_, &is_active_);
  };

  ABSL_ASSIGN_OR_RETURN(LiteRtLmConversation * conversation_ptr,
                        ResetAndCreateConversation(text));

  mediapipe::Timestamp task_logger_ts(tasks_logger_timestamp_++);
  tasks_logger_->RecordCpuInputArrival(task_logger_ts);
  absl::Cleanup log_invocation_end = [this, task_logger_ts] {
    tasks_logger_->RecordInvocationEnd(task_logger_ts);
  };

  std::string msg =
      CreateConversationTextUserMessage(text, ModeToString(mode_));
  LiteRtLmJsonResponse* response = nullptr;
  if (litert_lm_conversation_send_message(
          conversation_ptr, msg.c_str(), /*extra_context=*/nullptr,
          /*optional_args=*/nullptr, &response) != kLiteRtLmStatusOk) {
    const char* message = litert_lm_get_last_error_message();
    return absl::InternalError(
        absl::StrCat("Failed to generate content. ", message ? message : ""));
  }

  TextSummarizerResult result;
  const char* response_text = nullptr;
  if (litert_lm_json_response_get_string(response, &response_text) !=
      kLiteRtLmStatusOk) {
    litert_lm_json_response_delete(response);
    return absl::InternalError("Failed to read the generated content.");
  }

  ABSL_ASSIGN_OR_RETURN(result.summary, PostProcessResponse(response_text));

  litert_lm_json_response_delete(response);
  return result;
}

absl::Status TextSummarizer::SummarizeStreaming(
    absl::string_view text,
    absl::AnyInvocable<void(absl::StatusOr<TextSummarizerStreamResult>)>
        callback) {
  ABSL_RETURN_IF_ERROR(TryLockEngine(&mutex_, /*is_active=*/&is_active_));
  absl::Cleanup failure_cleanup = [this] {
    UnlockEngine(&mutex_, &is_active_);
  };

  ABSL_ASSIGN_OR_RETURN(LiteRtLmConversation * conversation_ptr,
                        ResetAndCreateConversation(text));

  mediapipe::Timestamp task_logger_ts(tasks_logger_timestamp_++);
  tasks_logger_->RecordCpuInputArrival(task_logger_ts);
  absl::Cleanup failure_logger_cleanup = [this, task_logger_ts] {
    tasks_logger_->RecordInvocationEnd(task_logger_ts);
  };

  auto context = std::make_unique<StreamContext>();
  context->callback = std::move(callback);
  context->input_msg =
      CreateConversationTextUserMessage(text, ModeToString(mode_));
  context->mutex = &mutex_;
  context->is_active = &is_active_;
  context->tasks_logger = tasks_logger_.get();
  context->task_logger_ts = task_logger_ts;

  int status = litert_lm_conversation_send_message_stream(
      conversation_ptr, context->input_msg.c_str(), /*extra_context=*/nullptr,
      /*optional_args=*/nullptr, &StaticStreamCallback, context.get());

  if (status != kLiteRtLmStatusOk) {
    // LiteRtLmStatusCode values are identical to absl::StatusCode values.
    const char* message = litert_lm_get_last_error_message();
    return absl::Status(
        static_cast<absl::StatusCode>(status),
        absl::StrCat("LiteRT LM engine error: ", message ? message : ""));
  }

  // Cancel the cleanup for UnlockEngine as the engine is now managed by the
  // streaming callback.
  std::move(failure_cleanup).Cancel();
  // Cancel the cleanup for logging as the end of invocation will be logged
  // in the StaticStreamCallback.
  std::move(failure_logger_cleanup).Cancel();

  // Release the context before the stream is driven below, so that
  // `FinishStream` can safely take ownership of and delete it from within the
  // callback.
  context.release();

#ifdef __EMSCRIPTEN__
  if (litert_lm_conversation_wait_until_done(conversation_ptr) != 0) {
    // The streaming callback may never run, so release the engine here to
    // avoid blocking `Close()` on `is_active_` forever.
    UnlockEngine(&mutex_, &is_active_);
    return absl::InternalError("Failed waiting for streaming completion.");
  }
#endif  // __EMSCRIPTEN__

  return absl::OkStatus();
}

absl::Status TextSummarizer::Close() {
  {
    absl::MutexLock lock(mutex_);
    // Wait for any active streaming session to finish to avoid possible access
    // to mutex, is_active, tasks_logger held by StreamContext.
    mutex_.Await(
        absl::Condition(+[](bool* active) { return !*active; }, &is_active_));

    conversation_.reset();
  }
  tasks_logger_->LogSessionEnd();
  engine_.reset();
  return absl::OkStatus();
}

absl::StatusOr<LiteRtLmConversation*>
TextSummarizer::ResetAndCreateConversation(absl::string_view text) {
  absl::MutexLock lock(mutex_);
  conversation_.reset();
  ABSL_ASSIGN_OR_RETURN(conversation_, CreateConversation(text));
  return conversation_.get();
}

absl::StatusOr<TextSummarizer::LiteRtLmConversationPtr>
TextSummarizer::CreateConversation(absl::string_view text) {
  if (!engine_) {
    return absl::InternalError("Engine is not initialized.");
  }
  if (text.empty()) {
    return absl::InvalidArgumentError(
        "Input text is empty. Please provide a non-empty text.");
  }
  LiteRtLmConversation* raw_conversation = nullptr;
  if (litert_lm_conversation_create(engine_.get(), nullptr,
                                    &raw_conversation) != kLiteRtLmStatusOk) {
    const char* message = litert_lm_get_last_error_message();
    return absl::InternalError(absl::StrCat("Failed to create conversation. ",
                                            message ? message : ""));
  }
  return TextSummarizer::LiteRtLmConversationPtr(raw_conversation);
}

}  // namespace mediapipe::tasks::text::text_summarizer
