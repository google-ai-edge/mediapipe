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

#include "mediapipe/tasks/cc/text/text_proofreader/text_proofreader_runner.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/functional/any_invocable.h"
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
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/text/text_proofreader/text_proofreader.h"
#include "mediapipe/tasks/cc/text/utils/genai_utils.h"

namespace mediapipe::tasks::text::text_proofreader {

namespace {

using ::mediapipe::tasks::core::BaseOptions;
using ::mediapipe::tasks::text::utils::CreateConversationTextUserMessage;
using ::mediapipe::tasks::text::utils::ExtractConversationTextResponse;
using ::mediapipe::tasks::text::utils::GetTrailingPunctuation;

constexpr int kDefaultMaxNumTokens = 8192;
constexpr int kPrefillChunkSize = 64;
constexpr int kMinEchoLoopNumRepetitions = 3;
constexpr int kMinEchoLoopMinChars = 15;
constexpr float kMinEchoLoopDetectRatio = 1.1;

absl::StatusOr<std::string> PostProcessResponse(absl::string_view response_text,
                                                bool trim_whitespace = true) {
  ABSL_ASSIGN_OR_RETURN(std::string decoded,
                        ExtractConversationTextResponse(response_text));
  if (trim_whitespace) {
    absl::StripLeadingAsciiWhitespace(&decoded);
  }
  return decoded;
}

}  // namespace

struct TextProofreaderRunner::StreamContext {
  TextProofreaderRunner* runner = nullptr;
  absl::AnyInvocable<void(absl::StatusOr<absl::string_view>, bool)>
      user_callback;
  bool trim_leading_whitespace = true;
  std::string original_text;
  std::string full_response;
  absl::Status error_status = absl::OkStatus();
};

absl::StatusOr<std::unique_ptr<TextProofreaderRunner>>
TextProofreaderRunner::Create(std::unique_ptr<TextProofreaderOptions> options) {
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

  litert_lm_engine_settings_set_prefill_chunk_size(settings, kPrefillChunkSize);

  litert_lm_engine_settings_set_parallel_file_section_loading(
      settings, options->parallel_file_section_loading);

#ifdef __EMSCRIPTEN__
  litert_lm_engine_settings_set_single_threaded_execution(settings, true);
#endif

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

  int max_num_tokens = options->max_num_tokens > 0 ? options->max_num_tokens
                                                   : kDefaultMaxNumTokens;
  return absl::WrapUnique(new TextProofreaderRunner(
      TextProofreaderRunner::LiteRtLmEnginePtr(engine), max_num_tokens));
}

TextProofreaderRunner::TextProofreaderRunner(LiteRtLmEnginePtr engine,
                                             int max_num_tokens)
    : engine_(std::move(engine)), max_num_tokens_(max_num_tokens) {}

TextProofreaderRunner::~TextProofreaderRunner() { Close().IgnoreError(); }

absl::Status TextProofreaderRunner::RunStreaming(
    absl::string_view text,
    absl::AnyInvocable<void(absl::StatusOr<absl::string_view>, bool)>
        result_callback) {
  {
    absl::MutexLock lock(mutex_);
    cancelled_ = false;
  }

  ABSL_ASSIGN_OR_RETURN(LiteRtLmConversation * conversation_ptr,
                        ResetAndCreateConversation(text));

  auto* context = new StreamContext();
  context->runner = this;
  context->user_callback = std::move(result_callback);
  context->trim_leading_whitespace = true;
  context->original_text = std::string(text);

  std::string input_msg = CreateConversationTextUserMessage(text);

  int status = litert_lm_conversation_send_message_stream(
      conversation_ptr, input_msg.c_str(), /*extra_context=*/nullptr,
      /*visual_token_budget=*/nullptr,
      &TextProofreaderRunner::StaticStreamCallback, context);

  if (status != kLiteRtLmStatusOk) {
    // LiteRtLmStatusCode values are identical to absl::StatusCode values.
    const char* message = litert_lm_get_last_error_message();
    absl::Status err(
        static_cast<absl::StatusCode>(status),
        absl::StrCat("LiteRT LM engine error: ", message ? message : ""));
    DeliverErrorAndCleanup(context, err);
    return err;
  }

#ifdef __EMSCRIPTEN__
  if (litert_lm_conversation_wait_until_done(conversation_ptr) !=
      kLiteRtLmStatusOk) {
    const char* message = litert_lm_get_last_error_message();
    return absl::InternalError(
        absl::StrCat("Failed to wait for the conversation to finish: ",
                     message ? message : ""));
  }
#endif

  return absl::OkStatus();
}

void TextProofreaderRunner::Cancel() {
  absl::MutexLock lock(mutex_);
  cancelled_ = true;
  if (conversation_) {
    litert_lm_conversation_cancel_process(conversation_.get());
  }
}

absl::Status TextProofreaderRunner::Close() {
  {
    absl::MutexLock lock(mutex_);
    if (!engine_) {
      return absl::OkStatus();
    }

    conversation_.reset();
  }
  engine_.reset();
  return absl::OkStatus();
}

absl::StatusOr<LiteRtLmConversation*>
TextProofreaderRunner::ResetAndCreateConversation(absl::string_view text) {
  {
    absl::MutexLock lock(mutex_);
    if (cancelled_) {
      return absl::CancelledError("Session was cancelled.");
    }
  }
  ABSL_ASSIGN_OR_RETURN(auto new_conversation, CreateConversation(text));

  LiteRtLmConversation* raw_conv = new_conversation.get();
  {
    absl::MutexLock lock(mutex_);
    conversation_.swap(new_conversation);
  }
  return raw_conv;
}

absl::StatusOr<TextProofreaderRunner::LiteRtLmConversationPtr>
TextProofreaderRunner::CreateConversation(absl::string_view text) {
  if (!engine_) {
    return absl::InternalError("Engine is not initialized.");
  }

  LiteRtLmConversation* raw_conversation = nullptr;
  if (litert_lm_conversation_create(engine_.get(), nullptr,
                                    &raw_conversation) != kLiteRtLmStatusOk) {
    const char* message = litert_lm_get_last_error_message();
    return absl::InternalError(absl::StrCat("Failed to create conversation. ",
                                            message ? message : ""));
  }
  return TextProofreaderRunner::LiteRtLmConversationPtr(raw_conversation);
}

void TextProofreaderRunner::StaticStreamCallback(
    void* callback_data, const LiteRtLmStreamChunk* chunk) {
  auto* context = reinterpret_cast<StreamContext*>(callback_data);
  const char* text = nullptr;
  bool is_final = false;
  const char* error_msg = nullptr;
  if (litert_lm_stream_chunk_get_text(chunk, &text) != kLiteRtLmStatusOk ||
      litert_lm_stream_chunk_is_final(chunk, &is_final) != kLiteRtLmStatusOk ||
      litert_lm_stream_chunk_get_error(chunk, &error_msg) !=
          kLiteRtLmStatusOk) {
    context->runner->HandleStreamCallback(context, absl::string_view(),
                                          /*is_final=*/true,
                                          "Failed to read stream chunk.");
    return;
  }
  context->runner->HandleStreamCallback(
      context, text ? absl::string_view(text) : absl::string_view(), is_final,
      error_msg ? absl::string_view(error_msg) : absl::string_view());
}

void TextProofreaderRunner::HandleStreamCallback(StreamContext* context,
                                                 absl::string_view chunk,
                                                 bool is_final,
                                                 absl::string_view error_msg) {
  if (!error_msg.empty()) {
    if (context->error_status.ok()) {
      context->error_status = absl::InternalError(error_msg);
    }
    is_final = true;
  } else if (!chunk.empty() && context->error_status.ok()) {
    auto processed_chunk =
        PostProcessResponse(chunk, context->trim_leading_whitespace);
    if (!processed_chunk.ok()) {
      context->error_status = processed_chunk.status();
      is_final = true;
    } else {
      std::string output_chunk = *processed_chunk;
      if (!output_chunk.empty()) {
        context->trim_leading_whitespace = false;
        absl::StrAppend(&context->full_response, output_chunk);
        if (context->user_callback) {
          context->user_callback(output_chunk, /*is_final=*/false);
        }
      }
    }
  }

  if (!is_final && context->error_status.ok() &&
      context->full_response.length() >
          context->original_text.length() * kMinEchoLoopDetectRatio &&
      mediapipe::tasks::text::utils::HasRepeatingSuffix(
          kMinEchoLoopMinChars, kMinEchoLoopNumRepetitions,
          context->full_response)) {
    Cancel();
    context->error_status =
        absl::InternalError("Low quality output (repeated output detected).");
  }

  if (is_final) {
    std::string appended_punct = "";
    if (context->error_status.ok()) {
      appended_punct = GetTrailingPunctuation(context->original_text,
                                              context->full_response);
    }

    auto user_callback = std::move(context->user_callback);
    absl::Status final_error = context->error_status;
    delete context;
    if (user_callback) {
      if (!final_error.ok()) {
        user_callback(final_error, true);
      } else {
        user_callback(appended_punct, true);
      }
    }
  }
}

void TextProofreaderRunner::DeliverErrorAndCleanup(StreamContext* context,
                                                   absl::Status status) {
  auto user_callback = std::move(context->user_callback);
  delete context;
  if (user_callback) {
    user_callback(status, /*is_final=*/true);
  }
}

}  // namespace mediapipe::tasks::text::text_proofreader
