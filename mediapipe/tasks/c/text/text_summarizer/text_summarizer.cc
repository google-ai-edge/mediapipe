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

#include "mediapipe/tasks/c/text/text_summarizer/text_summarizer.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"
#include "mediapipe/tasks/cc/text/text_summarizer/text_summarizer.h"

struct MpTextSummarizerInternal {
  std::unique_ptr<::mediapipe::tasks::text::text_summarizer::TextSummarizer>
      instance;
};

namespace mediapipe::tasks::c::text::text_summarizer {

namespace {
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::text::text_summarizer::TextSummarizer;
using ::mediapipe::tasks::text::text_summarizer::TextSummarizerResult;
using ::mediapipe::tasks::text::text_summarizer::TextSummarizerStreamResult;

TextSummarizer* GetCppSummarizer(MpTextSummarizerPtr wrapper) {
  ABSL_CHECK(wrapper != nullptr) << "TextSummarizer is null.";
  return wrapper->instance.get();
}

::mediapipe::tasks::text::text_summarizer::TextSummarizerOptions::Mode
ConvertCModeToCppMode(enum MpTextSummarizerMode c_mode) {
  using CppMode =
      ::mediapipe::tasks::text::text_summarizer::TextSummarizerOptions::Mode;
  switch (c_mode) {
    case MP_TEXT_SUMMARIZER_MODE_TLDR:
      return CppMode::TLDR;
    case MP_TEXT_SUMMARIZER_MODE_KEYPOINTS:
      return CppMode::KEYPOINTS;
    default:
      // Default to MP_TEXT_SUMMARIZER_MODE_KEYPOINTS for unspecified or unknown
      // modes.
      return CppMode::KEYPOINTS;
  }
}

}  // namespace

absl::Status CppTextSummarizerCreate(const MpTextSummarizerOptions* options,
                                     MpTextSummarizerPtr* summarizer) {
  if (!options) {
    return absl::InvalidArgumentError(
        "Failed to create TextSummarizer: Options must not be null.");
  }

  auto cpp_options = std::make_unique<
      ::mediapipe::tasks::text::text_summarizer::TextSummarizerOptions>();

  CppConvertToBaseOptions(options->base_options, &cpp_options->base_options);
  cpp_options->mode = ConvertCModeToCppMode(options->mode);
  cpp_options->max_num_tokens = options->max_num_tokens;
  if (options->cache_dir) {
    cpp_options->cache_dir = options->cache_dir;
  }
  cpp_options->min_log_severity = options->min_log_severity;

  ABSL_ASSIGN_OR_RETURN(auto cpp_summarizer,
                        TextSummarizer::Create(std::move(cpp_options)));

  *summarizer =
      new MpTextSummarizerInternal{.instance = std::move(cpp_summarizer)};
  return absl::OkStatus();
}

absl::Status CppTextSummarizerSummarize(MpTextSummarizerPtr summarizer,
                                        const char* input_text,
                                        MpTextSummarizerResult* result) {
  if (!summarizer) {
    return absl::InvalidArgumentError(
        "Failed to summarize: Summarizer must not be null.");
  }
  if (!input_text) {
    return absl::InvalidArgumentError(
        "Failed to summarize: Input text must not be null.");
  }
  if (!result) {
    return absl::InvalidArgumentError(
        "Failed to summarize: Result pointer must not be null.");
  }
  TextSummarizer* cpp_summarizer = GetCppSummarizer(summarizer);
  ABSL_ASSIGN_OR_RETURN(TextSummarizerResult cpp_result,
                        cpp_summarizer->Summarize(input_text));
  result->summary = strdup(cpp_result.summary.c_str());
  return absl::OkStatus();
}

absl::Status CppTextSummarizerSummarizeStreaming(
    MpTextSummarizerPtr summarizer, const char* input_text,
    MpTextSummarizerCallback callback, void* user_data) {
  if (!summarizer) {
    return absl::InvalidArgumentError(
        "Failed to summarize streaming: Summarizer must not be null.");
  }
  if (!input_text) {
    return absl::InvalidArgumentError(
        "Failed to summarize streaming: Input text must not be null.");
  }
  if (!callback) {
    return absl::InvalidArgumentError(
        "Failed to summarize streaming: Callback must not be null.");
  }
  TextSummarizer* cpp_summarizer = GetCppSummarizer(summarizer);
  return cpp_summarizer->SummarizeStreaming(
      input_text, [callback, user_data](
                      absl::StatusOr<TextSummarizerStreamResult> result_or) {
        if (!result_or.ok()) {
          std::string error_str = result_or.status().ToString();
          callback(user_data, nullptr, error_str.c_str());
        } else {
          const auto& result = result_or.value();
          MpTextSummarizerStreamResult c_result;
          c_result.chunk = const_cast<char*>(
              result.chunk.empty() ? nullptr : result.chunk.c_str());
          c_result.done = result.done;
          callback(user_data, &c_result, nullptr);
        }
      });
}

void CppTextSummarizerCloseResult(MpTextSummarizerResult* result) {
  if (!result) {
    return;
  }
  if (result->summary) {
    free(result->summary);
    result->summary = nullptr;
  }
}

absl::Status CppTextSummarizerClose(MpTextSummarizerPtr summarizer) {
  if (!summarizer) {
    return absl::OkStatus();
  }
  TextSummarizer* cpp_summarizer = GetCppSummarizer(summarizer);
  ABSL_RETURN_IF_ERROR(cpp_summarizer->Close());
  delete summarizer;
  return absl::OkStatus();
}

}  // namespace mediapipe::tasks::c::text::text_summarizer

extern "C" {

MP_EXPORT MpStatus
MpTextSummarizerCreate(const struct MpTextSummarizerOptions* options,
                       MpTextSummarizerPtr* summarizer, char** error_msg) {
  auto status =
      mediapipe::tasks::c::text::text_summarizer::CppTextSummarizerCreate(
          options, summarizer);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpTextSummarizerSummarize(
    MpTextSummarizerPtr summarizer, const char* input_text,
    struct MpTextSummarizerResult* result, char** error_msg) {
  auto status =
      mediapipe::tasks::c::text::text_summarizer::CppTextSummarizerSummarize(
          summarizer, input_text, result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpTextSummarizerSummarizeStreaming(
    MpTextSummarizerPtr summarizer, const char* input_text,
    MpTextSummarizerCallback callback, void* user_data, char** error_msg) {
  auto status = mediapipe::tasks::c::text::text_summarizer::
      CppTextSummarizerSummarizeStreaming(summarizer, input_text, callback,
                                          user_data);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT void MpTextSummarizerCloseResult(
    struct MpTextSummarizerResult* result) {
  mediapipe::tasks::c::text::text_summarizer::CppTextSummarizerCloseResult(
      result);
}

MP_EXPORT MpStatus MpTextSummarizerClose(MpTextSummarizerPtr summarizer,
                                         char** error_msg) {
  auto status =
      mediapipe::tasks::c::text::text_summarizer::CppTextSummarizerClose(
          summarizer);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

}  // extern "C"
