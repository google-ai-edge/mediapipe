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

#ifndef MEDIAPIPE_TASKS_C_TEXT_TEXT_SUMMARIZER_TEXT_SUMMARIZER_H_
#define MEDIAPIPE_TASKS_C_TEXT_TEXT_SUMMARIZER_TEXT_SUMMARIZER_H_

#include <stdbool.h>

#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/core/mp_status.h"

#ifndef MP_EXPORT
#if defined(_MSC_VER)
#define MP_EXPORT __declspec(dllexport)
#else
#define MP_EXPORT __attribute__((visibility("default")))
#endif  // _MSC_VER
#endif  // MP_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

// The opaque pointer to a MediaPipe TextSummarizer task.
typedef struct MpTextSummarizerInternal* MpTextSummarizerPtr;

enum MpTextSummarizerMode {
  MP_TEXT_SUMMARIZER_MODE_TLDR = 0,
  MP_TEXT_SUMMARIZER_MODE_KEYPOINTS = 1,
};

// The options for configuring a MediaPipe TextSummarizer task.
struct MpTextSummarizerOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  struct MpBaseOptions base_options;

  // The mode of the TextSummarizer task.
  enum MpTextSummarizerMode mode;

  // The maximum number of tokens for TextSummarizer task. If set, the
  // summarization will be truncated if the input and output exceed this value.
  //
  // If not set, then the default max_num_tokens is roughly 4000 tokens
  // due to the model's capacity.
  int max_num_tokens;

  // Cache directory for the model weight cache and program cache.
  // By default, it will be set by LiteRT-LM engine based on the model file
  // path.
  //
  // To disable caching, set this to ":nocache".
  char* cache_dir;

  // Minimum logging severity (default 4 = ERROR).
  int min_log_severity;
};

// The result of a summarization operation.
// The caller is responsible for calling MpTextSummarizerCloseResult() on the
// result populated by MpTextSummarizerSummarize() to free internal resources.
struct MpTextSummarizerResult {
  // The generated summary text.
  char* summary;
};

// The result of a streaming summarization operation.
// Results provided by the MpTextSummarizerSummarizeStreaming() callback are
// managed by the API and don't need to be explicitly freed.
struct MpTextSummarizerStreamResult {
  // The generated summary text.
  char* chunk;
  // Whether the summarization is done.
  bool done;
};

// Callback function for streaming summarization.
typedef void (*MpTextSummarizerCallback)(
    void* user_data, const struct MpTextSummarizerStreamResult* result,
    const char* error_msg);

// Creates a TextSummarizer from the provided `options`.
// If successful, returns `kMpOk` and sets `*summarizer` to the new
// `MpTextSummarizerPtr`.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus
MpTextSummarizerCreate(const struct MpTextSummarizerOptions* options,
                       MpTextSummarizerPtr* summarizer, char** error_msg);

// Performs summarization on the provided `input_text`.
// If successful, returns `kMpOk` and sets `*result` to the new
// `MpTextSummarizerResult`.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpTextSummarizerSummarize(
    MpTextSummarizerPtr summarizer, const char* input_text,
    struct MpTextSummarizerResult* result, char** error_msg);

// Performs summarization on the provided `input_text` and returns the result
// incrementally via the callback.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpTextSummarizerSummarizeStreaming(
    MpTextSummarizerPtr summarizer, const char* input_text,
    MpTextSummarizerCallback callback, void* user_data, char** error_msg);

// Frees the memory allocated inside a `MpTextSummarizerResult` result.
// Does not free the result pointer itself.
MP_EXPORT void MpTextSummarizerCloseResult(
    struct MpTextSummarizerResult* result);

// Shuts down the TextSummarizer when all the work is done. Frees all memory.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpTextSummarizerClose(MpTextSummarizerPtr summarizer,
                                         char** error_msg);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MEDIAPIPE_TASKS_C_TEXT_TEXT_SUMMARIZER_TEXT_SUMMARIZER_H_
