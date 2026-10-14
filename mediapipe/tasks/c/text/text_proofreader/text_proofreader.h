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

#ifndef MEDIAPIPE_TASKS_C_TEXT_TEXT_PROOFREADER_TEXT_PROOFREADER_H_
#define MEDIAPIPE_TASKS_C_TEXT_TEXT_PROOFREADER_TEXT_PROOFREADER_H_
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

// The opaque pointer to a MediaPipe TextProofreader task.
typedef struct MpTextProofreaderInternal* MpTextProofreaderPtr;

// The options for configuring a MediaPipe text proofreader task.
struct MpTextProofreaderOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  struct MpBaseOptions base_options;

  // The maximum number of tokens for proofreading tasks. If set, the
  // proofread output text will be truncated if the input and output exceed this
  // value.
  //
  // If not set, then the default max_num_tokens is roughly 4000 tokens
  // due to the model's capacity.
  int max_num_tokens;

  // Cache directory for the model weight cache and program cache.
  // By default, it will be set by LiteRT-LM engine based on the model file
  // path.
  //
  // To disable caching, set this to ":nocache".
  const char* cache_dir;

  // Minimum logging severity (default 4 = ERROR).
  int min_log_severity;
};

// The type of correction.
enum MpCorrectionType {
  // The text is the same in the original and proofread text.
  MP_CORRECTION_TYPE_SAME = 0,
  // The text was inserted in the proofread text.
  MP_CORRECTION_TYPE_INSERTION = 1,
  // The text was deleted from the original text.
  MP_CORRECTION_TYPE_DELETION = 2,
};

// A single correction.
struct MpCorrection {
  // The type of correction.
  enum MpCorrectionType type;
  // The text associated with the correction.
  const char* text;
};

// The result of a proofreading operation.
// The caller owns the memory allocated for MpTextProofreaderResult, and is
// responsible for calling MpTextProofreaderCloseResult() to free the result and
// its internal resources.
struct MpTextProofreaderResult {
  // The proofread text with corrections applied
  const char* proofread_text;

  // Count of elements in the array pointed to by the `corrections` field
  int corrections_count;

  // List of unchanged, deleted, and inserted text segments.
  // Concatenating all unchanged and deleted segments in array order gives the
  // original text. Concatenating all unchanged and inserted segments gives the
  // proofread text.
  struct MpCorrection* corrections;
};

// The result of a streaming proofreading operation.
// Results provided by the MpTextProofreaderProofreadStreaming() callback are
// managed by the API and don't need to be explicitly freed.
struct MpTextProofreaderStreamResult {
  // The proofread text chunk.
  // done=false: The chunk is a intermediate partial proofread text.
  // done=true: The chunk is the final proofread text.
  const char* chunk;

  // Count of elements in the array pointed to by the `corrections` field
  // This is only populated when done=true.
  int corrections_count;

  // List of unchanged, deleted, and inserted text segments.
  // Concatenating all unchanged and deleted segments in array order gives the
  // original text. Concatenating all unchanged and inserted segments gives the
  // proofread text.
  // This is only populated when done=true.
  struct MpCorrection* corrections;

  // Whether the proofreading is done.
  bool done;
};

// Callback function for streaming proofreading.
//
// Arguments:
// - `user_data`: the pointer to user-defined data provided when calling
//   `MpTextProofreaderProofreadStreaming`.
// - `result`: the pointer to the streaming proofreading result.
// - `error_msg`: the pointer to the error message in case of any failure.
//
// The passed arguments are only valid for the lifetime of the callback.
typedef void (*MpTextProofreaderCallback)(
    void* user_data, const struct MpTextProofreaderStreamResult* result,
    const char* error_msg);

// Creates a TextProofreader from the provided `options`.
// If successful, returns `kMpOk` and sets `*proofreader` to the new
// `MpTextProofreaderPtr`.
//
// The caller can free `options` as soon as the function returns.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus
MpTextProofreaderCreate(const struct MpTextProofreaderOptions* options,
                        MpTextProofreaderPtr* proofreader, char** error_msg);

// Performs proofreading on the provided `input_text`.
// If successful, returns `kMpOk` and sets `*result` to the new
// `MpTextProofreaderResult`.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpTextProofreaderProofread(
    MpTextProofreaderPtr proofreader, const char* input_text,
    struct MpTextProofreaderResult** result, char** error_msg);

// Performs proofreading on the provided `input_text` and returns the result
// in streaming mode via the callback.
//
// This function is non-blocking. The provided callback will be invoked on a
// different thread. The caller is responsible for proper synchronization
// (e.g., using `user_data` to notify the original thread that the operation
// has completed).
//
// Note: This function does not support concurrent streaming sessions.
// Calling it while another streaming session is active will be ignored.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpTextProofreaderProofreadStreaming(
    MpTextProofreaderPtr proofreader, const char* input_text,
    MpTextProofreaderCallback callback, void* user_data, char** error_msg);

// Frees the result and its internal resources.
MP_EXPORT void MpTextProofreaderCloseResult(
    struct MpTextProofreaderResult* result);

// Shuts down the TextProofreader when all the work is done. Frees all memory.
//
// The return value is always `kMpOk`.
// The parameter `error_msg` is ignored.
MP_EXPORT MpStatus MpTextProofreaderClose(MpTextProofreaderPtr proofreader,
                                          char** error_msg);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MEDIAPIPE_TASKS_C_TEXT_TEXT_PROOFREADER_TEXT_PROOFREADER_H_
