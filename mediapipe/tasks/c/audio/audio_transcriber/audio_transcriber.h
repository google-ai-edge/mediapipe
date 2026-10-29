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

#ifndef MEDIAPIPE_TASKS_C_AUDIO_AUDIO_TRANSCRIBER_AUDIO_TRANSCRIBER_H_
#define MEDIAPIPE_TASKS_C_AUDIO_AUDIO_TRANSCRIBER_AUDIO_TRANSCRIBER_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

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

// The opaque pointer to a MediaPipe AudioTranscriber task.
typedef struct MpAudioTranscriberInternal* MpAudioTranscriberPtr;

// Model architecture backend family for AudioTranscriber.
enum MpAudioTranscriberModelArchitecture {
  MP_AUDIO_TRANSCRIBER_MODEL_ARCHITECTURE_AUTO = 0,
  MP_AUDIO_TRANSCRIBER_MODEL_ARCHITECTURE_OMNI = 1,
};

// The options for configuring a MediaPipe AudioTranscriber task.
struct MpAudioTranscriberOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file or file descriptor, accelerator delegate, etc.
  struct MpBaseOptions base_options;

  // Input audio sample rate in Hz. Defaults to 16000 Hz when 0; negative
  // values are invalid.
  int sample_rate_hz;

  // Whether to emit interim (non-final) transcription segments during
  // streaming.
  bool enable_interim_results;

  // Model architecture family. Defaults to
  // MP_AUDIO_TRANSCRIBER_MODEL_ARCHITECTURE_AUTO (0).
  enum MpAudioTranscriberModelArchitecture model_architecture;

  // Optional system/user prompt override for generative ASR models (nullable).
  const char* prompt;

  // Optional directory path for caching compiled model artifacts (nullable).
  const char* cache_dir;
};

// A single transcribed segment emitted by AudioTranscriber.
struct MpTranscriptionSegment {
  char* transcript;
  bool is_final;
  int64_t from_timestamp_ms;
  int64_t to_timestamp_ms;
};

// The result of an AudioTranscriber invocation.
// The caller is responsible for calling MpAudioTranscriberCloseResult() to free
// the allocated segments and transcript strings.
struct MpAudioTranscriberResult {
  struct MpTranscriptionSegment* segments;
  uint32_t segments_count;
};

// Creates an AudioTranscriber from the provided `options`.
// If successful, returns `kMpOk` and sets `*transcriber` to the new
// `MpAudioTranscriberPtr`.
//
// To obtain a detailed error, `error_msg` must be a non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It is the caller's responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus
MpAudioTranscriberCreate(const struct MpAudioTranscriberOptions* options,
                         MpAudioTranscriberPtr* transcriber, char** error_msg);

// Transcribes a chunk of mono PCM float audio samples in [-1.0, 1.0].
// If successful, returns `kMpOk` and populates `*result`.
MP_EXPORT MpStatus MpAudioTranscriberTranscribe(
    MpAudioTranscriberPtr transcriber, const float* audio_samples,
    size_t num_samples, bool end_of_stream,
    struct MpAudioTranscriberResult* result, char** error_msg);

// Flushes any remaining buffered audio and finalizes the current stream.
// If successful, returns `kMpOk` and populates `*result`.
MP_EXPORT MpStatus MpAudioTranscriberFinishStream(
    MpAudioTranscriberPtr transcriber, struct MpAudioTranscriberResult* result,
    char** error_msg);

// Prewarms the underlying inference engine to reduce first-chunk latency.
MP_EXPORT MpStatus MpAudioTranscriberPrewarm(MpAudioTranscriberPtr transcriber,
                                             char** error_msg);

// Frees the internal segments and transcript strings of `result`.
MP_EXPORT void MpAudioTranscriberCloseResult(
    struct MpAudioTranscriberResult* result);

// Shuts down the AudioTranscriber and frees all owned memory.
MP_EXPORT MpStatus MpAudioTranscriberClose(MpAudioTranscriberPtr transcriber,
                                           char** error_msg);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MEDIAPIPE_TASKS_C_AUDIO_AUDIO_TRANSCRIBER_AUDIO_TRANSCRIBER_H_
