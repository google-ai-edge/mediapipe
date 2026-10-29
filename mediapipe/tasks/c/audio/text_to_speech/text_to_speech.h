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

#ifndef MEDIAPIPE_TASKS_C_AUDIO_TEXT_TO_SPEECH_TEXT_TO_SPEECH_H_
#define MEDIAPIPE_TASKS_C_AUDIO_TEXT_TO_SPEECH_TEXT_TO_SPEECH_H_

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

// Opaque handle for the MediaPipe TextToSpeech task instance.
typedef struct MpTextToSpeechInternal MpTextToSpeech;
typedef struct MpTextToSpeechInternal* MpTextToSpeechPtr;

// Opaque handle for an active MediaPipe TextToSpeech streaming synthesis
// session.
typedef struct MpTextToSpeechStreamInternal MpTextToSpeechStream;
typedef struct MpTextToSpeechStreamInternal* MpTextToSpeechStreamPtr;

// Voice metadata descriptor returned by MpTextToSpeechListVoices().
typedef struct MpTextToSpeechVoiceInfo {
  const char* voice_id;
  const char* display_name;
  const char* locale;
  int32_t sample_rate_hz;
  bool is_downloaded;
} MpTextToSpeechVoiceInfo;

// Container for voice discovery results. Caller frees with
// MpTextToSpeechVoiceListFree().
typedef struct MpTextToSpeechVoiceList {
  MpTextToSpeechVoiceInfo* voices;
  size_t num_voices;
} MpTextToSpeechVoiceList;

// Task initialization options for MediaPipe TextToSpeech (backed by LiteRT-LM
// Omni Kokoro-82M).
typedef struct MpTextToSpeechOptions {
  // Base options for configuring MediaPipe Tasks, specifying the single
  // .litertlm model file path and hardware delegate (CPU/GPU/NPU).
  struct MpBaseOptions base_options;

  // Optional default voice ID (NULL uses engine default, e.g. "af_heart").
  const char* default_voice_id;

  // Optional default BCP-47 locale (NULL infers from voice metadata).
  const char* default_locale;

  // Speaking rate multiplier in (0.0f, 4.0f] (1.0f = normal speed).
  float speaking_rate;

  // Target sequence length bucket for static-shape Kokoro models (0 = auto).
  int32_t target_bucket;

  // Optional directory for caching unpacked .litertlm assets and compiled
  // XNNPACK/GPU delegate weights.
  const char* cache_directory;
} MpTextToSpeechOptions;

// Per-utterance synthesis overrides passed to MpTextToSpeechSynthesize().
typedef struct MpTextToSpeechSynthesisOptions {
  // Optional voice override for this utterance (NULL uses engine default).
  const char* voice_id;
  // Optional locale override for this utterance (NULL infers from voice).
  const char* locale;
} MpTextToSpeechSynthesisOptions;

// Word-level timing alignment descriptor mapping an original text word to its
// synthesized audio time range in seconds.
typedef struct MpTextToSpeechWordTimestamp {
  // Heap-allocated null-terminated UTF-8 string for the original surface word.
  const char* word;
  // Zero-based byte offset of the start of `word` in the input text.
  int32_t start_char;
  // Zero-based byte offset of the end of `word` (exclusive) in the input text.
  int32_t end_char;
  // Estimated start time in seconds within the synthesized audio waveform.
  float start_sec;
  // Estimated end time in seconds within the synthesized audio waveform.
  float end_sec;
} MpTextToSpeechWordTimestamp;

// Result of a synchronous unary synthesis call (`MpTextToSpeechSynthesize`) or
// a single streaming chunk (`MpTextToSpeechStreamReadNext`).
// Caller frees internal sample memory using `MpTextToSpeechFreeAudioResult()`.
typedef struct MpTextToSpeechAudioResult {
  // Heap-allocated 32-bit mono float PCM audio samples in [-1.0f, 1.0f].
  float* samples;
  // Number of samples in `samples`.
  size_t num_samples;
  // Audio sample rate in Hz (e.g. 24000).
  int32_t sample_rate_hz;
  // Heap-allocated word-level timing alignments for read-along highlighting.
  MpTextToSpeechWordTimestamp* word_timestamps;
  // Number of entries in `word_timestamps`.
  size_t num_word_timestamps;
} MpTextToSpeechAudioResult;

// Push-mode C callback invoked for each synthesized PCM audio chunk.
// - On valid chunk: `status == kMpOk`, `samples` points to `num_samples`
//   float32 values owned by the callee for the duration of the callback,
//   `chunk_index` is the 0-based index, and `is_final` indicates the terminal
//   chunk.
// - On error: `status != kMpOk` and `error_msg` contains a diagnostic string.
// Returning `false` from the callback immediately cancels the in-flight stream.
typedef bool (*MpTextToSpeechStreamCallback)(
    void* user_data, const float* samples, size_t num_samples,
    int32_t sample_rate_hz, int32_t chunk_index, bool is_final, MpStatus status,
    const char* error_msg);

// Populates `options` with default values (`MP_DELEGATE_CPU`,
// `speaking_rate = 1.0f`, `target_bucket = 0`, NULL `default_voice_id` and
// `default_locale`).
MP_EXPORT void MpTextToSpeechOptionsInitDefault(MpTextToSpeechOptions* options);

// Populates `options` with default per-utterance values (NULL `voice_id` and
// `locale`).
MP_EXPORT void MpTextToSpeechSynthesisOptionsInitDefault(
    MpTextToSpeechSynthesisOptions* options);

// Creates a MediaPipe TextToSpeech task instance configured with `options`.
// On error, returns a non-OK `MpStatus` and (if `error_msg != NULL`) populates
// `*error_msg` with a heap-allocated string that must be freed with
// `MpTextToSpeechFreeError()`.
MP_EXPORT MpStatus MpTextToSpeechCreate(const MpTextToSpeechOptions* options,
                                        MpTextToSpeechPtr* tts_out,
                                        char** error_msg);

// Queries available voices for an initialized `MpTextToSpeechPtr` instance.
// Caller must release `voice_list_out` using `MpTextToSpeechVoiceListFree()`.
MP_EXPORT MpStatus MpTextToSpeechListVoices(
    MpTextToSpeechPtr tts, MpTextToSpeechVoiceList* voice_list_out,
    char** error_msg);

// Frees memory allocated by `MpTextToSpeechListVoices()`.
MP_EXPORT void MpTextToSpeechVoiceListFree(MpTextToSpeechVoiceList* voice_list);

// Canonical synchronous unary synthesis.
// Synthesizes `text` on the calling thread and populates `*result_out`.
// `options` may be NULL to use the engine's defaults.
// Caller must free `*result_out` with `MpTextToSpeechFreeAudioResult()`.
MP_EXPORT MpStatus MpTextToSpeechSynthesize(
    MpTextToSpeechPtr tts, const char* text,
    const MpTextToSpeechSynthesisOptions* options,
    MpTextToSpeechAudioResult* result_out, char** error_msg);

// Frees the `samples` and `word_timestamps` buffers inside
// `MpTextToSpeechAudioResult` and resets its fields.
MP_EXPORT void MpTextToSpeechFreeAudioResult(MpTextToSpeechAudioResult* result);

// Computes syllable- and acoustic-RMS-aligned word timestamps for `text`
// across `samples` (`num_samples` at `sample_rate_hz`) and populates
// `*result_out` (`word_timestamps` and `num_word_timestamps`). `samples` may be
// NULL to compute syllable-weighted timestamps from `num_samples` alone.
// Caller frees `*result_out` with `MpTextToSpeechFreeAudioResult()`.
MP_EXPORT MpStatus MpTextToSpeechComputeWordTimestamps(
    const char* text, const float* samples, size_t num_samples,
    int32_t sample_rate_hz, float time_offset_sec,
    MpTextToSpeechAudioResult* result_out, char** error_msg);

// Mode A: Pull-based & bidirectional streaming synthesis.
// Starts a new streaming synthesis session.
// - `initial_text`: Optional initial text to synthesize (may be NULL or "").
// - `options`: Optional per-utterance voice/locale overrides (may be NULL).
// - `finish_input`: Pass `true` for Level 1 (single-utterance text -> streaming
//   PCM chunks). Pass `false` for Level 2 (incremental
//   `MpTextToSpeechStreamPushText` + `MpTextToSpeechStreamFinishInput`).
MP_EXPORT MpStatus MpTextToSpeechStartStream(
    MpTextToSpeechPtr tts, const char* initial_text,
    const MpTextToSpeechSynthesisOptions* options, bool finish_input,
    MpTextToSpeechStreamPtr* stream_out, char** error_msg);

// Pulls the next synthesized PCM chunk synchronously on the calling thread.
// - Returns `kMpOk` and populates `*chunk_out` (`*chunk_index_out`,
//   `*is_final_out` if non-NULL) when a chunk is ready. Caller must free
//   `*chunk_out` with `MpTextToSpeechFreeAudioResult()`.
// - Returns `kMpOutOfRange` (`*num_samples == 0`) when the stream reaches EOS.
// - Returns `kMpCancelled` if cancelled via `MpTextToSpeechStreamCancel()` or
//   `MpTextToSpeechCancel()`.
MP_EXPORT MpStatus MpTextToSpeechStreamReadNext(
    MpTextToSpeechStreamPtr stream, MpTextToSpeechAudioResult* chunk_out,
    int32_t* chunk_index_out, bool* is_final_out, char** error_msg);

// Appends an incremental text fragment (e.g., from an LLM token stream) to an
// open `MpTextToSpeechStreamPtr` created with `finish_input = false`.
MP_EXPORT MpStatus MpTextToSpeechStreamPushText(MpTextToSpeechStreamPtr stream,
                                                const char* text_delta,
                                                char** error_msg);

// Marks the end of incremental text input on `stream`, flushing any remaining
// partial sentence into the synthesis pipeline.
MP_EXPORT MpStatus MpTextToSpeechStreamFinishInput(
    MpTextToSpeechStreamPtr stream, char** error_msg);

// Lock-free cancellation for a specific `MpTextToSpeechStreamPtr`. Safe to call
// from any thread while `MpTextToSpeechStreamReadNext` is blocked on another
// thread.
MP_EXPORT void MpTextToSpeechStreamCancel(MpTextToSpeechStreamPtr stream);

// Closes the `MpTextToSpeechStreamPtr` handle and releases native stream
// resources.
MP_EXPORT void MpTextToSpeechStreamClose(MpTextToSpeechStreamPtr stream);

// Mode B: Push-callback streaming convenience function.
// Synthesizes `text` and invokes `callback` for each generated PCM chunk on the
// calling thread.
MP_EXPORT MpStatus MpTextToSpeechSynthesizeStreamAsync(
    MpTextToSpeechPtr tts, const char* text,
    const MpTextToSpeechSynthesisOptions* options,
    MpTextToSpeechStreamCallback callback, void* user_data, char** error_msg);

// Thread-safe, lock-free barge-in cancellation. May be called concurrently from
// any thread while `MpTextToSpeechSynthesize` or `MpTextToSpeechStreamReadNext`
// is blocked on another thread.
MP_EXPORT void MpTextToSpeechCancel(MpTextToSpeechPtr tts);

// Closes the `MpTextToSpeechPtr` instance and releases all native resources.
MP_EXPORT void MpTextToSpeechClose(MpTextToSpeechPtr tts);

// Frees an error message string returned via `char** error_msg`.
MP_EXPORT void MpTextToSpeechFreeError(char* error_msg);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MEDIAPIPE_TASKS_C_AUDIO_TEXT_TO_SPEECH_TEXT_TO_SPEECH_H_
