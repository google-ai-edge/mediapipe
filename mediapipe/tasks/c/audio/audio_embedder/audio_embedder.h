// Copyright 2026 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MEDIAPIPE_TASKS_C_AUDIO_AUDIO_EMBEDDER_AUDIO_EMBEDDER_H_
#define MEDIAPIPE_TASKS_C_AUDIO_AUDIO_EMBEDDER_AUDIO_EMBEDDER_H_

#include "mediapipe/tasks/c/audio/core/common.h"
#include "mediapipe/tasks/c/components/containers/embedding_result.h"
#include "mediapipe/tasks/c/components/processors/embedder_options.h"
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

typedef struct MpAudioEmbedderInternal* MpAudioEmbedderPtr;

// The C representation of a list of audio embedding results.
// The caller must call `MpAudioEmbedderCloseResult` to free the memory.
typedef struct {
  struct MpEmbeddingResult* results;
  int results_count;
} MpAudioEmbedderResult;

// The options for configuring a MediaPipe AudioEmbedder task.
struct MpAudioEmbedderOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the
  // model file or accelerator options.
  struct MpBaseOptions base_options;

  // Options for configuring the embedder behavior, such as l2_normalize
  // and quantize.
  struct MpEmbedderOptions embedder_options;
};

// Creates an AudioEmbedder from the provided `options`.
// The caller is responsible for calling `MpAudioEmbedderClose` to release the
// embedder.
// To obtain a detailed error, error_msg must be non-null pointer to a char*,
// which will be populated with a newly-allocated error message upon failure.
// It's the caller responsibility to free the error message with
// `MpErrorFree()`.
//
// @param options The options for configuring the audio embedder.
// @param embedder_out A pointer to receive the created audio embedder.
// @param error_msg An optional pointer to receive the error message if the
//     creation fails. If set, this will be populated with a newly-allocated
//     error message upon failure. It's the caller responsibility to free the
//     error message with `MpErrorFree()`.
// @return An `MpStatus` indicating success or failure.
MP_EXPORT MpStatus MpAudioEmbedderCreate(struct MpAudioEmbedderOptions* options,
                                         MpAudioEmbedderPtr* embedder_out,
                                         char** error_msg);

// Performs audio embedding extraction on the provided audio clip. Only use
// this method when the AudioEmbedder is created with the audio clips running
// mode.
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
//
// @param embedder The audio embedder instance.
// @param audio_data The audio data to be embedded.
// @param result_out A pointer to receive the embedding result.
// @param error_msg An optional pointer to receive the error message if the
//     creation fails. If set, this will be populated with a newly-allocated
//     error message upon failure. It's the caller responsibility to free the
//     error message with `MpErrorFree()`.
// @return An `MpStatus` indicating success or failure.
MP_EXPORT MpStatus MpAudioEmbedderEmbed(MpAudioEmbedderPtr embedder,
                                        const MpAudioData* audio_data,
                                        MpAudioEmbedderResult* result_out,
                                        char** error_msg);

// Frees the memory allocated inside a MpAudioEmbedderResult result. Does not
// free the result pointer itself.
//
// @param result A pointer to the embedding result.
// @return An `MpStatus` indicating success or failure.
MP_EXPORT void MpAudioEmbedderCloseResult(MpAudioEmbedderResult* result);

// Shuts down the AudioEmbedder when all the work is done. Frees all memory.
//
// @param embedder The audio embedder instance to be closed.
// @param error_msg An optional pointer to receive the error message if the
//     creation fails. If set, this will be populated with a newly-allocated
//     error message upon failure. It's the caller responsibility to free the
//     error message with `MpErrorFree()`.
// @return An `MpStatus` indicating success or failure.
MP_EXPORT MpStatus MpAudioEmbedderClose(MpAudioEmbedderPtr embedder,
                                        char** error_msg);

// Utility function to compute cosine similarity [1] between two embeddings.
// Returns `kMpOk` on success, or an error status if e.g. the embeddings are
// of different types (quantized vs. float), have different sizes, or have a
// an L2-norm of 0.
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
// [1]: https://en.wikipedia.org/wiki/Cosine_similarity
MP_EXPORT MpStatus MpAudioEmbedderCosineSimilarity(const struct MpEmbedding* u,
                                                   const struct MpEmbedding* v,
                                                   double* similarity,
                                                   char** error_msg);

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_AUDIO_AUDIO_EMBEDDER_AUDIO_EMBEDDER_H_
