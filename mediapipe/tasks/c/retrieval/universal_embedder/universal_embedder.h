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

#ifndef MEDIAPIPE_TASKS_C_RETRIEVAL_UNIVERSAL_EMBEDDER_UNIVERSAL_EMBEDDER_H_
#define MEDIAPIPE_TASKS_C_RETRIEVAL_UNIVERSAL_EMBEDDER_UNIVERSAL_EMBEDDER_H_

#include <stdbool.h>

#include "mediapipe/tasks/c/audio/core/common.h"
#include "mediapipe/tasks/c/components/containers/embedding_result.h"
#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/vision/core/image.h"

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

typedef struct MpUniversalEmbedderInternal* MpUniversalEmbedderPtr;
typedef struct MpEmbeddingResult MpUniversalEmbedderResult;

// Activation data type for model execution.
enum MpActivationDataType {
  kMpActivationDataTypeDefault = 0,
  kMpActivationDataTypeFloat32 = 1,
  kMpActivationDataTypeFloat16 = 2,
  kMpActivationDataTypeInt16 = 3,
  kMpActivationDataTypeInt8 = 4,
};

// Options for configuring a MediaPipe Universal Embedder task.
struct MpUniversalEmbedderOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  struct MpBaseOptions base_options;

  // Whether to L2-normalize the output embedding vector.
  bool l2_normalize;

  // Optional maximum sequence length (in tokens) for text encoder signatures.
  // Set to 0 to disable or use the default.
  int max_input_length;

  // Optional vision tokens per image for vision encoder signatures.
  // Set to 0 to disable or use the default.
  int vision_tokens_per_image;

  // Activation data type for model execution.
  enum MpActivationDataType activation_data_type;

  // Optional directory path for storing compiled model cache artifacts.
  const char* cache_dir;
};

// Creates a UniversalEmbedder from the provided `options`.
// If successful, returns `kMpOk` and sets `*embedder` to the new
// `MpUniversalEmbedderPtr`.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus
MpUniversalEmbedderCreate(struct MpUniversalEmbedderOptions* options,
                          MpUniversalEmbedderPtr* embedder, char** error_msg);

// Performs embedding extraction on the input `utf8_str` text.
// If successful, returns `kMpOk` and sets `*result` to the new
// `MpUniversalEmbedderResult`.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpUniversalEmbedderEmbedText(
    MpUniversalEmbedderPtr embedder, const char* utf8_str,
    MpUniversalEmbedderResult* result, char** error_msg);

// Performs embedding extraction on the input image bytes.
// If successful, returns `kMpOk` and sets `*result` to the new
// `MpUniversalEmbedderResult`.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpUniversalEmbedderEmbedImage(
    MpUniversalEmbedderPtr embedder, const char* image_bytes,
    int image_bytes_size, MpUniversalEmbedderResult* result, char** error_msg);

// Performs embedding extraction on the input MpImage container.
// If successful, returns `kMpOk` and sets `*result` to the new
// `MpUniversalEmbedderResult`.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpUniversalEmbedderEmbedMpImage(
    MpUniversalEmbedderPtr embedder, MpImagePtr image,
    MpUniversalEmbedderResult* result, char** error_msg);

// Performs embedding extraction on the input audio float samples.
// If successful, returns `kMpOk` and sets `*result` to the new
// `MpUniversalEmbedderResult`.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpUniversalEmbedderEmbedAudio(
    MpUniversalEmbedderPtr embedder, const float* audio_data,
    int audio_data_size, MpUniversalEmbedderResult* result, char** error_msg);

// Performs embedding extraction on the input MpAudioData container.
// If successful, returns `kMpOk` and sets `*result` to the new
// `MpUniversalEmbedderResult`.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpUniversalEmbedderEmbedMpAudioData(
    MpUniversalEmbedderPtr embedder, const struct MpAudioData* audio_data,
    MpUniversalEmbedderResult* result, char** error_msg);

// Frees the memory allocated inside a MpUniversalEmbedderResult result. Does
// not free the result pointer itself.
MP_EXPORT void MpUniversalEmbedderCloseResult(
    MpUniversalEmbedderResult* result);

// Shuts down the UniversalEmbedder when all the work is done. Frees all memory.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpUniversalEmbedderClose(MpUniversalEmbedderPtr embedder,
                                            char** error_msg);

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_RETRIEVAL_UNIVERSAL_EMBEDDER_UNIVERSAL_EMBEDDER_H_
