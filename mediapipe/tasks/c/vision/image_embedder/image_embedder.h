/* Copyright 2023 The MediaPipe Authors.

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

#ifndef MEDIAPIPE_TASKS_C_VISION_IMAGE_EMBEDDER_IMAGE_EMBEDDER_H_
#define MEDIAPIPE_TASKS_C_VISION_IMAGE_EMBEDDER_IMAGE_EMBEDDER_H_

#include <stdint.h>

#include "mediapipe/tasks/c/components/containers/embedding_result.h"
#include "mediapipe/tasks/c/components/processors/embedder_options.h"
#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"

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

typedef struct MpImageEmbedderInternal* MpImageEmbedderPtr;
typedef struct MpEmbeddingResult ImageEmbedderResult;
typedef struct MpEmbeddingResult MpImageEmbedderResult;

// The user-defined result callback for processing live stream data. The result
// callback should only be specified when the running mode is set to
// MP_RUNNING_MODE_LIVE_STREAM.
//
// Arguments of the callback function:
//   - `user_data`: the `result_callback_user_data` of the options that the
//     image embedder was created with, passed through unchanged.
//   - `status`: `kMpOk` if a result is available, or the error otherwise.
//   - `error_msg`: the message of the error, or null if `status` is `kMpOk`.
//   - `result`: the result, or null if `status` is not `kMpOk`.
//   - `image`: the image that the result was obtained on. If `status` is not
//     `kMpOk`, an empty image that must not be used.
//   - `timestamp_ms`: the timestamp of that image. Unspecified if `status` is
//     not `kMpOk`.
//
// `error_msg`, `result` and `image` are only valid for the lifetime of the
// callback and must not be freed by the caller (do not pass `error_msg` to
// `MpErrorFree()`). The callback is invoked on a thread of the image embedder
// and must not call `MpImageEmbedderClose()`.
typedef void (*MpImageEmbedderResultCallbackFn)(
    void* user_data, MpStatus status, const char* error_msg,
    const ImageEmbedderResult* result, MpImagePtr image, int64_t timestamp_ms);

// The options for configuring a MediaPipe image embedder task.
//
// The struct has no default values, in C or in C++: a field that is never
// written is indeterminate. Zero-initialize the struct (`= {0}` in C, `= {}`
// in C++) and then set its fields; the suggested defaults are listed below.
// Zero is not a valid `running_mode`, so that field must always be set.
typedef struct MpImageEmbedderOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  struct MpBaseOptions base_options;

  // The running mode of the task. Suggested default: MP_RUNNING_MODE_IMAGE.
  // Image embedder has three running modes:
  // 1) The image mode for embedding image on single image inputs.
  // 2) The video mode for embedding image on the decoded frames of a video.
  // 3) The live stream mode for embedding image on the live stream of input
  // data, such as from camera. In this mode, the "result_callback" below must
  // be specified to receive the embedding results asynchronously.
  enum MpRunningMode running_mode;

  // Options for configuring the embedder behavior, such as l2_normalize and
  // quantize.
  struct MpEmbedderOptions embedder_options;

  // The user-defined result callback for processing live stream data. The
  // result callback should only be specified when the running mode is set to
  // MP_RUNNING_MODE_LIVE_STREAM. See
  // `MpImageEmbedderResultCallbackFn` for details.
  MpImageEmbedderResultCallbackFn result_callback;

  // The pointer that is passed as `user_data` to every invocation of
  // `result_callback`. Can be null. The image embedder neither dereferences nor
  // frees it. It must remain valid until `MpImageEmbedderClose()` returns.
  void* result_callback_user_data;
} MpImageEmbedderOptions;

// Creates an ImageEmbedder from the provided `options`.
// Returns kMpOk on success and sets `embedder` to the new ImageEmbedder.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpImageEmbedderCreate(struct MpImageEmbedderOptions* options,
                                         MpImageEmbedderPtr* embedder,
                                         char** error_msg);

// Performs embedding extraction on the input `image`.
//
// Returns kMpOk on success and sets `result` to the embedding result.
// You must free its memory by calling `MpImageEmbedderCloseResult`.
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpImageEmbedderEmbedImage(
    MpImageEmbedderPtr embedder, MpImagePtr image,
    const MpImageProcessingOptions* image_processing_options,
    ImageEmbedderResult* result, char** error_msg);

// Performs embedding extraction on the provided video frame.
// Only use this method when the ImageEmbedder is created with the video
// running mode.
// The image can be of any size with format RGB or RGBA. It's required to
// provide the video frame's timestamp (in milliseconds). The input timestamps
// must be monotonically increasing.
//
// Returns kMpOk on success and sets `result` to the embedding result.
// You must free its memory by calling `MpImageEmbedderCloseResult`.
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpImageEmbedderEmbedForVideo(
    MpImageEmbedderPtr embedder, MpImagePtr image,
    const MpImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, ImageEmbedderResult* result, char** error_msg);

// Sends live image data to embedder, and the results will be available via
// the `result_callback` provided in the ImageEmbedderOptions.
// Only use this method when the ImageEmbedder is created with the live
// stream running mode.
// The image can be of any size with format RGB or RGBA. It's required to
// provide a timestamp (in milliseconds) to indicate when the input image is
// sent to the object detector. The input timestamps must be monotonically
// increasing.
// See `MpImageEmbedderResultCallbackFn` for the arguments that
// `result_callback` receives.
//
// Returns kMpOk on success. To obtain a detailed error, `error_msg` must be
// non-null pointer to a `char*`, which will be populated with a newly-allocated
// error message upon failure. It's the caller responsibility to free the error
// message with `MpErrorFree()`.
MP_EXPORT MpStatus MpImageEmbedderEmbedAsync(
    MpImageEmbedderPtr embedder, MpImagePtr image,
    const MpImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, char** error_msg);

// Frees the memory allocated inside a ImageEmbedderResult result.
// Does not free the result pointer itself.
MP_EXPORT void MpImageEmbedderCloseResult(ImageEmbedderResult* result);

// Shuts down and frees the image embedder. In the live stream mode, waits for
// the pending results. Once this function returns, `result_callback` is not
// invoked anymore, and `embedder` is freed and must not be used, even if an
// error is returned.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpImageEmbedderClose(MpImageEmbedderPtr embedder,
                                        char** error_msg);

// Utility function to compute cosine similarity [1] between two embeddings.
// Returns kMpOk on success and sets `similarity` to the computed similarity.
// [1]: https://en.wikipedia.org/wiki/Cosine_similarity
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpImageEmbedderCosineSimilarity(const struct MpEmbedding* u,
                                                   const struct MpEmbedding* v,
                                                   double* similarity,
                                                   char** error_msg);

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_VISION_IMAGE_EMBEDDER_IMAGE_EMBEDDER_H_
