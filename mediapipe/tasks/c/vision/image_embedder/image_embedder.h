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

#include <cstdint>

#include "mediapipe/tasks/c/components/containers/embedding_result.h"
#include "mediapipe/tasks/c/components/processors/embedder_options.h"
#include "mediapipe/tasks/c/core/base_options.h"

#ifndef MP_EXPORT
#define MP_EXPORT __attribute__((visibility("default")))
#endif  // MP_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

typedef EmbeddingResult ImageEmbedderResult;

// Supported image formats.
enum ImageFormat {
  UNKNOWN = 0,
  SRGB = 1,
  SRGBA = 2,
  GRAY8 = 3,
  SBGRA = 11  // compatible with Flutter `bgra8888` format.
};

// Supported processing modes.
enum RunningMode {
  IMAGE = 1,
  VIDEO = 2,
  LIVE_STREAM = 3,
};

// Structure to hold image frame.
struct ImageFrame {
  enum ImageFormat format;
  const uint8_t* image_buffer;
  int width;
  int height;
};

// TODO: Add GPU buffer declaration and proccessing logic for it.
struct GpuBuffer {
  int width;
  int height;
};

// The object to contain an image, realizes `OneOf` concept.
struct MpImage {
  enum { IMAGE_FRAME, GPU_BUFFER } type;
  union {
    struct ImageFrame image_frame;
    struct GpuBuffer gpu_buffer;
  };
};

// The options for configuring a MediaPipe image embedder task.
struct ImageEmbedderOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  struct BaseOptions base_options;

  // The running mode of the task. Default to the image mode.
  // Image embedder has three running modes:
  // 1) The image mode for embedding image on single image inputs.
  // 2) The video mode for embedding image on the decoded frames of a video.
  // 3) The live stream mode for embedding image on the live stream of input
  // data, such as from camera. In this mode, the "result_callback" below must
  // be specified to receive the embedding results asynchronously.
  RunningMode running_mode;

  // Options for configuring the embedder behavior, such as l2_normalize and
  // quantize.
  struct EmbedderOptions embedder_options;

  // The user-defined result callback for processing live stream data.
  // The result callback should only be specified when the running mode is set
  // to RunningMode::LIVE_STREAM.
  typedef void (*result_callback_fn)(ImageEmbedderResult*, const MpImage*,
                                     int64_t);
  result_callback_fn result_callback;
};

// Creates an ImageEmbedder from provided `options`.
// Returns a pointer to the image embedder on success.
// If an error occurs, returns `nullptr` and sets the error parameter to an
// an error message (if `error_msg` is not nullptr). You must free the memory
// allocated for the error message.
MP_EXPORT void* image_embedder_create(struct ImageEmbedderOptions* options,
                                      char** error_msg = nullptr);

// Performs embedding extraction on the input `image`. Returns `0` on success.
// If an error occurs, returns an error code and sets the error parameter to an
// an error message (if `error_msg` is not nullptr). You must free the memory
// allocated for the error message.
//
// TODO: Add API for video and live stream processing.
MP_EXPORT int image_embedder_embed_image(void* embedder, const MpImage* image,
                                         ImageEmbedderResult* result,
                                         char** error_msg = nullptr);

// Frees the memory allocated inside a ImageEmbedderResult result.
// Does not free the result pointer itself.
MP_EXPORT void image_embedder_close_result(ImageEmbedderResult* result);

// Frees image embedder.
// If an error occurs, returns an error code and sets the error parameter to an
// an error message (if `error_msg` is not nullptr). You must free the memory
// allocated for the error message.
MP_EXPORT int image_embedder_close(void* embedder, char** error_msg = nullptr);

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_VISION_IMAGE_EMBEDDER_IMAGE_EMBEDDER_H_
