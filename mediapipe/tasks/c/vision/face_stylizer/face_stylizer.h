/* Copyright 2024 The MediaPipe Authors.

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

#ifndef MEDIAPIPE_TASKS_C_VISION_FACE_STYLIZER_FACE_STYLIZER_H_
#define MEDIAPIPE_TASKS_C_VISION_FACE_STYLIZER_FACE_STYLIZER_H_

#include <stdbool.h>

#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/vision/core/common.h"

#ifndef MP_EXPORT
#define MP_EXPORT __attribute__((visibility("default")))
#endif  // MP_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

// The options for configuring a MediaPipe face stylizer task.
struct FaceStylizerOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  struct BaseOptions base_options;
};

// Creates an FaceStylizer from the provided `options`.
// Returns a pointer to the face stylizer on success.
// If an error occurs, returns `nullptr` and sets the error parameter to an
// an error message (if `error_msg` is not `nullptr`). You must free the memory
// allocated for the error message.
MP_EXPORT void* face_stylizer_create(struct FaceStylizerOptions* options,
                                     char** error_msg);

// Performs face stylization on the input `image`. Returns `0` on
// success. If an error occurs, returns an error code and sets the error
// parameter to an an error message (if `error_msg` is not `nullptr`). You must
// free the memory allocated for the error message.
MP_EXPORT int face_stylizer_stylize_image(void* stylizer, const MpImage& image,
                                          MpImage* result, char** error_msg);

// Frees the memory allocated inside a FaceStylizerResult result.
// Does not free the result pointer itself.
MP_EXPORT void face_stylizer_close_result(MpImage* result);

// Frees face stylizer.
// If an error occurs, returns an error code and sets the error parameter to an
// an error message (if `error_msg` is not `nullptr`). You must free the memory
// allocated for the error message.
MP_EXPORT int face_stylizer_close(void* stylizer, char** error_msg);

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_VISION_FACE_STYLIZER_FACE_STYLIZER_H_
