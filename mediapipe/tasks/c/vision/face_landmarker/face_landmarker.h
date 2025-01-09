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

#ifndef MEDIAPIPE_TASKS_C_VISION_FACE_LANDMARKER_FACE_LANDMARKER_H_
#define MEDIAPIPE_TASKS_C_VISION_FACE_LANDMARKER_FACE_LANDMARKER_H_

#include <stdbool.h>

#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/c/vision/face_landmarker/face_landmarker_result.h"

#ifndef MP_EXPORT
#define MP_EXPORT __attribute__((visibility("default")))
#endif  // MP_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

// The options for configuring a MediaPipe face landmarker task.
struct FaceLandmarkerOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  struct BaseOptions base_options;

  // The running mode of the task. Default to the image mode.
  // FaceLandmarker has three running modes:
  // 1) The image mode for recognizing face landmarks on single image inputs.
  // 2) The video mode for recognizing face landmarks on the decoded frames of a
  //    video.
  // 3) The live stream mode for recognizing face landmarks on the live stream
  //    of input data, such as from camera. In this mode, the "result_callback"
  //    below must be specified to receive the detection results asynchronously.
  RunningMode running_mode;

  // The maximum number of faces can be detected by the FaceLandmarker.
  int num_faces = 1;

  // The minimum confidence score for the face detection to be considered
  // successful.
  float min_face_detection_confidence = 0.5;

  // The minimum confidence score of face presence score in the face landmark
  // detection.
  float min_face_presence_confidence = 0.5;

  // The minimum confidence score for the face tracking to be considered
  // successful.
  float min_tracking_confidence = 0.5;

  // Whether FaceLandmarker outputs face blendshapes classification. Face
  // blendshapes are used for rendering the 3D face model.
  bool output_face_blendshapes = false;

  // Whether FaceLandmarker outputs facial transformation_matrix. Facial
  // transformation matrix is used to transform the face landmarks in canonical
  // face to the detected face, so that users can apply face effects on the
  // detected landmarks.
  bool output_facial_transformation_matrixes = false;

  // The user-defined result callback for processing live stream data.
  // The result callback should only be specified when the running mode is set
  // to RunningMode::LIVE_STREAM. Arguments of the callback function include:
  // the pointer to recognition result, the image that result was obtained
  // on, the timestamp relevant to recognition results and pointer to error
  // message in case of any failure. The validity of the passed arguments is
  // true for the lifetime of the callback function.
  //
  // The passed `image` is only valid for the lifetime of the call. A caller is
  // responsible for closing the face landmarker result.
  typedef void (*result_callback_fn)(FaceLandmarkerResult* result,
                                     const MpImage* image, int64_t timestamp_ms,
                                     char* error_msg);
  result_callback_fn result_callback;
};

// Creates an FaceLandmarker from the provided `options`.
// Returns a pointer to the face landmarker on success.
// If an error occurs, returns `nullptr` and sets the error parameter to an
// an error message (if `error_msg` is not `nullptr`). You must free the memory
// allocated for the error message.
MP_EXPORT void* face_landmarker_create(struct FaceLandmarkerOptions* options,
                                       char** error_msg);

// Performs face landmark detection on the input `image`. Returns `0` on
// success. If an error occurs, returns an error code and sets the error
// parameter to an an error message (if `error_msg` is not `nullptr`). You must
// free the memory allocated for the error message.
MP_EXPORT int face_landmarker_detect_image(void* landmarker,
                                           const MpImage* image,
                                           FaceLandmarkerResult* result,
                                           char** error_msg);

// Performs face landmark detection on the provided video frame.
// Only use this method when the FaceLandmarker is created with the video
// running mode.
// The image can be of any size with format RGB or RGBA. It's required to
// provide the video frame's timestamp (in milliseconds). The input timestamps
// must be monotonically increasing.
// If an error occurs, returns an error code and sets the error parameter to an
// an error message (if `error_msg` is not `nullptr`). You must free the memory
// allocated for the error message.
// You need to invoke `face_landmarker_detect_for_video` after each invocation
// to free memory.
MP_EXPORT int face_landmarker_detect_for_video(void* landmarker,
                                               const MpImage* image,
                                               int64_t timestamp_ms,
                                               FaceLandmarkerResult* result,
                                               char** error_msg);

// Sends live image data to face landmark detection, and the results will be
// available via the `result_callback` provided in the FaceLandmarkerOptions.
// Only use this method when the FaceLandmarker is created with the live
// stream running mode.
// The image can be of any size with format RGB or RGBA. It's required to
// provide a timestamp (in milliseconds) to indicate when the input image is
// sent to the face landmarker. The input timestamps must be monotonically
// increasing.
// The `result_callback` provides:
//   - The recognition results as an FaceLandmarkerResult object.
//   - The const reference to the corresponding input image that the face
//     landmarker runs on. Note that the const reference to the image will no
//     longer be valid when the callback returns. To access the image data
//     outside of the callback, callers need to make a copy of the image.
//   - The input timestamp in milliseconds.
// If an error occurs, returns an error code and sets the error parameter to an
// an error message (if `error_msg` is not `nullptr`). You must free the memory
// allocated for the error message.
MP_EXPORT int face_landmarker_detect_async(void* landmarker,
                                           const MpImage* image,
                                           int64_t timestamp_ms,
                                           char** error_msg);

// Frees the memory allocated inside a FaceLandmarkerResult result.
// Does not free the result pointer itself.
MP_EXPORT void face_landmarker_close_result(FaceLandmarkerResult* result);

// Frees face landmarker.
// If an error occurs, returns an error code and sets the error parameter to an
// an error message (if `error_msg` is not `nullptr`). You must free the memory
// allocated for the error message.
MP_EXPORT int face_landmarker_close(void* landmarker, char** error_msg);

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_VISION_FACE_LANDMARKER_FACE_LANDMARKER_H_
