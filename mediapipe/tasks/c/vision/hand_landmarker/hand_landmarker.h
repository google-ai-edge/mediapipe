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

#ifndef MEDIAPIPE_TASKS_C_VISION_HAND_LANDMARKER_HAND_LANDMARKER_H_
#define MEDIAPIPE_TASKS_C_VISION_HAND_LANDMARKER_HAND_LANDMARKER_H_

#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/c/vision/hand_landmarker/hand_landmarker_result.h"

#ifndef MP_EXPORT
#define MP_EXPORT __attribute__((visibility("default")))
#endif  // MP_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

// The options for configuring a MediaPipe hand landmarker task.
struct HandLandmarkerOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  struct BaseOptions base_options;

  // The running mode of the task. Default to the image mode.
  // HandLandmarker has three running modes:
  // 1) The image mode for recognizing hand landmarks on single image inputs.
  // 2) The video mode for recognizing hand landmarks on the decoded frames of a
  //    video.
  // 3) The live stream mode for recognizing hand landmarks on the live stream
  //    of input data, such as from camera. In this mode, the "result_callback"
  //    below must be specified to receive the detection results asynchronously.
  RunningMode running_mode;

  // The maximum number of hands can be detected by the HandLandmarker.
  int num_hands = 1;

  // The minimum confidence score for the hand detection to be considered
  // successful.
  float min_hand_detection_confidence = 0.5;

  // The minimum confidence score of hand presence score in the hand landmark
  // detection.
  float min_hand_presence_confidence = 0.5;

  // The minimum confidence score for the hand tracking to be considered
  // successful.
  float min_tracking_confidence = 0.5;

  // The user-defined result callback for processing live stream data.
  // The result callback should only be specified when the running mode is set
  // to RunningMode::LIVE_STREAM. Arguments of the callback function include:
  // the pointer to recognition result, the image that result was obtained
  // on, the timestamp relevant to recognition results and pointer to error
  // message in case of any failure. The validity of the passed arguments is
  // true for the lifetime of the callback function.
  //
  // A caller is responsible for closing hand landmarker result.
  typedef void (*result_callback_fn)(const HandLandmarkerResult* result,
                                     const MpImage& image, int64_t timestamp_ms,
                                     char* error_msg);
  result_callback_fn result_callback;
};

// Creates an HandLandmarker from the provided `options`.
// Returns a pointer to the hand landmarker on success.
// If an error occurs, returns `nullptr` and sets the error parameter to an
// an error message (if `error_msg` is not `nullptr`). You must free the memory
// allocated for the error message.
MP_EXPORT void* hand_landmarker_create(struct HandLandmarkerOptions* options,
                                       char** error_msg);

// Performs hand landmark detection on the input `image`. Returns `0` on
// success. If an error occurs, returns an error code and sets the error
// parameter to an an error message (if `error_msg` is not `nullptr`). You must
// free the memory allocated for the error message.
MP_EXPORT int hand_landmarker_detect_image(void* landmarker,
                                           const MpImage& image,
                                           HandLandmarkerResult* result,
                                           char** error_msg);

// Performs hand landmark detection on the provided video frame.
// Only use this method when the HandLandmarker is created with the video
// running mode.
// The image can be of any size with format RGB or RGBA. It's required to
// provide the video frame's timestamp (in milliseconds). The input timestamps
// must be monotonically increasing.
// If an error occurs, returns an error code and sets the error parameter to an
// an error message (if `error_msg` is not `nullptr`). You must free the memory
// allocated for the error message.
MP_EXPORT int hand_landmarker_detect_for_video(void* landmarker,
                                               const MpImage& image,
                                               int64_t timestamp_ms,
                                               HandLandmarkerResult* result,
                                               char** error_msg);

// Sends live image data to hand landmark detection, and the results will be
// available via the `result_callback` provided in the HandLandmarkerOptions.
// Only use this method when the HandLandmarker is created with the live
// stream running mode.
// The image can be of any size with format RGB or RGBA. It's required to
// provide a timestamp (in milliseconds) to indicate when the input image is
// sent to the hand landmarker. The input timestamps must be monotonically
// increasing.
// The `result_callback` provides:
//   - The recognition results as an HandLandmarkerResult object.
//   - The const reference to the corresponding input image that the hand
//     landmarker runs on. Note that the const reference to the image will no
//     longer be valid when the callback returns. To access the image data
//     outside of the callback, callers need to make a copy of the image.
//   - The input timestamp in milliseconds.
// If an error occurs, returns an error code and sets the error parameter to an
// an error message (if `error_msg` is not `nullptr`). You must free the memory
// allocated for the error message.
MP_EXPORT int hand_landmarker_detect_async(void* landmarker,
                                           const MpImage& image,
                                           int64_t timestamp_ms,
                                           char** error_msg);

// Frees the memory allocated inside a HandLandmarkerResult result.
// Does not free the result pointer itself.
MP_EXPORT void hand_landmarker_close_result(HandLandmarkerResult* result);

// Frees hand landmarker.
// If an error occurs, returns an error code and sets the error parameter to an
// an error message (if `error_msg` is not `nullptr`). You must free the memory
// allocated for the error message.
MP_EXPORT int hand_landmarker_close(void* landmarker, char** error_msg);

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_VISION_HAND_LANDMARKER_HAND_LANDMARKER_H_
