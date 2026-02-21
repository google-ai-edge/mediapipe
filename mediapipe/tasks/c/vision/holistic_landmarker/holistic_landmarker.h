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

#ifndef MEDIAPIPE_TASKS_C_VISION_HOLISTIC_LANDMARKER_HOLISTIC_LANDMARKER_H_
#define MEDIAPIPE_TASKS_C_VISION_HOLISTIC_LANDMARKER_HOLISTIC_LANDMARKER_H_

#include <stdbool.h>

#include <cstdint>

#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/core/common.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/holistic_landmarker/holistic_landmarker_result.h"

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

typedef struct MpHolisticLandmarkerInternal* MpHolisticLandmarkerPtr;

// The options for configuring a MediaPipe holistic landmarker task.
struct HolisticLandmarkerOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  struct BaseOptions base_options;

  // The running mode of the task. Default to the image mode.
  // HolisticLandmarker has three running modes:
  // 1) The image mode for detecting holistic landmarks on single image inputs.
  // 2) The video mode for detecting holistic landmarks on the decoded frames of
  //    a video.
  // 3) The live stream mode for detecting holistic landmarks on the live stream
  //    of input data, such as from camera. In this mode, the "result_callback"
  //    below must be specified to receive the detection results asynchronously.
  RunningMode running_mode = RunningMode::IMAGE;

  // The minimum confidence score for the face detection to be considered
  // successful.
  float min_face_detection_confidence = 0.5f;

  // The minimum threshold for the face suppression score in the face detection.
  float min_face_suppression_threshold = 0.3f;

  // The minimum confidence score of face presence score in the face landmark
  // detection.
  float min_face_presence_confidence = 0.5f;

  // The minimum confidence score of hand presence score in the hand landmark
  // detection.
  float min_hand_landmarks_confidence = 0.5f;

  // The minimum confidence score for the pose detection to be considered
  // successful.
  float min_pose_detection_confidence = 0.5f;

  // The minimum threshold for the pose suppression score in the pose detection.
  float min_pose_suppression_threshold = 0.3f;

  // The minimum confidence score of pose presence score in the pose landmark
  // detection.
  float min_pose_presence_confidence = 0.5f;

  // Whether to output face blendshapes classification. Face blendshapes are
  // used for rendering animations of the face.
  bool output_face_blendshapes = false;

  // Whether to output segmentation masks.
  bool output_pose_segmentation_masks = false;

  // The user-defined result callback for processing live stream data.
  // The result callback should only be specified when the running mode is set
  // to RunningMode::LIVE_STREAM. Arguments of the callback function include:
  // the status code, the pointer to recognition result, the image that result
  // was obtained on, the timestamp relevant to recognition results and pointer
  // to error message in case of any failure. The data passed to the callback is
  // only valid for the lifetime of the callback and must not be freed by the
  // user.
  typedef void (*result_callback_fn)(MpStatus status,
                                     const HolisticLandmarkerResult* result,
                                     MpImagePtr image, int64_t timestamp_ms);
  result_callback_fn result_callback;
};

// Creates an HolisticLandmarker from the provided `options`.
//
// If successful, returns `kMpOk` and sets `*landmarker` to the new
// `MpHolisticLandmarkerPtr`. To obtain a detailed error, `error_msg` must be
// non-null pointer to a `char*`, which will be populated with a newly-allocated
// error message upon failure. It's the caller responsibility to free the error
// message with `MpErrorFree()`.
MP_EXPORT MpStatus MpHolisticLandmarkerCreate(
    struct HolisticLandmarkerOptions* options,
    MpHolisticLandmarkerPtr* landmarker, char** error_msg);

// Performs holistic landmark detection on the input `image`.
// If successful, returns `kMpOk` and sets `*result` to the new
// `HolisticLandmarkerResult`. To obtain a detailed error, `error_msg` must be
// non-null pointer to a `char*`, which will be populated with a newly-allocated
// error message upon failure. It's the caller responsibility to free the error
// message with `MpErrorFree()`.
MP_EXPORT MpStatus MpHolisticLandmarkerDetectImage(
    MpHolisticLandmarkerPtr landmarker, MpImagePtr image,
    const struct ImageProcessingOptions* options,
    HolisticLandmarkerResult* result, char** error_msg);

// Performs holistic landmark detection on the provided video frame.
// Only use this method when the HolisticLandmarker is created with the video
// running mode.
// The image can be of any size with format RGB or RGBA. It's required to
// provide the video frame's timestamp (in milliseconds). The input timestamps
// must be monotonically increasing.
// You need to invoke `MpHolisticLandmarkerCloseResult` after each invocation
// to free memory.
//
// Returns `kMpOk` on success. To obtain a detailed error, error_msg must be
// non-null pointer to a char*, which will be populated with a newly-allocated
// error message upon failure. It's the caller responsibility to free the error
// message with `MpErrorFree()`.
MP_EXPORT MpStatus MpHolisticLandmarkerDetectForVideo(
    MpHolisticLandmarkerPtr landmarker, MpImagePtr image,
    const struct ImageProcessingOptions* options, int64_t timestamp_ms,
    HolisticLandmarkerResult* result, char** error_msg);

// Sends live image data to holistic landmark detection, and the results will be
// available via the `result_callback` provided in the
// HolisticLandmarkerOptions. Only use this method when the HolisticLandmarker
// is created with the live stream running mode. The image can be of any size
// with format RGB or RGBA. It's required to provide a timestamp (in
// milliseconds) to indicate when the input image is sent to the holistic
// landmarker. The input timestamps must be monotonically increasing. The
// `result_callback` provides:
//   - The recognition results as an HolisticLandmarkerResult object.
//   - The const reference to the corresponding input image that the holistic
//     landmarker runs on. Note that the const reference to the image will no
//     longer be valid when the callback returns. To access the image data
//     outside of the callback, callers need to make a copy of the image.
//   - The input timestamp in milliseconds.
//
// Returns `kMpOk` on success. To obtain a detailed error, error_msg must be
// non-null pointer to a char*, which will be populated with a newly-allocated
// error message upon failure. It's the caller responsibility to free the error
// message with MpErrorFree().
MP_EXPORT MpStatus MpHolisticLandmarkerDetectAsync(
    MpHolisticLandmarkerPtr landmarker, MpImagePtr image,
    const struct ImageProcessingOptions* options, int64_t timestamp_ms,
    char** error_msg);

// Frees the memory allocated inside a HolisticLandmarkerResult result.
// Does not free the result pointer itself.
MP_EXPORT void MpHolisticLandmarkerCloseResult(
    HolisticLandmarkerResult* result);

// Frees holistic landmarker.
// Returns `kMpOk` on success. To obtain a detailed error, error_msg must be
// non-null pointer to a char*, which will be populated with a newly-allocated
// error message upon failure. It's the caller responsibility to free the error
// message with MpErrorFree().
MP_EXPORT MpStatus MpHolisticLandmarkerClose(MpHolisticLandmarkerPtr landmarker,
                                             char** error_msg);

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_VISION_HOLISTIC_LANDMARKER_HOLISTIC_LANDMARKER_H_
