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
#include <stdint.h>

#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/core/common.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
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

// The user-defined result callback for processing live stream data. The result
// callback should only be specified when the running mode is set to
// MP_RUNNING_MODE_LIVE_STREAM.
//
// Arguments of the callback function:
//   - `user_data`: the `result_callback_user_data` of the options that the
//     holistic landmarker was created with, passed through unchanged.
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
// `MpErrorFree()`). The callback is invoked on a thread of the holistic
// landmarker and must not call `MpHolisticLandmarkerClose()`.
typedef void (*MpHolisticLandmarkerResultCallback)(
    void* user_data, MpStatus status, const char* error_msg,
    const MpHolisticLandmarkerResult* result, MpImagePtr image,
    int64_t timestamp_ms);

// The options for configuring a MediaPipe holistic landmarker task.
//
// The struct has no default values, in C or in C++: a field that is never
// written is indeterminate. Zero-initialize the struct (`= {0}` in C, `= {}`
// in C++) and then set its fields; the suggested defaults are listed below.
// Zero is not a valid `running_mode`, so that field must always be set.
struct MpHolisticLandmarkerOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  struct MpBaseOptions base_options;

  // The running mode of the task. Suggested default: MP_RUNNING_MODE_IMAGE.
  // HolisticLandmarker has three running modes:
  // 1) The image mode for detecting holistic landmarks on single image inputs.
  // 2) The video mode for detecting holistic landmarks on the decoded frames of
  //    a video.
  // 3) The live stream mode for detecting holistic landmarks on the live stream
  //    of input data, such as from camera. In this mode, the "result_callback"
  //    below must be specified to receive the detection results asynchronously.
  enum MpRunningMode running_mode;

  // The minimum confidence score for the face detection to be considered
  // successful.
  // Suggested default: 0.5.
  float min_face_detection_confidence;

  // The minimum threshold for the face suppression score in the face detection.
  // Suggested default: 0.3.
  float min_face_suppression_threshold;

  // The minimum confidence score of face presence score in the face landmark
  // detection.
  // Suggested default: 0.5.
  float min_face_presence_confidence;

  // The minimum confidence score of hand presence score in the hand landmark
  // detection.
  // Suggested default: 0.5.
  float min_hand_landmarks_confidence;

  // The minimum confidence score for the pose detection to be considered
  // successful.
  // Suggested default: 0.5.
  float min_pose_detection_confidence;

  // The minimum threshold for the pose suppression score in the pose detection.
  // Suggested default: 0.3.
  float min_pose_suppression_threshold;

  // The minimum confidence score of pose presence score in the pose landmark
  // detection.
  // Suggested default: 0.5.
  float min_pose_presence_confidence;

  // Whether to output face blendshapes classification. Face blendshapes are
  // used for rendering animations of the face.
  // Suggested default: false.
  bool output_face_blendshapes;

  // Whether to output segmentation masks.
  // Suggested default: false.
  bool output_pose_segmentation_masks;

  // The user-defined result callback for processing live stream data. See
  // `MpHolisticLandmarkerResultCallback` for details.
  MpHolisticLandmarkerResultCallback result_callback;

  // The pointer that is passed as `user_data` to every invocation of
  // `result_callback`. Can be null. The holistic landmarker neither
  // dereferences nor frees it. It must remain valid until
  // `MpHolisticLandmarkerClose()` returns.
  void* result_callback_user_data;
};

// Creates an HolisticLandmarker from the provided `options`.
//
// If successful, returns `kMpOk` and sets `*landmarker` to the new
// `MpHolisticLandmarkerPtr`. To obtain a detailed error, `error_msg` must be
// non-null pointer to a `char*`, which will be populated with a newly-allocated
// error message upon failure. It's the caller responsibility to free the error
// message with `MpErrorFree()`.
MP_EXPORT MpStatus MpHolisticLandmarkerCreate(
    struct MpHolisticLandmarkerOptions* options,
    MpHolisticLandmarkerPtr* landmarker, char** error_msg);

// Performs holistic landmark detection on the input `image`.
// If successful, returns `kMpOk` and sets `*result` to the new
// `MpHolisticLandmarkerResult`. To obtain a detailed error, `error_msg` must be
// non-null pointer to a `char*`, which will be populated with a newly-allocated
// error message upon failure. It's the caller responsibility to free the error
// message with `MpErrorFree()`.
MP_EXPORT MpStatus MpHolisticLandmarkerDetectImage(
    MpHolisticLandmarkerPtr landmarker, MpImagePtr image,
    const struct MpImageProcessingOptions* options,
    MpHolisticLandmarkerResult* result, char** error_msg);

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
    const struct MpImageProcessingOptions* options, int64_t timestamp_ms,
    MpHolisticLandmarkerResult* result, char** error_msg);

// Sends live image data to holistic landmark detection, and the results will be
// available via the `result_callback` provided in the
// MpHolisticLandmarkerOptions. Only use this method when the HolisticLandmarker
// is created with the live stream running mode. The image can be of any size
// with format RGB or RGBA. It's required to provide a timestamp (in
// milliseconds) to indicate when the input image is sent to the holistic
// landmarker. The input timestamps must be monotonically increasing.
// See `MpHolisticLandmarkerResultCallback` for the arguments that
// `result_callback` receives.
//
// Returns `kMpOk` on success. To obtain a detailed error, error_msg must be
// non-null pointer to a char*, which will be populated with a newly-allocated
// error message upon failure. It's the caller responsibility to free the error
// message with MpErrorFree().
MP_EXPORT MpStatus MpHolisticLandmarkerDetectAsync(
    MpHolisticLandmarkerPtr landmarker, MpImagePtr image,
    const struct MpImageProcessingOptions* options, int64_t timestamp_ms,
    char** error_msg);

// Frees the memory allocated inside a MpHolisticLandmarkerResult result.
// Does not free the result pointer itself.
MP_EXPORT void MpHolisticLandmarkerCloseResult(
    MpHolisticLandmarkerResult* result);

// Shuts down and frees the holistic landmarker. In the live stream mode, waits
// for the pending results. Once this function returns, `result_callback` is not
// invoked anymore, and `landmarker` is freed and must not be used, even if an
// error is returned.
//
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
