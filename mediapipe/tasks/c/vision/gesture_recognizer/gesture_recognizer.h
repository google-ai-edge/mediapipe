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

#ifndef MEDIAPIPE_TASKS_C_VISION_GESTURE_RECOGNIZER_GESTURE_RECOGNIZER_H_
#define MEDIAPIPE_TASKS_C_VISION_GESTURE_RECOGNIZER_GESTURE_RECOGNIZER_H_

#include <stdint.h>

#include "mediapipe/tasks/c/components/processors/classifier_options.h"
#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/core/common.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/gesture_recognizer/gesture_recognizer_result.h"

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

typedef struct MpGestureRecognizerInternal* MpGestureRecognizerPtr;

// The user-defined result callback for processing live stream data. The result
// callback should only be specified when the running mode is set to
// MP_RUNNING_MODE_LIVE_STREAM.
//
// Arguments of the callback function:
//   - `user_data`: the `result_callback_user_data` of the options that the
//     gesture recognizer was created with, passed through unchanged.
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
// `MpErrorFree()`). The callback is invoked on a thread of the gesture
// recognizer and must not call `MpGestureRecognizerClose()`.
typedef void (*MpGestureRecognizerResultCallback)(
    void* user_data, MpStatus status, const char* error_msg,
    const MpGestureRecognizerResult* result, const MpImagePtr image,
    int64_t timestamp_ms);

// The options for configuring a MediaPipe gesture recognizer task.
//
// The struct has no default values, in C or in C++: a field that is never
// written is indeterminate. Zero-initialize the struct (`= {0}` in C, `= {}`
// in C++) and then set its fields; the suggested defaults are listed below.
// Zero is not a valid `running_mode`, so that field must always be set.
struct MpGestureRecognizerOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  struct MpBaseOptions base_options;

  // The running mode of the task. Suggested default: MP_RUNNING_MODE_IMAGE.
  // GestureRecognizer has three running modes:
  // 1) The image mode for recognizing hand gestures on single image inputs.
  // 2) The video mode for recognizing hand gestures on the decoded frames of a
  //    video.
  // 3) The live stream mode for recognizing hand gestures on the live stream of
  //    input data, such as from camera. In this mode, the "result_callback"
  //    below must be specified to receive the detection results asynchronously.
  enum MpRunningMode running_mode;

  // The maximum number of hands can be detected by the GestureRecognizer.
  // Suggested default: 1.
  int num_hands;

  // The minimum confidence score for the hand detection to be considered
  // successful.
  // Suggested default: 0.5.
  float min_hand_detection_confidence;

  // The minimum confidence score of hand presence score in the hand landmark
  // detection.
  // Suggested default: 0.5.
  float min_hand_presence_confidence;

  // The minimum confidence score for the hand tracking to be considered
  // successful.
  // Suggested default: 0.5.
  float min_tracking_confidence;

  // Options for configuring the canned gestures classifier, such as score
  // threshold, allow list and deny list of gestures. The categories for canned
  // gesture classifier are: ["None", "Closed_Fist", "Open_Palm",
  // "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"]
  struct MpClassifierOptions canned_gestures_classifier_options;

  // Options for configuring the custom gestures classifier, such as score
  // threshold, allow list and deny list of gestures.
  struct MpClassifierOptions custom_gestures_classifier_options;

  // The user-defined result callback for processing live stream data. The
  // result callback should only be specified when the running mode is set to
  // MP_RUNNING_MODE_LIVE_STREAM. See
  // `MpGestureRecognizerResultCallback` for details.
  MpGestureRecognizerResultCallback result_callback;

  // The pointer that is passed as `user_data` to every invocation of
  // `result_callback`. Can be null. The gesture recognizer neither dereferences
  // nor frees it. It must remain valid until `MpGestureRecognizerClose()`
  // returns.
  void* result_callback_user_data;
};

// Creates an GestureRecognizer from the provided `options`.
// If successful, returns `kMpOk` and sets `*recognizer` to the new
// `MpGestureRecognizerPtr`.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus
MpGestureRecognizerCreate(const struct MpGestureRecognizerOptions* options,
                          MpGestureRecognizerPtr* recognizer, char** error_msg);

// Performs gesture recognition on the input `image`.
// If successful, returns `kMpOk` and sets `*result` to the new
// `MpGestureRecognizerResult`.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpGestureRecognizerRecognizeImage(
    MpGestureRecognizerPtr recognizer, MpImagePtr image,
    const MpImageProcessingOptions* image_processing_options,
    MpGestureRecognizerResult* result, char** error_msg);

// Performs gesture recognition on the provided video frame.
// Only use this method when the GestureRecognizer is created with the video
// running mode.
// The image can be of any size with format RGB or RGBA. It's required to
// provide the video frame's timestamp (in milliseconds). The input timestamps
// must be monotonically increasing.
// If successful, returns `kMpOk` and sets `*result` to the new
// `MpGestureRecognizerResult`. To obtain a detailed error, `error_msg` must be
// non-null pointer to a `char*`, which will be populated with a newly-allocated
// error message upon failure. It's the caller responsibility to free the error
// message with `MpErrorFree()`.
MP_EXPORT MpStatus MpGestureRecognizerRecognizeForVideo(
    MpGestureRecognizerPtr recognizer, MpImagePtr image,
    const MpImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, MpGestureRecognizerResult* result, char** error_msg);

// Sends live image data to gesture recognition, and the results will be
// available via the `result_callback` provided in the
// MpGestureRecognizerOptions. Only use this method when the GestureRecognizer
// is created with the live stream running mode. The image can be of any size
// with format RGB or RGBA. It's required to provide a timestamp (in
// milliseconds) to indicate when the input image is sent to the gesture
// recognizer. The input timestamps must be monotonically increasing.
// See `MpGestureRecognizerResultCallback` for the arguments that
// `result_callback` receives.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpGestureRecognizerRecognizeAsync(
    MpGestureRecognizerPtr recognizer, MpImagePtr image,
    const MpImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, char** error_msg);

// Frees the memory allocated inside a MpGestureRecognizerResult result.
// Does not free the result pointer itself.
MP_EXPORT void MpGestureRecognizerCloseResult(
    MpGestureRecognizerResult* result);

// Shuts down and frees the gesture recognizer. In the live stream mode, waits
// for the pending results. Once this function returns, `result_callback` is not
// invoked anymore, and `recognizer` is freed and must not be used, even if an
// error is returned.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// 'MpErrorFree()`.
MP_EXPORT MpStatus MpGestureRecognizerClose(MpGestureRecognizerPtr recognizer,
                                            char** error_msg);

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_VISION_GESTURE_RECOGNIZER_GESTURE_RECOGNIZER_H_
