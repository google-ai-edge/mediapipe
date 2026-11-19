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

#ifndef MEDIAPIPE_TASKS_C_VISION_FACE_DETECTOR_FACE_DETECTOR_H_
#define MEDIAPIPE_TASKS_C_VISION_FACE_DETECTOR_FACE_DETECTOR_H_

#include <stdint.h>

#include "mediapipe/tasks/c/components/containers/detection_result.h"
#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/core/common.h"
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

typedef struct MpFaceDetectorInternal* MpFaceDetectorPtr;
typedef struct MpDetectionResult MpFaceDetectorResult;

// The user-defined result callback for processing live stream data. The result
// callback should only be specified when the running mode is set to
// MP_RUNNING_MODE_LIVE_STREAM.
//
// Arguments of the callback function:
//   - `user_data`: the `result_callback_user_data` of the options that the face
//     detector was created with, passed through unchanged.
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
// `MpErrorFree()`). The callback is invoked on a thread of the face detector
// and must not call `MpFaceDetectorClose()`.
typedef void (*MpFaceDetectorResultCallback)(void* user_data, MpStatus status,
                                             const char* error_msg,
                                             const MpFaceDetectorResult* result,
                                             const MpImagePtr image,
                                             int64_t timestamp_ms);

// The options for configuring a MediaPipe face detector task.
//
// The struct has no default values, in C or in C++: a field that is never
// written is indeterminate. Zero-initialize the struct (`= {0}` in C, `= {}`
// in C++) and then set its fields; the suggested defaults are listed below.
// Zero is not a valid `running_mode`, so that field must always be set.
struct MpFaceDetectorOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  struct MpBaseOptions base_options;

  // The running mode of the task. Suggested default: MP_RUNNING_MODE_IMAGE.
  // Face Detector has three running modes:
  // 1) The image mode for detecting faces on single image inputs.
  // 2) The video mode for detecting faces on the decoded frames of a video.
  // 3) The live stream mode for detecting faces on the live stream of input
  // data, such as from camera. In this mode, the "result_callback" below must
  // be specified to receive the detection results asynchronously.
  enum MpRunningMode running_mode;

  // The minimum confidence score for the face detection to be considered
  // successful.
  // Suggested default: 0.5.
  float min_detection_confidence;

  // The minimum non-maximum-suppression threshold for face detection to be
  // considered overlapped.
  // Suggested default: 0.3.
  float min_suppression_threshold;

  // Callback invoked asynchronously in live stream mode.
  // See `MpFaceDetectorResultCallback` for details.
  MpFaceDetectorResultCallback result_callback;

  // The pointer that is passed as `user_data` to every invocation of
  // `result_callback`. Can be null. The face detector neither dereferences nor
  // frees it. It must remain valid until `MpFaceDetectorClose()` returns.
  void* result_callback_user_data;
};

// Creates an FaceDetector from the provided `options`.
//
// If successful, returns `kMpOk` and sets `*detector` to the new
// `MpFaceDetectorPtr`. To obtain a detailed error, `error_msg` must be non-null
// pointer to a `char*`, which will be populated with a newly-allocated error
// message upon failure. It's the caller responsibility to free the error
// message with `free()`.
MP_EXPORT MpStatus MpFaceDetectorCreate(struct MpFaceDetectorOptions* options,
                                        MpFaceDetectorPtr* detector,
                                        char** error_msg);

// Performs face detection on the input `image`.
//
// If successful, returns `kMpOk` and sets `*result` to the new
// `MpFaceDetectorResult`. To obtain a detailed error, `error_msg` must be
// non-null pointer to a `char*`, which will be populated with a newly-allocated
// error message upon failure. It's the caller responsibility to free the error
// message with `free()`.
MP_EXPORT MpStatus
MpFaceDetectorDetectImage(MpFaceDetectorPtr detector, MpImagePtr image,
                          const struct MpImageProcessingOptions* options,
                          MpFaceDetectorResult* result, char** error_msg);

// Performs face detection on the provided video frame.
// Only use this method when the FaceDetector is created with the video
// running mode.
// The image can be of any size with format RGB or RGBA. It's required to
// provide the video frame's timestamp (in milliseconds). The input timestamps
// must be monotonically increasing.
//
// If successful, returns `kMpOk` and sets `*result` to the new
// `MpFaceDetectorResult`. To obtain a detailed error, error_msg must be
// non-null pointer to a char*, which will be populated with a newly-allocated
// error message upon failure. It's the caller responsibility to free the error
// message with free().
MP_EXPORT MpStatus MpFaceDetectorDetectForVideo(
    MpFaceDetectorPtr detector, MpImagePtr image,
    const struct MpImageProcessingOptions* options, int64_t timestamp_ms,
    MpFaceDetectorResult* result, char** error_msg);

// Sends live image data to face detection, and the results will be
// available via the `result_callback` provided in the MpFaceDetectorOptions.
// Only use this method when the FaceDetector is created with the live
// stream running mode.
// The image can be of any size with format RGB or RGBA. It's required to
// provide a timestamp (in milliseconds) to indicate when the input image is
// sent to the face detector. The input timestamps must be monotonically
// increasing.
// See `MpFaceDetectorResultCallback` for the arguments that `result_callback`
// receives.
//
// Returns `kMpOk` on success. To obtain a detailed error, error_msg must be
// non-null pointer to a char*, which will be populated with a newly-allocated
// error message upon failure. It's the caller responsibility to free the error
// message with free().
MP_EXPORT MpStatus
MpFaceDetectorDetectAsync(MpFaceDetectorPtr detector, MpImagePtr image,
                          const struct MpImageProcessingOptions* options,
                          int64_t timestamp_ms, char** error_msg);

// Frees the memory allocated inside a MpFaceDetectorResult result.
// Does not free the result pointer itself.
MP_EXPORT void MpFaceDetectorCloseResult(MpFaceDetectorResult* result);

// Shuts down and frees the face detector. In the live stream mode, waits for
// the pending results. Once this function returns, `result_callback` is not
// invoked anymore, and `detector` is freed and must not be used, even if an
// error is returned.
//
// Returns `kMpOk` on success. To obtain a detailed error, error_msg must be
// non-null pointer to a char*, which will be populated with a newly-allocated
// error message upon failure. It's the caller responsibility to free the error
// message with free().
MP_EXPORT MpStatus MpFaceDetectorClose(MpFaceDetectorPtr detector,
                                       char** error_msg);

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_VISION_FACE_DETECTOR_FACE_DETECTOR_H_
