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

#ifndef MEDIAPIPE_TASKS_C_VISION_OBJECT_DETECTOR_OBJECT_DETECTOR_H_
#define MEDIAPIPE_TASKS_C_VISION_OBJECT_DETECTOR_OBJECT_DETECTOR_H_

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

typedef struct MpObjectDetectorInternal* MpObjectDetectorPtr;
typedef struct MpDetectionResult MpObjectDetectorResult;

// The user-defined result callback for processing live stream data. The result
// callback should only be specified when the running mode is set to
// MP_RUNNING_MODE_LIVE_STREAM.
//
// Arguments of the callback function:
//   - `user_data`: the `result_callback_user_data` of the options that the
//     object detector was created with, passed through unchanged.
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
// `MpErrorFree()`). The callback is invoked on a thread of the object detector
// and must not call `MpObjectDetectorClose()`.
typedef void (*MpObjectDetectorResultCallback)(
    void* user_data, MpStatus status, const char* error_msg,
    const MpObjectDetectorResult* result, const MpImagePtr image,
    int64_t timestamp_ms);

// The options for configuring a MediaPipe object detector task.
//
// The struct has no default values, in C or in C++: a field that is never
// written is indeterminate. Zero-initialize the struct (`= {0}` in C, `= {}`
// in C++) and then set its fields; the suggested defaults are listed below.
// Zero is not a valid `running_mode`, so that field must always be set.
struct MpObjectDetectorOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  struct MpBaseOptions base_options;

  // The running mode of the task. Suggested default: MP_RUNNING_MODE_IMAGE.
  // Object detector has three running modes:
  // 1) The image mode for detecting objects on single image inputs.
  // 2) The video mode for detecting objects on the decoded frames of a video.
  // 3) The live stream mode for detecting objects on the live stream of input
  // data, such as from camera. In this mode, the "result_callback" below must
  // be specified to receive the detection results asynchronously.
  enum MpRunningMode running_mode;

  // The locale to use for display names specified through the TFLite Model
  // Metadata, if any. Defaults to English.
  const char* display_names_locale;

  // The maximum number of top-scored detection results to return. If < 0,
  // all available results will be returned. If 0, an invalid argument error is
  // returned.
  int max_results;

  // Score threshold to override the one provided in the model metadata (if
  // any). Results below this value are rejected.
  float score_threshold;

  // The allowlist of category names. If non-empty, detection results whose
  // category name is not in this set will be filtered out. Duplicate or unknown
  // category names are ignored. Mutually exclusive with category_denylist.
  const char** category_allowlist;
  // The number of elements in the category allowlist.
  uint32_t category_allowlist_count;

  // The denylist of category names. If non-empty, detection results whose
  // category name is in this set will be filtered out. Duplicate or unknown
  // category names are ignored. Mutually exclusive with category_allowlist.
  const char** category_denylist;
  // The number of elements in the category denylist.
  uint32_t category_denylist_count;

  // The user-defined result callback for processing live stream data. The
  // result callback should only be specified when the running mode is set to
  // MP_RUNNING_MODE_LIVE_STREAM. See
  // `MpObjectDetectorResultCallback` for details.
  MpObjectDetectorResultCallback result_callback;

  // The pointer that is passed as `user_data` to every invocation of
  // `result_callback`. Can be null. The object detector neither dereferences
  // nor frees it. It must remain valid until `MpObjectDetectorClose()` returns.
  void* result_callback_user_data;
};

// Creates an ObjectDetector from the provided `options`.
// Returns a pointer to the image detector on success.
// If an error occurs, returns `nullptr` and sets the error parameter to an
// an error message (if `error_msg` is not `nullptr`). You must free the memory
// allocated for the error message.
MP_EXPORT MpStatus
MpObjectDetectorCreate(struct MpObjectDetectorOptions* options,
                       MpObjectDetectorPtr* detector_out, char** error_msg);

// Performs image detection on the input `image`. Returns `0` on success.
// If an error occurs, returns an error code and sets the error parameter to an
// an error message (if `error_msg` is not `nullptr`). You must free the memory
// allocated for the error message.
MP_EXPORT MpStatus
MpObjectDetectorDetectImage(MpObjectDetectorPtr detector, MpImagePtr image,
                            const struct MpImageProcessingOptions* options,
                            MpObjectDetectorResult* result, char** error_msg);

// Performs image detection on the provided video frame.
// Only use this method when the ObjectDetector is created with the video
// running mode.
// The image can be of any size with format RGB or RGBA. It's required to
// provide the video frame's timestamp (in milliseconds). The input timestamps
// must be monotonically increasing.
// If an error occurs, returns an error code and sets the error parameter to an
// an error message (if `error_msg` is not `nullptr`). You must free the memory
// allocated for the error message.
MP_EXPORT MpStatus MpObjectDetectorDetectForVideo(
    MpObjectDetectorPtr detector, MpImagePtr image,
    const struct MpImageProcessingOptions* options, int64_t timestamp_ms,
    MpObjectDetectorResult* result, char** error_msg);

// Sends live image data to image detection, and the results will be
// available via the `result_callback` provided in the MpObjectDetectorOptions.
// Only use this method when the ObjectDetector is created with the live
// stream running mode.
// The image can be of any size with format RGB or RGBA. It's required to
// provide a timestamp (in milliseconds) to indicate when the input image is
// sent to the object detector. The input timestamps must be monotonically
// increasing.
// See `MpObjectDetectorResultCallback` for the arguments that `result_callback`
// receives.
// If an error occurs, returns an error code and sets the error parameter to an
// an error message (if `error_msg` is not `nullptr`). You must free the memory
// allocated for the error message.
MP_EXPORT MpStatus
MpObjectDetectorDetectAsync(MpObjectDetectorPtr detector, MpImagePtr image,
                            const struct MpImageProcessingOptions* options,
                            int64_t timestamp_ms, char** error_msg);

// Frees the memory allocated inside a MpObjectDetectorResult result.
// Does not free the result pointer itself.
MP_EXPORT void MpObjectDetectorCloseResult(MpObjectDetectorResult* result);

// Shuts down and frees the object detector. In the live stream mode, waits for
// the pending results. Once this function returns, `result_callback` is not
// invoked anymore, and `detector` is freed and must not be used, even if an
// error is returned.
//
// If an error occurs, returns an error code and sets the error parameter to an
// an error message (if `error_msg` is not `nullptr`). You must free the memory
// allocated for the error message.
MP_EXPORT MpStatus MpObjectDetectorClose(MpObjectDetectorPtr detector,
                                         char** error_msg);

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_VISION_OBJECT_DETECTOR_OBJECT_DETECTOR_H_
