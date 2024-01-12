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

#include "mediapipe/tasks/c/components/containers/detection_result.h"
#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/vision/core/common.h"

#ifndef MP_EXPORT
#define MP_EXPORT __attribute__((visibility("default")))
#endif  // MP_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

typedef DetectionResult ObjectDetectorResult;

// The options for configuring a MediaPipe object detector task.
struct ObjectDetectorOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  struct BaseOptions base_options;

  // The running mode of the task. Default to the image mode.
  // Object detector has three running modes:
  // 1) The image mode for detecting objects on single image inputs.
  // 2) The video mode for detecting objects on the decoded frames of a video.
  // 3) The live stream mode for detecting objects on the live stream of input
  // data, such as from camera. In this mode, the "result_callback" below must
  // be specified to receive the detection results asynchronously.
  RunningMode running_mode;

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

  // The user-defined result callback for processing live stream data.
  // The result callback should only be specified when the running mode is set
  // to RunningMode::LIVE_STREAM. Arguments of the callback function include:
  // the pointer to detection result, the image that result was obtained
  // on, the timestamp relevant to detection results and pointer to error
  // message in case of any failure. The validity of the passed arguments is
  // true for the lifetime of the callback function.
  //
  // A caller is responsible for closing object detector result.
  typedef void (*result_callback_fn)(const ObjectDetectorResult* result,
                                     const MpImage& image, int64_t timestamp_ms,
                                     char* error_msg);
  result_callback_fn result_callback;
};

// Creates an ObjectDetector from the provided `options`.
// Returns a pointer to the image detector on success.
// If an error occurs, returns `nullptr` and sets the error parameter to an
// an error message (if `error_msg` is not `nullptr`). You must free the memory
// allocated for the error message.
MP_EXPORT void* object_detector_create(struct ObjectDetectorOptions* options,
                                       char** error_msg);

// Performs image detection on the input `image`. Returns `0` on success.
// If an error occurs, returns an error code and sets the error parameter to an
// an error message (if `error_msg` is not `nullptr`). You must free the memory
// allocated for the error message.
MP_EXPORT int object_detector_detect_image(void* detector, const MpImage* image,
                                           ObjectDetectorResult* result,
                                           char** error_msg);

// Performs image detection on the provided video frame.
// Only use this method when the ObjectDetector is created with the video
// running mode.
// The image can be of any size with format RGB or RGBA. It's required to
// provide the video frame's timestamp (in milliseconds). The input timestamps
// must be monotonically increasing.
// If an error occurs, returns an error code and sets the error parameter to an
// an error message (if `error_msg` is not `nullptr`). You must free the memory
// allocated for the error message.
MP_EXPORT int object_detector_detect_for_video(void* detector,
                                               const MpImage* image,
                                               int64_t timestamp_ms,
                                               ObjectDetectorResult* result,
                                               char** error_msg);

// Sends live image data to image detection, and the results will be
// available via the `result_callback` provided in the ObjectDetectorOptions.
// Only use this method when the ObjectDetector is created with the live
// stream running mode.
// The image can be of any size with format RGB or RGBA. It's required to
// provide a timestamp (in milliseconds) to indicate when the input image is
// sent to the object detector. The input timestamps must be monotonically
// increasing.
// The `result_callback` provides:
//   - The detection results as an ObjectDetectorResult object.
//   - The const reference to the corresponding input image that the image
//     detector runs on. Note that the const reference to the image will no
//     longer be valid when the callback returns. To access the image data
//     outside of the callback, callers need to make a copy of the image.
//   - The input timestamp in milliseconds.
// If an error occurs, returns an error code and sets the error parameter to an
// an error message (if `error_msg` is not `nullptr`). You must free the memory
// allocated for the error message.
MP_EXPORT int object_detector_detect_async(void* detector, const MpImage* image,
                                           int64_t timestamp_ms,
                                           char** error_msg);

// Frees the memory allocated inside a ObjectDetectorResult result.
// Does not free the result pointer itself.
MP_EXPORT void object_detector_close_result(ObjectDetectorResult* result);

// Frees object detector.
// If an error occurs, returns an error code and sets the error parameter to an
// an error message (if `error_msg` is not `nullptr`). You must free the memory
// allocated for the error message.
MP_EXPORT int object_detector_close(void* detector, char** error_msg);

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_VISION_OBJECT_DETECTOR_OBJECT_DETECTOR_H_
