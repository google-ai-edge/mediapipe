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

#ifndef MEDIAPIPE_TASKS_C_VISION_IMAGE_SEGMENTER_IMAGE_SEGMENTER_H_
#define MEDIAPIPE_TASKS_C_VISION_IMAGE_SEGMENTER_IMAGE_SEGMENTER_H_

#include <stdbool.h>
#include <stdint.h>

#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/core/common.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/image_segmenter/image_segmenter_result.h"

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

typedef struct MpImageSegmenterInternal* MpImageSegmenterPtr;

// The user-defined result callback for processing live stream data. The result
// callback should only be specified when the running mode is set to
// MP_RUNNING_MODE_LIVE_STREAM.
//
// Arguments of the callback function:
//   - `user_data`: the `result_callback_user_data` of the options that the
//     image segmenter was created with, passed through unchanged.
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
// `MpErrorFree()`). The callback is invoked on a thread of the image segmenter
// and must not call `MpImageSegmenterClose()`.
typedef void (*MpImageSegmenterResultCallback)(
    void* user_data, MpStatus status, const char* error_msg,
    const MpImageSegmenterResult* result, MpImagePtr image,
    int64_t timestamp_ms);

// The options for configuring a mediapipe image segmenter task.
//
// The struct has no default values, in C or in C++: a field that is never
// written is indeterminate. Zero-initialize the struct (`= {0}` in C, `= {}`
// in C++) and then set its fields; the suggested defaults are listed below.
// Zero is not a valid `running_mode`, so that field must always be set.
struct MpImageSegmenterOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  struct MpBaseOptions base_options;

  // The running mode of the task. Suggested default: MP_RUNNING_MODE_IMAGE.
  // Image segmenter has three running modes:
  // 1) The image mode for segmenting image on single image inputs.
  // 2) The video mode for segmenting image on the decoded frames of a video.
  // 3) The live stream mode for segmenting image on the live stream of input
  // data, such as from camera. In this mode, the "result_callback" below must
  // be specified to receive the segmentation results asynchronously.
  enum MpRunningMode running_mode;

  // The locale to use for display names specified through the TFLite Model
  // Metadata, if any. Defaults to English.
  const char* display_names_locale;

  // Whether to output confidence masks.
  // Suggested default: true.
  bool output_confidence_masks;

  // Whether to output category mask.
  // Suggested default: false.
  bool output_category_mask;

  // The user-defined result callback for processing live stream data. See
  // `MpImageSegmenterResultCallback` for details.
  MpImageSegmenterResultCallback result_callback;

  // The pointer that is passed as `user_data` to every invocation of
  // `result_callback`. Can be null. The image segmenter neither dereferences
  // nor frees it. It must remain valid until `MpImageSegmenterClose()` returns.
  void* result_callback_user_data;
};

// Options for configuring runtime behavior of ImageSegmenter.
struct MpSegmentationOptions {
  // The width of the output segmentation masks.
  int output_width;

  // The height of the output segmentation masks.
  int output_height;
};

// Creates an ImageSegmenter from the provided `options`.
// Returns 'kMpOk' on success and sets `segmenter` to the created
// ImageSegmenter.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus
MpImageSegmenterCreate(struct MpImageSegmenterOptions* options,
                       MpImageSegmenterPtr* segmenter, char** error_msg);

// Performs image segmentation on the input `image`.
// Returns 'kMpOk' on success and sets `result` to the segmentation result.
// You must call `MpImageSegmenterCloseResult` to free its memory.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus
MpImageSegmenterSegmentImage(MpImageSegmenterPtr segmenter, MpImagePtr image,
                             const MpImageProcessingOptions* options,
                             MpImageSegmenterResult* result, char** error_msg);

// Performs image segmentation on the provided video frame.
// Only use this method when the ImageSegmenter is created with the video
// running mode.
// The image can be of any size with format RGB or RGBA. It's required to
// provide the video frame's timestamp (in milliseconds). The input timestamps
// must be monotonically increasing.
// Returns 'kMpOk' on success and sets `result` to the segmentation result.
// You must call `MpImageSegmenterCloseResult` to free its memory.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpImageSegmenterSegmentForVideo(
    MpImageSegmenterPtr segmenter, MpImagePtr image,
    const MpImageProcessingOptions* options, int64_t timestamp_ms,
    MpImageSegmenterResult* result, char** error_msg);

// Sends live image data to image segmentation, and the results will be
// available via the `result_callback` provided in the MpImageSegmenterOptions.
// Only use this method when the ImageSegmenter is created with the live
// stream running mode.
// The image can be of any size with format RGB or RGBA. It's required to
// provide a timestamp (in milliseconds) to indicate when the input image is
// sent to the image segmenter. The input timestamps must be monotonically
// increasing.
// See `MpImageSegmenterResultCallback` for the arguments that `result_callback`
// receives.
//
// Returns 'kMpOk' on success. To obtain a detailed error, `error_msg` must be
// non-null pointer to a `char*`, which will be populated with a newly-allocated
// error message upon failure. It's the caller responsibility to free the error
// message with `MpErrorFree()`.
MP_EXPORT MpStatus
MpImageSegmenterSegmentAsync(MpImageSegmenterPtr segmenter, MpImagePtr image,
                             const MpImageProcessingOptions* options,
                             int64_t timestamp_ms, char** error_msg);

// Frees the memory allocated inside a MpImageSegmenterResult result.
// Does not free the result pointer itself.
MP_EXPORT void MpImageSegmenterCloseResult(MpImageSegmenterResult* result);

// Shuts down and frees the image segmenter. In the live stream mode, waits for
// the pending results. Once this function returns, `result_callback` is not
// invoked anymore, and `segmenter` is freed and must not be used, even if an
// error is returned.
//
// Returns 'kMpOk' on success. To obtain a detailed error, `error_msg` must be
// non-null pointer to a `char*`, which will be populated with a newly-allocated
// error message upon failure. It's the caller responsibility to free the error
// message with `MpErrorFree()`.
MP_EXPORT MpStatus MpImageSegmenterClose(MpImageSegmenterPtr segmenter,
                                         char** error_msg);

// Gets the category label list of the ImageSegmenter can recognize.
// The index in the category mask corresponds to the category in the label list.
// The output mask list at index corresponds to the category in the label list.
// If there is no label map provided in the model file, an empty label list is
// returned.
//
// Returns 'kMpOk' on success and sets `label_list` to the label list. The
// caller is responsible for freeing the memory of the `label_list` by calling
// `MpStringListFree()`. To obtain a detailed error, `error_msg` must be
// non-null pointer to a `char*`, which will be populated with a newly-allocated
// error message upon failure. It's the caller responsibility to free the error
// message with `MpErrorFree()`.
MP_EXPORT MpStatus MpImageSegmenterGetLabels(MpImageSegmenterPtr segmenter,
                                             MpStringList* label_list,
                                             char** error_msg);

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_VISION_IMAGE_SEGMENTER_IMAGE_SEGMENTER_H_
