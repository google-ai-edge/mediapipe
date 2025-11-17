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

#include <cstdint>

#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/image_segmenter/image_segmenter_result.h"

#ifndef MP_EXPORT
#define MP_EXPORT __attribute__((visibility("default")))
#endif  // MP_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MpImageSegmenterInternal* MpImageSegmenterPtr;

// The options for configuring a mediapipe image segmenter task.
struct ImageSegmenterOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  struct BaseOptions base_options;

  // The running mode of the task. Default to the image mode.
  // Image segmenter has three running modes:
  // 1) The image mode for segmenting image on single image inputs.
  // 2) The video mode for segmenting image on the decoded frames of a video.
  // 3) The live stream mode for segmenting image on the live stream of input
  // data, such as from camera. In this mode, the "result_callback" below must
  // be specified to receive the segmentation results asynchronously.
  RunningMode running_mode;

  // The locale to use for display names specified through the TFLite Model
  // Metadata, if any. Defaults to English.
  const char* display_names_locale;

  // Whether to output confidence masks.
  bool output_confidence_masks = true;

  // Whether to output category mask.
  bool output_category_mask = false;

  // The user-defined result callback for processing live stream data.
  // The result callback should only be specified when the running mode is set
  // to RunningMode::LIVE_STREAM. Arguments of the callback function include:
  // the pointer to recognition result, the image that result was obtained
  // on, the timestamp relevant to recognition results and pointer to error
  // message in case of any failure. The validity of the passed arguments is
  // true for the lifetime of the callback function.
  //
  // The passed arguments are only valid for the lifetime of the callback.
  typedef void (*result_callback_fn)(MpStatus status,
                                     const ImageSegmenterResult* result,
                                     MpImagePtr image, int64_t timestamp_ms);
  result_callback_fn result_callback;
};

// Options for configuring runtime behavior of ImageSegmenter.
struct SegmentationOptions {
  // The width of the output segmentation masks.
  int output_width;

  // The height of the output segmentation masks.
  int output_height;
};

// Creates an ImageSegmenter from the provided `options`.
// Returns 'kMpOk' on success and sets `segmenter` to the created
// ImageSegmenter.
MP_EXPORT MpStatus MpImageSegmenterCreate(struct ImageSegmenterOptions* options,
                                          MpImageSegmenterPtr* segmenter);

// Performs image segmentation on the input `image`.
// Returns 'kMpOk' on success and sets `result` to the segmentation result.
// You must call `MpImageSegmenterCloseResult` to free its memory.
MP_EXPORT MpStatus MpImageSegmenterSegmentImage(
    MpImageSegmenterPtr segmenter, MpImagePtr image,
    const ImageProcessingOptions* options, ImageSegmenterResult* result);

// Performs image segmentation on the provided video frame.
// Only use this method when the ImageSegmenter is created with the video
// running mode.
// The image can be of any size with format RGB or RGBA. It's required to
// provide the video frame's timestamp (in milliseconds). The input timestamps
// must be monotonically increasing.
// Returns 'kMpOk' on success and sets `result` to the segmentation result.
// You must call `MpImageSegmenterCloseResult` to free its memory.
MP_EXPORT MpStatus MpImageSegmenterSegmentForVideo(
    MpImageSegmenterPtr segmenter, MpImagePtr image,
    const ImageProcessingOptions* options, int64_t timestamp_ms,
    ImageSegmenterResult* result);

// Sends live image data to image segmentation, and the results will be
// available via the `result_callback` provided in the ImageSegmenterOptions.
// Only use this method when the ImageSegmenter is created with the live
// stream running mode.
// The image can be of any size with format RGB or RGBA. It's required to
// provide a timestamp (in milliseconds) to indicate when the input image is
// sent to the image segmenter. The input timestamps must be monotonically
// increasing.
// The `result_callback` provides:
//   - The recognition results as an ImageSegmenterResult object.
//   - The const reference to the corresponding input image that the image
//     segmenter runs on. Note that the const reference to the image will no
//     longer be valid when the callback returns. To access the image data
//     outside of the callback, callers need to make a copy of the image.
//   - The input timestamp in milliseconds.
// Returns 'kMpOk' on success.
MP_EXPORT MpStatus MpImageSegmenterSegmentAsync(
    MpImageSegmenterPtr segmenter, MpImagePtr image,
    const ImageProcessingOptions* options, int64_t timestamp_ms);

// Frees the memory allocated inside a ImageSegmenterResult result.
// Does not free the result pointer itself.
MP_EXPORT void MpImageSegmenterCloseResult(ImageSegmenterResult* result);

// Shuts down the ImageSegmenter when all the work is done. Frees all memory.
// Returns 'kMpOk' on success.
MP_EXPORT MpStatus MpImageSegmenterClose(MpImageSegmenterPtr segmenter);

// Gets the category label list of the ImageSegmenter can recognize.
// The index in the category mask corresponds to the category in the label list.
// The output mask list at index corresponds to the category in the label list.
// If there is no label map provided in the model file, an empty label list is
// returned.
// The caller is responsible for freeing the memory of the `label_list`
// by calling `MpStringListFree`.
// Returns 'kMpOk' on success and sets `label_list` to the label list.
MP_EXPORT MpStatus MpImageSegmenterGetLabels(MpImageSegmenterPtr segmenter,
                                             MpStringList* label_list);

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_VISION_IMAGE_SEGMENTER_IMAGE_SEGMENTER_H_
