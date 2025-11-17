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

#ifndef MEDIAPIPE_TASKS_C_VISION_INTERACTIVE_SEGMENTER_INTERACTIVE_SEGMENTER_H_
#define MEDIAPIPE_TASKS_C_VISION_INTERACTIVE_SEGMENTER_INTERACTIVE_SEGMENTER_H_

#include <stdbool.h>
#include <stdint.h>

#include <cstdint>

#include "mediapipe/tasks/c/components/containers/keypoint.h"
#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/image_segmenter/image_segmenter_result.h"

#ifndef MP_EXPORT
#define MP_EXPORT __attribute__((visibility("default")))
#endif  // MP_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MpInteractiveSegmenterInternal* MpInteractiveSegmenterPtr;

// The options for configuring a mediapipe interactive segmenter task.
struct InteractiveSegmenterOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  struct BaseOptions base_options;

  // Whether to output confidence masks.
  bool output_confidence_masks = true;

  // Whether to output category mask.
  bool output_category_mask = false;
};

// The Region-Of-Interest (ROI) to interact with.
struct RegionOfInterest {
  // Specifies the format used to specify the region-of-interest. Note that
  // using `kUnspecified` is invalid.
  enum {
    kUnspecified = 0,  // Format not specified
    kKeypoint = 1,     // Using keypoint to represent ROI
    kScribble = 2      // Using scribble to represent ROI
  } format;

  // Represents the ROI in keypoint format, this should have a valid keypoint
  // with coordinates `x` and `y` if `format` is `kKeyPoint`; `nullptr` if not
  // present
  NormalizedKeypoint* keypoint;

  // Represents the ROI in scribble format, this should be not a `nullptr` if
  // `format` is `kScribble`; `nullptr` if not present
  NormalizedKeypoint* scribble;

  // Number of keypoints in scribble; 0 if not present
  uint32_t scribble_count;
};

// Creates an InteractiveSegmenter from the provided `options`.
// Return kMpOk on success and sets `segmenter` to the created
// InteractiveSegmenter.
MP_EXPORT MpStatus
MpInteractiveSegmenterCreate(struct InteractiveSegmenterOptions* options,
                             MpInteractiveSegmenterPtr* segmenter);

// Performs interactive segmentation on the input `image`.
// Return kMpOk on success and sets `result` to the segmentation result.
// You must call `MpInteractiveSegmenterCloseResult` to free the result's
// memory.
MP_EXPORT MpStatus MpInteractiveSegmenterSegmentImage(
    MpInteractiveSegmenterPtr segmenter, MpImagePtr image,
    const RegionOfInterest* roi,
    const ImageProcessingOptions* image_processing_options,
    ImageSegmenterResult* result);

// Frees the memory allocated inside a ImageSegmenterResult result.
// Does not free the result pointer itself.
MP_EXPORT void MpInteractiveSegmenterCloseResult(ImageSegmenterResult* result);

// Frees interactive segmenter.
// Returns kMpOk on success.
MP_EXPORT MpStatus
MpInteractiveSegmenterClose(MpInteractiveSegmenterPtr segmenter);

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_VISION_INTERACTIVE_SEGMENTER_INTERACTIVE_SEGMENTER_H_
