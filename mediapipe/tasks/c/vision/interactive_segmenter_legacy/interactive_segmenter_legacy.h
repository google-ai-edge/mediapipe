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

#ifndef MEDIAPIPE_TASKS_C_VISION_INTERACTIVE_SEGMENTER_LEGACY_INTERACTIVE_SEGMENTER_LEGACY_H_
#define MEDIAPIPE_TASKS_C_VISION_INTERACTIVE_SEGMENTER_LEGACY_INTERACTIVE_SEGMENTER_LEGACY_H_

#include <stdbool.h>
#include <stdint.h>

#include <cstdint>

#include "mediapipe/tasks/c/components/containers/keypoint.h"
#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/core/common.h"
#include "mediapipe/tasks/c/core/mp_status.h"
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

typedef struct MpInteractiveSegmenterLegacyInternal*
    MpInteractiveSegmenterLegacyPtr;

// The options for configuring a mediapipe interactive segmenter task.
struct MpInteractiveSegmenterLegacyOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  struct MpBaseOptions base_options;

  // Whether to output confidence masks.
  bool output_confidence_masks = true;

  // Whether to output category mask.
  bool output_category_mask = false;
};

// The format used to specify the region-of-interest.
enum MpRegionOfInterestFormat {
  MP_REGION_OF_INTEREST_FORMAT_UNSPECIFIED = 0,
  MP_REGION_OF_INTEREST_FORMAT_KEYPOINT = 1,
  MP_REGION_OF_INTEREST_FORMAT_SCRIBBLE = 2,
};

// The Region-Of-Interest (ROI) to interact with.
struct MpRegionOfInterest {
  // Specifies the format used to specify the region-of-interest. Note that
  // using `MP_REGION_OF_INTEREST_FORMAT_UNSPECIFIED` is invalid.
  enum MpRegionOfInterestFormat format;

  // Represents the ROI in keypoint format, this should have a valid keypoint
  // with coordinates `x` and `y` if `format` is
  // `MP_REGION_OF_INTEREST_FORMAT_KEYPOINT`; `nullptr` if not present
  MpNormalizedKeypoint* keypoint;

  // Represents the ROI in scribble format, this should be not a `nullptr` if
  // `format` is `MP_REGION_OF_INTEREST_FORMAT_SCRIBBLE`; `nullptr` if not
  // present
  MpNormalizedKeypoint* scribble;

  // Number of keypoints in scribble; 0 if not present
  uint32_t scribble_count;
};

// Creates an InteractiveSegmenterLegacy from the provided `options`.
// Return kMpOk on success and sets `segmenter` to the created
// InteractiveSegmenterLegacy.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpInteractiveSegmenterLegacyCreate(
    struct MpInteractiveSegmenterLegacyOptions* options,
    MpInteractiveSegmenterLegacyPtr* segmenter, char** error_msg);

// Performs interactive segmentation on the input `image`.
// Return kMpOk on success and sets `result` to the segmentation result.
// You must call `MpInteractiveSegmenterLegacyCloseResult` to free the result's
// memory.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpInteractiveSegmenterLegacySegmentImage(
    MpInteractiveSegmenterLegacyPtr segmenter, MpImagePtr image,
    const MpRegionOfInterest* roi,
    const MpImageProcessingOptions* image_processing_options,
    MpImageSegmenterResult* result, char** error_msg);

// Frees the memory allocated inside a MpImageSegmenterResult result.
// Does not free the result pointer itself.
MP_EXPORT void MpInteractiveSegmenterLegacyCloseResult(
    MpImageSegmenterResult* result);

// Frees interactive segmenter.
// Returns kMpOk on success.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpInteractiveSegmenterLegacyClose(
    MpInteractiveSegmenterLegacyPtr segmenter, char** error_msg);

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_VISION_INTERACTIVE_SEGMENTER_LEGACY_INTERACTIVE_SEGMENTER_LEGACY_H_
