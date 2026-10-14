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

#ifndef MEDIAPIPE_TASKS_C_VISION_INTERACTIVE_SEGMENTER_INTERACTIVE_SEGMENTER_H_
#define MEDIAPIPE_TASKS_C_VISION_INTERACTIVE_SEGMENTER_INTERACTIVE_SEGMENTER_H_

#include <stdbool.h>
#include <stdint.h>

#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/core/common.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/vision/core/image.h"

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

typedef struct MpInteractiveSegmenterInternal* MpInteractiveSegmenterPtr;

// The options for configuring a mediapipe interactive segmenter task.
struct MpInteractiveSegmenterOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  struct MpBaseOptions base_options;
};

struct MpStrokePoint {
  float x;
  float y;
};

// The brush mode representing the intent of the brush stroke.
enum MpBrushMode {
  // No-op.
  MP_BRUSH_MODE_UNSPECIFIED = 0,
  // Adds regions/objects of interest to the segmentation mask.
  MP_BRUSH_MODE_POSITIVE = 1,
  // Excludes regions/objects from the segmentation mask.
  MP_BRUSH_MODE_NEGATIVE = 2,
  // Draws a boundary around the object of interest to add to the segmentation
  // mask.
  MP_BRUSH_MODE_LASSO = 3,
};

struct MpStroke {
  enum MpBrushMode brush_mode;
  struct MpStrokePoint* points;
  uint32_t points_count;
  bool is_completed;
};

struct MpStrokes {
  struct MpStroke* strokes;
  uint32_t strokes_count;
};

// Creates an InteractiveSegmenter from the provided `options`.
//
// Returns `kMpOk` on success and sets `segmenter` to the created
// InteractiveSegmenter.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller's responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpInteractiveSegmenterCreate(
    const struct MpInteractiveSegmenterOptions* options,
    MpInteractiveSegmenterPtr* segmenter, char** error_msg);

// Sets the image to be segmented.
MP_EXPORT MpStatus MpInteractiveSegmenterSetImage(
    MpInteractiveSegmenterPtr segmenter, MpImagePtr image, char** error_msg);

// Performs segmentation on the previously set image using provided `strokes`.
//
// Returns `kMpOk` on success and sets `mask` to the segmentation result mask.
// You must call `MpImageFree` to free the `mask` memory.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller's responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpInteractiveSegmenterSegment(
    MpInteractiveSegmenterPtr segmenter, const struct MpStrokes* strokes,
    MpImagePtr* mask, char** error_msg);

// Frees interactive segmenter.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller's responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpInteractiveSegmenterClose(
    MpInteractiveSegmenterPtr segmenter, char** error_msg);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MEDIAPIPE_TASKS_C_VISION_INTERACTIVE_SEGMENTER_INTERACTIVE_SEGMENTER_H_
