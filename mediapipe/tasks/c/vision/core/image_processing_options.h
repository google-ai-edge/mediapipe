/* Copyright 2025 The MediaPipe Authors.

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

#ifndef MEDIAPIPE_TASKS_C_VISION_CORE_IMAGE_PREROCESSING_OPTIONS_H_
#define MEDIAPIPE_TASKS_C_VISION_CORE_IMAGE_PREROCESSING_OPTIONS_H_

#include "mediapipe/tasks/c/components/containers/rect.h"

#ifdef __cplusplus
extern "C" {
#endif

// Options for image processing.
// If both region_of_interest and rotation are specified, the crop around the
// region-of-interest is extracted first, then the specified rotation is applied
// to the crop.
typedef struct ImageProcessingOptions {
  // The optional region-of-interest to crop from the image.
  // If has_region_of_interest is 0, the full image is used.
  // Coordinates must be in [0,1] with 'left' < 'right' and 'top' < 'bottom'.
  int has_region_of_interest;
  MPRectF region_of_interest;

  // The rotation to apply to the image (or cropped region-of-interest),
  // in degrees clockwise. The rotation must be a multiple (positive or
  // negative) of 90°.
  int rotation_degrees;
} ImageProcessingOptions;

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_VISION_CORE_IMAGE_PREROCESSING_OPTIONS_H_
