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

#ifndef MEDIAPIPE_TASKS_C_VISION_IMAGE_SEGMENTER_RESULT_IMAGE_SEGMENTER_RESULT_H_
#define MEDIAPIPE_TASKS_C_VISION_IMAGE_SEGMENTER_RESULT_IMAGE_SEGMENTER_RESULT_H_

#include <cstdint>

#include "mediapipe/tasks/c/vision/core/common.h"

#ifndef MP_EXPORT
#define MP_EXPORT __attribute__((visibility("default")))
#endif  // MP_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

struct ImageSegmenterResult {
  // Multiple masks of float image in VEC32F1 format where, for each mask, each
  // pixel represents the prediction confidence, usually in the [0, 1] range.
  MpMask* confidence_masks;

  // Number of elements in the array `confidence_masks`.
  uint32_t confidence_masks_count;  // Number of elements in the array

  // Flag to indicate presence of confidence masks (0 for no, 1 for yes).
  uint32_t has_confidence_masks;

  // A category mask of uint8 image in GRAY8 format where each pixel represents
  // the class which the pixel in the original image was predicted to belong to.
  MpMask category_mask;

  // Flag to indicate presence of category mask (0 for no, 1 for yes).
  uint32_t has_category_mask;

  // The quality scores of the result masks, in the range of [0, 1]. Defaults to
  // `1` if the model doesn't output quality scores. Each element corresponds to
  // the score of the category in the model outputs.
  float* quality_scores;

  // Number of elements in the array `quality_scores`.
  uint32_t quality_scores_count;
};

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_VISION_IMAGE_SEGMENTER_RESULT_IMAGE_SEGMENTER_RESULT_H_
