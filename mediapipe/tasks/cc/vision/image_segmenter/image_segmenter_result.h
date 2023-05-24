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

#ifndef MEDIAPIPE_TASKS_CC_VISION_IMAGE_SEGMENTER_IMAGE_SEGMENTER_RESULT_H_
#define MEDIAPIPE_TASKS_CC_VISION_IMAGE_SEGMENTER_IMAGE_SEGMENTER_RESULT_H_

#include <optional>

#include "mediapipe/framework/formats/image.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace image_segmenter {

// The output result of ImageSegmenter
struct ImageSegmenterResult {
  // Multiple masks of float image in VEC32F1 format where, for each mask, each
  // pixel represents the prediction confidence, usually in the [0, 1] range.
  std::optional<std::vector<Image>> confidence_masks;
  // A category mask of uint8 image in GRAY8 format where each pixel represents
  // the class which the pixel in the original image was predicted to belong to.
  std::optional<Image> category_mask;
  // The quality scores of the result masks, in the range of [0, 1]. Defaults to
  // `1` if the model doesn't output quality scores. Each element corresponds to
  // the score of the category in the model outputs.
  std::vector<float> quality_scores;
};

}  // namespace image_segmenter
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_IMAGE_SEGMENTER_IMAGE_SEGMENTER_RESULT_H_
