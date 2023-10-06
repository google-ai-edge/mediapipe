/* Copyright 2022 The MediaPipe Authors.

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

#ifndef MEDIAPIPE_TASKS_CC_VISION_CORE_IMAGE_PROCESSING_OPTIONS_H_
#define MEDIAPIPE_TASKS_CC_VISION_CORE_IMAGE_PROCESSING_OPTIONS_H_

#include <optional>

#include "mediapipe/tasks/cc/components/containers/rect.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace core {

// Options for image processing.
//
// If both region-or-interest and rotation are specified, the crop around the
// region-of-interest is extracted first, then the specified rotation is applied
// to the crop.
struct ImageProcessingOptions {
  // The optional region-of-interest to crop from the image. If not specified,
  // the full image is used.
  //
  // Coordinates must be in [0,1] with 'left' < 'right' and 'top' < bottom.
  std::optional<components::containers::RectF> region_of_interest =
      std::nullopt;

  // The rotation to apply to the image (or cropped region-of-interest), in
  // degrees clockwise.
  //
  // The rotation must be a multiple (positive or negative) of 90°.
  int rotation_degrees = 0;
};

}  // namespace core
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_CORE_IMAGE_PROCESSING_OPTIONS_H_
