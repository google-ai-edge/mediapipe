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

#ifndef MEDIAPIPE_TASKS_CC_VISION_IMAGE_GENERATOR_IMAGE_GENERATOR_RESULT_H_
#define MEDIAPIPE_TASKS_CC_VISION_IMAGE_GENERATOR_IMAGE_GENERATOR_RESULT_H_

#include "mediapipe/framework/formats/image.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace image_generator {

// The result of ImageGenerator task.
struct ImageGeneratorResult {
  // The generated image.
  Image generated_image;

  // The condition_image used in the plugin model, only available if the
  // condition type is set in ImageGeneratorOptions.
  std::optional<Image> condition_image = std::nullopt;
};

}  // namespace image_generator
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_IMAGE_GENERATOR_IMAGE_GENERATOR_RESULT_H_
