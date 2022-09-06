/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

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

#ifndef MEDIAPIPE_TASKS_CC_VISION_UTILS_IMAGE_UTILS_H_
#define MEDIAPIPE_TASKS_CC_VISION_UTILS_IMAGE_UTILS_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/tensor.h"

namespace mediapipe {
namespace tasks {
namespace vision {

struct Shape {
  int height;
  int width;
  int channels;
};

// Decodes an image file and returns it as a mediapipe::Image object.
//
// Support a wide range of image formats (see stb_image.h for the full list), as
// long as the image data is grayscale (1 channel), RGB (3 channels) or RGBA (4
// channels).
//
// Note: this function is not optimized for speed, and thus shouldn't be used
// outside of tests or simple CLI demo tools.
absl::StatusOr<mediapipe::Image> DecodeImageFromFile(const std::string& path);

// Get the shape of a image-like tensor.
//
// The tensor should have dimension 2, 3 or 4, representing `[height x width]`,
// `[height x width x channels]`, or `[batch x height x width x channels]`.
absl::StatusOr<Shape> GetImageLikeTensorShape(const mediapipe::Tensor& tensor);

}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_UTILS_IMAGE_UTILS_H_
