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

#include "mediapipe/tasks/c/vision/core/image_processing_options_converter.h"

#include <optional>

#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/components/containers/rect.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"

namespace mediapipe::tasks::c::vision::core {

namespace {
using ::mediapipe::tasks::components::containers::RectF;
}  // namespace

void CppConvertToImageProcessingOptions(
    const ImageProcessingOptions& in,
    mediapipe::tasks::vision::core::ImageProcessingOptions* out) {
  out->rotation_degrees = in.rotation_degrees;
  if (in.has_region_of_interest) {
    out->region_of_interest.emplace(
        RectF{in.region_of_interest.left, in.region_of_interest.top,
              in.region_of_interest.right, in.region_of_interest.bottom});
  } else {
    out->region_of_interest.reset();
  }
}

}  // namespace mediapipe::tasks::c::vision::core
