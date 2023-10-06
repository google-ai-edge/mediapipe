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

#include "mediapipe/tasks/cc/components/containers/rect.h"

namespace mediapipe::tasks::components::containers {

RectF ToRectF(const Rect& rect, int image_height, int image_width) {
  return RectF{static_cast<float>(rect.left) / image_width,
               static_cast<float>(rect.top) / image_height,
               static_cast<float>(rect.right) / image_width,
               static_cast<float>(rect.bottom) / image_height};
}

Rect ToRect(const RectF& rect, int image_height, int image_width) {
  return Rect{static_cast<int>(rect.left * image_width),
              static_cast<int>(rect.top * image_height),
              static_cast<int>(rect.right * image_width),
              static_cast<int>(rect.bottom * image_height)};
}

}  // namespace mediapipe::tasks::components::containers
