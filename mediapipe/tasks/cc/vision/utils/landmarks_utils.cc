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

#include "mediapipe/tasks/cc/vision/utils/landmarks_utils.h"

#include <algorithm>
#include <vector>

#include "mediapipe/tasks/cc/components/containers/rect.h"

namespace mediapipe::tasks::vision::utils {

using ::mediapipe::tasks::components::containers::RectF;

float CalculateArea(const RectF& rect) {
  return (rect.right - rect.left) * (rect.bottom - rect.top);
}

float CalculateIntersectionArea(const RectF& a, const RectF& b) {
  const float intersection_left = std::max<float>(a.left, b.left);
  const float intersection_top = std::max<float>(a.top, b.top);
  const float intersection_right = std::min<float>(a.right, b.right);
  const float intersection_bottom = std::min<float>(a.bottom, b.bottom);

  return std::max<float>(intersection_bottom - intersection_top, 0.0) *
         std::max<float>(intersection_right - intersection_left, 0.0);
}

float CalculateIOU(const RectF& a, const RectF& b) {
  const float area_a = CalculateArea(a);
  const float area_b = CalculateArea(b);
  if (area_a <= 0 || area_b <= 0) return 0.0;

  const float intersection_area = CalculateIntersectionArea(a, b);
  return intersection_area / (area_a + area_b - intersection_area);
}

}  // namespace mediapipe::tasks::vision::utils
