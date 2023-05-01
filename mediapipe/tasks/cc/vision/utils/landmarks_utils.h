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

#ifndef MEDIAPIPE_TASKS_CC_VISION_UTILS_LANDMARKS_UTILS_H_
#define MEDIAPIPE_TASKS_CC_VISION_UTILS_LANDMARKS_UTILS_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "mediapipe/tasks/cc/components/containers/rect.h"

namespace mediapipe::tasks::vision::utils {

// Calculates intersection over union for two bounds.
float CalculateIOU(const components::containers::RectF& a,
                   const components::containers::RectF& b);

// Calculates area for face bound
float CalculateArea(const components::containers::RectF& rect);

// Calucates intersection area of two face bounds
float CalculateIntersectionArea(const components::containers::RectF& a,
                                const components::containers::RectF& b);
}  // namespace mediapipe::tasks::vision::utils

#endif  // MEDIAPIPE_TASKS_CC_VISION_UTILS_LANDMARKS_UTILS_H_
