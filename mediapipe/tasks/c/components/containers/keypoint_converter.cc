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

#include "mediapipe/tasks/c/components/containers/keypoint_converter.h"

#include <string.h>  // IWYU pragma: for open source compule

#include <cstdlib>

#include "mediapipe/tasks/c/components/containers/keypoint.h"
#include "mediapipe/tasks/cc/components/containers/keypoint.h"

namespace mediapipe::tasks::c::components::containers {

void CppConvertToNormalizedKeypoint(
    const mediapipe::tasks::components::containers::NormalizedKeypoint& in,
    NormalizedKeypoint* out) {
  out->x = in.x;
  out->y = in.y;

  out->label = in.label.has_value() ? strdup(in.label->c_str()) : nullptr;
  out->has_score = in.score.has_value();
  out->score = out->has_score ? in.score.value() : 0;
}

void CppCloseNormalizedKeypoint(NormalizedKeypoint* keypoint) {
  if (keypoint && keypoint->label) {
    free(keypoint->label);
    keypoint->label = nullptr;
  }
}

}  // namespace mediapipe::tasks::c::components::containers
