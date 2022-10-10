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

#ifndef MEDIAPIPE_TASKS_CC_COMPONENTS_CONTAINERS_LANDMARKS_DETECTION_H_
#define MEDIAPIPE_TASKS_CC_COMPONENTS_CONTAINERS_LANDMARKS_DETECTION_H_

#include <vector>

// Sturcts holding landmarks related data structure for hand landmarker, pose
// detector, face mesher, etc.
namespace mediapipe::tasks::components::containers {

// x and y are in [0,1] range with origin in top left in input image space.
// If model provides z, z is in the same scale as x. origin is in the center
// of the face.
struct Landmark {
  float x;
  float y;
  float z;
};

// [0, 1] range in input image space
struct Bound {
  float left;
  float top;
  float right;
  float bottom;
};

}  // namespace mediapipe::tasks::components::containers
#endif  // MEDIAPIPE_TASKS_CC_COMPONENTS_CONTAINERS_LANDMARKS_DETECTION_H_
