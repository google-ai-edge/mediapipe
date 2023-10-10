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

Defines topology of the GHUM Hand model. This is related only to the 3D model of
the hand, not the way NN model predicts joint rotations for it.
==============================================================================*/

#ifndef MEDIAPIPE_TASKS_CC_VISION_UTILS_GHUM_GHUM_HAND_TOPOLOGY_H_
#define MEDIAPIPE_TASKS_CC_VISION_UTILS_GHUM_GHUM_HAND_TOPOLOGY_H_

namespace mediapipe::tasks::vision::utils::ghum {

// GHUM hand 16 joint names in order they are produced by the HandRig pipeline.
enum class GhumHandJointName {
  kHand = 0,
  kIndex01,
  kIndex02,
  kIndex03,
  kMiddle01,
  kMiddle02,
  kMiddle03,
  kRing01,
  kRing02,
  kRing03,
  kPinky01,
  kPinky02,
  kPinky03,
  kThumb01,
  kThumb02,
  kThumb03,
};

}  // namespace mediapipe::tasks::vision::utils::ghum

#endif  // MEDIAPIPE_TASKS_CC_VISION_UTILS_GHUM_GHUM_HAND_TOPOLOGY_H_
