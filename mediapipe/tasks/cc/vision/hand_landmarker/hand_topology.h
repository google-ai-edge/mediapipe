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

#ifndef MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_HAND_TOPOLOGY_H_
#define MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_HAND_TOPOLOGY_H_

namespace mediapipe::tasks::vision::hand_landmarker {

// Hand model 21 landmark names.
enum class HandLandmarkName {
  kWrist = 0,
  kThumb1,
  kThumb2,
  kThumb3,
  kThumb4,
  kIndex1,
  kIndex2,
  kIndex3,
  kIndex4,
  kMiddle1,
  kMiddle2,
  kMiddle3,
  kMiddle4,
  kRing1,
  kRing2,
  kRing3,
  kRing4,
  kPinky1,
  kPinky2,
  kPinky3,
  kPinky4,
};

}  // namespace mediapipe::tasks::vision::hand_landmarker

#endif  // MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_HAND_TOPOLOGY_H_
