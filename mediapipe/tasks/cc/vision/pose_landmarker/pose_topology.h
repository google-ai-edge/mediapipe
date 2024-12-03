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

#ifndef MEDIAPIPE_TASKS_CC_VISION_POSE_LANDMARKER_POSE_TOPOLOGY_H_
#define MEDIAPIPE_TASKS_CC_VISION_POSE_LANDMARKER_POSE_TOPOLOGY_H_

namespace mediapipe::tasks::vision::pose_landmarker {

inline constexpr int kNumPoseLandmarks = 33;

// BlazePose 33 landmark names.
enum class PoseLandmarkName {
  kNose = 0,
  kLeftEyeInner,
  kLeftEye,
  kLeftEyeOuter,
  kRightEyeInner,
  kRightEye,
  kRightEyeOuter,
  kLeftEar,
  kRightEar,
  kMouthLeft,
  kMouthRight,
  kLeftShoulder,
  kRightShoulder,
  kLeftElbow,
  kRightElbow,
  kLeftWrist,
  kRightWrist,
  kLeftPinky1,
  kRightPinky1,
  kLeftIndex1,
  kRightIndex1,
  kLeftThumb2,
  kRightThumb2,
  kLeftHip,
  kRightHip,
  kLeftKnee,
  kRightKnee,
  kLeftAnkle,
  kRightAnkle,
  kLeftHeel,
  kRightHeel,
  kLeftFootIndex,
  kRightFootIndex,
};

}  // namespace mediapipe::tasks::vision::pose_landmarker

#endif  // MEDIAPIPE_TASKS_CC_VISION_POSE_LANDMARKER_POSE_TOPOLOGY_H_
