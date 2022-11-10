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

#ifndef MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_HAND_LANDMARKER_RESULT_H_
#define MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_HAND_LANDMARKER_RESULT_H_

#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace hand_landmarker {

// The hand landmarks detection result from HandLandmarker, where each vector
// element represents a single hand detected in the image.
struct HandLandmarkerResult {
  // Classification of handedness.
  std::vector<mediapipe::ClassificationList> handedness;
  // Detected hand landmarks in normalized image coordinates.
  std::vector<mediapipe::NormalizedLandmarkList> hand_landmarks;
  // Detected hand landmarks in world coordinates.
  std::vector<mediapipe::LandmarkList> hand_world_landmarks;
};

}  // namespace hand_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_HAND_LANDMARKER_RESULT_H_
