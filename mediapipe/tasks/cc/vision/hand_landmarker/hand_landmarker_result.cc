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

#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker_result.h"

#include <algorithm>

#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/tasks/cc/components/containers/classification_result.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace hand_landmarker {

HandLandmarkerResult ConvertToHandLandmarkerResult(
    const std::vector<mediapipe::ClassificationList>& handedness_proto,
    const std::vector<mediapipe::NormalizedLandmarkList>& hand_landmarks_proto,
    const std::vector<mediapipe::LandmarkList>& hand_world_landmarks_proto) {
  HandLandmarkerResult result;
  result.handedness.resize(handedness_proto.size());
  result.hand_landmarks.resize(hand_landmarks_proto.size());
  result.hand_world_landmarks.resize(hand_world_landmarks_proto.size());
  std::transform(handedness_proto.begin(), handedness_proto.end(),
                 result.handedness.begin(),
                 [](const mediapipe::ClassificationList& classification_list) {
                   return components::containers::ConvertToClassifications(
                       classification_list);
                 });
  std::transform(hand_landmarks_proto.begin(), hand_landmarks_proto.end(),
                 result.hand_landmarks.begin(),
                 components::containers::ConvertToNormalizedLandmarks);
  std::transform(hand_world_landmarks_proto.begin(),
                 hand_world_landmarks_proto.end(),
                 result.hand_world_landmarks.begin(),
                 components::containers::ConvertToLandmarks);
  return result;
}

}  // namespace hand_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
