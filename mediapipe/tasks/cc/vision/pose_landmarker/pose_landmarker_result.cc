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

#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarker_result.h"

#include <algorithm>

#include "mediapipe/tasks/cc/components/containers/landmark.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace pose_landmarker {

PoseLandmarkerResult ConvertToPoseLandmarkerResult(
    std::optional<std::vector<mediapipe::Image>> segmentation_masks,
    const std::vector<mediapipe::NormalizedLandmarkList>& pose_landmarks_proto,
    const std::vector<mediapipe::LandmarkList>& pose_world_landmarks_proto) {
  PoseLandmarkerResult result;
  result.segmentation_masks = segmentation_masks;

  result.pose_landmarks.resize(pose_landmarks_proto.size());
  result.pose_world_landmarks.resize(pose_world_landmarks_proto.size());
  std::transform(pose_landmarks_proto.begin(), pose_landmarks_proto.end(),
                 result.pose_landmarks.begin(),
                 components::containers::ConvertToNormalizedLandmarks);
  std::transform(pose_world_landmarks_proto.begin(),
                 pose_world_landmarks_proto.end(),
                 result.pose_world_landmarks.begin(),
                 components::containers::ConvertToLandmarks);
  return result;
}

}  // namespace pose_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
