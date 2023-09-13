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

#ifndef MEDIAPIPE_TASKS_CC_VISION_POSE_LANDMARKER_POSE_LANDMARKER_RESULT_H_
#define MEDIAPIPE_TASKS_CC_VISION_POSE_LANDMARKER_POSE_LANDMARKER_RESULT_H_

#include <vector>

// #include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace pose_landmarker {

// The pose landmarks detection result from PoseLandmarker, where each vector
// element represents a single pose detected in the image.
struct PoseLandmarkerResult {
  // Segmentation masks for pose.
  std::optional<std::vector<Image>> segmentation_masks;
  // Detected pose landmarks in normalized image coordinates.
  std::vector<components::containers::NormalizedLandmarks> pose_landmarks;
  // Detected pose landmarks in world coordinates.
  std::vector<components::containers::Landmarks> pose_world_landmarks;
};

PoseLandmarkerResult ConvertToPoseLandmarkerResult(
    std::optional<std::vector<mediapipe::Image>> segmentation_mask,
    const std::vector<mediapipe::NormalizedLandmarkList>& pose_landmarks_proto,
    const std::vector<mediapipe::LandmarkList>& pose_world_landmarks_proto);

}  // namespace pose_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_POSE_LANDMARKER_POSE_LANDMARKER_RESULT_H_
