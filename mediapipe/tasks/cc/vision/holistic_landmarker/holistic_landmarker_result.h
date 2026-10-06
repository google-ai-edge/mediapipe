/* Copyright 2026 The MediaPipe Authors.

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

#ifndef MEDIAPIPE_TASKS_CC_VISION_HOLISTIC_LANDMARKER_HOLISTIC_LANDMARKER_RESULT_H_
#define MEDIAPIPE_TASKS_CC_VISION_HOLISTIC_LANDMARKER_HOLISTIC_LANDMARKER_RESULT_H_

#include <optional>
#include <vector>

#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/tasks/cc/components/containers/category.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace holistic_landmarker {

// The holistic landmarks detection result from HolisticLandmarker.
struct HolisticLandmarkerResult {
  // Detected face landmarks in normalized image coordinates.
  components::containers::NormalizedLandmarks face_landmarks;
  // Optional face blendshapes.
  std::optional<std::vector<components::containers::Category>> face_blendshapes;
  // Detected pose landmarks in normalized image coordinates.
  components::containers::NormalizedLandmarks pose_landmarks;
  // Detected pose landmarks in world coordinates.
  components::containers::Landmarks pose_world_landmarks;
  // Left hand landmarks in normalized image coordinates.
  components::containers::NormalizedLandmarks left_hand_landmarks;
  // Right hand landmarks in normalized image coordinates.
  components::containers::NormalizedLandmarks right_hand_landmarks;
  // Left hand landmarks in world coordinates.
  components::containers::Landmarks left_hand_world_landmarks;
  // Right hand landmarks in world coordinates.
  components::containers::Landmarks right_hand_world_landmarks;
  // Segmentation masks for pose.
  std::optional<Image> pose_segmentation_masks;
};

}  // namespace holistic_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_HOLISTIC_LANDMARKER_HOLISTIC_LANDMARKER_RESULT_H_
