/* Copyright 2022 The MediaPipe Authors.

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
#ifndef MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_CALCULATORS_HAND_LANDMARKS_DEDUPLICATION_CALCULATOR_H_
#define MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_CALCULATORS_HAND_LANDMARKS_DEDUPLICATION_CALCULATOR_H_

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/vision/utils/landmarks_duplicates_finder.h"

namespace mediapipe::api2 {

// Create a DuplicatesFinder dedicated for finding hand duplications.
std::unique_ptr<tasks::vision::utils::DuplicatesFinder>
CreateHandDuplicatesFinder(bool start_from_the_end = false);

// Filter duplicate hand landmarks by finding the overlapped hands.
// Inputs:
//   MULTI_LANDMARKS - std::vector<NormalizedLandmarkList>
//     The hand landmarks to be filtered.
//   MULTI_ROIS - std::vector<NormalizedRect>
//     The regions where each encloses the landmarks of a single hand.
//   MULTI_WORLD_LANDMARKS - std::vector<LandmarkList>
//      The hand landmarks to be filtered in world coordinates.
//   MULTI_CLASSIFICATIONS - std::vector<ClassificationList>
//      The handedness of hands.
//   IMAGE_SIZE - std::pair<int, int>
//     The size of the image which the hand landmarks are detected on.
//
// Outputs:
//   MULTI_LANDMARKS - std::vector<NormalizedLandmarkList>
//     The hand landmarks with duplication removed.
//   MULTI_ROIS - std::vector<NormalizedRect>
//     The regions where each encloses the landmarks of a single hand with
//     duplicate hands removed.
//   MULTI_WORLD_LANDMARKS - std::vector<LandmarkList>
//      The hand landmarks with duplication removed in world coordinates.
//   MULTI_CLASSIFICATIONS - std::vector<ClassificationList>
//      The handedness of hands with duplicate hands removed.
//
// Example:
// node {
//   calculator: "HandLandmarksDeduplicationCalculator"
//   input_stream: "MULTI_LANDMARKS:landmarks_in"
//   input_stream: "MULTI_ROIS:rois_in"
//   input_stream: "MULTI_WORLD_LANDMARKS:world_landmarks_in"
//   input_stream: "MULTI_CLASSIFICATIONS:handedness_in"
//   input_stream: "IMAGE_SIZE:image_size"
//   output_stream: "MULTI_LANDMARKS:landmarks_out"
//   output_stream: "MULTI_ROIS:rois_out"
//   output_stream: "MULTI_WORLD_LANDMARKS:world_landmarks_out"
//   output_stream: "MULTI_CLASSIFICATIONS:handedness_out"
// }
class HandLandmarksDeduplicationCalculator : public Node {
 public:
  constexpr static Input<std::vector<mediapipe::NormalizedLandmarkList>>
      kInLandmarks{"MULTI_LANDMARKS"};
  constexpr static Input<std::vector<mediapipe::NormalizedRect>>::Optional
      kInRois{"MULTI_ROIS"};
  constexpr static Input<std::vector<mediapipe::LandmarkList>>::Optional
      kInWorldLandmarks{"MULTI_WORLD_LANDMARKS"};
  constexpr static Input<std::vector<mediapipe::ClassificationList>>::Optional
      kInClassifications{"MULTI_CLASSIFICATIONS"};
  constexpr static Input<std::pair<int, int>> kInSize{"IMAGE_SIZE"};

  constexpr static Output<std::vector<mediapipe::NormalizedLandmarkList>>
      kOutLandmarks{"MULTI_LANDMARKS"};
  constexpr static Output<std::vector<mediapipe::NormalizedRect>>::Optional
      kOutRois{"MULTI_ROIS"};
  constexpr static Output<std::vector<mediapipe::LandmarkList>>::Optional
      kOutWorldLandmarks{"MULTI_WORLD_LANDMARKS"};
  constexpr static Output<std::vector<mediapipe::ClassificationList>>::Optional
      kOutClassifications{"MULTI_CLASSIFICATIONS"};
  MEDIAPIPE_NODE_CONTRACT(kInLandmarks, kInRois, kInWorldLandmarks,
                          kInClassifications, kInSize, kOutLandmarks, kOutRois,
                          kOutWorldLandmarks, kOutClassifications);
  absl::Status Process(mediapipe::CalculatorContext* cc) override;
};

}  // namespace mediapipe::api2

#endif  // MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_CALCULATORS_HAND_LANDMARKS_DEDUPLICATION_CALCULATOR_H_
