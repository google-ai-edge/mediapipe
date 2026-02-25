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

#include <vector>

#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/vision/utils/landmarks_duplicates_finder.h"

namespace mediapipe {
namespace tasks {

// Create a DuplicatesFinder dedicated for finding hand duplications.
std::unique_ptr<tasks::vision::utils::DuplicatesFinder>
CreateHandDuplicatesFinder(bool start_from_the_end = false);

// Filter duplicate hand landmarks by finding the overlapped hands.
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
struct HandLandmarksDeduplicationNode
    : public api3::Node<"HandLandmarksDeduplicationCalculator"> {
  template <typename S>
  struct Contract {
    // The hand landmarks to be filtered.
    api3::Input<S, std::vector<mediapipe::NormalizedLandmarkList>> landmarks_in{
        "MULTI_LANDMARKS"};

    // The regions where each encloses the landmarks of a single hand.
    api3::Optional<api3::Input<S, std::vector<mediapipe::NormalizedRect>>>
        rois_in{"MULTI_ROIS"};

    // The hand landmarks to be filtered in world coordinates.
    api3::Optional<api3::Input<S, std::vector<mediapipe::LandmarkList>>>
        world_landmarks_in{"MULTI_WORLD_LANDMARKS"};

    // The handedness of hands.
    api3::Optional<api3::Input<S, std::vector<mediapipe::ClassificationList>>>
        classifications_in{"MULTI_CLASSIFICATIONS"};

    // The size of the image which the hand landmarks are detected on.
    api3::Input<S, std::pair<int, int>> input_size{"IMAGE_SIZE"};

    // The hand landmarks with duplication removed.
    api3::Output<S, std::vector<mediapipe::NormalizedLandmarkList>>
        landmarks_out{"MULTI_LANDMARKS"};

    // The regions where each encloses the landmarks of a single hand with
    // duplicate hands removed.
    api3::Optional<api3::Output<S, std::vector<mediapipe::NormalizedRect>>>
        rois_out{"MULTI_ROIS"};

    // The hand landmarks with duplication removed in world coordinates.
    api3::Optional<api3::Output<S, std::vector<mediapipe::LandmarkList>>>
        world_landmarks_out{"MULTI_WORLD_LANDMARKS"};

    // The handedness of hands with duplicate hands removed.
    api3::Optional<api3::Output<S, std::vector<mediapipe::ClassificationList>>>
        classifications_out{"MULTI_CLASSIFICATIONS"};
  };
};

}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_CALCULATORS_HAND_LANDMARKS_DEDUPLICATION_CALCULATOR_H_
