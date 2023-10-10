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

#include "mediapipe/tasks/cc/vision/utils/ghum/ghum_hand_utils.h"

#include <array>
#include <utility>

#include "absl/log/absl_check.h"
#include "mediapipe/calculators/util/set_joints_visibility_calculator.h"
#include "mediapipe/calculators/util/set_joints_visibility_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_topology.h"
#include "mediapipe/tasks/cc/vision/utils/ghum/ghum_hand_topology.h"

namespace mediapipe::tasks::vision::utils::ghum {

namespace {

using ::mediapipe::api2::SetJointsVisibilityCalculator;
using ::mediapipe::api2::builder::Stream;
using ::mediapipe::tasks::vision::hand_landmarker::HandLandmarkName;

// Includes mapping for all 16 GHUM Hand joints.
constexpr std::array<std::pair<GhumHandJointName, HandLandmarkName>, 16>
    kGhumHandJointsToHandLandmarksMapping = {{
        {GhumHandJointName::kHand, HandLandmarkName::kWrist},
        {GhumHandJointName::kIndex01, HandLandmarkName::kIndex1},
        {GhumHandJointName::kIndex02, HandLandmarkName::kIndex2},
        {GhumHandJointName::kIndex03, HandLandmarkName::kIndex3},
        {GhumHandJointName::kMiddle01, HandLandmarkName::kMiddle1},
        {GhumHandJointName::kMiddle02, HandLandmarkName::kMiddle2},
        {GhumHandJointName::kMiddle03, HandLandmarkName::kMiddle3},
        {GhumHandJointName::kRing01, HandLandmarkName::kRing1},
        {GhumHandJointName::kRing02, HandLandmarkName::kRing2},
        {GhumHandJointName::kRing03, HandLandmarkName::kRing3},
        {GhumHandJointName::kPinky01, HandLandmarkName::kPinky1},
        {GhumHandJointName::kPinky02, HandLandmarkName::kPinky2},
        {GhumHandJointName::kPinky03, HandLandmarkName::kPinky3},
        {GhumHandJointName::kThumb01, HandLandmarkName::kThumb1},
        {GhumHandJointName::kThumb02, HandLandmarkName::kThumb2},
        {GhumHandJointName::kThumb03, HandLandmarkName::kThumb3},
    }};

}  // namespace

Stream<JointList> SetGhumHandJointsVisibilityFromWorldLandmarks(
    Stream<JointList> ghum_hand_joints,
    Stream<LandmarkList> hand_world_landmarks,
    mediapipe::api2::builder::Graph& graph) {
  auto& node = graph.AddNode("SetJointsVisibilityCalculator");
  auto& opts = node.GetOptions<SetJointsVisibilityCalculatorOptions>();
  for (const auto& pair : kGhumHandJointsToHandLandmarksMapping) {
    // Sanity check to verify that all hand joints are set and are set in the
    // right order.
    ABSL_CHECK_EQ(static_cast<int>(pair.first), opts.mapping_size());
    auto* mapping = opts.add_mapping();
    mapping->mutable_copy()->set_idx(static_cast<int>(pair.second));
  }

  ghum_hand_joints.ConnectTo(node[SetJointsVisibilityCalculator::kInJoints]);
  hand_world_landmarks.ConnectTo(
      node[SetJointsVisibilityCalculator::kInLandmarks]);

  return node[SetJointsVisibilityCalculator::kOutJoints];
}

}  // namespace mediapipe::tasks::vision::utils::ghum
