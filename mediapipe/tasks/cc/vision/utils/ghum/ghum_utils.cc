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

#include "mediapipe/tasks/cc/vision/utils/ghum/ghum_utils.h"

#include <array>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "mediapipe/calculators/util/combine_joints_calculator.h"
#include "mediapipe/calculators/util/combine_joints_calculator.pb.h"
#include "mediapipe/calculators/util/set_joints_visibility_calculator.h"
#include "mediapipe/calculators/util/set_joints_visibility_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_topology.h"
#include "mediapipe/tasks/cc/vision/utils/ghum/ghum_topology.h"

namespace mediapipe::tasks::vision::utils::ghum {

namespace {

using ::mediapipe::api2::CombineJointsCalculator;
using ::mediapipe::api2::SetJointsVisibilityCalculator;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::SidePacket;
using ::mediapipe::api2::builder::Stream;
using ::mediapipe::tasks::vision::pose_landmarker::PoseLandmarkName;

}  // namespace

Stream<JointList> SetGhumJointsFromHundJoints(
    std::vector<HundToGhumJointsMapping>& mappings, Graph& graph) {
  auto& to_ghum = graph.AddNode("CombineJointsCalculator");
  auto& to_ghum_options = to_ghum.GetOptions<CombineJointsCalculatorOptions>();
  to_ghum_options.set_num_joints(63);

  // Default joint values for not mapped joints.
  for (float v : kGhumDefaultJointRotation) {
    to_ghum_options.mutable_default_joint()->add_rotation_6d(v);
  }
  to_ghum_options.mutable_default_joint()->set_visibility(1.0);

  for (int i = 0; i < mappings.size(); ++i) {
    auto* subset = to_ghum_options.add_joints_mapping();
    for (const auto& joint_name : mappings[i].ghum_joints_order) {
      subset->add_idx(static_cast<int>(joint_name));
    }
    mappings[i].hund_joints.ConnectTo(
        to_ghum[CombineJointsCalculator::kInJoints][i]);
  }

  return to_ghum[CombineJointsCalculator::kOutJoints];
}

Stream<JointList> SetGhumJointsVisibilityFromWorldLandmarks(
    Stream<JointList> ghum_joints, Stream<LandmarkList> pose_world_landmarks,
    Graph& graph) {
  auto& set_visibility = graph.AddNode("SetJointsVisibilityCalculator");
  auto& set_visibility_options =
      set_visibility.GetOptions<SetJointsVisibilityCalculatorOptions>();
  std::vector<std::pair<GhumJointName, std::vector<PoseLandmarkName>>> mapping =
      {
          {GhumJointName::kPelvis,
           {PoseLandmarkName::kLeftHip, PoseLandmarkName::kRightHip}},
          {GhumJointName::kSpine01,
           {PoseLandmarkName::kLeftHip, PoseLandmarkName::kRightHip}},
          {GhumJointName::kSpine02,
           {PoseLandmarkName::kLeftHip, PoseLandmarkName::kRightHip}},
          {GhumJointName::kSpine03,
           {PoseLandmarkName::kLeftShoulder, PoseLandmarkName::kRightShoulder}},
          {GhumJointName::kNeck01,
           {PoseLandmarkName::kLeftShoulder, PoseLandmarkName::kRightShoulder}},
          {GhumJointName::kHead01,
           {PoseLandmarkName::kLeftShoulder, PoseLandmarkName::kRightShoulder}},
          {GhumJointName::kJaw01,
           {PoseLandmarkName::kMouthLeft, PoseLandmarkName::kMouthRight}},
          {GhumJointName::kEye01L, {PoseLandmarkName::kLeftEye}},
          {GhumJointName::kEyelidup01L, {PoseLandmarkName::kLeftEye}},
          {GhumJointName::kEye01R, {PoseLandmarkName::kRightEye}},
          {GhumJointName::kEyelidup01R, {PoseLandmarkName::kRightEye}},
          {GhumJointName::kEyeliddown01L, {PoseLandmarkName::kLeftEye}},
          {GhumJointName::kEyeliddown01R, {PoseLandmarkName::kRightEye}},
          {GhumJointName::kClavicleL, {PoseLandmarkName::kLeftShoulder}},
          {GhumJointName::kUpperarmL, {PoseLandmarkName::kLeftShoulder}},
          {GhumJointName::kLowerarmL, {PoseLandmarkName::kLeftElbow}},
          {GhumJointName::kHandL, {PoseLandmarkName::kLeftWrist}},
          {GhumJointName::kIndex01L, {PoseLandmarkName::kLeftWrist}},
          {GhumJointName::kIndex02L, {PoseLandmarkName::kLeftWrist}},
          {GhumJointName::kIndex03L, {PoseLandmarkName::kLeftWrist}},
          {GhumJointName::kMiddle01L, {PoseLandmarkName::kLeftWrist}},
          {GhumJointName::kMiddle02L, {PoseLandmarkName::kLeftWrist}},
          {GhumJointName::kMiddle03L, {PoseLandmarkName::kLeftWrist}},
          {GhumJointName::kRing01L, {PoseLandmarkName::kLeftWrist}},
          {GhumJointName::kRing02L, {PoseLandmarkName::kLeftWrist}},
          {GhumJointName::kRing03L, {PoseLandmarkName::kLeftWrist}},
          {GhumJointName::kPinky01L, {PoseLandmarkName::kLeftWrist}},
          {GhumJointName::kPinky02L, {PoseLandmarkName::kLeftWrist}},
          {GhumJointName::kPinky03L, {PoseLandmarkName::kLeftWrist}},
          {GhumJointName::kThumb01L, {PoseLandmarkName::kLeftWrist}},
          {GhumJointName::kThumb02L, {PoseLandmarkName::kLeftWrist}},
          {GhumJointName::kThumb03L, {PoseLandmarkName::kLeftWrist}},
          {GhumJointName::kClavicleR, {PoseLandmarkName::kRightShoulder}},
          {GhumJointName::kUpperarmR, {PoseLandmarkName::kRightShoulder}},
          {GhumJointName::kLowerarmR, {PoseLandmarkName::kRightElbow}},
          {GhumJointName::kHandR, {PoseLandmarkName::kRightWrist}},
          {GhumJointName::kIndex01R, {PoseLandmarkName::kRightWrist}},
          {GhumJointName::kIndex02R, {PoseLandmarkName::kRightWrist}},
          {GhumJointName::kIndex03R, {PoseLandmarkName::kRightWrist}},
          {GhumJointName::kMiddle01R, {PoseLandmarkName::kRightWrist}},
          {GhumJointName::kMiddle02R, {PoseLandmarkName::kRightWrist}},
          {GhumJointName::kMiddle03R, {PoseLandmarkName::kRightWrist}},
          {GhumJointName::kRing01R, {PoseLandmarkName::kRightWrist}},
          {GhumJointName::kRing02R, {PoseLandmarkName::kRightWrist}},
          {GhumJointName::kRing03R, {PoseLandmarkName::kRightWrist}},
          {GhumJointName::kPinky01R, {PoseLandmarkName::kRightWrist}},
          {GhumJointName::kPinky02R, {PoseLandmarkName::kRightWrist}},
          {GhumJointName::kPinky03R, {PoseLandmarkName::kRightWrist}},
          {GhumJointName::kThumb01R, {PoseLandmarkName::kRightWrist}},
          {GhumJointName::kThumb02R, {PoseLandmarkName::kRightWrist}},
          {GhumJointName::kThumb03R, {PoseLandmarkName::kRightWrist}},
          {GhumJointName::kThighL, {PoseLandmarkName::kLeftHip}},
          {GhumJointName::kCalfL, {PoseLandmarkName::kLeftKnee}},
          {GhumJointName::kAnkleL, {PoseLandmarkName::kLeftAnkle}},
          {GhumJointName::kFootL, {PoseLandmarkName::kLeftAnkle}},
          {GhumJointName::kBallL, {PoseLandmarkName::kLeftHeel}},
          {GhumJointName::kToes01L, {PoseLandmarkName::kLeftFootIndex}},
          {GhumJointName::kThighR, {PoseLandmarkName::kRightHip}},
          {GhumJointName::kCalfR, {PoseLandmarkName::kRightKnee}},
          {GhumJointName::kAnkleR, {PoseLandmarkName::kRightAnkle}},
          {GhumJointName::kFootR, {PoseLandmarkName::kRightAnkle}},
          {GhumJointName::kBallR, {PoseLandmarkName::kRightHeel}},
          {GhumJointName::kToes01R, {PoseLandmarkName::kRightFootIndex}},
      };
  for (const auto& pair : mapping) {
    const auto& joint_name = pair.first;
    const auto& landmark_names = pair.second;

    CHECK_EQ(static_cast<int>(joint_name),
             set_visibility_options.mapping_size());
    auto* mapping = set_visibility_options.add_mapping();

    if (landmark_names.size() == 1) {
      mapping->mutable_copy()->set_idx(static_cast<int>(landmark_names[0]));
    } else {
      for (const auto& landmark_name : landmark_names) {
        mapping->mutable_highest()->add_idx(static_cast<int>(landmark_name));
      }
    }
  }

  ghum_joints.ConnectTo(
      set_visibility[SetJointsVisibilityCalculator::kInJoints]);
  pose_world_landmarks.ConnectTo(
      set_visibility[SetJointsVisibilityCalculator::kInLandmarks]);

  return set_visibility[SetJointsVisibilityCalculator::kOutJoints];
}

std::vector<std::array<float, 6>> GetGhumRestingJointRotationsSubset(
    absl::Span<const GhumJointName> ghum_joint_names) {
  std::vector<std::array<float, 6>> res;
  res.reserve(ghum_joint_names.size());
  for (GhumJointName ghum_joint_name : ghum_joint_names) {
    res.push_back(
        kGhumRestingJointRotations[static_cast<int>(ghum_joint_name)]);
  }
  return res;
}

}  // namespace mediapipe::tasks::vision::utils::ghum
