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

Utility methods for populating Ghum joints from joints produced by Hund Hand
Pose and Hund Hand models and pose landmarks.
==============================================================================*/

#ifndef MEDIAPIPE_TASKS_CC_VISION_UTILS_GHUM_GHUM_UTILS_H_
#define MEDIAPIPE_TASKS_CC_VISION_UTILS_GHUM_GHUM_UTILS_H_

#include <array>
#include <vector>

#include "absl/types/span.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/body_rig.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/tasks/cc/vision/utils/ghum/ghum_topology.h"

namespace mediapipe::tasks::vision::utils::ghum {

struct HundToGhumJointsMapping {
  // Joints produced by the HUND models.
  mediapipe::api2::builder::Stream<mediapipe::JointList> hund_joints;
  // Order of joints in GHUM topology.
  absl::Span<const GhumJointName> ghum_joints_order;
};

// Sets GHUM joints from given HUND joints according to the mappings and in
// specified order.
//
// All joints that are not defined will remain in kGhumDefaultJointRotation
// and with 1.0 visibility.
//
// All joints specified later in order will override those that were specified
// earlier.
mediapipe::api2::builder::Stream<mediapipe::JointList>
SetGhumJointsFromHundJoints(
    std::vector<HundToGhumJointsMapping>& hund_to_ghum_joints_mappings,
    mediapipe::api2::builder::Graph& graph);

// Sets visibility of 63 GHUM joints from 33 pose world landmarks.
mediapipe::api2::builder::Stream<mediapipe::JointList>
SetGhumJointsVisibilityFromWorldLandmarks(
    mediapipe::api2::builder::Stream<mediapipe::JointList> ghum_joints,
    mediapipe::api2::builder::Stream<mediapipe::LandmarkList>
        pose_world_landmarks,
    mediapipe::api2::builder::Graph& graph);

// Gets a subset of kGhumRestingJointRotations in 6D format.
std::vector<std::array<float, 6>> GetGhumRestingJointRotationsSubset(
    absl::Span<const GhumJointName> ghum_joint_names);

}  // namespace mediapipe::tasks::vision::utils::ghum

#endif  // MEDIAPIPE_TASKS_CC_VISION_UTILS_GHUM_GHUM_UTILS_H_
