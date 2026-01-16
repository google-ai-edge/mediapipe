// Copyright 2023 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/calculators/util/set_joints_visibility_calculator.h"

#include <algorithm>
#include <optional>
#include <utility>

#include "mediapipe/calculators/util/set_joints_visibility_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/body_rig.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace api2 {

namespace {}  // namespace

class SetJointsVisibilityCalculatorImpl
    : public NodeImpl<SetJointsVisibilityCalculator> {
 public:
  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options<SetJointsVisibilityCalculatorOptions>();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    // Skip if Joints are empty.
    if (kInJoints(cc).IsEmpty()) {
      return absl::OkStatus();
    }

    // Get joints.
    const JointList& in_joints = kInJoints(cc).Get();
    RET_CHECK_EQ(in_joints.joint_size(), options_.mapping_size())
        << "Number of joints doesn't match number of mappings";

    // Get landmarks.
    RET_CHECK(!kInLandmarks(cc).IsEmpty()) << "Landmarks must be provided";
    const LandmarkList& in_landmarks = kInLandmarks(cc).Get();

    // Set joints visibility.
    JointList out_joints;
    for (int i = 0; i < in_joints.joint_size(); ++i) {
      // Initialize output joint.
      Joint* out_joint = out_joints.add_joint();
      *out_joint = in_joints.joint(i);

      // Get visibility. But only if it exists in the source landmark(s).
      std::optional<float> visibility;
      auto& mapping = options_.mapping(i);
      if (mapping.has_unchanged()) {
        continue;
      } else if (mapping.has_copy()) {
        const int idx = mapping.copy().idx();
        RET_CHECK(idx >= 0 && idx < in_landmarks.landmark_size())
            << "Landmark index out of range";
        if (in_landmarks.landmark(idx).has_visibility()) {
          visibility = in_landmarks.landmark(idx).visibility();
        }
      } else if (mapping.has_highest()) {
        RET_CHECK_GT(mapping.highest().idx_size(), 0) << "No indexes provided";
        for (int idx : mapping.highest().idx()) {
          RET_CHECK(idx >= 0 && idx < in_landmarks.landmark_size())
              << "Landmark index out of range";
          if (in_landmarks.landmark(idx).has_visibility()) {
            const float landmark_visibility =
                in_landmarks.landmark(idx).visibility();
            visibility = visibility.has_value()
                             ? std::max(visibility.value(), landmark_visibility)
                             : landmark_visibility;
          }
        }
      } else {
        RET_CHECK_FAIL() << "Unknown mapping";
      }

      // Set visibility. But only if it was possible to obtain it.
      if (visibility.has_value()) {
        out_joint->set_visibility(visibility.value());
      }
    }

    kOutJoints(cc).Send(std::move(out_joints));
    return absl::OkStatus();
  }

 private:
  SetJointsVisibilityCalculatorOptions options_;
};
MEDIAPIPE_NODE_IMPLEMENTATION(SetJointsVisibilityCalculatorImpl);

}  // namespace api2
}  // namespace mediapipe
