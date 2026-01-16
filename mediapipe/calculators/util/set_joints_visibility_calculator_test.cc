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

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "absl/types/optional.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/body_rig.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace api2 {
namespace {

using Node = ::mediapipe::CalculatorGraphConfig::Node;

struct SetJointsVisibilityTestCase {
  std::string test_name;
  std::string mapping;
  std::vector<absl::optional<float>> in_joint_visibilities;
  std::vector<absl::optional<float>> landmark_visibilities;
  std::vector<absl::optional<float>> out_joint_visibilities;
};

using SetJointsVisibilityTest =
    ::testing::TestWithParam<SetJointsVisibilityTestCase>;

TEST_P(SetJointsVisibilityTest, SetJointsVisibilityTest) {
  const SetJointsVisibilityTestCase& tc = GetParam();

  // Prepare graph.
  mediapipe::CalculatorRunner runner(ParseTextProtoOrDie<Node>(absl::Substitute(
      R"(
      calculator: "SetJointsVisibilityCalculator"
      input_stream: "JOINTS:joints"
      input_stream: "LANDMARKS:landmarks"
      output_stream: "JOINTS:joints_with_visibility"
      options: {
        [mediapipe.SetJointsVisibilityCalculatorOptions.ext] {
          mapping: [
            $0
          ]
        }
      }
  )",
      tc.mapping)));

  // Prepare joints.
  JointList in_joints;
  for (auto vis_opt : tc.in_joint_visibilities) {
    Joint* joint = in_joints.add_joint();
    if (vis_opt) {
      joint->set_visibility(vis_opt.value());
    }
  }

  // Prepare landmarks.
  LandmarkList landmarks;
  for (auto vis_opt : tc.landmark_visibilities) {
    Landmark* lmk = landmarks.add_landmark();
    if (vis_opt) {
      lmk->set_visibility(vis_opt.value());
    }
  }

  // Send joints and landmarks to the graph.
  runner.MutableInputs()->Tag("JOINTS").packets.push_back(
      MakePacket<JointList>(std::move(in_joints)).At(mediapipe::Timestamp(0)));
  runner.MutableInputs()
      ->Tag("LANDMARKS")
      .packets.push_back(MakePacket<LandmarkList>(std::move(landmarks))
                             .At(mediapipe::Timestamp(0)));

  // Run the graph.
  MP_ASSERT_OK(runner.Run());

  const auto& output_packets = runner.Outputs().Tag("JOINTS").packets;
  EXPECT_EQ(1, output_packets.size());

  const auto& out_joints = output_packets[0].Get<JointList>();
  EXPECT_EQ(out_joints.joint_size(), tc.out_joint_visibilities.size());
  for (int i = 0; i < out_joints.joint_size(); ++i) {
    const Joint& joint = out_joints.joint(i);
    auto expected_vis_opt = tc.out_joint_visibilities[i];
    if (expected_vis_opt) {
      EXPECT_NEAR(joint.visibility(), expected_vis_opt.value(), 1e-5);
    } else {
      EXPECT_FALSE(joint.has_visibility());
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    SetJointsVisibilityTests, SetJointsVisibilityTest,
    testing::ValuesIn<SetJointsVisibilityTestCase>({
        {"Empty_LandmarksAndJoints", "", {}, {}, {}},
        {"Empty_Joints", "", {}, {1, 2, 3}, {}},
        {"Empty_Landmarks",
         "{ unchanged: {} }, { unchanged: {} }, { unchanged: {} }",
         {1, 2, absl::nullopt},
         {},
         {1, 2, absl::nullopt}},

        {"Mapping_Unchanged", "{ unchanged: {} }", {1}, {2}, {1}},
        {"Mapping_Unchanged_UnsetJointVisRemainsUnset",
         "{ unchanged: {} }",
         {absl::nullopt},
         {2},
         {absl::nullopt}},

        {"Mapping_Copy", "{ copy: { idx: 0 } }", {1}, {2}, {2}},
        {"Mapping_Copy_UnsetLmkVisResultsIntoZeroJointVis",
         "{ copy: { idx: 0 } }",
         {absl::nullopt},
         {absl::nullopt},
         {0}},

        {"Mapping_Highest",
         "{ highest: { idx: [0, 1, 2] } }",
         {absl::nullopt},
         {2, 4, 3},
         {4}},
        {"Mapping_Highest_UnsetLmkIsIgnored",
         "{ highest: { idx: [0, 1, 2] } }",
         {absl::nullopt},
         {-2, absl::nullopt, -3},
         {-2}},
    }),
    [](const testing::TestParamInfo<SetJointsVisibilityTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace api2
}  // namespace mediapipe
