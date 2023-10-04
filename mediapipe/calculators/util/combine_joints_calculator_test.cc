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

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/body_rig.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace api2 {
namespace {

using Node = ::mediapipe::CalculatorGraphConfig::Node;

Joint MakeJoint(const std::vector<float>& rotation_6d,
                std::optional<float> visibility) {
  Joint joint;
  for (float r : rotation_6d) {
    joint.add_rotation_6d(r);
  }
  if (visibility) {
    joint.set_visibility(visibility.value());
  }
  return joint;
}

struct CombineJointsTestCase {
  std::string test_name;
  int num_joints;
  std::string joints_mapping;
  std::vector<std::vector<Joint>> in_joints;
  std::vector<Joint> out_joints;
};

using CombineJointsTest = ::testing::TestWithParam<CombineJointsTestCase>;

TEST_P(CombineJointsTest, CombineJointsTest) {
  const CombineJointsTestCase& tc = GetParam();

  std::string input_joint_streams = "";
  for (int i = 0; i < tc.in_joints.size(); ++i) {
    input_joint_streams +=
        absl::StrFormat("input_stream: \"JOINTS:%d:joints_%d\"\n", i, i);
  }

  // Prepare graph.
  mediapipe::CalculatorRunner runner(ParseTextProtoOrDie<Node>(absl::Substitute(
      R"(
      calculator: "CombineJointsCalculator"
      $0
      output_stream: "JOINTS:combined_joints"
      options: {
        [mediapipe.CombineJointsCalculatorOptions.ext] {
          num_joints: $1
          joints_mapping: [ $2 ]
          default_joint: {
            rotation_6d: [1, 0, 0, 1, 0, 0]
            visibility: 1.0
          }
        }
      }
  )",
      input_joint_streams, tc.num_joints, tc.joints_mapping)));

  // Prepare and send joints.
  for (int i = 0; i < tc.in_joints.size(); ++i) {
    JointList in_joints;
    for (const auto& joint : tc.in_joints[i]) {
      *in_joints.add_joint() = joint;
    }
    runner.MutableInputs()
        ->Get("JOINTS", i)
        .packets.push_back(MakePacket<JointList>(std::move(in_joints))
                               .At(mediapipe::Timestamp(0)));
  }

  // Run the graph.
  MP_ASSERT_OK(runner.Run());

  const auto& output_packets = runner.Outputs().Tag("JOINTS").packets;
  EXPECT_EQ(1, output_packets.size());

  const auto& out_joints = output_packets[0].Get<JointList>();
  EXPECT_EQ(out_joints.joint_size(), tc.out_joints.size());
  for (int i = 0; i < out_joints.joint_size(); ++i) {
    const Joint& actual = out_joints.joint(i);
    const Joint& expected = tc.out_joints[i];

    EXPECT_EQ(actual.rotation_6d_size(), expected.rotation_6d_size())
        << "Unexpected joint #" << i << " rotation";
    for (int j = 0; j < actual.rotation_6d_size(); ++j) {
      EXPECT_NEAR(actual.rotation_6d(j), expected.rotation_6d(j), 1e-5)
          << "Unexpected joint #" << i << " rotation";
    }

    EXPECT_EQ(actual.has_visibility(), expected.has_visibility())
        << "Unexpected joint #" << i << " visibility";
    if (actual.has_visibility()) {
      EXPECT_NEAR(actual.visibility(), expected.visibility(), 1e-5)
          << "Unexpected joint #" << i << " visibility";
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    CombineJointsTests, CombineJointsTest,
    testing::ValuesIn<CombineJointsTestCase>({
        {"Empty_NoOutJoints", 0, "{ idx: [] }", {{}}, {}},
        {"Empty_SingleOutJoint",
         1,
         "{ idx: [] }",
         {{}},
         {MakeJoint({1, 0, 0, 1, 0, 0}, 1)}},

        {"Single_SetFirst",
         2,
         "{ idx: [0] }",
         {{MakeJoint({3, 3, 3, 3, 3, 3}, 4)}},
         {MakeJoint({3, 3, 3, 3, 3, 3}, 4), MakeJoint({1, 0, 0, 1, 0, 0}, 1)}},
        {"Single_SetBoth",
         2,
         "{ idx: [0, 1] }",
         {{MakeJoint({3, 3, 3, 3, 3, 3}, 4), MakeJoint({7, 7, 7, 7, 7, 7}, 8)}},
         {MakeJoint({3, 3, 3, 3, 3, 3}, 4), MakeJoint({7, 7, 7, 7, 7, 7}, 8)}},
        {"Single_SetBoth_ReverseOrder",
         2,
         "{ idx: [1, 0] }",
         {{MakeJoint({3, 3, 3, 3, 3, 3}, 4), MakeJoint({7, 7, 7, 7, 7, 7}, 8)}},
         {MakeJoint({7, 7, 7, 7, 7, 7}, 8), MakeJoint({3, 3, 3, 3, 3, 3}, 4)}},

        {"Double_NoOverwrite",
         3,
         "{ idx: [0] }, { idx: [1] }",
         {{MakeJoint({3, 3, 3, 3, 3, 3}, 4)},
          {MakeJoint({7, 7, 7, 7, 7, 7}, 8)}},
         {MakeJoint({3, 3, 3, 3, 3, 3}, 4), MakeJoint({7, 7, 7, 7, 7, 7}, 8),
          MakeJoint({1, 0, 0, 1, 0, 0}, 1)}},
        {"Double_OverwriteSecond",
         3,
         "{ idx: [0, 1] }, { idx: [1, 2] }",
         {{MakeJoint({3, 3, 3, 3, 3, 3}, 4), MakeJoint({4, 4, 4, 4, 4, 4}, 5)},
          {MakeJoint({6, 6, 6, 6, 6, 6}, 7), MakeJoint({8, 8, 8, 8, 8, 8}, 9)}},
         {MakeJoint({3, 3, 3, 3, 3, 3}, 4), MakeJoint({6, 6, 6, 6, 6, 6}, 7),
          MakeJoint({8, 8, 8, 8, 8, 8}, 9)}},
    }),
    [](const testing::TestParamInfo<CombineJointsTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace api2
}  // namespace mediapipe
