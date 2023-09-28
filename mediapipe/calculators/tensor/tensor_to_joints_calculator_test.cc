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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/body_rig.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {
namespace api2 {
namespace {

using Node = ::mediapipe::CalculatorGraphConfig::Node;

struct TensorToJointsTestCase {
  std::string test_name;
  int num_joints;
  int start_index;
  std::vector<float> raw_values;
  std::vector<std::vector<float>> expected_rotations;
};

using TensorToJointsTest = ::testing::TestWithParam<TensorToJointsTestCase>;

TEST_P(TensorToJointsTest, TensorToJointsTest) {
  const TensorToJointsTestCase& tc = GetParam();

  // Prepare graph.
  mediapipe::CalculatorRunner runner(ParseTextProtoOrDie<Node>(absl::Substitute(
      R"(
      calculator: "TensorToJointsCalculator"
      input_stream: "TENSOR:tensor"
      output_stream: "JOINTS:joints"
      options: {
        [mediapipe.TensorToJointsCalculatorOptions.ext] {
          num_joints: $0
          start_index: $1
        }
      }
  )",
      tc.num_joints, tc.start_index)));

  // Prepare tensor.
  Tensor tensor(Tensor::ElementType::kFloat32,
                Tensor::Shape{1, 1, static_cast<int>(tc.raw_values.size()), 1});
  float* tensor_buffer = tensor.GetCpuWriteView().buffer<float>();
  ASSERT_NE(tensor_buffer, nullptr);
  for (int i = 0; i < tc.raw_values.size(); ++i) {
    tensor_buffer[i] = tc.raw_values[i];
  }

  // Send tensor to the graph.
  runner.MutableInputs()->Tag("TENSOR").packets.push_back(
      mediapipe::MakePacket<Tensor>(std::move(tensor)).At(Timestamp(0)));

  // Run the graph.
  MP_ASSERT_OK(runner.Run());

  const auto& output_packets = runner.Outputs().Tag("JOINTS").packets;
  EXPECT_EQ(1, output_packets.size());

  const auto& joints = output_packets[0].Get<JointList>();
  EXPECT_EQ(joints.joint_size(), tc.expected_rotations.size());
  for (int i = 0; i < joints.joint_size(); ++i) {
    const Joint& joint = joints.joint(i);
    std::vector<float> expected_rotation_6d = tc.expected_rotations[i];
    EXPECT_EQ(joint.rotation_6d_size(), expected_rotation_6d.size())
        << "Unexpected joint #" << i << " rotation";
    for (int j = 0; j < joint.rotation_6d_size(); ++j) {
      EXPECT_EQ(joint.rotation_6d(j), expected_rotation_6d[j])
          << "Unexpected joint #" << i << " rotation";
    }
    EXPECT_FALSE(joint.has_visibility());
  }
}

INSTANTIATE_TEST_SUITE_P(
    TensorToJointsTests, TensorToJointsTest,
    testing::ValuesIn<TensorToJointsTestCase>({
        {"Empty", 0, 3, {0, 0, 0}, {}},

        {"Single",
         1,
         3,
         {0, 0, 0, 10, 11, 12, 13, 14, 15},
         {{10, 11, 12, 13, 14, 15}}},

        {"Double",
         2,
         3,
         {0, 0, 0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21},
         {{10, 11, 12, 13, 14, 15}, {16, 17, 18, 19, 20, 21}}},
    }),
    [](const testing::TestParamInfo<TensorToJointsTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace api2
}  // namespace mediapipe
