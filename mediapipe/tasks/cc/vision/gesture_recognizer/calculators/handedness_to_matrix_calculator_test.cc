/* Copyright 2025 The MediaPipe Authors.

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
#include "mediapipe/tasks/cc/vision/gesture_recognizer/calculators/handedness_to_matrix_calculator.h"

#include <string>

#include "mediapipe/framework/api3/function_runner.h"
#include "mediapipe/framework/api3/graph.h"
#include "mediapipe/framework/api3/packet.h"
#include "mediapipe/framework/api3/stream.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

using ::mediapipe::api3::GenericGraph;
using ::mediapipe::api3::Packet;
using ::mediapipe::api3::Runner;
using ::mediapipe::api3::Stream;

mediapipe::ClassificationList ClassificationForHandedness(float handedness) {
  mediapipe::ClassificationList result;
  auto* h = result.add_classification();
  if (handedness < 0.5f) {
    h->set_label("Left");
    h->set_score(1.0f - handedness);
  } else {
    h->set_label("Right");
    h->set_score(handedness);
  }
  return result;
}

struct HandednessToMatrixCalculatorTestCase {
  std::string test_name;
  float handedness;
};

using HandednessToMatrixCalculatorTest =
    testing::TestWithParam<HandednessToMatrixCalculatorTestCase>;

TEST_P(HandednessToMatrixCalculatorTest, OutputsCorrectResult) {
  const HandednessToMatrixCalculatorTestCase& test_case = GetParam();

  MP_ASSERT_OK_AND_ASSIGN(
      auto runner,
      Runner::For([&](GenericGraph& graph,
                      Stream<mediapipe::ClassificationList> handedness)
                      -> Stream<Matrix> {
        auto& node = graph.AddNode<tasks::HandednessToMatrixNode>();
        node.in_handedness.Set(handedness);
        return node.out_handedness_matrix.Get();
      }).Create());

  mediapipe::ClassificationList input_handedness =
      ClassificationForHandedness(test_case.handedness);

  MP_ASSERT_OK_AND_ASSIGN(
      Packet<Matrix> output_packet,
      runner.Run(
          api3::MakePacket<mediapipe::ClassificationList>(input_handedness)));

  ASSERT_TRUE(output_packet);
  const Matrix& handedness = output_packet.GetOrDie();
  ASSERT_EQ(1, handedness.cols());
  ASSERT_EQ(1, handedness.rows());
  EXPECT_NEAR(handedness(0, 0), test_case.handedness, .001f);
}

INSTANTIATE_TEST_CASE_P(
    HandednessToMatrixCalculatorTests, HandednessToMatrixCalculatorTest,
    testing::ValuesIn<HandednessToMatrixCalculatorTestCase>(
        {{/* test_name= */ "TestWithLeftHand", /* handedness= */ 0.01f},
         {/* test_name= */ "TestWithRightHand", /* handedness= */ 0.99f}}),
    [](const testing::TestParamInfo<
        HandednessToMatrixCalculatorTest::ParamType>& info) {
      return info.param.test_name;
    });

TEST(HandednessToMatrixCalculatorTest, HasCorrectRegistrationName) {
  EXPECT_EQ(tasks::HandednessToMatrixNode::GetRegistrationName(),
            "HandednessToMatrixCalculator");
}

}  // namespace
}  // namespace mediapipe
