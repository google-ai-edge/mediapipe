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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

namespace {

constexpr char kHandednessTag[] = "HANDEDNESS";
constexpr char kHandednessMatrixTag[] = "HANDEDNESS_MATRIX";

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

  auto node_config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
      R"pb(
        calculator: "HandednessToMatrixCalculator"
        input_stream: "HANDEDNESS:handedness"
        output_stream: "HANDEDNESS_MATRIX:handedness_matrix"
      )pb");
  CalculatorRunner runner(node_config);

  auto input_handedness = std::make_unique<mediapipe::ClassificationList>();
  *input_handedness = ClassificationForHandedness(test_case.handedness);
  runner.MutableInputs()
      ->Tag(kHandednessTag)
      .packets.push_back(Adopt(input_handedness.release()).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";

  const auto handedness =
      runner.Outputs().Tag(kHandednessMatrixTag).packets[0].Get<Matrix>();
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

}  // namespace

}  // namespace mediapipe
