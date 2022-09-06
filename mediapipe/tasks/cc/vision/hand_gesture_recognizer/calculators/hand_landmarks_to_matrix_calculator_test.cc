/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

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
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace tasks {
namespace vision {

namespace {

constexpr char kHandLandmarksTag[] = "HAND_LANDMARKS";
constexpr char kHandWorldLandmarksTag[] = "HAND_WORLD_LANDMARKS";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kLandmarksMatrixTag[] = "LANDMARKS_MATRIX";
constexpr char kNumHandLandmarks = 21;

template <class LandmarkListT>
LandmarkListT BuildPseudoHandLandmarks(int offset = 0) {
  LandmarkListT landmarks;
  for (int i = 0; i < kNumHandLandmarks; ++i) {
    auto* landmark = landmarks.add_landmark();
    landmark->set_x((offset + i) * 0.01 + 0.001);
    landmark->set_y((offset + i) * 0.01 + 0.002);
    landmark->set_z((offset + i) * 0.01 + 0.003);
  }
  return landmarks;
}

struct HandLandmarks2dToMatrixCalculatorTestCase {
  std::string test_name;
  int hand_offset;
};

using HandLandmarks2dToMatrixCalculatorTest =
    testing::TestWithParam<HandLandmarks2dToMatrixCalculatorTestCase>;

TEST_P(HandLandmarks2dToMatrixCalculatorTest, OutputsCorrectResult) {
  const HandLandmarks2dToMatrixCalculatorTestCase& test_case = GetParam();

  auto node_config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
      R"pb(
        calculator: "HandLandmarksToMatrixCalculator"
        input_stream: "HAND_LANDMARKS:hand_landmarks"
        input_stream: "IMAGE_SIZE:image_size"
        output_stream: "LANDMARKS_MATRIX:landmarks_matrix"
      )pb");
  CalculatorRunner runner(node_config);

  auto hand_landmarks = std::make_unique<NormalizedLandmarkList>();
  *hand_landmarks =
      BuildPseudoHandLandmarks<NormalizedLandmarkList>(test_case.hand_offset);

  runner.MutableInputs()
      ->Tag(kHandLandmarksTag)
      .packets.push_back(Adopt(hand_landmarks.release()).At(Timestamp(0)));
  auto image_size = std::make_unique<std::pair<int, int>>(640, 480);
  runner.MutableInputs()
      ->Tag(kImageSizeTag)
      .packets.push_back(Adopt(image_size.release()).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";

  const auto hand =
      runner.Outputs().Tag(kLandmarksMatrixTag).packets[0].Get<Matrix>();
  ASSERT_EQ(21, hand.cols());
  ASSERT_EQ(3, hand.rows());
  EXPECT_NEAR(hand(0, 2), 0.1f, 0.001f);
  EXPECT_NEAR(hand(1, 5), 0.1875f, 0.001f);
}

INSTANTIATE_TEST_CASE_P(
    HandLandmarksToMatrixCalculatorTests, HandLandmarks2dToMatrixCalculatorTest,
    testing::ValuesIn<HandLandmarks2dToMatrixCalculatorTestCase>(
        {{.test_name = "TestWithHandOffset0", .hand_offset = 0},
         {.test_name = "TestWithHandOffset21", .hand_offset = 21}}),
    [](const testing::TestParamInfo<
        HandLandmarks2dToMatrixCalculatorTest::ParamType>& info) {
      return info.param.test_name;
    });

struct HandLandmarksWorld3dToMatrixCalculatorTestCase {
  std::string test_name;
  int hand_offset;
};

using HandLandmarksWorld3dToMatrixCalculatorTest =
    testing::TestWithParam<HandLandmarksWorld3dToMatrixCalculatorTestCase>;

TEST_P(HandLandmarksWorld3dToMatrixCalculatorTest, OutputsCorrectResult) {
  const HandLandmarksWorld3dToMatrixCalculatorTestCase& test_case = GetParam();

  auto node_config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
      R"pb(
        calculator: "HandLandmarksToMatrixCalculator"
        input_stream: "HAND_WORLD_LANDMARKS:hand_landmarks"
        input_stream: "IMAGE_SIZE:image_size"
        output_stream: "LANDMARKS_MATRIX:landmarks_matrix"
      )pb");
  CalculatorRunner runner(node_config);

  auto hand_landmarks = std::make_unique<LandmarkList>();
  *hand_landmarks =
      BuildPseudoHandLandmarks<LandmarkList>(test_case.hand_offset);

  runner.MutableInputs()
      ->Tag(kHandWorldLandmarksTag)
      .packets.push_back(Adopt(hand_landmarks.release()).At(Timestamp(0)));
  auto image_size = std::make_unique<std::pair<int, int>>(640, 480);
  runner.MutableInputs()
      ->Tag(kImageSizeTag)
      .packets.push_back(Adopt(image_size.release()).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";

  const auto hand =
      runner.Outputs().Tag(kLandmarksMatrixTag).packets[0].Get<Matrix>();
  ASSERT_EQ(21, hand.cols());
  ASSERT_EQ(3, hand.rows());
  EXPECT_NEAR(hand(0, 2), 0.1f, 0.001f);
  EXPECT_NEAR(hand(1, 5), 0.25f, 0.001f);
}

INSTANTIATE_TEST_CASE_P(
    HandLandmarksToMatrixCalculatorTests,
    HandLandmarksWorld3dToMatrixCalculatorTest,
    testing::ValuesIn<HandLandmarksWorld3dToMatrixCalculatorTestCase>(
        {{.test_name = "TestWithHandOffset0", .hand_offset = 0},
         {.test_name = "TestWithHandOffset21", .hand_offset = 21}}),
    [](const testing::TestParamInfo<
        HandLandmarksWorld3dToMatrixCalculatorTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace

}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
