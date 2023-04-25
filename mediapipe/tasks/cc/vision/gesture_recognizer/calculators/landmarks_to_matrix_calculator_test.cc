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

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

namespace {

using ::mediapipe::NormalizedRect;

constexpr char kLandmarksTag[] = "LANDMARKS";
constexpr char kWorldLandmarksTag[] = "WORLD_LANDMARKS";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kLandmarksMatrixTag[] = "LANDMARKS_MATRIX";
constexpr char kNormRectTag[] = "NORM_RECT";

template <class LandmarkListT>
LandmarkListT BuildPseudoLandmarks(int num_landmarks, int offset = 0) {
  LandmarkListT landmarks;
  for (int i = 0; i < num_landmarks; ++i) {
    auto* landmark = landmarks.add_landmark();
    landmark->set_x((offset + i) * 0.01 + 0.001);
    landmark->set_y((offset + i) * 0.01 + 0.002);
    landmark->set_z((offset + i) * 0.01 + 0.003);
  }
  return landmarks;
}

struct Landmarks2dToMatrixCalculatorTestCase {
  std::string test_name;
  int base_offset;
  int object_normalization_origin_offset = -1;
  float expected_cell_0_2;
  float expected_cell_1_5;
  float rotation;
};

using Landmarks2dToMatrixCalculatorTest =
    testing::TestWithParam<Landmarks2dToMatrixCalculatorTestCase>;

TEST_P(Landmarks2dToMatrixCalculatorTest, OutputsCorrectResult) {
  const Landmarks2dToMatrixCalculatorTestCase& test_case = GetParam();

  auto node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(absl::Substitute(
          R"pb(
            calculator: "LandmarksToMatrixCalculator"
            input_stream: "LANDMARKS:landmarks"
            input_stream: "IMAGE_SIZE:image_size"
            input_stream: "NORM_RECT:norm_rect"
            output_stream: "LANDMARKS_MATRIX:landmarks_matrix"
            options {
              [mediapipe.LandmarksToMatrixCalculatorOptions.ext] {
                object_normalization: $0
                object_normalization_origin_offset: $1
              }
            }
          )pb",
          test_case.object_normalization_origin_offset >= 0 ? "true" : "false",
          test_case.object_normalization_origin_offset));
  CalculatorRunner runner(node_config);

  auto landmarks = std::make_unique<NormalizedLandmarkList>();
  *landmarks =
      BuildPseudoLandmarks<NormalizedLandmarkList>(21, test_case.base_offset);

  runner.MutableInputs()
      ->Tag(kLandmarksTag)
      .packets.push_back(Adopt(landmarks.release()).At(Timestamp(0)));
  auto image_size = std::make_unique<std::pair<int, int>>(640, 480);
  runner.MutableInputs()
      ->Tag(kImageSizeTag)
      .packets.push_back(Adopt(image_size.release()).At(Timestamp(0)));
  auto norm_rect = std::make_unique<NormalizedRect>();
  norm_rect->set_rotation(test_case.rotation);
  runner.MutableInputs()
      ->Tag(kNormRectTag)
      .packets.push_back(Adopt(norm_rect.release()).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";

  const auto matrix =
      runner.Outputs().Tag(kLandmarksMatrixTag).packets[0].Get<Matrix>();
  ASSERT_EQ(21, matrix.cols());
  ASSERT_EQ(3, matrix.rows());
  EXPECT_NEAR(matrix(0, 2), test_case.expected_cell_0_2, 1e-4f);
  EXPECT_NEAR(matrix(1, 5), test_case.expected_cell_1_5, 1e-4f);
}

INSTANTIATE_TEST_CASE_P(
    LandmarksToMatrixCalculatorTests, Landmarks2dToMatrixCalculatorTest,
    testing::ValuesIn<Landmarks2dToMatrixCalculatorTestCase>(
        {{/* test_name= */ "TestWithOffset0",
          /* base_offset= */ 0,
          /* object_normalization_origin_offset= */ 0,
          /* expected_cell_0_2= */ 0.1f,
          /* expected_cell_1_5= */ 0.1875f,
          /* rotation= */ 0},
         {/* test_name= */ "TestWithOffset21",
          /* base_offset= */ 21,
          /* object_normalization_origin_offset= */ 0,
          /* expected_cell_0_2= */ 0.1f,
          /* expected_cell_1_5= */ 0.1875f,
          /* rotation= */ 0},
         {/* test_name= */ "TestWithRotation",
          /* base_offset= */ 0,
          /* object_normalization_origin_offset= */ 0,
          /* expected_cell_0_2= */ 0.075f,
          /* expected_cell_1_5= */ -0.25f,
          /* rotation= */ M_PI / 2.0}}),
    [](const testing::TestParamInfo<
        Landmarks2dToMatrixCalculatorTest::ParamType>& info) {
      return info.param.test_name;
    });

struct LandmarksWorld3dToMatrixCalculatorTestCase {
  std::string test_name;
  int base_offset;
  int object_normalization_origin_offset = -1;
  float expected_cell_0_2;
  float expected_cell_1_5;
  float rotation;
};

using LandmarksWorld3dToMatrixCalculatorTest =
    testing::TestWithParam<LandmarksWorld3dToMatrixCalculatorTestCase>;

TEST_P(LandmarksWorld3dToMatrixCalculatorTest, OutputsCorrectResult) {
  const LandmarksWorld3dToMatrixCalculatorTestCase& test_case = GetParam();

  auto node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(absl::Substitute(
          R"pb(
            calculator: "LandmarksToMatrixCalculator"
            input_stream: "WORLD_LANDMARKS:landmarks"
            input_stream: "IMAGE_SIZE:image_size"
            input_stream: "NORM_RECT:norm_rect"
            output_stream: "LANDMARKS_MATRIX:landmarks_matrix"
            options {
              [mediapipe.LandmarksToMatrixCalculatorOptions.ext] {
                object_normalization: $0
                object_normalization_origin_offset: $1
              }
            }
          )pb",
          test_case.object_normalization_origin_offset >= 0 ? "true" : "false",
          test_case.object_normalization_origin_offset));
  CalculatorRunner runner(node_config);

  auto landmarks = std::make_unique<LandmarkList>();
  *landmarks = BuildPseudoLandmarks<LandmarkList>(21, test_case.base_offset);

  runner.MutableInputs()
      ->Tag(kWorldLandmarksTag)
      .packets.push_back(Adopt(landmarks.release()).At(Timestamp(0)));
  auto image_size = std::make_unique<std::pair<int, int>>(640, 480);
  runner.MutableInputs()
      ->Tag(kImageSizeTag)
      .packets.push_back(Adopt(image_size.release()).At(Timestamp(0)));
  auto norm_rect = std::make_unique<NormalizedRect>();
  norm_rect->set_rotation(test_case.rotation);
  runner.MutableInputs()
      ->Tag(kNormRectTag)
      .packets.push_back(Adopt(norm_rect.release()).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";

  const auto matrix =
      runner.Outputs().Tag(kLandmarksMatrixTag).packets[0].Get<Matrix>();
  ASSERT_EQ(21, matrix.cols());
  ASSERT_EQ(3, matrix.rows());
  EXPECT_NEAR(matrix(0, 2), test_case.expected_cell_0_2, 1e-4f);
  EXPECT_NEAR(matrix(1, 5), test_case.expected_cell_1_5, 1e-4f);
}

INSTANTIATE_TEST_CASE_P(
    LandmarksToMatrixCalculatorTests, LandmarksWorld3dToMatrixCalculatorTest,
    testing::ValuesIn<LandmarksWorld3dToMatrixCalculatorTestCase>(
        {{/* test_name= */ "TestWithOffset0",
          /* base_offset= */ 0,
          /* object_normalization_origin_offset= */ 0,
          /* expected_cell_0_2= */ 0.1f,
          /* expected_cell_1_5= */ 0.25,
          /* rotation= */ 0},
         {/* test_name= */ "TestWithOffset21",
          /* base_offset= */ 21,
          /* object_normalization_origin_offset= */ 0,
          /* expected_cell_0_2= */ 0.1f,
          /* expected_cell_1_5= */ 0.25,
          /* rotation= */ 0},
         {/* test_name= */ "NoObjectNormalization",
          /* base_offset= */ 0,
          /* object_normalization_origin_offset= */ -1,
          /* expected_cell_0_2= */ 0.021f,
          /* expected_cell_1_5= */ 0.052f,
          /* rotation= */ 0},
         {/* test_name= */ "TestWithRotation",
          /* base_offset= */ 0,
          /* object_normalization_origin_offset= */ 0,
          /* expected_cell_0_2= */ 0.1f,
          /* expected_cell_1_5= */ -0.25f,
          /* rotation= */ M_PI / 2.0}}),
    [](const testing::TestParamInfo<
        LandmarksWorld3dToMatrixCalculatorTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace

}  // namespace mediapipe
