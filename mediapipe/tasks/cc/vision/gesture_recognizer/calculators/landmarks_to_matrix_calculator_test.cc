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
#include "mediapipe/tasks/cc/vision/gesture_recognizer/calculators/landmarks_to_matrix_calculator.h"

#include <string>
#include <utility>

#include "mediapipe/framework/api3/function_runner.h"
#include "mediapipe/framework/api3/graph.h"
#include "mediapipe/framework/api3/packet.h"
#include "mediapipe/framework/api3/stream.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

using api3::GenericGraph;
using api3::Packet;
using api3::Runner;
using api3::Stream;
using mediapipe::LandmarkList;
using mediapipe::LandmarksToMatrixCalculatorOptions;
using mediapipe::NormalizedLandmarkList;
using tasks::LandmarksToMatrixNode;

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

TEST_P(Landmarks2dToMatrixCalculatorTest, OutputsCorrectResult2d) {
  const Landmarks2dToMatrixCalculatorTestCase& params = GetParam();

  // Initialize the runner
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner,
      Runner::For([&params](
                      GenericGraph& graph,
                      Stream<NormalizedLandmarkList> landmarks,
                      Stream<std::pair<int, int>> image_size,
                      Stream<NormalizedRect> norm_rect) -> Stream<Matrix> {
        auto& node = graph.AddNode<LandmarksToMatrixNode>();
        {
          LandmarksToMatrixCalculatorOptions& opts = *node.options.Mutable();
          opts.set_object_normalization(
              params.object_normalization_origin_offset >= 0);
          opts.set_object_normalization_origin_offset(
              params.object_normalization_origin_offset);
        }
        node.landmarks.Set(landmarks);
        node.image_size.Set(image_size);
        node.norm_rect.Set(norm_rect);
        return node.landmarks_matrix.Get();
      }).Create());

  // Initialize the inputs
  NormalizedLandmarkList landmarks =
      BuildPseudoLandmarks<NormalizedLandmarkList>(21, params.base_offset);
  std::pair<int, int> image_size(640, 480);
  NormalizedRect norm_rect;
  norm_rect.set_rotation(params.rotation);

  // Run the graph and get the output
  MP_ASSERT_OK_AND_ASSIGN(
      Packet<Matrix> output_packet,
      runner.Run(api3::MakePacket<NormalizedLandmarkList>(landmarks),
                 api3::MakePacket<std::pair<int, int>>(image_size),
                 api3::MakePacket<NormalizedRect>(norm_rect)));

  // Verify the output
  ASSERT_TRUE(output_packet);
  const Matrix& output_matrix = output_packet.GetOrDie();
  ASSERT_EQ(output_matrix.cols(), 21);
  ASSERT_EQ(output_matrix.rows(), 3);
  EXPECT_NEAR(output_matrix(0, 2), params.expected_cell_0_2, 1e-4f);
  EXPECT_NEAR(output_matrix(1, 5), params.expected_cell_1_5, 1e-4f);
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
  const LandmarksWorld3dToMatrixCalculatorTestCase& params = GetParam();

  // Initialize the runner
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner,
      Runner::For([&params](
                      GenericGraph& graph, Stream<LandmarkList> landmarks,
                      Stream<std::pair<int, int>> image_size,
                      Stream<NormalizedRect> norm_rect) -> Stream<Matrix> {
        auto& node = graph.AddNode<LandmarksToMatrixNode>();
        {
          LandmarksToMatrixCalculatorOptions& opts = *node.options.Mutable();
          opts.set_object_normalization(
              params.object_normalization_origin_offset >= 0);
          opts.set_object_normalization_origin_offset(
              params.object_normalization_origin_offset);
        }
        node.world_landmarks.Set(landmarks);
        node.image_size.Set(image_size);
        node.norm_rect.Set(norm_rect);
        return node.landmarks_matrix.Get();
      }).Create());

  // Initialize the inputs
  LandmarkList landmarks =
      BuildPseudoLandmarks<LandmarkList>(21, params.base_offset);
  std::pair<int, int> image_size(640, 480);
  NormalizedRect norm_rect;
  norm_rect.set_rotation(params.rotation);

  // Run the graph and get the output
  MP_ASSERT_OK_AND_ASSIGN(
      Packet<Matrix> output_packet,
      runner.Run(api3::MakePacket<LandmarkList>(landmarks),
                 api3::MakePacket<std::pair<int, int>>(image_size),
                 api3::MakePacket<NormalizedRect>(norm_rect)));

  // Verify the output
  ASSERT_TRUE(output_packet);
  const Matrix& output_matrix = output_packet.GetOrDie();
  ASSERT_EQ(output_matrix.cols(), 21);
  ASSERT_EQ(output_matrix.rows(), 3);
  EXPECT_NEAR(output_matrix(0, 2), params.expected_cell_0_2, 1e-4f);
  EXPECT_NEAR(output_matrix(1, 5), params.expected_cell_1_5, 1e-4f);
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

TEST(LandmarksToMatrixCalculatorTest, HasCorrectRegistrationName) {
  EXPECT_EQ(tasks::LandmarksToMatrixNode::GetRegistrationName(),
            "LandmarksToMatrixCalculator");
}

}  // namespace
}  // namespace mediapipe
