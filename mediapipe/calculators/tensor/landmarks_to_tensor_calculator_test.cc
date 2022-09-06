// Copyright 2021 The MediaPipe Authors.
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

#include <vector>

#include "absl/memory/memory.h"
#include "mediapipe/calculators/tensor/landmarks_to_tensor_calculator.pb.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

using ::mediapipe::ParseTextProtoOrDie;
using Node = ::mediapipe::CalculatorGraphConfig::Node;

void RunLandmarks(mediapipe::CalculatorRunner* runner,
                  const LandmarkList& landmarks) {
  runner->MutableInputs()
      ->Tag("LANDMARKS")
      .packets.push_back(MakePacket<LandmarkList>(landmarks).At(Timestamp(0)));
  MP_ASSERT_OK(runner->Run());
}

void RunNormLandmarks(mediapipe::CalculatorRunner* runner,
                      const NormalizedLandmarkList& landmarks,
                      const std::pair<int, int> image_size) {
  runner->MutableInputs()
      ->Tag("NORM_LANDMARKS")
      .packets.push_back(
          MakePacket<NormalizedLandmarkList>(landmarks).At(Timestamp(0)));
  runner->MutableInputs()
      ->Tag("IMAGE_SIZE")
      .packets.push_back(
          MakePacket<std::pair<int, int>>(image_size).At(Timestamp(0)));
  MP_ASSERT_OK(runner->Run());
}

const Tensor& GetOutputTensor(mediapipe::CalculatorRunner* runner) {
  const auto& output_packets = runner->Outputs().Tag("TENSORS").packets;
  EXPECT_EQ(output_packets.size(), 1);

  const auto& tensors = output_packets[0].Get<std::vector<Tensor>>();
  EXPECT_EQ(tensors.size(), 1);

  return tensors[0];
}

void ValidateTensor(const Tensor& tensor,
                    const std::vector<int>& expected_shape,
                    const std::vector<float>& expected_values) {
  EXPECT_EQ(tensor.shape().dims, expected_shape);
  EXPECT_EQ(tensor.shape().num_elements(), expected_values.size());

  auto* tensor_buffer = tensor.GetCpuReadView().buffer<float>();
  const std::vector<float> tensor_values(
      tensor_buffer, tensor_buffer + tensor.shape().num_elements());
  EXPECT_THAT(tensor_values, testing::ElementsAreArray(expected_values));
}

TEST(LandmarksToTensorCalculatorTest, AllAttributes) {
  mediapipe::CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "LandmarksToTensorCalculator"
    input_stream: "LANDMARKS:landmarks"
    output_stream: "TENSORS:tensors"
    options: {
      [mediapipe.LandmarksToTensorCalculatorOptions.ext] {
        attributes: [ X, Y, Z, VISIBILITY, PRESENCE ]
      }
    }
  )pb"));

  LandmarkList landmarks;
  auto* landmark1 = landmarks.add_landmark();
  landmark1->set_x(1.0f);
  landmark1->set_y(2.0f);
  landmark1->set_z(3.0f);
  landmark1->set_visibility(4.0f);
  landmark1->set_presence(5.0f);
  auto* landmark2 = landmarks.add_landmark();
  landmark2->set_x(6.0f);
  landmark2->set_y(7.0f);
  landmark2->set_z(8.0f);
  landmark2->set_visibility(9.0f);
  landmark2->set_presence(10.0f);

  RunLandmarks(&runner, landmarks);
  const auto& tensor = GetOutputTensor(&runner);
  ValidateTensor(tensor, /*expected_shape=*/{1, 2, 5}, /*expected_values=*/
                 {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});
}

TEST(LandmarksToTensorCalculatorTest, XYZAttributes) {
  mediapipe::CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "LandmarksToTensorCalculator"
    input_stream: "LANDMARKS:landmarks"
    output_stream: "TENSORS:tensors"
    options: {
      [mediapipe.LandmarksToTensorCalculatorOptions.ext] {
        attributes: [ X, Y, Z ]
      }
    }
  )pb"));

  LandmarkList landmarks;
  auto* landmark1 = landmarks.add_landmark();
  landmark1->set_x(1.0f);
  landmark1->set_y(2.0f);
  landmark1->set_z(3.0f);
  auto* landmark2 = landmarks.add_landmark();
  landmark2->set_x(6.0f);
  landmark2->set_y(7.0f);
  landmark2->set_z(8.0f);

  RunLandmarks(&runner, landmarks);
  const auto& tensor = GetOutputTensor(&runner);
  ValidateTensor(tensor, /*expected_shape=*/{1, 2, 3}, /*expected_values=*/
                 {1.0f, 2.0f, 3.0f, 6.0f, 7.0f, 8.0f});
}

TEST(LandmarksToTensorCalculatorTest, XYZAttributes_Flatten) {
  mediapipe::CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "LandmarksToTensorCalculator"
    input_stream: "LANDMARKS:landmarks"
    output_stream: "TENSORS:tensors"
    options: {
      [mediapipe.LandmarksToTensorCalculatorOptions.ext] {
        attributes: [ X, Y, Z ]
        flatten: true
      }
    }
  )pb"));

  LandmarkList landmarks;
  auto* landmark1 = landmarks.add_landmark();
  landmark1->set_x(1.0f);
  landmark1->set_y(2.0f);
  landmark1->set_z(3.0f);
  auto* landmark2 = landmarks.add_landmark();
  landmark2->set_x(6.0f);
  landmark2->set_y(7.0f);
  landmark2->set_z(8.0f);

  RunLandmarks(&runner, landmarks);
  const auto& tensor = GetOutputTensor(&runner);
  ValidateTensor(tensor, /*expected_shape=*/{1, 6}, /*expected_values=*/
                 {1.0f, 2.0f, 3.0f, 6.0f, 7.0f, 8.0f});
}

TEST(LandmarksToTensorCalculatorTest, NormalizedLandmarks) {
  mediapipe::CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "LandmarksToTensorCalculator"
    input_stream: "NORM_LANDMARKS:landmarks"
    input_stream: "IMAGE_SIZE:image_size"
    output_stream: "TENSORS:tensors"
    options: {
      [mediapipe.LandmarksToTensorCalculatorOptions.ext] {
        attributes: [ X, Y, Z, VISIBILITY, PRESENCE ]
      }
    }
  )pb"));

  NormalizedLandmarkList landmarks;
  auto* landmark1 = landmarks.add_landmark();
  landmark1->set_x(0.1f);
  landmark1->set_y(0.5f);
  landmark1->set_z(1.0f);
  landmark1->set_visibility(4.0f);
  landmark1->set_presence(5.0f);

  std::pair<int, int> image_size{200, 100};

  RunNormLandmarks(&runner, landmarks, image_size);
  const auto& tensor = GetOutputTensor(&runner);
  ValidateTensor(tensor, /*expected_shape=*/{1, 1, 5}, /*expected_values=*/
                 {20.0f, 50.0f, 200.0f, 4.0f, 5.0f});
}

}  // namespace
}  // namespace mediapipe
