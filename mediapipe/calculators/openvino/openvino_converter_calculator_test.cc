// Copyright (c) 2023 Intel Corporation
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
//

#include <random>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/substitute.h"
#include "mediapipe/calculators/openvino/openvino_converter_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"  // NOLINT
#include "mediapipe/framework/tool/validate_type.h"

#include <openvino/openvino.hpp>

namespace mediapipe {

using RandomEngine = std::mt19937_64;
using testing::Eq;
const uint32 kSeed = 1234;
const int kNumSizes = 8;
const int sizes[kNumSizes][2] = {{1, 1}, {12, 1}, {1, 9},   {2, 2},
                                 {5, 3}, {7, 13}, {16, 32}, {101, 2}};

class OpenVINOConverterCalculatorTest : public ::testing::Test {
 protected:
  std::unique_ptr<CalculatorGraph> graph_;
};

TEST_F(OpenVINOConverterCalculatorTest, CustomDivAndSub) {
  CalculatorGraph graph;
  // Run the calculator and verify that one output is generated.
  CalculatorGraphConfig graph_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "input_image"
        node {
          calculator: "OpenVINOConverterCalculator"
          input_stream: "IMAGE:input_image"
          output_stream: "TENSORS:tensor"
          options {
            [mediapipe.OpenVINOConverterCalculatorOptions.ext] {
              enable_normalization: true
              use_custom_normalization: true
              custom_div: 2.0
              custom_sub: 33.0
            }
          }
        }
      )pb");
  std::vector<Packet> output_packets;
  tool::AddVectorSink("tensor", &graph_config, &output_packets);

  // Run the graph.
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));
  auto input_image = absl::make_unique<ImageFrame>(ImageFormat::GRAY8, 1, 1);
  cv::Mat mat = mediapipe::formats::MatView(input_image.get());
  mat.at<uint8>(0, 0) = 200;
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_image", Adopt(input_image.release()).At(Timestamp(0))));

  // Wait until the calculator done processing.
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Get and process results.
  const std::vector<ov::Tensor>& tensor_vec =
      output_packets[0].Get<std::vector<ov::Tensor>>();
  EXPECT_EQ(1, tensor_vec.size());

  const ov::Tensor tensor = tensor_vec[0];
  EXPECT_EQ(ov::element::f32, tensor.get_element_type());
  EXPECT_FLOAT_EQ(67.0f, tensor.data<float>()[0]);

  // Fully close graph at end, otherwise calculator+tensors are destroyed
  // after calling WaitUntilDone().
  MP_ASSERT_OK(graph.CloseInputStream("input_image"));
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST_F(OpenVINOConverterCalculatorTest, SetOutputRange) {
  std::vector<std::pair<float, float>> range_values = {
      std::make_pair(0.0, 1.0), std::make_pair(-1.0, 1.0),
      std::make_pair(-0.5, 0.5)};
  for (std::pair<float, float> range : range_values) {
    CalculatorGraph graph;
    CalculatorGraphConfig graph_config =
        mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
            absl::Substitute(R"(
        input_stream: "input_image"
        node {
          calculator: "OpenVINOConverterCalculator"
          input_stream: "IMAGE:input_image"
          output_stream: "TENSORS:tensor"
          options {
            [mediapipe.OpenVINOConverterCalculatorOptions.ext] {
              enable_normalization: true
              output_tensor_float_range {
                min: $0
                max: $1
              }
            }
          }
        }
        )",
                             /*$0=*/range.first,
                             /*$1=*/range.second));
    std::vector<Packet> output_packets;
    tool::AddVectorSink("tensor", &graph_config, &output_packets);

    // Run the graph.
    MP_ASSERT_OK(graph.Initialize(graph_config));
    MP_ASSERT_OK(graph.StartRun({}));
    auto input_image = absl::make_unique<ImageFrame>(ImageFormat::GRAY8, 1, 1);
    cv::Mat mat = mediapipe::formats::MatView(input_image.get());
    mat.at<uint8>(0, 0) = 200;
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "input_image", Adopt(input_image.release()).At(Timestamp(0))));

    // Wait until the calculator finishes processing.
    MP_ASSERT_OK(graph.WaitUntilIdle());
    EXPECT_THAT(output_packets.size(), Eq(1));

    // Get and process results.
    const std::vector<ov::Tensor>& tensor_vec =
        output_packets[0].Get<std::vector<ov::Tensor>>();
    EXPECT_THAT(tensor_vec.size(), Eq(1));

    const ov::Tensor tensor = tensor_vec[0];

    // Calculate the expected normalized value:
    float normalized_value =
        range.first + (200 * (range.second - range.first)) / 255.0;

    EXPECT_EQ(tensor.get_element_type(), ov::element::f32);
    EXPECT_THAT(normalized_value,
                testing::FloatNear(tensor.data<float>()[0],
                                   2.0f * std::abs(tensor.data<float>()[0]) *
                                       std::numeric_limits<float>::epsilon()));

    // Fully close graph at end, otherwise calculator+tensors are destroyed
    // after calling WaitUntilDone().
    MP_ASSERT_OK(graph.CloseInputStream("input_image"));
    MP_ASSERT_OK(graph.WaitUntilDone());
  }
}

// TODO: add test *without* normalization

}  // namespace mediapipe
