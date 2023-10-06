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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/calculators/tensors_to_segmentation_calculator.pb.h"

namespace mediapipe {

namespace {

using ::mediapipe::Image;
using ::mediapipe::Tensor;
using ::testing::HasSubstr;

constexpr std::array<float, 4> kTestValues = {0.2, 1.5, -0.6, 3.4};

constexpr std::array<float, 4> kExpectedSoftmaxValues = {0.03372, 0.12374,
                                                         0.01515, 0.82737};

constexpr std::array<float, 4> kExpectedSigmoidValues = {0.54983, 0.81757,
                                                         0.35434, 0.96770};

void PushTensorsToRunner(int tensor_height, int tensor_width,
                         const std::vector<float>& test_values,
                         CalculatorRunner* runner) {
  // Creates input tensor.
  auto tensors = absl::make_unique<std::vector<Tensor>>();
  tensors->emplace_back(Tensor::ElementType::kFloat32,
                        Tensor::Shape{tensor_height, tensor_width,
                                      static_cast<int>(test_values.size())});
  // Fills in tensor data.
  auto view = tensors->back().GetCpuWriteView();
  float* tensor_buffer = view.buffer<float>();
  ASSERT_NE(tensor_buffer, nullptr);
  const int tensor_size = tensor_height * tensor_width;
  const int channels = test_values.size();
  for (int i = 0; i < tensor_size; ++i) {
    absl::Span<float> channel_buffer =
        absl::MakeSpan(tensor_buffer + i * channels, channels);
    std::copy(test_values.begin(), test_values.end(), channel_buffer.begin());
  }
  // Pushs input to the runner.
  auto& input_stream_packets = runner->MutableInputs()->Tag("TENSORS").packets;
  input_stream_packets.push_back(
      mediapipe::Adopt(tensors.release()).At(Timestamp(0)));
}

std::vector<Packet> GetPackets(const CalculatorRunner& runner) {
  std::vector<Packet> mask_packets;
  for (int i = 0; i < runner.Outputs().NumEntries(); ++i) {
    EXPECT_EQ(runner.Outputs().Get("CONFIDENCE_MASK", i).packets.size(), 1);
    mask_packets.push_back(
        runner.Outputs().Get("CONFIDENCE_MASK", i).packets[0]);
  }
  return mask_packets;
}

MATCHER_P4(FloatImagePacket, expected_height, expected_width, expected_value,
           buffer_indices, "") {
  const auto& segmented_mask = arg.template Get<Image>();
  auto image_frame_ptr = segmented_mask.GetImageFrameSharedPtr();
  const float* data_buffer =
      reinterpret_cast<const float*>(image_frame_ptr->PixelData());
  return (segmented_mask.width() == expected_width &&
          segmented_mask.height() == expected_height &&
          std::all_of(buffer_indices.begin(), buffer_indices.end(), [&](int i) {
            return std::abs(data_buffer[i] - expected_value) < 1e-5;
          }));
}

MATCHER_P4(Uint8ImagePacket, expected_height, expected_width, expected_value,
           buffer_indices, "") {
  const auto& segmented_mask = arg.template Get<Image>();
  auto image_frame_ptr = segmented_mask.GetImageFrameSharedPtr();
  const uint8_t* data_buffer =
      reinterpret_cast<const uint8_t*>(image_frame_ptr->PixelData());
  return (segmented_mask.width() == expected_width &&
          segmented_mask.height() == expected_height &&
          std::all_of(buffer_indices.begin(), buffer_indices.end(),
                      [&](int i) { return data_buffer[i] == expected_value; }));
}

}  // namespace

TEST(TensorsToSegmentationCalculatorTest, FailsInvalidTensorDimensionOne) {
  CalculatorRunner runner(
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(
          R"pb(
            calculator: "mediapipe.tasks.TensorsToSegmentationCalculator"
            input_stream: "TENSORS:tensors"
            output_stream: "CONFIDENCE_MASK:segmentation"
            options {
              [mediapipe.tasks.TensorsToSegmentationCalculatorOptions.ext] {
                segmenter_options { activation: SOFTMAX }
              }
            }
          )pb"));
  auto tensors = absl::make_unique<std::vector<Tensor>>();
  tensors->emplace_back(Tensor::ElementType::kFloat32, Tensor::Shape{2});
  auto& input_stream_packets = runner.MutableInputs()->Tag("TENSORS").packets;
  input_stream_packets.push_back(
      mediapipe::Adopt(tensors.release()).At(Timestamp(0)));
  absl::Status status = runner.Run();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              HasSubstr("Tensor should have 2, 3, or 4 dims"));
}

TEST(TensorsToSegmentationCalculatorTest, FailsInvalidTensorDimensionFive) {
  CalculatorRunner runner(
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(
          R"pb(
            calculator: "mediapipe.tasks.TensorsToSegmentationCalculator"
            input_stream: "TENSORS:tensors"
            output_stream: "CONFIDENCE_MASK:segmentation"
            options {
              [mediapipe.tasks.TensorsToSegmentationCalculatorOptions.ext] {
                segmenter_options { activation: SOFTMAX }
              }
            }
          )pb"));
  auto tensors = absl::make_unique<std::vector<Tensor>>();
  tensors->emplace_back(Tensor::ElementType::kFloat32,
                        Tensor::Shape{2, 2, 1, 3, 5});
  auto& input_stream_packets = runner.MutableInputs()->Tag("TENSORS").packets;
  input_stream_packets.push_back(
      mediapipe::Adopt(tensors.release()).At(Timestamp(0)));
  absl::Status status = runner.Run();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              HasSubstr("Tensor should have 2, 3, or 4 dims"));
}

TEST(TensorsToSegmentationCalculatorTest, SucceedsConfidenceMaskWithSoftmax) {
  CalculatorRunner runner(
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(
          R"pb(
            calculator: "mediapipe.tasks.TensorsToSegmentationCalculator"
            input_stream: "TENSORS:tensors"
            output_stream: "CONFIDENCE_MASK:0:segmented_mask_0"
            output_stream: "CONFIDENCE_MASK:1:segmented_mask_1"
            output_stream: "CONFIDENCE_MASK:2:segmented_mask_2"
            output_stream: "CONFIDENCE_MASK:3:segmented_mask_3"
            options {
              [mediapipe.tasks.TensorsToSegmentationCalculatorOptions.ext] {
                segmenter_options { activation: SOFTMAX }
              }
            }
          )pb"));

  const int tensor_height = 2;
  const int tensor_width = 5;
  const int tensor_channels = kTestValues.size();
  PushTensorsToRunner(
      tensor_height, tensor_width,
      std::vector<float>(kTestValues.begin(), kTestValues.end()), &runner);
  MP_ASSERT_OK(runner.Run());
  ASSERT_EQ(runner.Outputs().NumEntries(), tensor_channels);
  const std::vector<int> buffer_indices = {0};
  std::vector<Packet> packets = GetPackets(runner);
  EXPECT_THAT(packets,
              testing::ElementsAre(
                  FloatImagePacket(tensor_height, tensor_width,
                                   kExpectedSoftmaxValues[0], buffer_indices),
                  FloatImagePacket(tensor_height, tensor_width,
                                   kExpectedSoftmaxValues[1], buffer_indices),
                  FloatImagePacket(tensor_height, tensor_width,
                                   kExpectedSoftmaxValues[2], buffer_indices),
                  FloatImagePacket(tensor_height, tensor_width,
                                   kExpectedSoftmaxValues[3], buffer_indices)));

  // VerifyRunnerResult<float>(tensor_height, tensor_width, tensor_channels,
  // kExpectedSoftmaxValues, runner);
}

TEST(TensorsToSegmentationCalculatorTest, SucceedsConfidenceMaskWithNone) {
  CalculatorRunner runner(
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(
          R"pb(
            calculator: "mediapipe.tasks.TensorsToSegmentationCalculator"
            input_stream: "TENSORS:tensors"
            output_stream: "CONFIDENCE_MASK:0:segmented_mask_0"
            output_stream: "CONFIDENCE_MASK:1:segmented_mask_1"
            output_stream: "CONFIDENCE_MASK:2:segmented_mask_2"
            output_stream: "CONFIDENCE_MASK:3:segmented_mask_3"
            options {
              [mediapipe.tasks.TensorsToSegmentationCalculatorOptions.ext] {
                segmenter_options { activation: NONE }
              }
            }
          )pb"));

  const int tensor_height = 3;
  const int tensor_width = 4;
  const int tensor_channels = kTestValues.size();
  PushTensorsToRunner(
      tensor_height, tensor_width,
      std::vector<float>(kTestValues.begin(), kTestValues.end()), &runner);
  MP_ASSERT_OK(runner.Run());
  ASSERT_EQ(runner.Outputs().NumEntries(), tensor_channels);
  const std::vector<int> buffer_indices = {0};
  std::vector<Packet> packets = GetPackets(runner);
  EXPECT_THAT(packets, testing::ElementsAre(
                           FloatImagePacket(tensor_height, tensor_width,
                                            kTestValues[0], buffer_indices),
                           FloatImagePacket(tensor_height, tensor_width,
                                            kTestValues[1], buffer_indices),
                           FloatImagePacket(tensor_height, tensor_width,
                                            kTestValues[2], buffer_indices),
                           FloatImagePacket(tensor_height, tensor_width,
                                            kTestValues[3], buffer_indices)));
}

TEST(TensorsToSegmentationCalculatorTest, SucceedsConfidenceMaskWithSigmoid) {
  CalculatorRunner runner(
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(
          R"pb(
            calculator: "mediapipe.tasks.TensorsToSegmentationCalculator"
            input_stream: "TENSORS:tensors"
            output_stream: "CONFIDENCE_MASK:0:segmented_mask_0"
            output_stream: "CONFIDENCE_MASK:1:segmented_mask_1"
            output_stream: "CONFIDENCE_MASK:2:segmented_mask_2"
            output_stream: "CONFIDENCE_MASK:3:segmented_mask_3"
            options {
              [mediapipe.tasks.TensorsToSegmentationCalculatorOptions.ext] {
                segmenter_options { activation: SIGMOID }
              }
            }
          )pb"));

  const int tensor_height = 4;
  const int tensor_width = 6;
  const int tensor_channels = kTestValues.size();
  PushTensorsToRunner(
      tensor_height, tensor_width,
      std::vector<float>(kTestValues.begin(), kTestValues.end()), &runner);
  MP_ASSERT_OK(runner.Run());
  ASSERT_EQ(runner.Outputs().NumEntries(), tensor_channels);
  const std::vector<int> buffer_indices = {0};
  std::vector<Packet> packets = GetPackets(runner);
  EXPECT_THAT(packets,
              testing::ElementsAre(
                  FloatImagePacket(tensor_height, tensor_width,
                                   kExpectedSigmoidValues[0], buffer_indices),
                  FloatImagePacket(tensor_height, tensor_width,
                                   kExpectedSigmoidValues[1], buffer_indices),
                  FloatImagePacket(tensor_height, tensor_width,
                                   kExpectedSigmoidValues[2], buffer_indices),
                  FloatImagePacket(tensor_height, tensor_width,
                                   kExpectedSigmoidValues[3], buffer_indices)));
}

TEST(TensorsToSegmentationCalculatorTest, SucceedsCategoryMask) {
  CalculatorRunner runner(
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(
          R"pb(
            calculator: "mediapipe.tasks.TensorsToSegmentationCalculator"
            input_stream: "TENSORS:tensors"
            output_stream: "CONFIDENCE_MASK:0:segmented_mask_0"
            output_stream: "CONFIDENCE_MASK:1:segmented_mask_1"
            output_stream: "CONFIDENCE_MASK:2:segmented_mask_2"
            output_stream: "CONFIDENCE_MASK:3:segmented_mask_3"
            output_stream: "CATEGORY_MASK:segmentation"
            options {
              [mediapipe.tasks.TensorsToSegmentationCalculatorOptions.ext] {
                segmenter_options { activation: NONE }
              }
            }
          )pb"));

  const int tensor_height = 2;
  const int tensor_width = 5;
  PushTensorsToRunner(
      tensor_height, tensor_width,
      std::vector<float>(kTestValues.begin(), kTestValues.end()), &runner);
  MP_ASSERT_OK(runner.Run());
  ASSERT_EQ(runner.Outputs().NumEntries(), 5);
  // Largest element index is 3.
  const int expected_index = 3;
  const std::vector<int> buffer_indices = {0};
  std::vector<Packet> packets = runner.Outputs().Tag("CATEGORY_MASK").packets;
  EXPECT_THAT(packets, testing::ElementsAre(
                           Uint8ImagePacket(tensor_height, tensor_width,
                                            expected_index, buffer_indices)));
}

TEST(TensorsToSegmentationCalculatorTest, SucceedsCategoryMaskResize) {
  CalculatorRunner runner(
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(
          R"pb(
            calculator: "mediapipe.tasks.TensorsToSegmentationCalculator"
            input_stream: "TENSORS:tensors"
            input_stream: "OUTPUT_SIZE:size"
            output_stream: "CONFIDENCE_MASK:0:segmented_mask_0"
            output_stream: "CONFIDENCE_MASK:1:segmented_mask_1"
            output_stream: "CONFIDENCE_MASK:2:segmented_mask_2"
            output_stream: "CONFIDENCE_MASK:3:segmented_mask_3"
            output_stream: "CATEGORY_MASK:segmentation"
            options {
              [mediapipe.tasks.TensorsToSegmentationCalculatorOptions.ext] {
                segmenter_options { activation: NONE }
              }
            }
          )pb"));

  const int input_height = 1;
  const int input_width = 4;
  const int output_height = 2;
  const int output_width = 8;

  PushTensorsToRunner(
      input_height, input_width,
      std::vector<float>(kTestValues.begin(), kTestValues.end()), &runner);
  runner.MutableInputs()
      ->Tag("OUTPUT_SIZE")
      .packets.push_back(mediapipe::MakePacket<std::pair<int, int>>(
                             std::make_pair(output_width, output_height))
                             .At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());

  // Largest element index is 3.
  // Upscale x2, so the expected value should distribute to 4 elements.
  const int expected_index = 3;
  const std::vector<int> buffer_indices = {
      0 * output_width + 0, 0 * output_width + 1, 1 * output_width + 0,
      1 * output_width + 1};
  std::vector<Packet> packets = runner.Outputs().Tag("CATEGORY_MASK").packets;
  EXPECT_THAT(packets, testing::ElementsAre(
                           Uint8ImagePacket(output_height, output_width,
                                            expected_index, buffer_indices)));
}

}  // namespace mediapipe
