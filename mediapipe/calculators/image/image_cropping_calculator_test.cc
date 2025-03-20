// Copyright 2020 The MediaPipe Authors.
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

#include "mediapipe/calculators/image/image_cropping_calculator.h"

#include <cmath>
#include <memory>

#include "mediapipe/calculators/image/image_cropping_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/calculator_state.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/tag_map.h"
#include "mediapipe/framework/tool/tag_map_helper.h"

namespace mediapipe {

namespace {

constexpr int input_width = 100;
constexpr int input_height = 100;

constexpr char kRectTag[] = "RECT";
constexpr char kHeightTag[] = "HEIGHT";
constexpr char kWidthTag[] = "WIDTH";

std::unique_ptr<mediapipe::ImageFrame> GetInputFrame(int width, int height,
                                                     int channel) {
  const int total_size = width * height * channel;

  auto image_format = channel == 4 ? mediapipe::ImageFormat::SRGBA
                                   : mediapipe::ImageFormat::SRGB;

  auto input_frame = std::make_unique<mediapipe::ImageFrame>(
      image_format, width, height, /*alignment_boundary =*/1);
  for (int i = 0; i < total_size; ++i) {
    input_frame->MutablePixelData()[i] = i % 256;
  }

  return input_frame;
}

// Test identity function, where cropping size is same as input size
TEST(ImageCroppingCalculatorTest, IdentityFunctionCropWithOriginalSize) {
  auto calculator_node =
      ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(
          absl::Substitute(
              R"pb(
                calculator: "ImageCroppingCalculator"
                input_stream: "IMAGE:input_frames"
                output_stream: "IMAGE:cropped_output_frames"
                options: {
                  [mediapipe.ImageCroppingCalculatorOptions.ext] {
                    width: $0
                    height: $1
                  }
                }
              )pb",
              input_width, input_height));
  mediapipe::CalculatorRunner runner(calculator_node);

  // Input frame.
  const auto input_frame = GetInputFrame(input_width, input_height, 3);
  auto input_frame_packet =
      mediapipe::MakePacket<mediapipe::ImageFrame>(std::move(*input_frame));
  runner.MutableInputs()->Tag("IMAGE").packets.push_back(
      input_frame_packet.At(mediapipe::Timestamp(1)));

  MP_ASSERT_OK(runner.Run());

  const auto& outputs = runner.Outputs();
  EXPECT_EQ(outputs.NumEntries(), 1);
  const auto& output_image =
      outputs.Tag("IMAGE").packets[0].Get<mediapipe::ImageFrame>();

  const auto expected_output = GetInputFrame(input_width, input_height, 3);
  cv::Mat output_mat = formats::MatView(&output_image);
  cv::Mat expected_mat = formats::MatView(expected_output.get());
  double max_diff = cv::norm(expected_mat, output_mat, cv::NORM_INF);
  EXPECT_EQ(max_diff, 0);
}  // TEST

// Test identity function, where cropping size is same as input size.
// When an image has an odd number for its size, its center falls on a
// fractional pixel. As a result, the values for center_x and center_y need to
// be of type float.
TEST(ImageCroppingCalculatorTest, IdentityFunctionCropWithOddSize) {
  const int input_width = 99;
  const int input_height = 99;

  auto calculator_node =
      ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(
          absl::Substitute(
              R"pb(
                calculator: "ImageCroppingCalculator"
                input_stream: "IMAGE:input_frames"
                output_stream: "IMAGE:cropped_output_frames"
                options: {
                  [mediapipe.ImageCroppingCalculatorOptions.ext] {
                    width: $0
                    height: $1
                  }
                }
              )pb",
              input_width, input_height));
  mediapipe::CalculatorRunner runner(calculator_node);

  // Input frame.
  const auto input_frame = GetInputFrame(input_width, input_height, 3);
  auto input_frame_packet =
      mediapipe::MakePacket<mediapipe::ImageFrame>(std::move(*input_frame));
  runner.MutableInputs()->Tag("IMAGE").packets.push_back(
      input_frame_packet.At(mediapipe::Timestamp(1)));

  MP_ASSERT_OK(runner.Run());

  const auto& outputs = runner.Outputs();
  EXPECT_EQ(outputs.NumEntries(), 1);
  const auto& output_image =
      outputs.Tag("IMAGE").packets[0].Get<mediapipe::ImageFrame>();

  const auto expected_output = GetInputFrame(input_width, input_height, 3);
  cv::Mat output_mat = formats::MatView(&output_image);
  cv::Mat expected_mat = formats::MatView(expected_output.get());
  double max_diff = cv::norm(expected_mat, output_mat, cv::NORM_INF);
  EXPECT_EQ(max_diff, 0);
}  // TEST

// Test identity function on GPU, where cropping size is same as input size.
TEST(ImageCroppingCalculatorTest, IdentityFunctionCropWithOriginalSizeGPU) {
  mediapipe::CalculatorGraphConfig config =
      ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(absl::Substitute(
          R"pb(
            input_stream: "input_frames"
            node {
              calculator: "ImageFrameToGpuBufferCalculator"
              input_stream: "input_frames"
              output_stream: "input_frames_gpu"
            }
            node {
              calculator: "ImageCroppingCalculator"
              input_stream: "IMAGE_GPU:input_frames_gpu"
              output_stream: "IMAGE_GPU:cropped_output_frames_gpu"
              options: {
                [mediapipe.ImageCroppingCalculatorOptions.ext] {
                  width: $0
                  height: $1
                }
              }
            }
            node {
              calculator: "GpuBufferToImageFrameCalculator"
              input_stream: "cropped_output_frames_gpu"
              output_stream: "cropped_output_frames"
            }
          )pb",
          input_width, input_height));

  std::vector<Packet> output_packets;
  tool::AddVectorSink("cropped_output_frames", &config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));

  // Input frame.
  const auto input_frame = GetInputFrame(input_width, input_height, 4);
  auto input_frame_packet =
      mediapipe::MakePacket<mediapipe::ImageFrame>(std::move(*input_frame));

  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_frames", input_frame_packet.At(mediapipe::Timestamp(1))));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Get and process results.
  const ImageFrame& output_image = output_packets[0].Get<ImageFrame>();

  const auto expected_output = GetInputFrame(input_width, input_height, 4);
  cv::Mat output_mat = formats::MatView(&output_image);
  cv::Mat expected_mat = formats::MatView(expected_output.get());
  double max_diff = cv::norm(expected_mat, output_mat, cv::NORM_INF);

  EXPECT_EQ(max_diff, 0);
}  // TEST

// Test normal case, where norm_width and norm_height in options are set.
TEST(ImageCroppingCalculatorTest, GetCroppingDimensionsNormal) {
  auto calculator_node =
      ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(
          R"pb(
            calculator: "ImageCroppingCalculator"
            input_stream: "IMAGE_GPU:input_frames"
            output_stream: "IMAGE_GPU:cropped_output_frames"
            options: {
              [mediapipe.ImageCroppingCalculatorOptions.ext] {
                norm_width: 0.6
                norm_height: 0.6
                norm_center_x: 0.5
                norm_center_y: 0.5
                rotation: 0.3
              }
            }
          )pb");

  auto calculator_state = std::make_unique<CalculatorState>(
      "Node", 0, "Calculator", calculator_node, /*profiling_context=*/nullptr,
      /*graph_service_manager=*/nullptr);
  auto cc = std::make_unique<CalculatorContext>(calculator_state.get(),
                                                tool::CreateTagMap({}).value(),
                                                tool::CreateTagMap({}).value());

  RectSpec expectRect = {
      .width = 60,
      .height = 60,
      .center_x = 50,
      .center_y = 50,
      .rotation = 0.3,
  };
  EXPECT_EQ(ImageCroppingCalculator::GetCropSpecs(cc.get(), input_width,
                                                  input_height),
            expectRect);
}  // TEST

// Test when (width height) + (norm_width norm_height) are set in options.
// width and height should take precedence.
TEST(ImageCroppingCalculatorTest, RedundantSpecInOptions) {
  auto calculator_node =
      ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(
          R"pb(
            calculator: "ImageCroppingCalculator"
            input_stream: "IMAGE_GPU:input_frames"
            output_stream: "IMAGE_GPU:cropped_output_frames"
            options: {
              [mediapipe.ImageCroppingCalculatorOptions.ext] {
                width: 50
                height: 50
                norm_width: 0.6
                norm_height: 0.6
                norm_center_x: 0.5
                norm_center_y: 0.5
                rotation: 0.3
              }
            }
          )pb");

  auto calculator_state = std::make_unique<CalculatorState>(
      "Node", 0, "Calculator", calculator_node, /*profiling_context=*/nullptr,
      /*graph_service_manager=*/nullptr);
  auto cc = std::make_unique<CalculatorContext>(calculator_state.get(),
                                                tool::CreateTagMap({}).value(),
                                                tool::CreateTagMap({}).value());
  RectSpec expectRect = {
      .width = 50,
      .height = 50,
      .center_x = 50,
      .center_y = 50,
      .rotation = 0.3,
  };
  EXPECT_EQ(ImageCroppingCalculator::GetCropSpecs(cc.get(), input_width,
                                                  input_height),
            expectRect);
}  // TEST

// Test when WIDTH HEIGHT are set from input stream,
// and options has norm_width/height set.
// WIDTH HEIGHT from input stream should take precedence.
TEST(ImageCroppingCalculatorTest, RedundantSpectWithInputStream) {
  auto calculator_node =
      ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(
          R"pb(
            calculator: "ImageCroppingCalculator"
            input_stream: "IMAGE_GPU:input_frames"
            input_stream: "WIDTH:crop_width"
            input_stream: "HEIGHT:crop_height"
            output_stream: "IMAGE_GPU:cropped_output_frames"
            options: {
              [mediapipe.ImageCroppingCalculatorOptions.ext] {
                width: 50
                height: 50
                norm_width: 0.6
                norm_height: 0.6
                norm_center_x: 0.5
                norm_center_y: 0.5
                rotation: 0.3
              }
            }
          )pb");

  auto calculator_state = std::make_unique<CalculatorState>(
      "Node", 0, "Calculator", calculator_node, /*profiling_context=*/nullptr,
      /*graph_service_manager=*/nullptr);
  auto inputTags = tool::CreateTagMap({
                                          "HEIGHT:0:crop_height",
                                          "WIDTH:0:crop_width",
                                      })
                       .value();
  auto cc = std::make_unique<CalculatorContext>(
      calculator_state.get(), inputTags, tool::CreateTagMap({}).value());
  auto& inputs = cc->Inputs();
  inputs.Tag(kHeightTag).Value() = MakePacket<int>(1);
  inputs.Tag(kWidthTag).Value() = MakePacket<int>(1);
  RectSpec expectRect = {
      .width = 1,
      .height = 1,
      .center_x = 50,
      .center_y = 50,
      .rotation = 0.3,
  };
  EXPECT_EQ(ImageCroppingCalculator::GetCropSpecs(cc.get(), input_width,
                                                  input_height),
            expectRect);
}  // TEST

// Test when RECT is set from input stream,
// and options has norm_width/height set.
// RECT from input stream should take precedence.
TEST(ImageCroppingCalculatorTest, RedundantSpecWithInputStream) {
  auto calculator_node =
      ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(
          R"pb(
            calculator: "ImageCroppingCalculator"
            input_stream: "IMAGE_GPU:input_frames"
            input_stream: "RECT:rect"
            output_stream: "IMAGE_GPU:cropped_output_frames"
            options: {
              [mediapipe.ImageCroppingCalculatorOptions.ext] {
                width: 50
                height: 50
                norm_width: 0.6
                norm_height: 0.6
                norm_center_x: 0.5
                norm_center_y: 0.5
                rotation: 0.3
              }
            }
          )pb");

  auto calculator_state = std::make_unique<CalculatorState>(
      "Node", 0, "Calculator", calculator_node, /*profiling_context=*/nullptr,
      /*graph_service_manager=*/nullptr);
  auto inputTags = tool::CreateTagMap({
                                          "RECT:0:rect",
                                      })
                       .value();
  auto cc = std::make_unique<CalculatorContext>(
      calculator_state.get(), inputTags, tool::CreateTagMap({}).value());
  auto& inputs = cc->Inputs();
  Rect rect = ParseTextProtoOrDie<Rect>(
      R"pb(
        width: 1 height: 1 x_center: 40 y_center: 40 rotation: 0.5
      )pb");
  inputs.Tag(kRectTag).Value() = MakePacket<Rect>(rect);
  RectSpec expectRect = {
      .width = 1,
      .height = 1,
      .center_x = 40,
      .center_y = 40,
      .rotation = 0.5,
  };
  EXPECT_EQ(ImageCroppingCalculator::GetCropSpecs(cc.get(), input_width,
                                                  input_height),
            expectRect);
}  // TEST

}  // namespace
}  // namespace mediapipe
