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
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/gtest.h"
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

// Test normal case, where norm_width and norm_height in options are set.
TEST(ImageCroppingCalculatorTest, GetCroppingDimensionsNormal) {
  auto calculator_node =
      ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(
          R"(
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
          )");

  auto calculator_state = absl::make_unique<CalculatorState>(
      "Node", 0, "Calculator", calculator_node, nullptr);
  auto cc = absl::make_unique<CalculatorContext>(
      calculator_state.get(), tool::CreateTagMap({}).ValueOrDie(),
      tool::CreateTagMap({}).ValueOrDie());

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
          R"(
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
          )");

  auto calculator_state = absl::make_unique<CalculatorState>(
      "Node", 0, "Calculator", calculator_node, nullptr);
  auto cc = absl::make_unique<CalculatorContext>(
      calculator_state.get(), tool::CreateTagMap({}).ValueOrDie(),
      tool::CreateTagMap({}).ValueOrDie());
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
          R"(
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
          )");

  auto calculator_state = absl::make_unique<CalculatorState>(
      "Node", 0, "Calculator", calculator_node, nullptr);
  auto inputTags = tool::CreateTagMap({
                                          "HEIGHT:0:crop_height",
                                          "WIDTH:0:crop_width",
                                      })
                       .ValueOrDie();
  auto cc = absl::make_unique<CalculatorContext>(
      calculator_state.get(), inputTags, tool::CreateTagMap({}).ValueOrDie());
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
          R"(
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
          )");

  auto calculator_state = absl::make_unique<CalculatorState>(
      "Node", 0, "Calculator", calculator_node, nullptr);
  auto inputTags = tool::CreateTagMap({
                                          "RECT:0:rect",
                                      })
                       .ValueOrDie();
  auto cc = absl::make_unique<CalculatorContext>(
      calculator_state.get(), inputTags, tool::CreateTagMap({}).ValueOrDie());
  auto& inputs = cc->Inputs();
  mediapipe::Rect rect = ParseTextProtoOrDie<mediapipe::Rect>(
      R"(
        width: 1 height: 1 x_center: 40 y_center: 40 rotation: 0.5
      )");
  inputs.Tag(kRectTag).Value() = MakePacket<mediapipe::Rect>(rect);
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
