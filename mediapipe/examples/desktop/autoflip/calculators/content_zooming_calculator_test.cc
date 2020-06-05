// Copyright 2019 The MediaPipe Authors.
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

#include <memory>

#include "mediapipe/examples/desktop/autoflip/autoflip_messages.pb.h"
#include "mediapipe/examples/desktop/autoflip/calculators/content_zooming_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/benchmark.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace autoflip {
namespace {

const char kConfigA[] = R"(
    calculator: "ContentZoomingCalculator"
    input_stream: "VIDEO:camera_frames"
    input_stream: "DETECTIONS:detection_set"
    output_stream: "BORDERS:borders"
    )";

const char kConfigB[] = R"(
    calculator: "ContentZoomingCalculator"
    input_stream: "VIDEO:camera_frames"
    input_stream: "DETECTIONS:detection_set"
    output_stream: "BORDERS:borders"
    options: {
      [mediapipe.autoflip.ContentZoomingCalculatorOptions.ext]: {
        target_size {
          width: 1000
          height: 500
        }
      }
    }
    )";

const char kConfigC[] = R"(
    calculator: "ContentZoomingCalculator"
    input_stream: "VIDEO_SIZE:size"
    input_stream: "DETECTIONS:detection_set"
    output_stream: "BORDERS:borders"
    )";

void CheckBorder(const StaticFeatures& static_features, int width, int height,
                 int top_border, int bottom_border) {
  ASSERT_EQ(2, static_features.border().size());
  auto part = static_features.border(0);
  EXPECT_EQ(part.border_position().x(), 0);
  EXPECT_EQ(part.border_position().y(), 0);
  EXPECT_EQ(part.border_position().width(), width);
  EXPECT_EQ(part.border_position().height(), top_border);
  EXPECT_EQ(Border::TOP, part.relative_position());

  part = static_features.border(1);
  EXPECT_EQ(part.border_position().x(), 0);
  EXPECT_EQ(part.border_position().y(), height - bottom_border);
  EXPECT_EQ(part.border_position().width(), width);
  EXPECT_EQ(part.border_position().height(), bottom_border);
  EXPECT_EQ(Border::BOTTOM, part.relative_position());
}

TEST(ContentZoomingCalculatorTest, ZoomTest) {
  auto runner = ::absl::make_unique<CalculatorRunner>(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigA));
  auto detection_set = std::make_unique<DetectionSet>();
  auto* detection = detection_set->add_detections();
  detection->set_only_required(true);
  auto* location = detection->mutable_location_normalized();
  location->set_height(.1);
  location->set_width(.1);
  location->set_x(.4);
  location->set_y(.5);

  auto input_frame =
      ::absl::make_unique<ImageFrame>(ImageFormat::SRGB, 1000, 1000);
  runner->MutableInputs()->Tag("VIDEO").packets.push_back(
      Adopt(input_frame.release()).At(Timestamp(0)));

  runner->MutableInputs()
      ->Tag("DETECTIONS")
      .packets.push_back(Adopt(detection_set.release()).At(Timestamp(0)));

  // Run the calculator.
  MP_ASSERT_OK(runner->Run());

  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag("BORDERS").packets;
  ASSERT_EQ(1, output_packets.size());
  const auto& static_features = output_packets[0].Get<StaticFeatures>();
  CheckBorder(static_features, 1000, 1000, 495, 395);
}

TEST(ContentZoomingCalculatorTest, MinAspectBorderValues) {
  auto runner = ::absl::make_unique<CalculatorRunner>(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigB));
  auto detection_set = std::make_unique<DetectionSet>();
  auto* detection = detection_set->add_detections();
  detection->set_only_required(true);
  auto* location = detection->mutable_location_normalized();
  location->set_height(1);
  location->set_width(1);
  location->set_x(0);
  location->set_y(0);

  auto input_frame =
      ::absl::make_unique<ImageFrame>(ImageFormat::SRGB, 1000, 1000);
  runner->MutableInputs()->Tag("VIDEO").packets.push_back(
      Adopt(input_frame.release()).At(Timestamp(0)));

  runner->MutableInputs()
      ->Tag("DETECTIONS")
      .packets.push_back(Adopt(detection_set.release()).At(Timestamp(0)));

  // Run the calculator.
  MP_ASSERT_OK(runner->Run());

  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag("BORDERS").packets;
  ASSERT_EQ(1, output_packets.size());
  const auto& static_features = output_packets[0].Get<StaticFeatures>();
  CheckBorder(static_features, 1000, 1000, 250, 250);
}

TEST(ContentZoomingCalculatorTest, TwoFacesWide) {
  auto runner = ::absl::make_unique<CalculatorRunner>(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigA));
  auto detection_set = std::make_unique<DetectionSet>();
  auto* detection = detection_set->add_detections();
  detection->set_only_required(true);
  auto* location = detection->mutable_location_normalized();
  location->set_height(.2);
  location->set_width(.2);
  location->set_x(.2);
  location->set_y(.4);

  location = detection->mutable_location_normalized();
  location->set_height(.2);
  location->set_width(.2);
  location->set_x(.6);
  location->set_y(.4);

  auto input_frame =
      ::absl::make_unique<ImageFrame>(ImageFormat::SRGB, 1000, 1000);
  runner->MutableInputs()->Tag("VIDEO").packets.push_back(
      Adopt(input_frame.release()).At(Timestamp(0)));

  runner->MutableInputs()
      ->Tag("DETECTIONS")
      .packets.push_back(Adopt(detection_set.release()).At(Timestamp(0)));

  // Run the calculator.
  MP_ASSERT_OK(runner->Run());

  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag("BORDERS").packets;
  ASSERT_EQ(1, output_packets.size());
  const auto& static_features = output_packets[0].Get<StaticFeatures>();

  CheckBorder(static_features, 1000, 1000, 389, 389);
}

TEST(ContentZoomingCalculatorTest, NoDetectionOnInit) {
  auto runner = ::absl::make_unique<CalculatorRunner>(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigA));
  auto detection_set = std::make_unique<DetectionSet>();

  auto input_frame =
      ::absl::make_unique<ImageFrame>(ImageFormat::SRGB, 1000, 1000);
  runner->MutableInputs()->Tag("VIDEO").packets.push_back(
      Adopt(input_frame.release()).At(Timestamp(0)));

  runner->MutableInputs()
      ->Tag("DETECTIONS")
      .packets.push_back(Adopt(detection_set.release()).At(Timestamp(0)));

  // Run the calculator.
  MP_ASSERT_OK(runner->Run());

  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag("BORDERS").packets;
  ASSERT_EQ(1, output_packets.size());
  const auto& static_features = output_packets[0].Get<StaticFeatures>();

  CheckBorder(static_features, 1000, 1000, 0, 0);
}

TEST(ContentZoomingCalculatorTest, ZoomTestPairSize) {
  auto runner = ::absl::make_unique<CalculatorRunner>(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigC));
  auto detection_set = std::make_unique<DetectionSet>();
  auto* detection = detection_set->add_detections();
  detection->set_only_required(true);
  auto* location = detection->mutable_location_normalized();
  location->set_height(.1);
  location->set_width(.1);
  location->set_x(.4);
  location->set_y(.5);

  auto input_size = ::absl::make_unique<std::pair<int, int>>(1000, 1000);
  runner->MutableInputs()
      ->Tag("VIDEO_SIZE")
      .packets.push_back(Adopt(input_size.release()).At(Timestamp(0)));

  runner->MutableInputs()
      ->Tag("DETECTIONS")
      .packets.push_back(Adopt(detection_set.release()).At(Timestamp(0)));

  // Run the calculator.
  MP_ASSERT_OK(runner->Run());

  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag("BORDERS").packets;
  ASSERT_EQ(1, output_packets.size());
  const auto& static_features = output_packets[0].Get<StaticFeatures>();
  CheckBorder(static_features, 1000, 1000, 495, 395);
}

}  // namespace
}  // namespace autoflip

}  // namespace mediapipe
