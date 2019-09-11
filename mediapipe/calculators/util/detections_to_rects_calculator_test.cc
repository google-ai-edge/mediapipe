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

#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/message_matchers.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

Detection DetectionWithLocationData(int32 xmin, int32 ymin, int32 width,
                                    int32 height) {
  Detection detection;
  LocationData* location_data = detection.mutable_location_data();
  location_data->set_format(LocationData::BOUNDING_BOX);
  location_data->mutable_bounding_box()->set_xmin(xmin);
  location_data->mutable_bounding_box()->set_ymin(ymin);
  location_data->mutable_bounding_box()->set_width(width);
  location_data->mutable_bounding_box()->set_height(height);
  return detection;
}

Detection DetectionWithRelativeLocationData(double xmin, double ymin,
                                            double width, double height) {
  Detection detection;
  LocationData* location_data = detection.mutable_location_data();
  location_data->set_format(LocationData::RELATIVE_BOUNDING_BOX);
  location_data->mutable_relative_bounding_box()->set_xmin(xmin);
  location_data->mutable_relative_bounding_box()->set_ymin(ymin);
  location_data->mutable_relative_bounding_box()->set_width(width);
  location_data->mutable_relative_bounding_box()->set_height(height);
  return detection;
}

TEST(DetectionsToRectsCalculatorTest, DetectionToRect) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
    calculator: "DetectionsToRectsCalculator"
    input_stream: "DETECTION:detection"
    output_stream: "RECT:rect"
  )"));

  auto detection = absl::make_unique<Detection>(
      DetectionWithLocationData(100, 200, 300, 400));

  runner.MutableInputs()
      ->Tag("DETECTION")
      .packets.push_back(
          Adopt(detection.release()).At(Timestamp::PostStream()));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output = runner.Outputs().Tag("RECT").packets;
  ASSERT_EQ(1, output.size());
  const auto& rect = output[0].Get<Rect>();
  EXPECT_EQ(rect.width(), 300);
  EXPECT_EQ(rect.height(), 400);
  EXPECT_EQ(rect.x_center(), 250);
  EXPECT_EQ(rect.y_center(), 400);
}

TEST(DetectionsToRectsCalculatorTest, DetectionToNormalizedRect) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
    calculator: "DetectionsToRectsCalculator"
    input_stream: "DETECTION:detection"
    output_stream: "NORM_RECT:rect"
  )"));

  auto detection = absl::make_unique<Detection>(
      DetectionWithRelativeLocationData(0.1, 0.2, 0.3, 0.4));

  runner.MutableInputs()
      ->Tag("DETECTION")
      .packets.push_back(
          Adopt(detection.release()).At(Timestamp::PostStream()));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output = runner.Outputs().Tag("NORM_RECT").packets;
  ASSERT_EQ(1, output.size());
  const auto& rect = output[0].Get<NormalizedRect>();
  EXPECT_FLOAT_EQ(rect.width(), 0.3);
  EXPECT_FLOAT_EQ(rect.height(), 0.4);
  EXPECT_FLOAT_EQ(rect.x_center(), 0.25);
  EXPECT_FLOAT_EQ(rect.y_center(), 0.4);
}

TEST(DetectionsToRectsCalculatorTest, DetectionsToRect) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
    calculator: "DetectionsToRectsCalculator"
    input_stream: "DETECTIONS:detections"
    output_stream: "RECT:rect"
  )"));

  auto detections(absl::make_unique<std::vector<Detection>>());
  detections->push_back(DetectionWithLocationData(100, 200, 300, 400));
  detections->push_back(DetectionWithLocationData(200, 300, 400, 500));

  runner.MutableInputs()
      ->Tag("DETECTIONS")
      .packets.push_back(
          Adopt(detections.release()).At(Timestamp::PostStream()));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output = runner.Outputs().Tag("RECT").packets;
  ASSERT_EQ(1, output.size());
  const auto& rect = output[0].Get<Rect>();
  EXPECT_EQ(rect.width(), 300);
  EXPECT_EQ(rect.height(), 400);
  EXPECT_EQ(rect.x_center(), 250);
  EXPECT_EQ(rect.y_center(), 400);
}

TEST(DetectionsToRectsCalculatorTest, DetectionsToNormalizedRect) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
    calculator: "DetectionsToRectsCalculator"
    input_stream: "DETECTIONS:detections"
    output_stream: "NORM_RECT:rect"
  )"));

  auto detections(absl::make_unique<std::vector<Detection>>());
  detections->push_back(DetectionWithRelativeLocationData(0.1, 0.2, 0.3, 0.4));
  detections->push_back(DetectionWithRelativeLocationData(0.2, 0.3, 0.4, 0.5));

  runner.MutableInputs()
      ->Tag("DETECTIONS")
      .packets.push_back(
          Adopt(detections.release()).At(Timestamp::PostStream()));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output = runner.Outputs().Tag("NORM_RECT").packets;
  ASSERT_EQ(1, output.size());
  const auto& rect = output[0].Get<NormalizedRect>();
  EXPECT_FLOAT_EQ(rect.width(), 0.3);
  EXPECT_FLOAT_EQ(rect.height(), 0.4);
  EXPECT_FLOAT_EQ(rect.x_center(), 0.25);
  EXPECT_FLOAT_EQ(rect.y_center(), 0.4);
}

TEST(DetectionsToRectsCalculatorTest, DetectionsToRects) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
    calculator: "DetectionsToRectsCalculator"
    input_stream: "DETECTIONS:detections"
    output_stream: "RECTS:rect"
  )"));

  auto detections(absl::make_unique<std::vector<Detection>>());
  detections->push_back(DetectionWithLocationData(100, 200, 300, 400));
  detections->push_back(DetectionWithLocationData(200, 300, 400, 500));

  runner.MutableInputs()
      ->Tag("DETECTIONS")
      .packets.push_back(
          Adopt(detections.release()).At(Timestamp::PostStream()));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output = runner.Outputs().Tag("RECTS").packets;
  ASSERT_EQ(1, output.size());
  const auto& rects = output[0].Get<std::vector<Rect>>();
  EXPECT_EQ(rects.size(), 2);
  EXPECT_EQ(rects[0].width(), 300);
  EXPECT_EQ(rects[0].height(), 400);
  EXPECT_EQ(rects[0].x_center(), 250);
  EXPECT_EQ(rects[0].y_center(), 400);
  EXPECT_EQ(rects[1].width(), 400);
  EXPECT_EQ(rects[1].height(), 500);
  EXPECT_EQ(rects[1].x_center(), 400);
  EXPECT_EQ(rects[1].y_center(), 550);
}

TEST(DetectionsToRectsCalculatorTest, DetectionsToNormalizedRects) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
    calculator: "DetectionsToRectsCalculator"
    input_stream: "DETECTIONS:detections"
    output_stream: "NORM_RECTS:rect"
  )"));

  auto detections(absl::make_unique<std::vector<Detection>>());
  detections->push_back(DetectionWithRelativeLocationData(0.1, 0.2, 0.3, 0.4));
  detections->push_back(DetectionWithRelativeLocationData(0.2, 0.3, 0.4, 0.5));

  runner.MutableInputs()
      ->Tag("DETECTIONS")
      .packets.push_back(
          Adopt(detections.release()).At(Timestamp::PostStream()));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output =
      runner.Outputs().Tag("NORM_RECTS").packets;
  ASSERT_EQ(1, output.size());
  const auto& rects = output[0].Get<std::vector<NormalizedRect>>();
  EXPECT_EQ(rects.size(), 2);
  EXPECT_FLOAT_EQ(rects[0].width(), 0.3);
  EXPECT_FLOAT_EQ(rects[0].height(), 0.4);
  EXPECT_FLOAT_EQ(rects[0].x_center(), 0.25);
  EXPECT_FLOAT_EQ(rects[0].y_center(), 0.4);
  EXPECT_FLOAT_EQ(rects[1].width(), 0.4);
  EXPECT_FLOAT_EQ(rects[1].height(), 0.5);
  EXPECT_FLOAT_EQ(rects[1].x_center(), 0.4);
  EXPECT_FLOAT_EQ(rects[1].y_center(), 0.55);
}

TEST(DetectionsToRectsCalculatorTest, DetectionToRects) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
    calculator: "DetectionsToRectsCalculator"
    input_stream: "DETECTION:detection"
    output_stream: "RECTS:rect"
  )"));

  auto detection = absl::make_unique<Detection>(
      DetectionWithLocationData(100, 200, 300, 400));

  runner.MutableInputs()
      ->Tag("DETECTION")
      .packets.push_back(
          Adopt(detection.release()).At(Timestamp::PostStream()));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output = runner.Outputs().Tag("RECTS").packets;
  ASSERT_EQ(1, output.size());
  const auto& rects = output[0].Get<std::vector<Rect>>();
  EXPECT_EQ(rects.size(), 1);
  EXPECT_EQ(rects[0].width(), 300);
  EXPECT_EQ(rects[0].height(), 400);
  EXPECT_EQ(rects[0].x_center(), 250);
  EXPECT_EQ(rects[0].y_center(), 400);
}

TEST(DetectionsToRectsCalculatorTest, DetectionToNormalizedRects) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
    calculator: "DetectionsToRectsCalculator"
    input_stream: "DETECTION:detection"
    output_stream: "NORM_RECTS:rect"
  )"));

  auto detection = absl::make_unique<Detection>(
      DetectionWithRelativeLocationData(0.1, 0.2, 0.3, 0.4));

  runner.MutableInputs()
      ->Tag("DETECTION")
      .packets.push_back(
          Adopt(detection.release()).At(Timestamp::PostStream()));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output =
      runner.Outputs().Tag("NORM_RECTS").packets;
  ASSERT_EQ(1, output.size());
  const auto& rects = output[0].Get<std::vector<NormalizedRect>>();
  EXPECT_EQ(rects.size(), 1);
  EXPECT_FLOAT_EQ(rects[0].width(), 0.3);
  EXPECT_FLOAT_EQ(rects[0].height(), 0.4);
  EXPECT_FLOAT_EQ(rects[0].x_center(), 0.25);
  EXPECT_FLOAT_EQ(rects[0].y_center(), 0.4);
}

TEST(DetectionsToRectsCalculatorTest, WrongInputToRect) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
    calculator: "DetectionsToRectsCalculator"
    input_stream: "DETECTIONS:detections"
    output_stream: "RECT:rect"
  )"));

  auto detections(absl::make_unique<std::vector<Detection>>());
  detections->push_back(DetectionWithRelativeLocationData(0.1, 0.2, 0.3, 0.4));

  runner.MutableInputs()
      ->Tag("DETECTIONS")
      .packets.push_back(
          Adopt(detections.release()).At(Timestamp::PostStream()));

  ASSERT_THAT(
      runner.Run().message(),
      testing::HasSubstr("Only Detection with formats of BOUNDING_BOX"));
}

TEST(DetectionsToRectsCalculatorTest, WrongInputToNormalizedRect) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
    calculator: "DetectionsToRectsCalculator"
    input_stream: "DETECTIONS:detections"
    output_stream: "NORM_RECT:rect"
  )"));

  auto detections(absl::make_unique<std::vector<Detection>>());
  detections->push_back(DetectionWithLocationData(100, 200, 300, 400));

  runner.MutableInputs()
      ->Tag("DETECTIONS")
      .packets.push_back(
          Adopt(detections.release()).At(Timestamp::PostStream()));

  ASSERT_THAT(runner.Run().message(),
              testing::HasSubstr(
                  "Only Detection with formats of RELATIVE_BOUNDING_BOX"));
}

}  // namespace mediapipe
