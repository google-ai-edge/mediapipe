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

#include <algorithm>
#include <memory>
#include <vector>

#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

constexpr char kNormRectsTag[] = "NORM_RECTS";
constexpr char kRectsTag[] = "RECTS";
constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kRectTag[] = "RECT";
constexpr char kDetectionTag[] = "DETECTION";

using ::mediapipe::NormalizedRect;
using ::mediapipe::Rect;

MATCHER_P4(RectEq, x_center, y_center, width, height, "") {
  return testing::Value(arg.x_center(), testing::Eq(x_center)) &&
         testing::Value(arg.y_center(), testing::Eq(y_center)) &&
         testing::Value(arg.width(), testing::Eq(width)) &&
         testing::Value(arg.height(), testing::Eq(height));
}

MATCHER_P4(NormRectEq, x_center, y_center, width, height, "") {
  return testing::Value(arg.x_center(), testing::FloatEq(x_center)) &&
         testing::Value(arg.y_center(), testing::FloatEq(y_center)) &&
         testing::Value(arg.width(), testing::FloatEq(width)) &&
         testing::Value(arg.height(), testing::FloatEq(height));
}

Detection DetectionWithLocationData(int32_t xmin, int32_t ymin, int32_t width,
                                    int32_t height) {
  Detection detection;
  LocationData* location_data = detection.mutable_location_data();
  location_data->set_format(LocationData::BOUNDING_BOX);
  location_data->mutable_bounding_box()->set_xmin(xmin);
  location_data->mutable_bounding_box()->set_ymin(ymin);
  location_data->mutable_bounding_box()->set_width(width);
  location_data->mutable_bounding_box()->set_height(height);
  return detection;
}

Detection DetectionWithKeyPoints(
    const std::vector<std::pair<float, float>>& key_points) {
  Detection detection;
  LocationData* location_data = detection.mutable_location_data();
  std::for_each(key_points.begin(), key_points.end(),
                [location_data](std::pair<float, float> kp) {
                  auto* new_kp = location_data->add_relative_keypoints();
                  new_kp->set_x(kp.first);
                  new_kp->set_y(kp.second);
                });
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
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "DetectionsToRectsCalculator"
    input_stream: "DETECTION:detection"
    output_stream: "RECT:rect"
  )pb"));

  auto detection = absl::make_unique<Detection>(
      DetectionWithLocationData(100, 200, 300, 400));

  runner.MutableInputs()
      ->Tag(kDetectionTag)
      .packets.push_back(
          Adopt(detection.release()).At(Timestamp::PostStream()));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output = runner.Outputs().Tag(kRectTag).packets;
  ASSERT_EQ(1, output.size());
  const auto& rect = output[0].Get<Rect>();
  EXPECT_THAT(rect, RectEq(250, 400, 300, 400));
}

absl::StatusOr<Rect> RunDetectionKeyPointsToRectCalculation(
    Detection detection, std::pair<int, int> image_size) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "DetectionsToRectsCalculator"
    input_stream: "DETECTION:detection"
    input_stream: "IMAGE_SIZE:image_size"
    output_stream: "RECT:rect"
    options: {
      [mediapipe.DetectionsToRectsCalculatorOptions.ext] {
        conversion_mode: USE_KEYPOINTS
      }
    }
  )pb"));

  runner.MutableInputs()
      ->Tag(kDetectionTag)
      .packets.push_back(MakePacket<Detection>(std::move(detection))
                             .At(Timestamp::PostStream()));
  runner.MutableInputs()
      ->Tag(kImageSizeTag)
      .packets.push_back(MakePacket<std::pair<int, int>>(image_size)
                             .At(Timestamp::PostStream()));

  MP_RETURN_IF_ERROR(runner.Run());
  const std::vector<Packet>& output = runner.Outputs().Tag(kRectTag).packets;
  RET_CHECK_EQ(output.size(), 1);
  return output[0].Get<Rect>();
}

TEST(DetectionsToRectsCalculatorTest, DetectionKeyPointsToRect) {
  auto status_or_value = RunDetectionKeyPointsToRectCalculation(
      /*detection=*/DetectionWithKeyPoints({{0.0f, 0.0f}, {1.0f, 1.0f}}),
      /*image_size=*/{640, 480});
  EXPECT_THAT(status_or_value.value(), RectEq(320, 240, 640, 480));

  status_or_value = RunDetectionKeyPointsToRectCalculation(
      /*detection=*/DetectionWithKeyPoints({{0.25f, 0.25f}, {0.75f, 0.75f}}),
      /*image_size=*/{640, 480});
  MP_ASSERT_OK(status_or_value);
  EXPECT_THAT(status_or_value.value(), RectEq(320, 240, 320, 240));

  status_or_value = RunDetectionKeyPointsToRectCalculation(
      /*detection=*/DetectionWithKeyPoints({{0.0f, 0.0f}, {0.5f, 0.5f}}),
      /*image_size=*/{640, 480});
  MP_ASSERT_OK(status_or_value);
  EXPECT_THAT(status_or_value.value(), RectEq(160, 120, 320, 240));

  status_or_value = RunDetectionKeyPointsToRectCalculation(
      /*detection=*/DetectionWithKeyPoints({{0.5f, 0.5f}, {1.0f, 1.0f}}),
      /*image_size=*/{640, 480});
  MP_ASSERT_OK(status_or_value);
  EXPECT_THAT(status_or_value.value(), RectEq(480, 360, 320, 240));

  status_or_value = RunDetectionKeyPointsToRectCalculation(
      /*detection=*/DetectionWithKeyPoints({{0.25f, 0.25f}, {0.75f, 0.75f}}),
      /*image_size=*/{0, 0});
  MP_ASSERT_OK(status_or_value);
  EXPECT_THAT(status_or_value.value(), RectEq(0, 0, 0, 0));
}

TEST(DetectionsToRectsCalculatorTest, DetectionToNormalizedRect) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "DetectionsToRectsCalculator"
    input_stream: "DETECTION:detection"
    output_stream: "NORM_RECT:rect"
  )pb"));

  auto detection = absl::make_unique<Detection>(
      DetectionWithRelativeLocationData(0.1, 0.2, 0.3, 0.4));

  runner.MutableInputs()
      ->Tag(kDetectionTag)
      .packets.push_back(
          Adopt(detection.release()).At(Timestamp::PostStream()));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output =
      runner.Outputs().Tag(kNormRectTag).packets;
  ASSERT_EQ(1, output.size());
  const auto& rect = output[0].Get<NormalizedRect>();
  EXPECT_THAT(rect, NormRectEq(0.25f, 0.4f, 0.3f, 0.4f));
}

absl::StatusOr<NormalizedRect> RunDetectionKeyPointsToNormRectCalculation(
    Detection detection) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "DetectionsToRectsCalculator"
    input_stream: "DETECTION:detection"
    output_stream: "NORM_RECT:rect"
    options: {
      [mediapipe.DetectionsToRectsCalculatorOptions.ext] {
        conversion_mode: USE_KEYPOINTS
      }
    }
  )pb"));

  runner.MutableInputs()
      ->Tag(kDetectionTag)
      .packets.push_back(MakePacket<Detection>(std::move(detection))
                             .At(Timestamp::PostStream()));

  MP_RETURN_IF_ERROR(runner.Run());
  const std::vector<Packet>& output =
      runner.Outputs().Tag(kNormRectTag).packets;
  RET_CHECK_EQ(output.size(), 1);
  return output[0].Get<NormalizedRect>();
}

TEST(DetectionsToRectsCalculatorTest, DetectionKeyPointsToNormalizedRect) {
  NormalizedRect rect;

  auto status_or_value = RunDetectionKeyPointsToNormRectCalculation(
      /*detection=*/DetectionWithKeyPoints(
          {{0.0f, 0.0f}, {0.5f, 0.5f}, {1.0f, 1.0f}}));
  MP_ASSERT_OK(status_or_value);
  EXPECT_THAT(status_or_value.value(), RectEq(0.5f, 0.5f, 1.0f, 1.0f));

  status_or_value = RunDetectionKeyPointsToNormRectCalculation(
      /*detection=*/DetectionWithKeyPoints(
          {{0.25f, 0.25f}, {0.75f, 0.25f}, {0.75f, 0.75f}}));
  EXPECT_THAT(status_or_value.value(), RectEq(0.5f, 0.5f, 0.5f, 0.5f));

  status_or_value = RunDetectionKeyPointsToNormRectCalculation(
      /*detection=*/DetectionWithKeyPoints({{0.0f, 0.0f}, {0.5f, 0.5f}}));
  MP_ASSERT_OK(status_or_value);
  EXPECT_THAT(status_or_value.value(), RectEq(0.25f, 0.25f, 0.5f, 0.5f));

  status_or_value = RunDetectionKeyPointsToNormRectCalculation(
      /*detection=*/DetectionWithKeyPoints({{0.5f, 0.5f}, {1.0f, 1.0f}}));
  MP_ASSERT_OK(status_or_value);
  EXPECT_THAT(status_or_value.value(), RectEq(0.75f, 0.75f, 0.5f, 0.5f));
}

TEST(DetectionsToRectsCalculatorTest, DetectionsToRect) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "DetectionsToRectsCalculator"
    input_stream: "DETECTIONS:detections"
    output_stream: "RECT:rect"
  )pb"));

  auto detections(absl::make_unique<std::vector<Detection>>());
  detections->push_back(DetectionWithLocationData(100, 200, 300, 400));
  detections->push_back(DetectionWithLocationData(200, 300, 400, 500));

  runner.MutableInputs()
      ->Tag(kDetectionsTag)
      .packets.push_back(
          Adopt(detections.release()).At(Timestamp::PostStream()));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output = runner.Outputs().Tag(kRectTag).packets;
  ASSERT_EQ(1, output.size());
  const auto& rect = output[0].Get<Rect>();
  EXPECT_THAT(rect, RectEq(250, 400, 300, 400));
}

TEST(DetectionsToRectsCalculatorTest, DetectionsToNormalizedRect) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "DetectionsToRectsCalculator"
    input_stream: "DETECTIONS:detections"
    output_stream: "NORM_RECT:rect"
  )pb"));

  auto detections(absl::make_unique<std::vector<Detection>>());
  detections->push_back(DetectionWithRelativeLocationData(0.1, 0.2, 0.3, 0.4));
  detections->push_back(DetectionWithRelativeLocationData(0.2, 0.3, 0.4, 0.5));

  runner.MutableInputs()
      ->Tag(kDetectionsTag)
      .packets.push_back(
          Adopt(detections.release()).At(Timestamp::PostStream()));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output =
      runner.Outputs().Tag(kNormRectTag).packets;
  ASSERT_EQ(1, output.size());
  const auto& rect = output[0].Get<NormalizedRect>();
  EXPECT_THAT(rect, NormRectEq(0.25f, 0.4f, 0.3f, 0.4f));
}

TEST(DetectionsToRectsCalculatorTest, DetectionsToRects) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "DetectionsToRectsCalculator"
    input_stream: "DETECTIONS:detections"
    output_stream: "RECTS:rect"
  )pb"));

  auto detections(absl::make_unique<std::vector<Detection>>());
  detections->push_back(DetectionWithLocationData(100, 200, 300, 400));
  detections->push_back(DetectionWithLocationData(200, 300, 400, 500));

  runner.MutableInputs()
      ->Tag(kDetectionsTag)
      .packets.push_back(
          Adopt(detections.release()).At(Timestamp::PostStream()));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output = runner.Outputs().Tag(kRectsTag).packets;
  ASSERT_EQ(1, output.size());
  const auto& rects = output[0].Get<std::vector<Rect>>();
  ASSERT_EQ(rects.size(), 2);
  EXPECT_THAT(rects[0], RectEq(250, 400, 300, 400));
  EXPECT_THAT(rects[1], RectEq(400, 550, 400, 500));
}

TEST(DetectionsToRectsCalculatorTest, DetectionsToNormalizedRects) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "DetectionsToRectsCalculator"
    input_stream: "DETECTIONS:detections"
    output_stream: "NORM_RECTS:rect"
  )pb"));

  auto detections(absl::make_unique<std::vector<Detection>>());
  detections->push_back(DetectionWithRelativeLocationData(0.1, 0.2, 0.3, 0.4));
  detections->push_back(DetectionWithRelativeLocationData(0.2, 0.3, 0.4, 0.5));

  runner.MutableInputs()
      ->Tag(kDetectionsTag)
      .packets.push_back(
          Adopt(detections.release()).At(Timestamp::PostStream()));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output =
      runner.Outputs().Tag(kNormRectsTag).packets;
  ASSERT_EQ(1, output.size());
  const auto& rects = output[0].Get<std::vector<NormalizedRect>>();
  ASSERT_EQ(rects.size(), 2);
  EXPECT_THAT(rects[0], NormRectEq(0.25f, 0.4f, 0.3f, 0.4f));
  EXPECT_THAT(rects[1], NormRectEq(0.4f, 0.55f, 0.4f, 0.5f));
}

TEST(DetectionsToRectsCalculatorTest, DetectionToRects) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "DetectionsToRectsCalculator"
    input_stream: "DETECTION:detection"
    output_stream: "RECTS:rect"
  )pb"));

  auto detection = absl::make_unique<Detection>(
      DetectionWithLocationData(100, 200, 300, 400));

  runner.MutableInputs()
      ->Tag(kDetectionTag)
      .packets.push_back(
          Adopt(detection.release()).At(Timestamp::PostStream()));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output = runner.Outputs().Tag(kRectsTag).packets;
  ASSERT_EQ(1, output.size());
  const auto& rects = output[0].Get<std::vector<Rect>>();
  EXPECT_EQ(rects.size(), 1);
  EXPECT_THAT(rects[0], RectEq(250, 400, 300, 400));
}

TEST(DetectionsToRectsCalculatorTest, DetectionToNormalizedRects) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "DetectionsToRectsCalculator"
    input_stream: "DETECTION:detection"
    output_stream: "NORM_RECTS:rect"
  )pb"));

  auto detection = absl::make_unique<Detection>(
      DetectionWithRelativeLocationData(0.1, 0.2, 0.3, 0.4));

  runner.MutableInputs()
      ->Tag(kDetectionTag)
      .packets.push_back(
          Adopt(detection.release()).At(Timestamp::PostStream()));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output =
      runner.Outputs().Tag(kNormRectsTag).packets;
  ASSERT_EQ(1, output.size());
  const auto& rects = output[0].Get<std::vector<NormalizedRect>>();
  ASSERT_EQ(rects.size(), 1);
  EXPECT_THAT(rects[0], NormRectEq(0.25f, 0.4f, 0.3f, 0.4f));
}

TEST(DetectionsToRectsCalculatorTest, WrongInputToRect) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "DetectionsToRectsCalculator"
    input_stream: "DETECTIONS:detections"
    output_stream: "RECT:rect"
  )pb"));

  auto detections(absl::make_unique<std::vector<Detection>>());
  detections->push_back(DetectionWithRelativeLocationData(0.1, 0.2, 0.3, 0.4));

  runner.MutableInputs()
      ->Tag(kDetectionsTag)
      .packets.push_back(
          Adopt(detections.release()).At(Timestamp::PostStream()));

  ASSERT_THAT(
      runner.Run().message(),
      testing::HasSubstr("Only Detection with formats of BOUNDING_BOX"));
}

TEST(DetectionsToRectsCalculatorTest, WrongInputToNormalizedRect) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "DetectionsToRectsCalculator"
    input_stream: "DETECTIONS:detections"
    output_stream: "NORM_RECT:rect"
  )pb"));

  auto detections(absl::make_unique<std::vector<Detection>>());
  detections->push_back(DetectionWithLocationData(100, 200, 300, 400));

  runner.MutableInputs()
      ->Tag(kDetectionsTag)
      .packets.push_back(
          Adopt(detections.release()).At(Timestamp::PostStream()));

  ASSERT_THAT(runner.Run().message(),
              testing::HasSubstr(
                  "Only Detection with formats of RELATIVE_BOUNDING_BOX"));
}

}  // namespace
}  // namespace mediapipe
