// Copyright 2022 The MediaPipe Authors.
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
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

constexpr char kDetectionTag[] = "DETECTION";
constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kPixelDetectionTag[] = "PIXEL_DETECTION";
constexpr char kPixelDetectionListTag[] = "PIXEL_DETECTION_LIST";
constexpr char kPixelDetectionsTag[] = "PIXEL_DETECTIONS";
constexpr char kRelativeDetectionListTag[] = "RELATIVE_DETECTION_LIST";
constexpr char kRelativeDetectionsTag[] = "RELATIVE_DETECTIONS";

Detection DetectionWithBoundingBox(int32_t xmin, int32_t ymin, int32_t width,
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

Detection DetectionWithRelativeBoundingBox(float xmin, float ymin, float width,
                                           float height) {
  Detection detection;
  LocationData* location_data = detection.mutable_location_data();
  location_data->set_format(LocationData::RELATIVE_BOUNDING_BOX);
  location_data->mutable_relative_bounding_box()->set_xmin(xmin);
  location_data->mutable_relative_bounding_box()->set_ymin(ymin);
  location_data->mutable_relative_bounding_box()->set_width(width);
  location_data->mutable_relative_bounding_box()->set_height(height);
  return detection;
}

std::vector<Detection> ConvertToDetectionVector(
    const DetectionList& detection_list) {
  std::vector<Detection> output;
  for (const auto& detection : detection_list.detection()) {
    output.push_back(detection);
  }
  return output;
}

void CheckBoundingBox(const Detection& output, const Detection& expected) {
  const auto& output_bbox = output.location_data().bounding_box();
  const auto& expected_bbox = output.location_data().bounding_box();
  EXPECT_THAT(output_bbox.xmin(), testing::Eq(expected_bbox.xmin()));
  EXPECT_THAT(output_bbox.ymin(), testing::Eq(expected_bbox.ymin()));
  EXPECT_THAT(output_bbox.width(), testing::Eq(expected_bbox.width()));
  EXPECT_THAT(output_bbox.height(), testing::Eq(expected_bbox.height()));
}

void CheckRelativeBoundingBox(const Detection& output,
                              const Detection& expected) {
  const auto& output_bbox = output.location_data().relative_bounding_box();
  const auto& expected_bbox = output.location_data().relative_bounding_box();
  EXPECT_THAT(output_bbox.xmin(), testing::FloatEq(expected_bbox.xmin()));
  EXPECT_THAT(output_bbox.ymin(), testing::FloatEq(expected_bbox.ymin()));
  EXPECT_THAT(output_bbox.width(), testing::FloatEq(expected_bbox.width()));
  EXPECT_THAT(output_bbox.height(), testing::FloatEq(expected_bbox.height()));
}

void CheckOutputDetections(const std::vector<Detection>& expected,
                           const std::vector<Detection>& output) {
  ASSERT_EQ(output.size(), expected.size());
  for (int i = 0; i < output.size(); ++i) {
    auto output_format = output[i].location_data().format();
    ASSERT_TRUE(output_format == LocationData::RELATIVE_BOUNDING_BOX ||
                output_format == LocationData::BOUNDING_BOX);
    ASSERT_EQ(output_format, expected[i].location_data().format());
    if (output_format == LocationData::RELATIVE_BOUNDING_BOX) {
      CheckRelativeBoundingBox(output[i], expected[i]);
    }
    if (output_format == LocationData::BOUNDING_BOX) {
      CheckBoundingBox(output[i], expected[i]);
    }
  }
}

TEST(DetectionsTransformationCalculatorTest, MissingImageSize) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "DetectionTransformationCalculator"
    input_stream: "DETECTIONS:detections"
    output_stream: "PIXEL_DETECTION:detection"
  )pb"));

  auto status = runner.Run();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              testing::HasSubstr("Image size must be provided"));
}

TEST(DetectionsTransformationCalculatorTest, WrongOutputType) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "DetectionTransformationCalculator"
    input_stream: "DETECTIONS:detections"
    input_stream: "IMAGE_SIZE:image_size"
    output_stream: "PIXEL_DETECTION:detection"
  )pb"));

  auto status = runner.Run();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              testing::HasSubstr("Output must be a container of detections"));
}

TEST(DetectionsTransformationCalculatorTest, WrongLocationDataFormat) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "DetectionTransformationCalculator"
    input_stream: "DETECTION:input_detection"
    input_stream: "IMAGE_SIZE:image_size"
    output_stream: "PIXEL_DETECTION:output_detection"
  )pb"));

  Detection detection;
  detection.mutable_location_data()->set_format(LocationData::GLOBAL);
  runner.MutableInputs()
      ->Tag(kDetectionTag)
      .packets.push_back(MakePacket<Detection>(detection).At(Timestamp(0)));
  std::pair<int, int> image_size({2000, 1000});
  runner.MutableInputs()
      ->Tag(kImageSizeTag)
      .packets.push_back(
          MakePacket<std::pair<int, int>>(image_size).At(Timestamp(0)));

  auto status = runner.Run();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              testing::HasSubstr("location data format must be either "
                                 "RELATIVE_BOUNDING_BOX or BOUNDING_BOX"));
}

TEST(DetectionsTransformationCalculatorTest, EmptyDetection) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "DetectionTransformationCalculator"
    input_stream: "DETECTION:input_detection"
    input_stream: "IMAGE_SIZE:image_size"
    output_stream: "PIXEL_DETECTION:output_detection"
  )pb"));

  std::pair<int, int> image_size({2000, 1000});
  runner.MutableInputs()
      ->Tag(kImageSizeTag)
      .packets.push_back(
          MakePacket<std::pair<int, int>>(image_size).At(Timestamp(0)));

  auto status = runner.Run();
  ASSERT_TRUE(status.ok());
}

TEST(DetectionsTransformationCalculatorTest, EmptyDetections) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "DetectionTransformationCalculator"
    input_stream: "DETECTIONS:input_detection"
    input_stream: "IMAGE_SIZE:image_size"
    output_stream: "PIXEL_DETECTIONS:output_detection"
  )pb"));

  std::pair<int, int> image_size({2000, 1000});
  runner.MutableInputs()
      ->Tag(kImageSizeTag)
      .packets.push_back(
          MakePacket<std::pair<int, int>>(image_size).At(Timestamp(0)));

  auto status = runner.Run();
  ASSERT_TRUE(status.ok());
}

TEST(DetectionsTransformationCalculatorTest,
     ConvertBoundingBoxToRelativeBoundingBox) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "DetectionTransformationCalculator"
    input_stream: "DETECTIONS:input_detections"
    input_stream: "IMAGE_SIZE:image_size"
    output_stream: "RELATIVE_DETECTIONS:output_detections"
    output_stream: "RELATIVE_DETECTION_LIST:output_detection_list"
  )pb"));

  auto detections(absl::make_unique<std::vector<Detection>>());
  detections->push_back(DetectionWithBoundingBox(100, 200, 400, 300));
  detections->push_back(DetectionWithBoundingBox(0, 0, 2000, 1000));
  std::pair<int, int> image_size({2000, 1000});
  runner.MutableInputs()
      ->Tag(kDetectionsTag)
      .packets.push_back(Adopt(detections.release()).At(Timestamp(0)));
  runner.MutableInputs()
      ->Tag(kImageSizeTag)
      .packets.push_back(
          MakePacket<std::pair<int, int>>(image_size).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());

  std::vector<Detection> expected(
      {DetectionWithRelativeBoundingBox(0.05, 0.2, 0.2, 0.3),
       DetectionWithRelativeBoundingBox(0, 0, 1, 1)});
  const std::vector<Packet>& detections_output =
      runner.Outputs().Tag(kRelativeDetectionsTag).packets;
  ASSERT_EQ(1, detections_output.size());
  CheckOutputDetections(expected,
                        detections_output[0].Get<std::vector<Detection>>());

  const std::vector<Packet>& detection_list_output =
      runner.Outputs().Tag(kRelativeDetectionListTag).packets;
  ASSERT_EQ(1, detection_list_output.size());
  CheckOutputDetections(
      expected,
      ConvertToDetectionVector(detection_list_output[0].Get<DetectionList>()));
}

TEST(DetectionsTransformationCalculatorTest,
     ConvertRelativeBoundingBoxToBoundingBox) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "DetectionTransformationCalculator"
    input_stream: "DETECTIONS:input_detections"
    input_stream: "IMAGE_SIZE:image_size"
    output_stream: "PIXEL_DETECTIONS:output_detections"
    output_stream: "PIXEL_DETECTION_LIST:output_detection_list"
  )pb"));

  auto detections(absl::make_unique<std::vector<Detection>>());
  detections->push_back(DetectionWithRelativeBoundingBox(0.1, 0.2, 0.3, 0.4));
  detections->push_back(DetectionWithRelativeBoundingBox(0, 0, 1, 1));
  std::pair<int, int> image_size({2000, 1000});
  runner.MutableInputs()
      ->Tag(kDetectionsTag)
      .packets.push_back(Adopt(detections.release()).At(Timestamp(0)));
  runner.MutableInputs()
      ->Tag(kImageSizeTag)
      .packets.push_back(
          MakePacket<std::pair<int, int>>(image_size).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());

  std::vector<Detection> expected({DetectionWithBoundingBox(100, 200, 400, 300),
                                   DetectionWithBoundingBox(0, 0, 2000, 1000)});
  const std::vector<Packet>& detections_output =
      runner.Outputs().Tag(kPixelDetectionsTag).packets;
  ASSERT_EQ(1, detections_output.size());
  CheckOutputDetections(expected,
                        detections_output[0].Get<std::vector<Detection>>());

  const std::vector<Packet>& detection_list_output =
      runner.Outputs().Tag(kPixelDetectionListTag).packets;
  ASSERT_EQ(1, detection_list_output.size());
  CheckOutputDetections(
      expected,
      ConvertToDetectionVector(detection_list_output[0].Get<DetectionList>()));
}

TEST(DetectionsTransformationCalculatorTest, ConvertSingleDetection) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "DetectionTransformationCalculator"
    input_stream: "DETECTION:input_detection"
    input_stream: "IMAGE_SIZE:image_size"
    output_stream: "PIXEL_DETECTION:outpu_detection"
    output_stream: "PIXEL_DETECTIONS:output_detections"
    output_stream: "PIXEL_DETECTION_LIST:output_detection_list"
  )pb"));

  runner.MutableInputs()
      ->Tag(kDetectionTag)
      .packets.push_back(MakePacket<Detection>(DetectionWithRelativeBoundingBox(
                                                   0.05, 0.2, 0.2, 0.3))
                             .At(Timestamp(0)));
  std::pair<int, int> image_size({2000, 1000});
  runner.MutableInputs()
      ->Tag(kImageSizeTag)
      .packets.push_back(
          MakePacket<std::pair<int, int>>(image_size).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());

  std::vector<Detection> expected(
      {DetectionWithBoundingBox(100, 200, 400, 300)});
  const std::vector<Packet>& detection_output =
      runner.Outputs().Tag(kPixelDetectionTag).packets;
  ASSERT_EQ(1, detection_output.size());
  CheckOutputDetections(expected, {detection_output[0].Get<Detection>()});

  const std::vector<Packet>& detections_output =
      runner.Outputs().Tag(kPixelDetectionsTag).packets;
  ASSERT_EQ(1, detections_output.size());
  CheckOutputDetections(expected,
                        detections_output[0].Get<std::vector<Detection>>());

  const std::vector<Packet>& detection_list_output =
      runner.Outputs().Tag(kPixelDetectionListTag).packets;
  ASSERT_EQ(1, detection_list_output.size());
  CheckOutputDetections(
      expected,
      ConvertToDetectionVector(detection_list_output[0].Get<DetectionList>()));
}

}  // namespace
}  // namespace mediapipe
