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

#include "absl/memory/memory.h"
#include "mediapipe/calculators/util/detections_to_render_data_calculator.pb.h"
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
#include "mediapipe/util/color.pb.h"
#include "mediapipe/util/render_data.pb.h"

namespace mediapipe {

constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kRenderDataTag[] = "RENDER_DATA";
constexpr char kDetectionListTag[] = "DETECTION_LIST";

// Error tolerance for pixels, distances, etc.
static constexpr double kErrorTolerance = 1e-5;

void VerifyRenderAnnotationColorThickness(
    const RenderAnnotation& annotation,
    const DetectionsToRenderDataCalculatorOptions& options) {
  EXPECT_THAT(annotation.color(), EqualsProto(options.color()));
  EXPECT_EQ(annotation.thickness(), options.thickness());
}

LocationData CreateLocationData(int32_t xmin, int32_t ymin, int32_t width,
                                int32_t height) {
  LocationData location_data;
  location_data.set_format(LocationData::BOUNDING_BOX);
  location_data.mutable_bounding_box()->set_xmin(xmin);
  location_data.mutable_bounding_box()->set_ymin(ymin);
  location_data.mutable_bounding_box()->set_width(width);
  location_data.mutable_bounding_box()->set_height(height);
  return location_data;
}

LocationData CreateRelativeLocationData(double xmin, double ymin, double width,
                                        double height) {
  LocationData location_data;
  location_data.set_format(LocationData::RELATIVE_BOUNDING_BOX);
  location_data.mutable_relative_bounding_box()->set_xmin(xmin);
  location_data.mutable_relative_bounding_box()->set_ymin(ymin);
  location_data.mutable_relative_bounding_box()->set_width(width);
  location_data.mutable_relative_bounding_box()->set_height(height);
  return location_data;
}

Detection CreateDetection(const std::vector<std::string>& labels,
                          const std::vector<int32_t>& label_ids,
                          const std::vector<float>& scores,
                          const LocationData& location_data,
                          const std::string& feature_tag) {
  Detection detection;
  for (const auto& label : labels) {
    detection.add_label(label);
  }
  for (const auto& label_id : label_ids) {
    detection.add_label_id(label_id);
  }
  for (const auto& score : scores) {
    detection.add_score(score);
  }
  *(detection.mutable_location_data()) = location_data;
  detection.set_feature_tag(feature_tag);
  return detection;
}

TEST(DetectionsToRenderDataCalculatorTest, OnlyDetecctionList) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "DetectionsToRenderDataCalculator"
    input_stream: "DETECTION_LIST:detection_list"
    output_stream: "RENDER_DATA:render_data"
  )pb"));

  LocationData location_data = CreateLocationData(100, 200, 300, 400);
  auto detections(absl::make_unique<DetectionList>());
  *(detections->add_detection()) =
      CreateDetection({"label1"}, {}, {0.3}, location_data, "feature_tag");

  runner.MutableInputs()
      ->Tag(kDetectionListTag)
      .packets.push_back(
          Adopt(detections.release()).At(Timestamp::PostStream()));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output =
      runner.Outputs().Tag(kRenderDataTag).packets;
  ASSERT_EQ(1, output.size());
  const auto& actual = output[0].Get<RenderData>();
  EXPECT_EQ(actual.render_annotations_size(), 3);
  // Labels
  EXPECT_EQ(actual.render_annotations(0).text().display_text(), "label1,0.3,");
  // Feature tag
  EXPECT_EQ(actual.render_annotations(1).text().display_text(), "feature_tag");
  // Location data
  EXPECT_EQ(actual.render_annotations(2).rectangle().left(), 100);
  EXPECT_EQ(actual.render_annotations(2).rectangle().right(), 100 + 300);
  EXPECT_EQ(actual.render_annotations(2).rectangle().top(), 200);
  EXPECT_EQ(actual.render_annotations(2).rectangle().bottom(), 200 + 400);
}

TEST(DetectionsToRenderDataCalculatorTest, OnlyDetecctionVector) {
  CalculatorRunner runner{ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "DetectionsToRenderDataCalculator"
    input_stream: "DETECTIONS:detections"
    output_stream: "RENDER_DATA:render_data"
  )pb")};

  LocationData location_data = CreateLocationData(100, 200, 300, 400);
  auto detections(absl::make_unique<std::vector<Detection>>());
  detections->push_back(
      CreateDetection({"label1"}, {}, {0.3}, location_data, "feature_tag"));

  runner.MutableInputs()
      ->Tag(kDetectionsTag)
      .packets.push_back(
          Adopt(detections.release()).At(Timestamp::PostStream()));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output =
      runner.Outputs().Tag(kRenderDataTag).packets;
  ASSERT_EQ(1, output.size());
  const auto& actual = output[0].Get<RenderData>();
  EXPECT_EQ(actual.render_annotations_size(), 3);
  // Labels
  EXPECT_EQ(actual.render_annotations(0).text().display_text(), "label1,0.3,");
  // Feature tag
  EXPECT_EQ(actual.render_annotations(1).text().display_text(), "feature_tag");
  // Location data
  EXPECT_EQ(actual.render_annotations(2).rectangle().left(), 100);
  EXPECT_EQ(actual.render_annotations(2).rectangle().right(), 100 + 300);
  EXPECT_EQ(actual.render_annotations(2).rectangle().top(), 200);
  EXPECT_EQ(actual.render_annotations(2).rectangle().bottom(), 200 + 400);
}

TEST(DetectionsToRenderDataCalculatorTest, BothDetecctionListAndVector) {
  CalculatorRunner runner{ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "DetectionsToRenderDataCalculator"
    input_stream: "DETECTION_LIST:detection_list"
    input_stream: "DETECTIONS:detections"
    output_stream: "RENDER_DATA:render_data"
  )pb")};

  LocationData location_data1 = CreateLocationData(100, 200, 300, 400);
  auto detection_list(absl::make_unique<DetectionList>());
  *(detection_list->add_detection()) =
      CreateDetection({"label1"}, {}, {0.3}, location_data1, "feature_tag1");
  runner.MutableInputs()
      ->Tag(kDetectionListTag)
      .packets.push_back(
          Adopt(detection_list.release()).At(Timestamp::PostStream()));

  LocationData location_data2 = CreateLocationData(600, 700, 800, 900);
  auto detections(absl::make_unique<std::vector<Detection>>());
  detections->push_back(
      CreateDetection({"label2"}, {}, {0.6}, location_data2, "feature_tag2"));
  runner.MutableInputs()
      ->Tag(kDetectionsTag)
      .packets.push_back(
          Adopt(detections.release()).At(Timestamp::PostStream()));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& actual =
      runner.Outputs().Tag(kRenderDataTag).packets;
  ASSERT_EQ(1, actual.size());
  // Check the feature tag for item from detection list.
  EXPECT_EQ(
      actual[0].Get<RenderData>().render_annotations(1).text().display_text(),
      "feature_tag1");
  // Check the feature tag for item from detection vector.
  EXPECT_EQ(
      actual[0].Get<RenderData>().render_annotations(4).text().display_text(),
      "feature_tag2");
}

TEST(DetectionsToRenderDataCalculatorTest, ProduceEmptyPacket) {
  // Check when produce_empty_packet is false.
  CalculatorRunner runner1{
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "DetectionsToRenderDataCalculator"
        input_stream: "DETECTION_LIST:detection_list"
        input_stream: "DETECTIONS:detections"
        output_stream: "RENDER_DATA:render_data"
        options {
          [mediapipe.DetectionsToRenderDataCalculatorOptions.ext] {
            produce_empty_packet: false
          }
        }
      )pb")};

  auto detection_list1(absl::make_unique<DetectionList>());
  runner1.MutableInputs()
      ->Tag(kDetectionListTag)
      .packets.push_back(
          Adopt(detection_list1.release()).At(Timestamp::PostStream()));

  auto detections1(absl::make_unique<std::vector<Detection>>());
  runner1.MutableInputs()
      ->Tag(kDetectionsTag)
      .packets.push_back(
          Adopt(detections1.release()).At(Timestamp::PostStream()));

  MP_ASSERT_OK(runner1.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& exact1 =
      runner1.Outputs().Tag(kRenderDataTag).packets;
  ASSERT_EQ(0, exact1.size());

  // Check when produce_empty_packet is true.
  CalculatorRunner runner2{
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "DetectionsToRenderDataCalculator"
        input_stream: "DETECTION_LIST:detection_list"
        input_stream: "DETECTIONS:detections"
        output_stream: "RENDER_DATA:render_data"
        options {
          [mediapipe.DetectionsToRenderDataCalculatorOptions.ext] {
            produce_empty_packet: true
          }
        }
      )pb")};

  auto detection_list2(absl::make_unique<DetectionList>());
  runner2.MutableInputs()
      ->Tag(kDetectionListTag)
      .packets.push_back(
          Adopt(detection_list2.release()).At(Timestamp::PostStream()));

  auto detections2(absl::make_unique<std::vector<Detection>>());
  runner2.MutableInputs()
      ->Tag(kDetectionsTag)
      .packets.push_back(
          Adopt(detections2.release()).At(Timestamp::PostStream()));

  MP_ASSERT_OK(runner2.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& exact2 =
      runner2.Outputs().Tag(kRenderDataTag).packets;
  ASSERT_EQ(1, exact2.size());
  EXPECT_EQ(exact2[0].Get<RenderData>().render_annotations_size(), 0);
}

}  // namespace mediapipe
