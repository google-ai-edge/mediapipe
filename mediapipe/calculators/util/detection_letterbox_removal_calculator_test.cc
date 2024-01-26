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

#include <cstdint>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/validate_type.h"

namespace mediapipe {

constexpr char kLetterboxPaddingTag[] = "LETTERBOX_PADDING";
constexpr char kDetectionsTag[] = "DETECTIONS";

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

CalculatorGraphConfig::Node GetDefaultNode() {
  return ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "DetectionLetterboxRemovalCalculator"
    input_stream: "DETECTIONS:detections"
    input_stream: "LETTERBOX_PADDING:letterbox_padding"
    output_stream: "DETECTIONS:adjusted_detections"
  )pb");
}

TEST(DetectionLetterboxRemovalCalculatorTest, PaddingLeftRight) {
  CalculatorRunner runner(GetDefaultNode());

  LocationData location_data =
      CreateRelativeLocationData(0.25f, 0.25f, 0.25f, 0.25f);
  const std::string label = "detected_object";

  auto detections = absl::make_unique<std::vector<Detection>>();
  detections->push_back(
      CreateDetection({label}, {}, {0.3f}, location_data, "feature_tag"));
  runner.MutableInputs()
      ->Tag(kDetectionsTag)
      .packets.push_back(
          Adopt(detections.release()).At(Timestamp::PostStream()));

  auto padding = absl::make_unique<std::array<float, 4>>(
      std::array<float, 4>{0.2f, 0.f, 0.3f, 0.f});
  runner.MutableInputs()
      ->Tag(kLetterboxPaddingTag)
      .packets.push_back(Adopt(padding.release()).At(Timestamp::PostStream()));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output =
      runner.Outputs().Tag(kDetectionsTag).packets;
  ASSERT_EQ(1, output.size());
  const auto& output_detections = output[0].Get<std::vector<Detection>>();

  EXPECT_EQ(output_detections.size(), 1);
  const auto& output_detection = output_detections[0];

  EXPECT_EQ(output_detection.label_size(), 1);
  EXPECT_EQ(output_detection.label(0), label);
  EXPECT_EQ(output_detection.label_id_size(), 0);
  EXPECT_EQ(output_detection.score_size(), 1);
  EXPECT_EQ(output_detection.score(0), 0.3f);

  EXPECT_EQ(output_detection.location_data().format(),
            LocationData::RELATIVE_BOUNDING_BOX);
  EXPECT_THAT(output_detection.location_data().relative_bounding_box().xmin(),
              testing::FloatNear(0.1f, 1e-5));
  EXPECT_THAT(output_detection.location_data().relative_bounding_box().ymin(),
              testing::FloatNear(0.25f, 1e-5));
  EXPECT_THAT(output_detection.location_data().relative_bounding_box().width(),
              testing::FloatNear(0.5f, 1e-5));
  EXPECT_THAT(output_detection.location_data().relative_bounding_box().height(),
              testing::FloatNear(0.25f, 1e-5));
}

TEST(DetectionLetterboxRemovalCalculatorTest, PaddingTopBottom) {
  CalculatorRunner runner(GetDefaultNode());

  LocationData location_data =
      CreateRelativeLocationData(0.25f, 0.25f, 0.25f, 0.25f);
  const std::string label = "detected_object";

  auto detections = absl::make_unique<std::vector<Detection>>();
  detections->push_back(
      CreateDetection({label}, {}, {0.3f}, location_data, "feature_tag"));
  runner.MutableInputs()
      ->Tag(kDetectionsTag)
      .packets.push_back(
          Adopt(detections.release()).At(Timestamp::PostStream()));

  auto padding = absl::make_unique<std::array<float, 4>>(
      std::array<float, 4>{0.f, 0.2f, 0.f, 0.3f});
  runner.MutableInputs()
      ->Tag(kLetterboxPaddingTag)
      .packets.push_back(Adopt(padding.release()).At(Timestamp::PostStream()));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output =
      runner.Outputs().Tag(kDetectionsTag).packets;
  ASSERT_EQ(1, output.size());
  const auto& output_detections = output[0].Get<std::vector<Detection>>();

  EXPECT_EQ(output_detections.size(), 1);
  const auto& output_detection = output_detections[0];

  EXPECT_EQ(output_detection.location_data().format(),
            LocationData::RELATIVE_BOUNDING_BOX);
  EXPECT_THAT(output_detection.location_data().relative_bounding_box().xmin(),
              testing::FloatNear(0.25f, 1e-5));
  EXPECT_THAT(output_detection.location_data().relative_bounding_box().ymin(),
              testing::FloatNear(0.1f, 1e-5));
  EXPECT_THAT(output_detection.location_data().relative_bounding_box().width(),
              testing::FloatNear(0.25f, 1e-5));
  EXPECT_THAT(output_detection.location_data().relative_bounding_box().height(),
              testing::FloatNear(0.5f, 1e-5));
}

}  // namespace mediapipe
