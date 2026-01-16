// Copyright 2021 The MediaPipe Authors.
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

#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

using ::testing::ElementsAre;

absl::Status RunScoreGraph(std::vector<Detection>& input_detections,
                           std::vector<Detection>* output_detections) {
  CalculatorRunner runner(R"pb(
    calculator: "FilterDetectionsCalculator"
    input_stream: "INPUT_DETECTIONS:input_detections"
    output_stream: "OUTPUT_DETECTIONS:output_detections"
    options {
      [mediapipe.FilterDetectionsCalculatorOptions.ext] { min_score: 0.5 }
    }
  )pb");

  const Timestamp input_timestamp = Timestamp(0);
  runner.MutableInputs()
      ->Tag("INPUT_DETECTIONS")
      .packets.push_back(MakePacket<std::vector<Detection>>(input_detections)
                             .At(input_timestamp));
  MP_RETURN_IF_ERROR(runner.Run()) << "Calculator run failed.";

  const std::vector<Packet>& output_packets =
      runner.Outputs().Tag("OUTPUT_DETECTIONS").packets;
  RET_CHECK_EQ(output_packets.size(), 1);

  *output_detections = output_packets[0].Get<std::vector<Detection>>();
  return absl::OkStatus();
}

TEST(FilterDetectionsCalculatorTest, TestFilterDetectionsScore) {
  std::vector<Detection> input_detections;
  Detection d1, d2;
  d1.add_score(0.2);
  d2.add_score(0.8);
  input_detections.push_back(d1);
  input_detections.push_back(d2);

  std::vector<Detection> output_detections;
  MP_EXPECT_OK(RunScoreGraph(input_detections, &output_detections));

  EXPECT_THAT(output_detections, ElementsAre(mediapipe::EqualsProto(d2)));
}

TEST(FilterDetectionsCalculatorTest, TestFilterDetectionsScoreMultiple) {
  std::vector<Detection> input_detections;
  Detection d1, d2, d3, d4;
  d1.add_score(0.3);
  d2.add_score(0.4);
  d3.add_score(0.5);
  d4.add_score(0.6);
  input_detections.push_back(d1);
  input_detections.push_back(d2);
  input_detections.push_back(d3);
  input_detections.push_back(d4);

  std::vector<Detection> output_detections;
  MP_EXPECT_OK(RunScoreGraph(input_detections, &output_detections));

  EXPECT_THAT(output_detections, ElementsAre(mediapipe::EqualsProto(d3),
                                             mediapipe::EqualsProto(d4)));
}

TEST(FilterDetectionsCalculatorTest, TestFilterDetectionsEmpty) {
  std::vector<Detection> input_detections;

  std::vector<Detection> output_detections;
  MP_EXPECT_OK(RunScoreGraph(input_detections, &output_detections));

  EXPECT_EQ(output_detections.size(), 0);
}

absl::Status RunSizeGraph(std::vector<Detection>& input_detections,
                          std::pair<int, int> image_dimensions,
                          std::vector<Detection>* output_detections) {
  CalculatorRunner runner(R"pb(
    calculator: "FilterDetectionsCalculator"
    input_stream: "INPUT_DETECTIONS:input_detections"
    input_stream: "IMAGE_SIZE:image_dimensions"
    output_stream: "OUTPUT_DETECTIONS:output_detections"
    options {
      [mediapipe.FilterDetectionsCalculatorOptions.ext] { min_pixel_size: 50 }
    }
  )pb");

  const Timestamp input_timestamp = Timestamp(0);
  runner.MutableInputs()
      ->Tag("INPUT_DETECTIONS")
      .packets.push_back(MakePacket<std::vector<Detection>>(input_detections)
                             .At(input_timestamp));
  runner.MutableInputs()
      ->Tag("IMAGE_SIZE")
      .packets.push_back(MakePacket<std::pair<int, int>>(image_dimensions)
                             .At(input_timestamp));
  MP_RETURN_IF_ERROR(runner.Run()) << "Calculator run failed.";

  const std::vector<Packet>& output_packets =
      runner.Outputs().Tag("OUTPUT_DETECTIONS").packets;
  RET_CHECK_EQ(output_packets.size(), 1);

  *output_detections = output_packets[0].Get<std::vector<Detection>>();
  return absl::OkStatus();
}

TEST(FilterDetectionsCalculatorTest, TestFilterDetectionsMinSize) {
  std::vector<Detection> input_detections;
  Detection d1, d2, d3, d4, d5;
  d1.mutable_location_data()->mutable_relative_bounding_box()->set_height(0.5);
  d1.mutable_location_data()->mutable_relative_bounding_box()->set_width(0.49);
  d2.mutable_location_data()->mutable_relative_bounding_box()->set_height(0.4);
  d2.mutable_location_data()->mutable_relative_bounding_box()->set_width(0.4);
  d3.mutable_location_data()->mutable_relative_bounding_box()->set_height(0.49);
  d3.mutable_location_data()->mutable_relative_bounding_box()->set_width(0.5);
  d4.mutable_location_data()->mutable_relative_bounding_box()->set_height(0.49);
  d4.mutable_location_data()->mutable_relative_bounding_box()->set_width(0.49);
  d5.mutable_location_data()->mutable_relative_bounding_box()->set_height(0.5);
  d5.mutable_location_data()->mutable_relative_bounding_box()->set_width(0.5);
  input_detections.push_back(d1);
  input_detections.push_back(d2);
  input_detections.push_back(d3);
  input_detections.push_back(d4);
  input_detections.push_back(d5);

  std::vector<Detection> output_detections;
  MP_EXPECT_OK(RunSizeGraph(input_detections, {100, 100}, &output_detections));

  EXPECT_THAT(output_detections, ElementsAre(mediapipe::EqualsProto(d1),
                                             mediapipe::EqualsProto(d3),
                                             mediapipe::EqualsProto(d5)));
}

}  // namespace
}  // namespace mediapipe
