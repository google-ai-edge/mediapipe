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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

constexpr char kGraphConfig[] = R"(
        input_stream: "input_detection"
        input_stream: "classification_list"
        output_stream: "output_detection"
        node {
          calculator: "DetectionClassificationsMergerCalculator"
          input_stream: "INPUT_DETECTION:input_detection"
          input_stream: "CLASSIFICATION_LIST:classification_list"
          output_stream: "OUTPUT_DETECTION:output_detection"
        }
      )";

constexpr char kInputDetection[] = R"(
        label: "entity"
        label_id: 1
        score: 0.9
        location_data {
          format: BOUNDING_BOX
          bounding_box { xmin: 50 ymin: 60 width: 70 height: 80 }
        }
        display_name: "Entity"
     )";

// Checks that the input Detection is returned unchanged if the input
// ClassificationList does not contain any result.
TEST(DetectionClassificationsMergerCalculator, SucceedsWithNoClassification) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(kGraphConfig);

  // Prepare input packets.
  const Detection& input_detection =
      ParseTextProtoOrDie<Detection>(kInputDetection);
  Packet input_detection_packet =
      MakePacket<Detection>(input_detection).At(Timestamp(0));
  const ClassificationList& classification_list =
      ParseTextProtoOrDie<ClassificationList>("");
  Packet classification_list_packet =
      MakePacket<ClassificationList>(classification_list).At(Timestamp(0));

  // Catch output.
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output_detection", &graph_config, &output_packets);

  // Run the graph.
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(
      graph.AddPacketToInputStream("input_detection", input_detection_packet));
  MP_ASSERT_OK(graph.AddPacketToInputStream("classification_list",
                                            classification_list_packet));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Get and validate output.
  EXPECT_THAT(output_packets, testing::SizeIs(1));
  const Detection& output_detection = output_packets[0].Get<Detection>();
  EXPECT_THAT(output_detection, mediapipe::EqualsProto(input_detection));
}

// Checks that merging succeeds when the input ClassificationList includes
// labels and display names.
TEST(DetectionClassificationsMergerCalculator,
     SucceedsWithLabelsAndDisplayNames) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(kGraphConfig);

  // Prepare input packets.
  const Detection& input_detection =
      ParseTextProtoOrDie<Detection>(kInputDetection);
  Packet input_detection_packet =
      MakePacket<Detection>(input_detection).At(Timestamp(0));
  const ClassificationList& classification_list =
      ParseTextProtoOrDie<ClassificationList>(R"pb(
        classification { index: 11 score: 0.5 label: "dog" display_name: "Dog" }
        classification { index: 12 score: 0.4 label: "fox" display_name: "Fox" }
      )pb");
  Packet classification_list_packet =
      MakePacket<ClassificationList>(classification_list).At(Timestamp(0));

  // Catch output.
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output_detection", &graph_config, &output_packets);

  // Run the graph.
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(
      graph.AddPacketToInputStream("input_detection", input_detection_packet));
  MP_ASSERT_OK(graph.AddPacketToInputStream("classification_list",
                                            classification_list_packet));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Get and validate output.
  EXPECT_THAT(output_packets, testing::SizeIs(1));
  const Detection& output_detection = output_packets[0].Get<Detection>();
  EXPECT_THAT(output_detection,
              mediapipe::EqualsProto(ParseTextProtoOrDie<Detection>(R"pb(
                label: "dog"
                label: "fox"
                label_id: 11
                label_id: 12
                score: 0.5
                score: 0.4
                location_data {
                  format: BOUNDING_BOX
                  bounding_box { xmin: 50 ymin: 60 width: 70 height: 80 }
                }
                display_name: "Dog"
                display_name: "Fox"
              )pb")));
}

// Checks that merging succeeds when the input ClassificationList doesn't
// include labels and display names.
TEST(DetectionClassificationsMergerCalculator,
     SucceedsWithoutLabelsAndDisplayNames) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(kGraphConfig);

  // Prepare input packets.
  const Detection& input_detection =
      ParseTextProtoOrDie<Detection>(kInputDetection);
  Packet input_detection_packet =
      MakePacket<Detection>(input_detection).At(Timestamp(0));
  const ClassificationList& classification_list =
      ParseTextProtoOrDie<ClassificationList>(R"pb(
        classification { index: 11 score: 0.5 }
        classification { index: 12 score: 0.4 }
      )pb");
  Packet classification_list_packet =
      MakePacket<ClassificationList>(classification_list).At(Timestamp(0));

  // Catch output.
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output_detection", &graph_config, &output_packets);

  // Run the graph.
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(
      graph.AddPacketToInputStream("input_detection", input_detection_packet));
  MP_ASSERT_OK(graph.AddPacketToInputStream("classification_list",
                                            classification_list_packet));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Get and validate output.
  EXPECT_THAT(output_packets, testing::SizeIs(1));
  const Detection& output_detection = output_packets[0].Get<Detection>();
  EXPECT_THAT(output_detection,
              mediapipe::EqualsProto(ParseTextProtoOrDie<Detection>(R"pb(
                label_id: 11
                label_id: 12
                score: 0.5
                score: 0.4
                location_data {
                  format: BOUNDING_BOX
                  bounding_box { xmin: 50 ymin: 60 width: 70 height: 80 }
                }
              )pb")));
}

// Checks that merging fails if the input ClassificationList misses mandatory
// "index" field.
TEST(DetectionClassificationsMergerCalculator, FailsWithMissingIndex) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(kGraphConfig);

  // Prepare input packets.
  const Detection& input_detection =
      ParseTextProtoOrDie<Detection>(kInputDetection);
  Packet input_detection_packet =
      MakePacket<Detection>(input_detection).At(Timestamp(0));
  const ClassificationList& classification_list =
      ParseTextProtoOrDie<ClassificationList>(R"pb(
        classification { score: 0.5 label: "dog" }
      )pb");
  Packet classification_list_packet =
      MakePacket<ClassificationList>(classification_list).At(Timestamp(0));

  // Catch output.
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output_detection", &graph_config, &output_packets);

  // Run the graph.
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(
      graph.AddPacketToInputStream("input_detection", input_detection_packet));
  MP_ASSERT_OK(graph.AddPacketToInputStream("classification_list",
                                            classification_list_packet));
  ASSERT_EQ(graph.WaitUntilIdle().code(), absl::StatusCode::kInvalidArgument);
}

// Checks that merging fails if the input ClassificationList misses mandatory
// "score" field.
TEST(DetectionClassificationsMergerCalculator, FailsWithMissingScore) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(kGraphConfig);

  // Prepare input packets.
  const Detection& input_detection =
      ParseTextProtoOrDie<Detection>(kInputDetection);
  Packet input_detection_packet =
      MakePacket<Detection>(input_detection).At(Timestamp(0));
  const ClassificationList& classification_list =
      ParseTextProtoOrDie<ClassificationList>(R"pb(
        classification { index: 11 label: "dog" }
      )pb");
  Packet classification_list_packet =
      MakePacket<ClassificationList>(classification_list).At(Timestamp(0));

  // Catch output.
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output_detection", &graph_config, &output_packets);

  // Run the graph.
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(
      graph.AddPacketToInputStream("input_detection", input_detection_packet));
  MP_ASSERT_OK(graph.AddPacketToInputStream("classification_list",
                                            classification_list_packet));
  ASSERT_EQ(graph.WaitUntilIdle().code(), absl::StatusCode::kInvalidArgument);
}

// Checks that merging fails if the input ClassificationList has an
// inconsistent number of labels.
TEST(DetectionClassificationsMergerCalculator,
     FailsWithInconsistentNumberOfLabels) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(kGraphConfig);

  // Prepare input packets.
  const Detection& input_detection =
      ParseTextProtoOrDie<Detection>(kInputDetection);
  Packet input_detection_packet =
      MakePacket<Detection>(input_detection).At(Timestamp(0));
  const ClassificationList& classification_list =
      ParseTextProtoOrDie<ClassificationList>(R"pb(
        classification { index: 11 score: 0.5 label: "dog" display_name: "Dog" }
        classification { index: 12 score: 0.4 display_name: "Fox" }
      )pb");
  Packet classification_list_packet =
      MakePacket<ClassificationList>(classification_list).At(Timestamp(0));

  // Catch output.
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output_detection", &graph_config, &output_packets);

  // Run the graph.
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(
      graph.AddPacketToInputStream("input_detection", input_detection_packet));
  MP_ASSERT_OK(graph.AddPacketToInputStream("classification_list",
                                            classification_list_packet));
  ASSERT_EQ(graph.WaitUntilIdle().code(), absl::StatusCode::kInvalidArgument);
}

// Checks that merging fails if the input ClassificationList has an
// inconsistent number of display names.
TEST(DetectionClassificationsMergerCalculator,
     FailsWithInconsistentNumberOfDisplayNames) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(kGraphConfig);

  // Prepare input packets.
  const Detection& input_detection =
      ParseTextProtoOrDie<Detection>(kInputDetection);
  Packet input_detection_packet =
      MakePacket<Detection>(input_detection).At(Timestamp(0));
  const ClassificationList& classification_list =
      ParseTextProtoOrDie<ClassificationList>(R"pb(
        classification { index: 11 score: 0.5 label: "dog" }
        classification { index: 12 score: 0.4 label: "fox" display_name: "Fox" }
      )pb");
  Packet classification_list_packet =
      MakePacket<ClassificationList>(classification_list).At(Timestamp(0));

  // Catch output.
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output_detection", &graph_config, &output_packets);

  // Run the graph.
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(
      graph.AddPacketToInputStream("input_detection", input_detection_packet));
  MP_ASSERT_OK(graph.AddPacketToInputStream("classification_list",
                                            classification_list_packet));
  ASSERT_EQ(graph.WaitUntilIdle().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace mediapipe
