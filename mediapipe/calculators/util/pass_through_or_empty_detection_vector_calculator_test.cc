#include "mediapipe/calculators/util/pass_through_or_empty_detection_vector_calculator.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

CalculatorGraphConfig GetGraphConfig() {
  mediapipe::api2::builder::Graph graph;
  mediapipe::api2::builder::Stream<std::string> tick =
      graph.In("TICK").SetName("tick").Cast<std::string>();
  mediapipe::api2::builder::Stream<std::vector<mediapipe::Detection>>
      detections = graph.In("DETECTIONS")
                       .SetName("input_detections")
                       .Cast<std::vector<mediapipe::Detection>>();

  mediapipe::api2::builder::Stream<std::vector<mediapipe::Detection>>
      output_detections =
          PassThroughOrEmptyDetectionVector(detections, tick, graph);
  output_detections.SetName("output_detections");

  return graph.GetConfig();
}

absl::Status SendTick(CalculatorGraph& graph, int at) {
  return graph.AddPacketToInputStream(
      "tick",
      mediapipe::MakePacket<std::string>("tick").At(mediapipe::Timestamp(at)));
}

absl::Status SendDetections(CalculatorGraph& graph,
                            std::vector<mediapipe::Detection> detections,
                            int at) {
  return graph.AddPacketToInputStream(
      "input_detections",
      mediapipe::MakePacket<std::vector<mediapipe::Detection>>(
          std::move(detections))
          .At(mediapipe::Timestamp(at)));
}

TEST(PassThroughOrEmptyDetectionVectorCalculatorTest, PassThrough) {
  CalculatorGraphConfig graph_config = GetGraphConfig();
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output_detections", &graph_config, &output_packets);

  CalculatorGraph calculator_graph(graph_config);
  MP_ASSERT_OK(calculator_graph.StartRun({}));

  // Sending empty vector.
  MP_ASSERT_OK(SendTick(calculator_graph, /*at=*/1));
  MP_ASSERT_OK(SendDetections(calculator_graph,
                              std::vector<mediapipe::Detection>{},
                              /*at=*/1));
  MP_ASSERT_OK(calculator_graph.WaitUntilIdle());

  ASSERT_EQ(output_packets.size(), 1);
  EXPECT_TRUE(
      output_packets[0].Get<std::vector<mediapipe::Detection>>().empty());

  // Sending non empty vector.
  output_packets.clear();
  mediapipe::Detection detection;
  detection.set_detection_id(1000);

  MP_ASSERT_OK(SendTick(calculator_graph, /*at=*/2));
  MP_ASSERT_OK(SendDetections(calculator_graph, {detection}, /*at=*/2));
  MP_ASSERT_OK(calculator_graph.WaitUntilIdle());

  ASSERT_EQ(output_packets.size(), 1);
}

TEST(PassThroughOrEmptyDetectionVectorCalculatorTest, OrEmptyVector) {
  CalculatorGraphConfig graph_config = GetGraphConfig();
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output_detections", &graph_config, &output_packets);

  CalculatorGraph calculator_graph(graph_config);
  MP_ASSERT_OK(calculator_graph.StartRun({}));

  mediapipe::Detection detection;
  detection.set_detection_id(1000);
  MP_ASSERT_OK(SendTick(calculator_graph, /*at=*/1));
  MP_ASSERT_OK(SendDetections(calculator_graph, {detection}, /*at=*/1));
  MP_ASSERT_OK(SendTick(calculator_graph, /*at=*/2));
  MP_ASSERT_OK(SendTick(calculator_graph, /*at=*/3));
  MP_ASSERT_OK(SendTick(calculator_graph, /*at=*/4));
  // This should trigger trigger calculator at 2, 3, 4 as detections are not
  // expected.
  MP_ASSERT_OK(SendDetections(calculator_graph,
                              std::vector<mediapipe::Detection>{},
                              /*at=*/5));
  MP_ASSERT_OK(calculator_graph.WaitUntilIdle());

  ASSERT_EQ(output_packets.size(), 4);

  for (int i = 1; i < output_packets.size(); ++i) {
    EXPECT_TRUE(
        output_packets[i].Get<std::vector<mediapipe::Detection>>().empty());
  }
}

}  // namespace
}  // namespace mediapipe
