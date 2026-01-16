// Copyright 2025 The MediaPipe Authors.
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

#include "mediapipe/calculators/core/vector_size_calculator.h"

#include <vector>

#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

using ::testing::ElementsAre;

MATCHER_P2(PacketEq, val, timestamp, "") {
  const Packet& packet = arg;
  return packet.Get<int>() == val && packet.Timestamp() == timestamp;
}

using TestIntVectorSizeCalculator = api2::VectorSizeCalculator<int>;
MEDIAPIPE_REGISTER_NODE(TestIntVectorSizeCalculator);

CalculatorGraphConfig CreateCalculatorGraphConfig() {
  return ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "vector_stream"
    output_stream: "size_stream"
    node {
      calculator: "TestIntVectorSizeCalculator"
      input_stream: "VECTOR:vector_stream"
      output_stream: "SIZE:size_stream"
    }
  )pb");
}

void AddInputVector(CalculatorGraph& graph, const std::vector<int>& input,
                    int timestamp) {
  auto image_packet = MakePacket<std::vector<int>>(input);
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "vector_stream", image_packet.At(Timestamp(timestamp))));
}

TEST(TestIntVectorSizeCalculator, EmptyVectorWithOutputSizeZero) {
  CalculatorGraphConfig graph_config = CreateCalculatorGraphConfig();
  std::vector<Packet> output_packets;
  tool::AddVectorSink("size_stream", &graph_config, &output_packets);

  const std::vector<int> input = {};

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));

  AddInputVector(graph, /*input=*/input, /*timestamp=*/1);
  MP_ASSERT_OK(graph.WaitUntilIdle());
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());

  EXPECT_THAT(output_packets, ElementsAre(PacketEq(0, Timestamp(1))));
}

TEST(TestIntVectorSizeCalculator, SingleVectorInput) {
  CalculatorGraphConfig graph_config = CreateCalculatorGraphConfig();
  std::vector<Packet> output_packets;
  tool::AddVectorSink("size_stream", &graph_config, &output_packets);

  const std::vector<int> input = {1, 2, 3};

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));

  AddInputVector(graph, /*input=*/input, /*timestamp=*/1);
  MP_ASSERT_OK(graph.WaitUntilIdle());
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());

  EXPECT_THAT(output_packets, ElementsAre(PacketEq(3, Timestamp(1))));
}

TEST(TestIntVectorSizeCalculator, MultipleVectorInputs) {
  CalculatorGraphConfig graph_config = CreateCalculatorGraphConfig();
  std::vector<Packet> output_packets;
  tool::AddVectorSink("size_stream", &graph_config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));

  AddInputVector(graph, /*input=*/{1, 2, 3}, /*timestamp=*/1);
  AddInputVector(graph, /*input=*/{5, 6, 7, 8}, /*timestamp=*/2);
  AddInputVector(graph, /*input=*/{9, 10}, /*timestamp=*/3);

  MP_ASSERT_OK(graph.WaitUntilIdle());
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());

  EXPECT_THAT(output_packets,
              ElementsAre(PacketEq(3, Timestamp(1)), PacketEq(4, Timestamp(2)),
                          PacketEq(2, Timestamp(3))));
}

}  // namespace mediapipe
