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

#include "absl/status/status.h"
#include "absl/types/optional.h"
#include "mediapipe/calculators/core/split_vector_calculator.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

typedef SplitVectorCalculator<int, false> SplitIntVectorCalculator;
REGISTER_CALCULATOR(SplitIntVectorCalculator);

namespace {

// Graph with default input stream handler, and the input selection is driven
// by an input stream. All MuxCalculator inputs are present at each timestamp.
constexpr char kTestGraphConfig1[] = R"pb(
  input_stream: "input"
  output_stream: "test_output"
  node {
    calculator: "SplitIntVectorCalculator"
    input_stream: "input"
    output_stream: "stream0"
    output_stream: "stream1"
    output_stream: "stream2"
    output_stream: "input_select"
    options {
      [mediapipe.SplitVectorCalculatorOptions.ext] {
        ranges: { begin: 0 end: 1 }
        ranges: { begin: 1 end: 2 }
        ranges: { begin: 2 end: 3 }
        ranges: { begin: 3 end: 4 }
        element_only: true
      }
    }
  }
  node {
    calculator: "MuxCalculator"
    input_stream: "INPUT:0:stream0"
    input_stream: "INPUT:1:stream1"
    input_stream: "INPUT:2:stream2"
    input_stream: "SELECT:input_select"
    output_stream: "OUTPUT:test_output"
    input_stream_handler { input_stream_handler: "DefaultInputStreamHandler" }
  }
)pb";

// Graph with default input stream handler, and the input selection is driven
// by an input side packet. All MuxCalculator inputs are present at each
// timestamp.
constexpr char kTestGraphConfig2[] = R"pb(
  input_side_packet: "input_selector"
  input_stream: "input"
  output_stream: "test_output"
  node {
    calculator: "SplitIntVectorCalculator"
    input_stream: "input"
    output_stream: "stream0"
    output_stream: "stream1"
    output_stream: "stream2"
    options {
      [mediapipe.SplitVectorCalculatorOptions.ext] {
        ranges: { begin: 0 end: 1 }
        ranges: { begin: 1 end: 2 }
        ranges: { begin: 2 end: 3 }
        element_only: true
      }
    }
  }
  node {
    calculator: "MuxCalculator"
    input_stream: "INPUT:0:stream0"
    input_stream: "INPUT:1:stream1"
    input_stream: "INPUT:2:stream2"
    input_side_packet: "SELECT:input_selector"
    output_stream: "OUTPUT:test_output"
    input_stream_handler { input_stream_handler: "DefaultInputStreamHandler" }
  }
)pb";

// Graph with mux input stream handler, and the input selection is driven
// by an input stream. Only one MuxCalculator input is present at each
// timestamp.
constexpr char kTestGraphConfig3[] = R"pb(
  input_stream: "input"
  output_stream: "test_output"
  node {
    calculator: "RoundRobinDemuxCalculator"
    input_stream: "input"
    output_stream: "OUTPUT:0:stream0"
    output_stream: "OUTPUT:1:stream1"
    output_stream: "OUTPUT:2:stream2"
    output_stream: "SELECT:input_select"
  }
  node {
    calculator: "MuxCalculator"
    input_stream: "INPUT:0:stream0"
    input_stream: "INPUT:1:stream1"
    input_stream: "INPUT:2:stream2"
    input_stream: "SELECT:input_select"
    output_stream: "OUTPUT:test_output"
  }
)pb";

constexpr char kOutputName[] = "test_output";
constexpr char kInputName[] = "input";
constexpr char kInputSelector[] = "input_selector";

// Helper to run a graph with the given inputs and generate outputs, asserting
// each step along the way.
// Inputs:
//   graph_config_proto - graph config protobuf
//   extra_side_packets - input side packets name to value map
//   input_stream_name - name of the input
void RunGraph(const std::string& graph_config_proto,
              const std::map<std::string, Packet>& extra_side_packets,
              const std::string& input_stream_name, int num_input_packets,
              std::function<Packet(int)> input_fn,
              const std::string& output_stream_name,
              std::function<absl::Status(const Packet&)> output_fn) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(graph_config_proto);
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream(output_stream_name, output_fn));
  MP_ASSERT_OK(graph.StartRun(extra_side_packets));
  for (int i = 0; i < num_input_packets; ++i) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(input_stream_name, input_fn(i)));
  }
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST(MuxCalculatorTest, InputStreamSelector_DefaultInputStreamHandler) {
  // Input and handling.
  std::vector<std::vector<int>> input_packets = {
      {1, 1, 2, 1},           {3, 5, 8, 2},       {13, 21, 34, 0},
      {55, 89, 144, 2},       {233, 377, 610, 0}, {987, 1597, 2584, 1},
      {4181, 6765, 10946, 2},
  };
  int packet_time_stamp = 22;
  // This function will return the i-th input packet.
  auto input_fn = [&packet_time_stamp, &input_packets](int i) -> Packet {
    return MakePacket<std::vector<int>>(input_packets[i])
        .At(Timestamp(packet_time_stamp++));
  };

  // Output and handling.
  std::vector<int> output;
  // This function collects the output from the packet.
  auto output_fn = [&output](const Packet& p) -> absl::Status {
    output.push_back(p.Get<int>());
    return absl::OkStatus();
  };

  RunGraph(kTestGraphConfig1, {}, kInputName, input_packets.size(), input_fn,
           kOutputName, output_fn);
  EXPECT_THAT(output, testing::ElementsAre(1, 8, 13, 144, 233, 1597, 10946));
}

TEST(MuxCalculatorTest, InputSidePacketSelector_DefaultInputStreamHandler) {
  // Input and handling.
  std::vector<std::vector<int>> input_packets = {
      {1, 1, 2},       {3, 5, 8},         {13, 21, 34},        {55, 89, 144},
      {233, 377, 610}, {987, 1597, 2584}, {4181, 6765, 10946},
  };
  int packet_time_stamp = 22;
  // This function will return the i-th input packet.
  auto input_fn = [&packet_time_stamp, &input_packets](int i) -> Packet {
    return MakePacket<std::vector<int>>(input_packets[i])
        .At(Timestamp(packet_time_stamp++));
  };

  // Output and handling.
  std::vector<int> output;
  // This function collects the output from the packet.
  auto output_fn = [&output](const Packet& p) -> absl::Status {
    output.push_back(p.Get<int>());
    return absl::OkStatus();
  };

  RunGraph(kTestGraphConfig2, {{kInputSelector, MakePacket<int>(0)}},
           kInputName, input_packets.size(), input_fn, kOutputName, output_fn);
  EXPECT_THAT(output, testing::ElementsAre(1, 3, 13, 55, 233, 987, 4181));

  output.clear();
  RunGraph(kTestGraphConfig2, {{kInputSelector, MakePacket<int>(1)}},
           kInputName, input_packets.size(), input_fn, kOutputName, output_fn);
  EXPECT_THAT(output, testing::ElementsAre(1, 5, 21, 89, 377, 1597, 6765));

  output.clear();
  RunGraph(kTestGraphConfig2, {{kInputSelector, MakePacket<int>(2)}},
           kInputName, input_packets.size(), input_fn, kOutputName, output_fn);
  EXPECT_THAT(output, testing::ElementsAre(2, 8, 34, 144, 610, 2584, 10946));
}

TEST(MuxCalculatorTest, InputStreamSelector_MuxInputStreamHandler) {
  // Input and handling.
  std::vector<int> input_packets = {1,   1,   2,    3,    5,    8,    13,
                                    21,  34,  55,   89,   144,  233,  377,
                                    610, 987, 1597, 2584, 4181, 6765, 10946};
  int packet_time_stamp = 22;
  // This function will return the i-th input packet.
  auto input_fn = [&packet_time_stamp, &input_packets](int i) -> Packet {
    return MakePacket<int>(input_packets[i]).At(Timestamp(packet_time_stamp++));
  };

  // Output and handling.
  std::vector<int> output;
  // This function collects the output from the packet.
  auto output_fn = [&output](const Packet& p) -> absl::Status {
    output.push_back(p.Get<int>());
    return absl::OkStatus();
  };

  RunGraph(kTestGraphConfig3, {}, kInputName, input_packets.size(), input_fn,
           kOutputName, output_fn);
  EXPECT_EQ(output, input_packets);
}

constexpr char kDualInputGraphConfig[] = R"pb(
  input_stream: "input_0"
  input_stream: "input_1"
  input_stream: "input_select"
  output_stream: "test_output"
  node {
    calculator: "MuxCalculator"
    input_stream: "INPUT:0:input_0"
    input_stream: "INPUT:1:input_1"
    input_stream: "SELECT:input_select"
    output_stream: "OUTPUT:test_output"
  }
)pb";

TEST(MuxCalculatorTest, DiscardSkippedInputs_MuxInputStreamHandler) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
          kDualInputGraphConfig);
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));

  std::shared_ptr<int> output;
  MP_ASSERT_OK(
      graph.ObserveOutputStream("test_output", [&output](const Packet& p) {
        output = p.Get<std::shared_ptr<int>>();
        return absl::OkStatus();
      }));

  MP_ASSERT_OK(graph.StartRun({}));

  auto one = std::make_shared<int>(1);
  auto two = std::make_shared<int>(2);
  auto three = std::make_shared<int>(3);
  std::weak_ptr<int> one_weak = one;
  std::weak_ptr<int> two_weak = two;

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_0",
      MakePacket<std::shared_ptr<int>>(std::move(one)).At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_1",
      MakePacket<std::shared_ptr<int>>(std::move(two)).At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_1",
      MakePacket<std::shared_ptr<int>>(std::move(three)).At(Timestamp(1))));
  EXPECT_EQ(one, nullptr);
  EXPECT_EQ(two, nullptr);
  EXPECT_EQ(three, nullptr);

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_select", MakePacket<int>(0).At(Timestamp(0))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_EQ(*output, 1);
  EXPECT_NE(one_weak.lock(), nullptr);
  EXPECT_EQ(two_weak.lock(), nullptr);

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_select", MakePacket<int>(1).At(Timestamp(1))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_EQ(*output, 3);

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

}  // namespace

class PassThroughAndTsBoundUpdateNode : public mediapipe::api2::Node {
 public:
  static constexpr mediapipe::api2::Input<int> kInValue{"VALUE"};
  static constexpr mediapipe::api2::Output<int> kOutValue{"VALUE"};
  static constexpr mediapipe::api2::Output<int> kOutTsBoundUpdate{
      "TS_BOUND_UPDATE"};
  MEDIAPIPE_NODE_CONTRACT(kInValue, kOutValue, kOutTsBoundUpdate);

  absl::Status Process(CalculatorContext* cc) override {
    kOutValue(cc).Send(kInValue(cc));
    kOutTsBoundUpdate(cc).SetNextTimestampBound(
        cc->InputTimestamp().NextAllowedInStream());
    return absl::OkStatus();
  }
};
MEDIAPIPE_REGISTER_NODE(PassThroughAndTsBoundUpdateNode);

class ToOptionalNode : public mediapipe::api2::Node {
 public:
  static constexpr mediapipe::api2::Input<int> kTick{"TICK"};
  static constexpr mediapipe::api2::Input<int> kInValue{"VALUE"};
  static constexpr mediapipe::api2::Output<absl::optional<int>> kOutValue{
      "OUTPUT"};
  MEDIAPIPE_NODE_CONTRACT(kTick, kInValue, kOutValue);

  absl::Status Process(CalculatorContext* cc) override {
    if (kInValue(cc).IsEmpty()) {
      kOutValue(cc).Send(absl::nullopt);
    } else {
      kOutValue(cc).Send({kInValue(cc).Get()});
    }
    return absl::OkStatus();
  }
};
MEDIAPIPE_REGISTER_NODE(ToOptionalNode);

namespace {

TEST(MuxCalculatorTest, HandleTimestampBoundUpdates) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"pb(
            input_stream: "select"
            node {
              calculator: "PassThroughAndTsBoundUpdateNode"
              input_stream: "VALUE:select"
              output_stream: "VALUE:select_ps"
              output_stream: "TS_BOUND_UPDATE:ts_bound_update"
            }
            node {
              calculator: "MuxCalculator"
              input_stream: "INPUT:0:select_ps"
              input_stream: "INPUT:1:ts_bound_update"
              input_stream: "SELECT:select"
              output_stream: "OUTPUT:select_or_ts_bound_update"
            }
            node {
              calculator: "ToOptionalNode"
              input_stream: "TICK:select"
              input_stream: "VALUE:select_or_ts_bound_update"
              output_stream: "OUTPUT:output"
            }
          )pb");
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output", &config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));

  auto send_value_fn = [&](int value, Timestamp ts) -> absl::Status {
    MP_RETURN_IF_ERROR(
        graph.AddPacketToInputStream("select", MakePacket<int>(value).At(ts)));
    return graph.WaitUntilIdle();
  };

  MP_ASSERT_OK(send_value_fn(0, Timestamp(1)));
  ASSERT_EQ(output_packets.size(), 1);
  EXPECT_EQ(output_packets[0].Get<absl::optional<int>>(), 0);

  MP_ASSERT_OK(send_value_fn(1, Timestamp(2)));
  ASSERT_EQ(output_packets.size(), 2);
  EXPECT_EQ(output_packets[1].Get<absl::optional<int>>(), absl::nullopt);

  MP_ASSERT_OK(send_value_fn(0, Timestamp(3)));
  ASSERT_EQ(output_packets.size(), 3);
  EXPECT_EQ(output_packets[2].Get<absl::optional<int>>(), 0);

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST(MuxCalculatorTest, HandlesCloseGracefully) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"pb(
            input_stream: "select"
            input_stream: "value_0"
            input_stream: "value_1"
            node {
              calculator: "MuxCalculator"
              input_stream: "SELECT:select"
              input_stream: "INPUT:0:value_0"
              input_stream: "INPUT:1:value_1"
              output_stream: "OUTPUT:output"
            }
          )pb");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));

  // Observe packets.
  std::vector<Packet> output_packets;
  MP_ASSERT_OK(graph.ObserveOutputStream(
      "output",
      [&output_packets](const Packet& p) -> absl::Status {
        output_packets.push_back(p);
        return absl::OkStatus();
      },
      /*observe_timestamp_bounds=*/true));

  // Start graph.
  MP_ASSERT_OK(graph.StartRun({}));

  // Add single packet wait for completion and close.
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "value_0", MakePacket<int>(0).At(Timestamp(1000))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());

  EXPECT_TRUE(output_packets.empty());
}

TEST(MuxCalculatorTest, HandlesCloseGracefullyWithDeafultInputStreamHandler) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"pb(
            # This is required in order for EXPECT_DEATH to work everywhere
            executor { name: "" type: "ApplicationThreadExecutor" }

            input_stream: "select"
            input_stream: "value_0"
            input_stream: "value_1"
            node {
              calculator: "MuxCalculator"
              input_stream: "SELECT:select"
              input_stream: "INPUT:0:value_0"
              input_stream: "INPUT:1:value_1"
              output_stream: "OUTPUT:output"
              input_stream_handler {
                input_stream_handler: "DefaultInputStreamHandler"
              }
            }
          )pb");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));

  // Observe packets.
  std::vector<Packet> output_packets;
  MP_ASSERT_OK(graph.ObserveOutputStream(
      "output",
      [&output_packets](const Packet& p) -> absl::Status {
        output_packets.push_back(p);
        return absl::OkStatus();
      },
      /*observe_timestamp_bounds=*/true));

  // Start graph.
  MP_ASSERT_OK(graph.StartRun({}));

  // Add single packet wait for completion and close.
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "value_0", MakePacket<int>(0).At(Timestamp(1000))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_EQ(output_packets.size(), 1);
  EXPECT_TRUE(output_packets[0].IsEmpty());
}

}  // namespace

}  // namespace mediapipe
