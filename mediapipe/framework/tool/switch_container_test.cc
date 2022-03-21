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

#include "absl/strings/str_replace.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/proto_ns.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/subgraph.h"
#include "mediapipe/framework/tool/node_chain_subgraph.pb.h"
#include "mediapipe/framework/tool/subgraph_expansion.h"

namespace mediapipe {
namespace {

// A Calculator that outputs thrice the value of its input packet (an int).
// It also accepts a side packet tagged "TIMEZONE", but doesn't use it.
class TripleIntCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<int>().Optional();
    cc->Outputs().Index(0).SetSameAs(&cc->Inputs().Index(0)).Optional();
    cc->InputSidePackets().Index(0).Set<int>().Optional();
    cc->OutputSidePackets()
        .Index(0)
        .SetSameAs(&cc->InputSidePackets().Index(0))
        .Optional();
    cc->InputSidePackets().Tag("TIMEZONE").Set<int>().Optional();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final {
    cc->SetOffset(TimestampDiff(0));
    if (cc->OutputSidePackets().HasTag("")) {
      cc->OutputSidePackets().Index(0).Set(
          MakePacket<int>(cc->InputSidePackets().Index(0).Get<int>() * 3));
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    int value = cc->Inputs().Index(0).Value().Get<int>();
    cc->Outputs().Index(0).Add(new int(3 * value), cc->InputTimestamp());
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(TripleIntCalculator);

// A testing example of a SwitchContainer containing two subnodes.
// Note that the input and output tags supplied to the container node,
// must match the input and output tags required by the subnodes.
CalculatorGraphConfig SubnodeContainerExample(const std::string& options = "") {
  std::string config = R"pb(
    input_stream: "foo"
    input_stream: "enable"
    input_side_packet: "timezone"
    node {
      calculator: "SwitchContainer"
      input_stream: "ENABLE:enable"
      input_stream: "foo"
      output_stream: "bar"
      options {
        [mediapipe.SwitchContainerOptions.ext] {
          contained_node: { calculator: "TripleIntCalculator" }
          contained_node: { calculator: "PassThroughCalculator" } $options
        }
      }
    }
    node {
      calculator: "PassThroughCalculator"
      input_stream: "foo"
      input_stream: "bar"
      output_stream: "output_foo"
      output_stream: "output_bar"
    }
  )pb";

  return mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
      absl::StrReplaceAll(config, {{"$options", options}}));
}

// A testing example of a SwitchContainer containing two subnodes.
// Note that the side-input and side-output tags supplied to the container node,
// must match the side-input and side-output tags required by the subnodes.
CalculatorGraphConfig SideSubnodeContainerExample() {
  return mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_side_packet: "foo"
    input_side_packet: "enable"
    output_side_packet: "output_bar"
    node {
      calculator: "SwitchContainer"
      input_side_packet: "ENABLE:enable"
      input_side_packet: "foo"
      output_side_packet: "bar"
      options {
        [mediapipe.SwitchContainerOptions.ext] {
          contained_node: { calculator: "TripleIntCalculator" }
          contained_node: { calculator: "PassThroughCalculator" }
        }
      }
    }
    node {
      calculator: "PassThroughCalculator"
      input_side_packet: "foo"
      input_side_packet: "bar"
      output_side_packet: "output_foo"
      output_side_packet: "output_bar"
    }
  )pb");
}

// Runs the test container graph with a few input packets.
void RunTestContainer(CalculatorGraphConfig supergraph,
                      bool send_bounds = false) {
  CalculatorGraph graph;
  std::vector<Packet> out_foo, out_bar;
  tool::AddVectorSink("output_foo", &supergraph, &out_foo);
  tool::AddVectorSink("output_bar", &supergraph, &out_bar);
  MP_ASSERT_OK(graph.Initialize(supergraph, {}));
  MP_ASSERT_OK(graph.StartRun({{"timezone", MakePacket<int>(3)}}));

  if (!send_bounds) {
    // Send enable == true signal at 5000 us.
    const int64 enable_ts = 5000;
    MP_EXPECT_OK(graph.AddPacketToInputStream(
        "enable", MakePacket<bool>(true).At(Timestamp(enable_ts))));
    MP_ASSERT_OK(graph.WaitUntilIdle());
  }

  const int packet_count = 10;
  // Send int value packets at {10K, 20K, 30K, ..., 100K}.
  for (uint64 t = 1; t <= packet_count; ++t) {
    if (send_bounds) {
      MP_EXPECT_OK(graph.AddPacketToInputStream(
          "enable", MakePacket<bool>(true).At(Timestamp(t * 10000))));
      MP_ASSERT_OK(graph.WaitUntilIdle());
    }
    MP_EXPECT_OK(graph.AddPacketToInputStream(
        "foo", MakePacket<int>(t).At(Timestamp(t * 10000))));
    MP_ASSERT_OK(graph.WaitUntilIdle());
    // The inputs are sent to the input stream "foo", they should pass through.
    EXPECT_EQ(out_foo.size(), t);
    // Since "enable == true" for ts 10K...100K us, the second contained graph
    // i.e. the one containing the PassThroughCalculator should output the
    // input values without changing them.
    EXPECT_EQ(out_bar.size(), t);
    EXPECT_EQ(out_bar.back().Get<int>(), t);
  }

  if (!send_bounds) {
    // Send enable == false signal at 105K us.
    MP_EXPECT_OK(graph.AddPacketToInputStream(
        "enable", MakePacket<bool>(false).At(Timestamp(105000))));
    MP_ASSERT_OK(graph.WaitUntilIdle());
  }

  // Send int value packets at {110K, 120K, ..., 200K}.
  for (uint64 t = 11; t <= packet_count * 2; ++t) {
    if (send_bounds) {
      MP_EXPECT_OK(graph.AddPacketToInputStream(
          "enable", MakePacket<bool>(false).At(Timestamp(t * 10000))));
      MP_ASSERT_OK(graph.WaitUntilIdle());
    }
    MP_EXPECT_OK(graph.AddPacketToInputStream(
        "foo", MakePacket<int>(t).At(Timestamp(t * 10000))));
    MP_ASSERT_OK(graph.WaitUntilIdle());
    // The inputs are sent to the input stream "foo", they should pass through.
    EXPECT_EQ(out_foo.size(), t);
    // Since "enable == false" for ts 110K...200K us, the first contained graph
    // i.e. the one containing the TripleIntCalculator should output the values
    // after tripling them.
    EXPECT_EQ(out_bar.size(), t);
    EXPECT_EQ(out_bar.back().Get<int>(), t * 3);
  }

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
  EXPECT_EQ(out_foo.size(), packet_count * 2);
  EXPECT_EQ(out_bar.size(), packet_count * 2);
}

// Runs the test side-packet container graph with input side-packets.
void RunTestSideContainer(CalculatorGraphConfig supergraph) {
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(supergraph, {}));
  MP_ASSERT_OK(graph.StartRun({
      {"enable", MakePacket<bool>(false)},
      {"foo", MakePacket<int>(4)},
  }));
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
  Packet side_output = graph.GetOutputSidePacket("output_bar").value();
  EXPECT_EQ(side_output.Get<int>(), 12);

  MP_ASSERT_OK(graph.StartRun({
      {"enable", MakePacket<bool>(true)},
      {"foo", MakePacket<int>(4)},
  }));
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
  side_output = graph.GetOutputSidePacket("output_bar").value();
  EXPECT_EQ(side_output.Get<int>(), 4);
}

// Rearrange the Node messages within a CalculatorGraphConfig message.
CalculatorGraphConfig OrderNodes(const CalculatorGraphConfig& config,
                                 std::vector<int> order) {
  auto result = config;
  result.clear_node();
  for (int i = 0; i < order.size(); ++i) {
    *result.add_node() = config.node(order[i]);
  }
  return result;
}

// Shows the SwitchContainer container applied to a pair of simple subnodes.
TEST(SwitchContainerTest, ApplyToSubnodes) {
  EXPECT_TRUE(SubgraphRegistry::IsRegistered("SwitchContainer"));
  CalculatorGraphConfig supergraph = SubnodeContainerExample();
  CalculatorGraphConfig expected_graph =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          name: "switchcontainer__SwitchDemuxCalculator"
          calculator: "SwitchDemuxCalculator"
          input_stream: "ENABLE:enable"
          input_stream: "foo"
          output_stream: "C0__:switchcontainer__c0__foo"
          output_stream: "C1__:switchcontainer__c1__foo"
          options {
            [mediapipe.SwitchContainerOptions.ext] {}
          }
        }
        node {
          name: "switchcontainer__TripleIntCalculator"
          calculator: "TripleIntCalculator"
          input_stream: "switchcontainer__c0__foo"
          output_stream: "switchcontainer__c0__bar"
        }
        node {
          name: "switchcontainer__PassThroughCalculator"
          calculator: "PassThroughCalculator"
          input_stream: "switchcontainer__c1__foo"
          output_stream: "switchcontainer__c1__bar"
        }
        node {
          name: "switchcontainer__SwitchMuxCalculator"
          calculator: "SwitchMuxCalculator"
          input_stream: "ENABLE:enable"
          input_stream: "C0__:switchcontainer__c0__bar"
          input_stream: "C1__:switchcontainer__c1__bar"
          output_stream: "bar"
          options {
            [mediapipe.SwitchContainerOptions.ext] {}
          }
        }
        node {
          calculator: "PassThroughCalculator"
          input_stream: "foo"
          input_stream: "bar"
          output_stream: "output_foo"
          output_stream: "output_bar"
        }
        input_stream: "foo"
        input_stream: "enable"
        input_side_packet: "timezone"
      )pb");
  expected_graph = OrderNodes(expected_graph, {4, 0, 3, 1, 2});
  MP_EXPECT_OK(tool::ExpandSubgraphs(&supergraph));
  EXPECT_THAT(supergraph, mediapipe::EqualsProto(expected_graph));
}

// Shows the SwitchContainer container runs with a pair of simple subnodes.
TEST(SwitchContainerTest, RunsWithSubnodes) {
  EXPECT_TRUE(SubgraphRegistry::IsRegistered("SwitchContainer"));
  CalculatorGraphConfig supergraph = SubnodeContainerExample();
  MP_EXPECT_OK(tool::ExpandSubgraphs(&supergraph));
  RunTestContainer(supergraph);
}

// Shows the SwitchContainer  does not allow input_stream_handler overwrite.
TEST(SwitchContainerTest, ValidateInputStreamHandler) {
  EXPECT_TRUE(SubgraphRegistry::IsRegistered("SwitchContainer"));
  CalculatorGraph graph;
  CalculatorGraphConfig supergraph = SideSubnodeContainerExample();
  *supergraph.mutable_input_stream_handler()->mutable_input_stream_handler() =
      "DefaultInputStreamHandler";
  MP_ASSERT_OK(graph.Initialize(supergraph, {}));
  CalculatorGraphConfig expected_graph = mediapipe::ParseTextProtoOrDie<
      CalculatorGraphConfig>(R"pb(
    node {
      name: "switchcontainer__SwitchDemuxCalculator"
      calculator: "SwitchDemuxCalculator"
      input_side_packet: "ENABLE:enable"
      input_side_packet: "foo"
      output_side_packet: "C0__:switchcontainer__c0__foo"
      output_side_packet: "C1__:switchcontainer__c1__foo"
      options {
        [mediapipe.SwitchContainerOptions.ext] {}
      }
      input_stream_handler { input_stream_handler: "DefaultInputStreamHandler" }
    }
    node {
      name: "switchcontainer__TripleIntCalculator"
      calculator: "TripleIntCalculator"
      input_side_packet: "switchcontainer__c0__foo"
      output_side_packet: "switchcontainer__c0__bar"
      input_stream_handler { input_stream_handler: "DefaultInputStreamHandler" }
    }
    node {
      name: "switchcontainer__PassThroughCalculator"
      calculator: "PassThroughCalculator"
      input_side_packet: "switchcontainer__c1__foo"
      output_side_packet: "switchcontainer__c1__bar"
      input_stream_handler { input_stream_handler: "DefaultInputStreamHandler" }
    }
    node {
      name: "switchcontainer__SwitchMuxCalculator"
      calculator: "SwitchMuxCalculator"
      input_side_packet: "ENABLE:enable"
      input_side_packet: "C0__:switchcontainer__c0__bar"
      input_side_packet: "C1__:switchcontainer__c1__bar"
      output_side_packet: "bar"
      options {
        [mediapipe.SwitchContainerOptions.ext] {}
      }
      input_stream_handler { input_stream_handler: "DefaultInputStreamHandler" }
    }
    node {
      calculator: "PassThroughCalculator"
      input_side_packet: "foo"
      input_side_packet: "bar"
      output_side_packet: "output_foo"
      output_side_packet: "output_bar"
      input_stream_handler { input_stream_handler: "DefaultInputStreamHandler" }
    }
    input_stream_handler { input_stream_handler: "DefaultInputStreamHandler" }
    executor {}
    input_side_packet: "foo"
    input_side_packet: "enable"
    output_side_packet: "output_bar"
  )pb");
  EXPECT_THAT(graph.Config(), mediapipe::EqualsProto(expected_graph));
}

TEST(SwitchContainerTest, RunsWithInputStreamHandler) {
  EXPECT_TRUE(SubgraphRegistry::IsRegistered("SwitchContainer"));
  CalculatorGraphConfig supergraph =
      SubnodeContainerExample(R"pb(synchronize_io: true)pb");
  MP_EXPECT_OK(tool::ExpandSubgraphs(&supergraph));
  LOG(INFO) << supergraph.DebugString();
  RunTestContainer(supergraph, true);
}

// Shows the SwitchContainer container applied to a pair of simple subnodes.
TEST(SwitchContainerTest, ApplyToSideSubnodes) {
  EXPECT_TRUE(SubgraphRegistry::IsRegistered("SwitchContainer"));
  CalculatorGraphConfig supergraph = SideSubnodeContainerExample();
  CalculatorGraphConfig expected_graph =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_side_packet: "foo"
        input_side_packet: "enable"
        output_side_packet: "output_bar"
        node {
          name: "switchcontainer__SwitchDemuxCalculator"
          calculator: "SwitchDemuxCalculator"
          input_side_packet: "ENABLE:enable"
          input_side_packet: "foo"
          output_side_packet: "C0__:switchcontainer__c0__foo"
          output_side_packet: "C1__:switchcontainer__c1__foo"
          options {
            [mediapipe.SwitchContainerOptions.ext] {}
          }
        }
        node {
          name: "switchcontainer__TripleIntCalculator"
          calculator: "TripleIntCalculator"
          input_side_packet: "switchcontainer__c0__foo"
          output_side_packet: "switchcontainer__c0__bar"
        }
        node {
          name: "switchcontainer__PassThroughCalculator"
          calculator: "PassThroughCalculator"
          input_side_packet: "switchcontainer__c1__foo"
          output_side_packet: "switchcontainer__c1__bar"
        }
        node {
          name: "switchcontainer__SwitchMuxCalculator"
          calculator: "SwitchMuxCalculator"
          input_side_packet: "ENABLE:enable"
          input_side_packet: "C0__:switchcontainer__c0__bar"
          input_side_packet: "C1__:switchcontainer__c1__bar"
          output_side_packet: "bar"
          options {
            [mediapipe.SwitchContainerOptions.ext] {}
          }
        }
        node {
          calculator: "PassThroughCalculator"
          input_side_packet: "foo"
          input_side_packet: "bar"
          output_side_packet: "output_foo"
          output_side_packet: "output_bar"
        }
      )pb");
  expected_graph = OrderNodes(expected_graph, {4, 0, 3, 1, 2});
  MP_EXPECT_OK(tool::ExpandSubgraphs(&supergraph));
  EXPECT_THAT(supergraph, mediapipe::EqualsProto(expected_graph));
}

// Shows the SwitchContainer container runs with a pair of simple subnodes.
TEST(SwitchContainerTest, RunWithSideSubnodes) {
  EXPECT_TRUE(SubgraphRegistry::IsRegistered("SwitchContainer"));
  CalculatorGraphConfig supergraph = SideSubnodeContainerExample();
  MP_EXPECT_OK(tool::ExpandSubgraphs(&supergraph));
  RunTestSideContainer(supergraph);
}

// Shows validation of SwitchContainer container side inputs.
TEST(SwitchContainerTest, ValidateSideInputs) {
  EXPECT_TRUE(SubgraphRegistry::IsRegistered("SwitchContainer"));
  CalculatorGraphConfig supergraph =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_side_packet: "foo"
        input_side_packet: "enable"
        output_side_packet: "output_bar"
        node {
          calculator: "SwitchContainer"
          input_side_packet: "ENABLE:enable"
          input_side_packet: "SELECT:enable"
          input_side_packet: "foo"
          output_side_packet: "bar"
          options {
            [mediapipe.SwitchContainerOptions.ext] {
              contained_node: { calculator: "TripleIntCalculator" }
              contained_node: { calculator: "PassThroughCalculator" }
            }
          }
        }
        node {
          calculator: "PassThroughCalculator"
          input_side_packet: "foo"
          input_side_packet: "bar"
          output_side_packet: "output_foo"
          output_side_packet: "output_bar"
        }
      )pb");
  auto status = tool::ExpandSubgraphs(&supergraph);
  EXPECT_EQ(std::pair(status.code(), std::string(status.message())),
            std::pair(absl::StatusCode::kInvalidArgument,
                      std::string("Only one of SwitchContainer inputs "
                                  "'ENABLE' and 'SELECT' can be specified")));
}

}  // namespace
}  // namespace mediapipe
