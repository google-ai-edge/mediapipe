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

#include "mediapipe/framework/api3/subgraph.h"

#include <string>

#include "absl/status/status.h"
#include "mediapipe/framework/api3/calculator.h"
#include "mediapipe/framework/api3/calculator_context.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/graph.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/api3/side_packet.h"
#include "mediapipe/framework/api3/stream.h"
#include "mediapipe/framework/api3/subgraph_context.h"
#include "mediapipe/framework/api3/testing/foo.pb.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::api3 {
namespace {

using ::testing::HasSubstr;

template <typename S>
struct SharedContract {
  SideInput<S, int> side_in{"SIDE_IN"};
  SideOutput<S, std::string> side_out{"SIDE_OUT"};

  Input<S, int> in{"IN"};
  Output<S, std::string> out{"OUT"};

  Optional<SideInput<S, int>> optional_side_in{"OPTIONAL_SIDE_IN"};
  Optional<SideOutput<S, std::string>> optional_side_out{"OPTIONAL_SIDE_OUT"};

  Optional<Input<S, int>> optional_in{"OPTIONAL_IN"};
  Optional<Output<S, std::string>> optional_out{"OPTIONAL_OUT"};

  Repeated<SideInput<S, int>> repeated_side_in{"REPEATED_SIDE_IN"};
  Repeated<SideOutput<S, std::string>> repeated_side_out{"REPEATED_SIDE_OUT"};

  Repeated<Input<S, int>> repeated_in{"REPEATED_IN"};
  Repeated<Output<S, std::string>> repeated_out{"REPEATED_OUT"};

  Options<S, FooOptions> options;
};

struct CalculatorNode : Node<"Calculator"> {
  template <typename S>
  using Contract = SharedContract<S>;
};

class CalculatorNodeImpl
    : public Calculator<CalculatorNode, CalculatorNodeImpl> {};

struct SubgraphNode : Node<"Subgraph"> {
  template <typename S>
  using Contract = SharedContract<S>;
};

class SubgraphNodeImpl : public Subgraph<SubgraphNode, SubgraphNodeImpl> {
 public:
  absl::Status Expand(GenericGraph& graph,
                      SubgraphContext<SubgraphNode>& sc) final {
    auto& node = graph.AddNode<CalculatorNode>();
    *node.options.Mutable() = sc.options.Get();

    node.side_in.Set(sc.side_in.Get());
    node.in.Set(sc.in.Get());
    sc.side_out.Set(node.side_out.Get());
    sc.out.Set(node.out.Get());

    if (sc.optional_side_in.IsConnected()) {
      node.optional_side_in.Set(sc.optional_side_in.Get());
    }
    if (sc.optional_in.IsConnected()) {
      node.optional_in.Set(sc.optional_in.Get());
    }
    if (sc.optional_side_out.IsConnected()) {
      sc.optional_side_out.Set(node.optional_side_out.Get());
    }
    if (sc.optional_out.IsConnected()) {
      sc.optional_out.Set(node.optional_out.Get());
    }

    for (int i = 0; i < sc.repeated_side_in.Count(); ++i) {
      node.repeated_side_in.Add(sc.repeated_side_in.At(i).Get());
    }
    for (int i = 0; i < sc.repeated_side_out.Count(); ++i) {
      sc.repeated_side_out.At(i).Set(node.repeated_side_out.Add());
    }
    for (int i = 0; i < sc.repeated_in.Count(); ++i) {
      node.repeated_in.Add(sc.repeated_in.At(i).Get());
    }
    for (int i = 0; i < sc.repeated_out.Count(); ++i) {
      sc.repeated_out.At(i).Set(node.repeated_out.Add());
    }

    return absl::OkStatus();
  }
};

template <typename S>
struct AllPortsContract {
  SideInput<S, int> side_in{"SIDE_IN"};
  SideOutput<S, std::string> side_out{"SIDE_OUT"};

  Input<S, int> in{"IN"};
  Output<S, std::string> out{"OUT"};

  Optional<SideInput<S, int>> optional_side_in{"OPTIONAL_SIDE_IN"};
  Optional<SideOutput<S, std::string>> optional_side_out{"OPTIONAL_SIDE_OUT"};

  Optional<Input<S, int>> optional_in{"OPTIONAL_IN"};
  Optional<Output<S, std::string>> optional_out{"OPTIONAL_OUT"};

  Repeated<SideInput<S, int>> repeated_side_in{"REPEATED_SIDE_IN"};
  Repeated<SideOutput<S, std::string>> repeated_side_out{"REPEATED_SIDE_OUT"};

  Repeated<Input<S, int>> repeated_in{"REPEATED_IN"};
  Repeated<Output<S, std::string>> repeated_out{"REPEATED_OUT"};
};

TEST(SubgraphTest, CanUseSubgraphWithDifferentPortTypes) {
  Graph<AllPortsContract> graph;

  SidePacket<int> side_in = graph.side_in.Get();
  Stream<int> in = graph.in.Get();
  SidePacket<int> optional_side_in = graph.optional_side_in.Get();
  Stream<int> optional_in = graph.optional_in.Get();
  SidePacket<int> repeated_side_in_0 = graph.repeated_side_in.Add();
  SidePacket<int> repeated_side_in_1 = graph.repeated_side_in.Add();
  SidePacket<int> repeated_side_in_2 = graph.repeated_side_in.Add();
  Stream<int> repeated_in_0 = graph.repeated_in.Add();
  Stream<int> repeated_in_1 = graph.repeated_in.Add();

  // Add a subgraph node.
  auto& node = graph.AddNode<SubgraphNode>();
  node.options.Mutable()->set_a(42);
  node.side_in.Set(side_in);
  node.in.Set(in);
  node.optional_side_in.Set(optional_side_in);
  node.optional_in.Set(optional_in);
  node.repeated_side_in.Add(repeated_side_in_0);
  node.repeated_side_in.Add(repeated_side_in_1);
  node.repeated_side_in.Add(repeated_side_in_2);
  node.repeated_in.Add(repeated_in_0);
  node.repeated_in.Add(repeated_in_1);

  graph.side_out.Set(node.side_out.Get());
  graph.out.Set(node.out.Get());
  graph.optional_side_out.Set(node.optional_side_out.Get());
  graph.optional_out.Set(node.optional_out.Get());
  graph.repeated_side_out.Add(node.repeated_side_out.Add());
  graph.repeated_side_out.Add(node.repeated_side_out.Add());
  graph.repeated_side_out.Add(node.repeated_side_out.Add());
  graph.repeated_out.Add(node.repeated_out.Add());
  graph.repeated_out.Add(node.repeated_out.Add());

  MP_ASSERT_OK_AND_ASSIGN(mediapipe::CalculatorGraphConfig config,
                          graph.GetConfig());

  // Expand and check the expanded graph.
  ValidatedGraphConfig validated_config;
  MP_ASSERT_OK(validated_config.Initialize(config));

  // Expecting subgraph to be expanded to a calculator.
  EXPECT_THAT(
      validated_config.Config(),
      EqualsProto(ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(R"pb(
        input_stream: "IN:__stream_0"
        input_stream: "OPTIONAL_IN:__stream_1"
        input_stream: "REPEATED_IN:0:__stream_2"
        input_stream: "REPEATED_IN:1:__stream_3"
        output_stream: "OPTIONAL_OUT:__stream_9"
        output_stream: "OUT:__stream_10"
        output_stream: "REPEATED_OUT:0:__stream_11"
        output_stream: "REPEATED_OUT:1:__stream_12"
        input_side_packet: "OPTIONAL_SIDE_IN:__side_packet_4"
        input_side_packet: "REPEATED_SIDE_IN:0:__side_packet_5"
        input_side_packet: "REPEATED_SIDE_IN:1:__side_packet_6"
        input_side_packet: "REPEATED_SIDE_IN:2:__side_packet_7"
        input_side_packet: "SIDE_IN:__side_packet_8"
        output_side_packet: "OPTIONAL_SIDE_OUT:__side_packet_13"
        output_side_packet: "REPEATED_SIDE_OUT:0:__side_packet_14"
        output_side_packet: "REPEATED_SIDE_OUT:1:__side_packet_15"
        output_side_packet: "REPEATED_SIDE_OUT:2:__side_packet_16"
        output_side_packet: "SIDE_OUT:__side_packet_17"

        node {
          name: "subgraph__Calculator"
          calculator: "Calculator"
          input_stream: "IN:__stream_0"
          input_stream: "OPTIONAL_IN:__stream_1"
          input_stream: "REPEATED_IN:0:__stream_2"
          input_stream: "REPEATED_IN:1:__stream_3"
          output_stream: "OPTIONAL_OUT:__stream_9"
          output_stream: "OUT:__stream_10"
          output_stream: "REPEATED_OUT:0:__stream_11"
          output_stream: "REPEATED_OUT:1:__stream_12"
          input_side_packet: "OPTIONAL_SIDE_IN:__side_packet_4"
          input_side_packet: "REPEATED_SIDE_IN:0:__side_packet_5"
          input_side_packet: "REPEATED_SIDE_IN:1:__side_packet_6"
          input_side_packet: "REPEATED_SIDE_IN:2:__side_packet_7"
          input_side_packet: "SIDE_IN:__side_packet_8"
          output_side_packet: "OPTIONAL_SIDE_OUT:__side_packet_13"
          output_side_packet: "REPEATED_SIDE_OUT:0:__side_packet_14"
          output_side_packet: "REPEATED_SIDE_OUT:1:__side_packet_15"
          output_side_packet: "REPEATED_SIDE_OUT:2:__side_packet_16"
          output_side_packet: "SIDE_OUT:__side_packet_17"
          node_options {
            [type.googleapis.com/mediapipe.FooOptions] { a: 42 }
          }
        }

        executor {}
      )pb")));
}

template <typename S>
struct RequiredPortsOnlyContract {
  SideInput<S, int> side_in{"SIDE_IN"};
  SideOutput<S, std::string> side_out{"SIDE_OUT"};

  Input<S, int> in{"IN"};
  Output<S, std::string> out{"OUT"};
};

TEST(SubgraphTest, CanUseSubgraphWithMissingOptionalAndRepeatedPorts) {
  Graph<RequiredPortsOnlyContract> graph;

  SidePacket<int> side_in = graph.side_in.Get();
  Stream<int> in = graph.in.Get();

  auto& node = graph.AddNode<SubgraphNode>();
  node.side_in.Set(side_in);
  node.in.Set(in);

  graph.side_out.Set(node.side_out.Get());
  graph.out.Set(node.out.Get());

  MP_ASSERT_OK_AND_ASSIGN(mediapipe::CalculatorGraphConfig config,
                          graph.GetConfig());

  // Expand and check the expanded graph.
  ValidatedGraphConfig validated_config;
  MP_ASSERT_OK(validated_config.Initialize(config));

  EXPECT_THAT(
      validated_config.Config(),
      EqualsProto(ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(R"pb(
        input_stream: "IN:__stream_0"
        output_stream: "OUT:__stream_2"
        input_side_packet: "SIDE_IN:__side_packet_1"
        output_side_packet: "SIDE_OUT:__side_packet_3"

        node {
          name: "subgraph__Calculator"
          calculator: "Calculator"
          input_stream: "IN:__stream_0"
          output_stream: "OUT:__stream_2"
          input_side_packet: "SIDE_IN:__side_packet_1"
          output_side_packet: "SIDE_OUT:__side_packet_3"
          node_options {
            [type.googleapis.com/mediapipe.FooOptions] {}
          }
        }

        executor {}
      )pb")));
}

TEST(SubgraphTest, FailsToValidateWithMissingRequiredPorts) {
  Graph<RequiredPortsOnlyContract> graph;

  SidePacket<int> side_in = graph.side_in.Get();

  auto& node = graph.AddNode<SubgraphNode>();
  node.side_in.Set(side_in);

  graph.side_out.Set(node.side_out.Get());

  MP_ASSERT_OK_AND_ASSIGN(mediapipe::CalculatorGraphConfig config,
                          graph.GetConfig());

  // Expand and check the expanded graph.
  ValidatedGraphConfig validated_config;
  EXPECT_THAT(validated_config.Initialize(config),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("GetContract failed to validate")));
}

}  // namespace
}  // namespace mediapipe::api3
