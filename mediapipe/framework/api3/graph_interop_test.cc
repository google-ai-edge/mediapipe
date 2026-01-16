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

#include <optional>
#include <string>
#include <utility>

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/graph.h"
#include "mediapipe/framework/api3/port_test_nodes.h"
#include "mediapipe/framework/api3/side_packet.h"
#include "mediapipe/framework/api3/stream.h"
#include "mediapipe/framework/api3/testing/generator.pb.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/stream_handler/fixed_size_input_stream_handler.pb.h"
#include "mediapipe/framework/testdata/night_light_calculator.pb.h"
#include "mediapipe/framework/testdata/sky_light_calculator.pb.h"

namespace mediapipe {
namespace {

TEST(GraphInteropTest, CanInteropWithApi2) {
  api2::builder::Graph api2_graph;

  api2::builder::Stream<int> api2_in = api2_graph.In(0).Cast<int>();
  api2::builder::SidePacket<std::string> api2_side_in =
      api2_graph.SideIn(0).Cast<std::string>();

  std::optional<api2::builder::Stream<int>> api2_foo_out;
  std::optional<api2::builder::SidePacket<std::string>> api2_foo_side_out;
  {
    // Converting graph from API2 to API3
    api3::GenericGraph& api3_graph = api2_graph;

    // Converting stream & side packet from API2 to API3.
    api3::Stream<int> api3_in = api2_in;
    api3::SidePacket<std::string> api3_side_in = api2_side_in;

    // Add API3 node.
    auto& foo = api3_graph.AddNode<api3::FooNode>();
    foo.input.Set(api3_in.SetName("in"));
    foo.side_input.Set(api3_side_in.SetName("side_in"));
    api3::Stream<int> api3_foo_out = foo.output.Get();
    api3::SidePacket<std::string> api3_foo_side_out = foo.side_output.Get();

    // Converting stream & side packet from API3 to API2.
    api2_foo_out = api2::builder::Stream<int>(api3_foo_out);
    api2_foo_side_out =
        api2::builder::SidePacket<std::string>(api3_foo_side_out);
  }

  api2_foo_out->SetName("out").ConnectTo(api2_graph.Out(0));
  api2_foo_side_out->SetName("side_out").ConnectTo(api2_graph.SideOut(0));

  CalculatorGraphConfig expected_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "Foo"
          input_stream: "INPUT:in"
          output_stream: "OUTPUT:out"
          input_side_packet: "SIDE_INPUT:side_in"
          output_side_packet: "SIDE_OUTPUT:side_out"
        }
        input_stream: "in"
        output_stream: "out"
        input_side_packet: "side_in"
        output_side_packet: "side_out"
      )pb");
  EXPECT_THAT(api2_graph.GetConfig(), EqualsProto(expected_config));
}

std::pair<api3::Stream<int>, api3::SidePacket<std::string>> RunFoo(
    api3::GenericGraph& graph, api3::Stream<int> in,
    api3::SidePacket<std::string> side_in) {
  auto& node = graph.AddNode<api3::FooNode>();
  node.input.Set(in);
  node.side_input.Set(side_in);
  return {node.output.Get(), node.side_output.Get()};
}

TEST(GraphInteropTest, CanInteropWithApi2AndUtilityFunctions) {
  api2::builder::Graph api2_graph;

  api2::builder::Stream<int> api2_in =
      api2_graph.In(0).SetName("in").Cast<int>();
  api2::builder::SidePacket<std::string> api2_side_in =
      api2_graph.SideIn(0).SetName("side_in").Cast<std::string>();

  auto result = RunFoo(api2_graph, api2_in, api2_side_in);

  api2::builder::Stream<int> api2_foo_out(result.first);
  api2::builder::SidePacket<std::string> api2_foo_side_out(result.second);

  api2_foo_out.SetName("out").ConnectTo(api2_graph.Out(0));
  api2_foo_side_out.SetName("side_out").ConnectTo(api2_graph.SideOut(0));

  CalculatorGraphConfig expected_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "Foo"
          input_stream: "INPUT:in"
          output_stream: "OUTPUT:out"
          input_side_packet: "SIDE_INPUT:side_in"
          output_side_packet: "SIDE_OUTPUT:side_out"
        }
        input_stream: "in"
        output_stream: "out"
        input_side_packet: "side_in"
        output_side_packet: "side_out"
      )pb");
  EXPECT_THAT(api2_graph.GetConfig(), EqualsProto(expected_config));
}

}  // namespace
}  // namespace mediapipe
