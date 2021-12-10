#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/api2/test_contracts.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/subgraph_expansion.h"

namespace mediapipe {
namespace api2 {
namespace test {

class FooBarImpl1 : public SubgraphImpl<FooBar1, FooBarImpl1> {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      const SubgraphOptions& /*options*/) {
    builder::Graph graph;
    auto& foo = graph.AddNode("Foo");
    auto& bar = graph.AddNode("Bar");
    graph.In(kIn) >> foo.In("BASE");
    foo.Out("OUT") >> bar.In("IN");
    bar.Out("OUT") >> graph.Out(kOut);
    return graph.GetConfig();
  }
};

class FooBarImpl2 : public SubgraphImpl<FooBar2, FooBarImpl2> {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      const SubgraphOptions& /*options*/) {
    builder::Graph graph;
    auto& foo = graph.AddNode<Foo>();
    auto& bar = graph.AddNode<Bar>();
    graph.In(kIn) >> foo.In(MPP_TAG("BASE"));
    foo.Out(MPP_TAG("OUT")) >> bar.In(MPP_TAG("IN"));
    bar.Out(MPP_TAG("OUT")) >> graph.Out(kOut);
    return graph.GetConfig();
  }
};

TEST(SubgraphTest, SubgraphConfig) {
  CalculatorGraphConfig subgraph = FooBarImpl1().GetConfig({}).value();
  const CalculatorGraphConfig expected_graph =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "IN:__stream_0"
        output_stream: "OUT:__stream_2"
        node {
          calculator: "Foo"
          input_stream: "BASE:__stream_0"
          output_stream: "OUT:__stream_1"
        }
        node {
          calculator: "Bar"
          input_stream: "IN:__stream_1"
          output_stream: "OUT:__stream_2"
        }
      )pb");
  EXPECT_THAT(subgraph, EqualsProto(expected_graph));
}

TEST(SubgraphTest, TypedSubgraphConfig) {
  CalculatorGraphConfig subgraph = FooBarImpl2().GetConfig({}).value();
  const CalculatorGraphConfig expected_graph =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "IN:__stream_0"
        output_stream: "OUT:__stream_2"
        node {
          calculator: "Foo"
          input_stream: "BASE:__stream_0"
          output_stream: "OUT:__stream_1"
        }
        node {
          calculator: "Bar"
          input_stream: "IN:__stream_1"
          output_stream: "OUT:__stream_2"
        }
      )pb");
  EXPECT_THAT(subgraph, EqualsProto(expected_graph));
}

TEST(SubgraphTest, ProtoApiConfig) {
  CalculatorGraphConfig graph;
  graph.add_input_stream("IN:__stream_0");
  graph.add_output_stream("OUT:__stream_2");
  auto* foo = graph.add_node();
  foo->set_calculator("Foo");
  foo->add_input_stream("BASE:__stream_0");
  foo->add_output_stream("OUT:__stream_1");
  auto* bar = graph.add_node();
  bar->set_calculator("Bar");
  bar->add_input_stream("IN:__stream_1");
  bar->add_output_stream("OUT:__stream_2");

  const CalculatorGraphConfig expected_graph =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "IN:__stream_0"
        output_stream: "OUT:__stream_2"
        node {
          calculator: "Foo"
          input_stream: "BASE:__stream_0"
          output_stream: "OUT:__stream_1"
        }
        node {
          calculator: "Bar"
          input_stream: "IN:__stream_1"
          output_stream: "OUT:__stream_2"
        }
      )pb");
  EXPECT_THAT(graph, EqualsProto(expected_graph));
}

TEST(SubgraphTest, ExpandSubgraphs) {
  CalculatorGraphConfig supergraph =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          name: "simple_source"
          calculator: "SomeSourceCalculator"
          output_stream: "foo"
        }
        node {
          calculator: "FooBar"
          input_stream: "IN:foo"
          output_stream: "OUT:output"
        }
      )pb");
  const CalculatorGraphConfig expected_graph =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          name: "simple_source"
          calculator: "SomeSourceCalculator"
          output_stream: "foo"
        }
        node {
          name: "foobar__Foo"
          calculator: "Foo"
          input_stream: "BASE:foo"
          output_stream: "OUT:foobar____stream_1"
        }
        node {
          name: "foobar__Bar"
          calculator: "Bar"
          input_stream: "IN:foobar____stream_1"
          output_stream: "OUT:output"
        }
      )pb");
  MP_EXPECT_OK(tool::ExpandSubgraphs(&supergraph));
  EXPECT_THAT(supergraph, EqualsProto(expected_graph));
}

}  // namespace test
}  // namespace api2
}  // namespace mediapipe
