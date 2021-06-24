#include "mediapipe/framework/api2/builder.h"

#include "absl/strings/substitute.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/api2/tag.h"
#include "mediapipe/framework/api2/test_contracts.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/message_matchers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace api2 {
namespace test {

TEST(BuilderTest, BuildGraph) {
  builder::Graph graph;
  auto& foo = graph.AddNode("Foo");
  auto& bar = graph.AddNode("Bar");
  graph.In("IN").SetName("base") >> foo.In("BASE");
  graph.SideIn("SIDE").SetName("side") >> foo.SideIn("SIDE");
  foo.Out("OUT") >> bar.In("IN");
  bar.Out("OUT").SetName("out") >> graph.Out("OUT");

  CalculatorGraphConfig expected =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "IN:base"
        input_side_packet: "SIDE:side"
        output_stream: "OUT:out"
        node {
          calculator: "Foo"
          input_stream: "BASE:base"
          input_side_packet: "SIDE:side"
          output_stream: "OUT:__stream_0"
        }
        node {
          calculator: "Bar"
          input_stream: "IN:__stream_0"
          output_stream: "OUT:out"
        }
      )pb");
  EXPECT_THAT(graph.GetConfig(), EqualsProto(expected));
}

template <class FooT>
void BuildGraphTypedTest() {
  builder::Graph graph;
  auto& foo = graph.AddNode<FooT>();
  auto& bar = graph.AddNode<Bar>();
  graph.In("IN").SetName("base") >> foo.In(MPP_TAG("BASE"));
  graph.SideIn("SIDE").SetName("side") >> foo.SideIn(MPP_TAG("BIAS"));
  foo.Out(MPP_TAG("OUT")) >> bar.In(MPP_TAG("IN"));
  bar.Out(MPP_TAG("OUT")).SetName("out") >> graph.Out("OUT");

  CalculatorGraphConfig expected =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
          absl::Substitute(R"(
        input_stream: "IN:base"
        input_side_packet: "SIDE:side"
        output_stream: "OUT:out"
        node {
          calculator: "$0"
          input_stream: "BASE:base"
          input_side_packet: "BIAS:side"
          output_stream: "OUT:__stream_0"
        }
        node {
          calculator: "Bar"
          input_stream: "IN:__stream_0"
          output_stream: "OUT:out"
        }
      )",
                           FooT::kCalculatorName));
  EXPECT_THAT(graph.GetConfig(), EqualsProto(expected));
}

TEST(BuilderTest, BuildGraphTyped) { BuildGraphTypedTest<Foo>(); }

TEST(BuilderTest, BuildGraphTyped2) { BuildGraphTypedTest<Foo2>(); }

TEST(BuilderTest, FanOut) {
  builder::Graph graph;
  auto& foo = graph.AddNode("Foo");
  auto& adder = graph.AddNode("FloatAdder");
  graph.In("IN").SetName("base") >> foo.In("BASE");
  foo.Out("OUT") >> adder.In("IN")[0];
  foo.Out("OUT") >> adder.In("IN")[1];
  adder.Out("OUT").SetName("out") >> graph.Out("OUT");

  CalculatorGraphConfig expected =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "IN:base"
        output_stream: "OUT:out"
        node {
          calculator: "Foo"
          input_stream: "BASE:base"
          output_stream: "OUT:__stream_0"
        }
        node {
          calculator: "FloatAdder"
          input_stream: "IN:0:__stream_0"
          input_stream: "IN:1:__stream_0"
          output_stream: "OUT:out"
        }
      )pb");
  EXPECT_THAT(graph.GetConfig(), EqualsProto(expected));
}

TEST(BuilderTest, TypedMultiple) {
  builder::Graph graph;
  auto& foo = graph.AddNode<Foo>();
  auto& adder = graph.AddNode<FloatAdder>();
  graph.In("IN").SetName("base") >> foo.In(MPP_TAG("BASE"));
  foo.Out(MPP_TAG("OUT")) >> adder.In(MPP_TAG("IN"))[0];
  foo.Out(MPP_TAG("OUT")) >> adder.In(MPP_TAG("IN"))[1];
  adder.Out(MPP_TAG("OUT")).SetName("out") >> graph.Out("OUT");

  CalculatorGraphConfig expected =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "IN:base"
        output_stream: "OUT:out"
        node {
          calculator: "Foo"
          input_stream: "BASE:base"
          output_stream: "OUT:__stream_0"
        }
        node {
          calculator: "FloatAdder"
          input_stream: "IN:0:__stream_0"
          input_stream: "IN:1:__stream_0"
          output_stream: "OUT:out"
        }
      )pb");
  EXPECT_THAT(graph.GetConfig(), EqualsProto(expected));
}

TEST(BuilderTest, TypedByPorts) {
  builder::Graph graph;
  auto& foo = graph.AddNode<Foo>();
  auto& adder = graph.AddNode<FloatAdder>();

  graph[FooBar1::kIn].SetName("base") >> foo[Foo::kBase];
  foo[Foo::kOut] >> adder[FloatAdder::kIn][0];
  foo[Foo::kOut] >> adder[FloatAdder::kIn][1];
  adder[FloatAdder::kOut].SetName("out") >> graph[FooBar1::kOut];

  CalculatorGraphConfig expected =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "IN:base"
        output_stream: "OUT:out"
        node {
          calculator: "Foo"
          input_stream: "BASE:base"
          output_stream: "OUT:__stream_0"
        }
        node {
          calculator: "FloatAdder"
          input_stream: "IN:0:__stream_0"
          input_stream: "IN:1:__stream_0"
          output_stream: "OUT:out"
        }
      )pb");
  EXPECT_THAT(graph.GetConfig(), EqualsProto(expected));
}

TEST(BuilderTest, PacketGenerator) {
  builder::Graph graph;
  auto& generator = graph.AddPacketGenerator("FloatGenerator");
  graph.SideIn("IN") >> generator.SideIn("IN");
  generator.SideOut("OUT") >> graph.SideOut("OUT");

  CalculatorGraphConfig expected =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_side_packet: "IN:__side_packet_0"
        output_side_packet: "OUT:__side_packet_1"
        packet_generator {
          packet_generator: "FloatGenerator"
          input_side_packet: "IN:__side_packet_0"
          output_side_packet: "OUT:__side_packet_1"
        }
      )pb");
  EXPECT_THAT(graph.GetConfig(), EqualsProto(expected));
}

TEST(BuilderTest, EmptyTag) {
  builder::Graph graph;
  auto& foo = graph.AddNode("Foo");
  graph.In("A").SetName("a") >> foo.In("")[0];
  graph.In("C").SetName("c") >> foo.In("")[2];
  graph.In("B").SetName("b") >> foo.In("")[1];
  foo.Out("")[0].SetName("x") >> graph.Out("ONE");
  foo.Out("")[1].SetName("y") >> graph.Out("TWO");

  CalculatorGraphConfig expected =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "A:a"
        input_stream: "B:b"
        input_stream: "C:c"
        output_stream: "ONE:x"
        output_stream: "TWO:y"
        node {
          calculator: "Foo"
          input_stream: "a"
          input_stream: "b"
          input_stream: "c"
          output_stream: "x"
          output_stream: "y"
        }
      )pb");
  EXPECT_THAT(graph.GetConfig(), EqualsProto(expected));
}

}  // namespace test
}  // namespace api2
}  // namespace mediapipe
