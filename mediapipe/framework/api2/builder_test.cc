#include "mediapipe/framework/api2/builder.h"

#include <functional>

#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/api2/tag.h"
#include "mediapipe/framework/api2/test_contracts.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::api2::builder {
namespace {

using ::mediapipe::api2::test::Bar;
using ::mediapipe::api2::test::FloatAdder;
using ::mediapipe::api2::test::Foo;
using ::mediapipe::api2::test::Foo2;
using ::mediapipe::api2::test::FooBar1;

TEST(BuilderTest, BuildGraph) {
  Graph graph;
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

TEST(BuilderTest, CopyableSource) {
  Graph graph;
  Source<int> a = graph.In("A").SetName("a").Cast<int>();
  Source<int> b = graph.In("B").SetName("b").Cast<int>();
  SideSource<float> side_a =
      graph.SideIn("SIDE_A").SetName("side_a").Cast<float>();
  SideSource<float> side_b =
      graph.SideIn("SIDE_B").SetName("side_b").Cast<float>();
  Destination<int> out = graph.Out("OUT").Cast<int>();
  SideDestination<float> side_out = graph.SideOut("SIDE_OUT").Cast<float>();

  Source<int> input = a;
  input = b;
  SideSource<float> side_input = side_b;
  side_input = side_a;

  input >> out;
  side_input >> side_out;

  CalculatorGraphConfig expected =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "A:a"
        input_stream: "B:b"
        output_stream: "OUT:b"
        input_side_packet: "SIDE_A:side_a"
        input_side_packet: "SIDE_B:side_b"
        output_side_packet: "SIDE_OUT:side_a"
      )pb");
  EXPECT_THAT(graph.GetConfig(), EqualsProto(expected));
}

TEST(BuilderTest, BuildGraphWithFunctions) {
  Graph graph;

  Source<int> base = graph.In("IN").SetName("base").Cast<int>();
  SideSource<float> side = graph.SideIn("SIDE").SetName("side").Cast<float>();

  auto foo_fn = [](Source<int> base, SideSource<float> side, Graph& graph) {
    auto& foo = graph.AddNode("Foo");
    base >> foo.In("BASE");
    side >> foo.SideIn("SIDE");
    return foo.Out("OUT")[0].Cast<double>();
  };
  Source<double> foo_out = foo_fn(base, side, graph);

  auto bar_fn = [](Source<double> in, Graph& graph) {
    auto& bar = graph.AddNode("Bar");
    in >> bar.In("IN");
    return bar.Out("OUT")[0].Cast<double>();
  };
  Source<double> bar_out = bar_fn(foo_out, graph);

  bar_out.SetName("out") >> graph.Out("OUT");

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
  Graph graph;
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

TEST(BuilderTest, BuildGraphTyped) { BuildGraphTypedTest<test::Foo>(); }

TEST(BuilderTest, BuildGraphTyped2) { BuildGraphTypedTest<test::Foo2>(); }

TEST(BuilderTest, FanOut) {
  Graph graph;
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
  Graph graph;
  auto& foo = graph.AddNode<test::Foo>();
  auto& adder = graph.AddNode<test::FloatAdder>();
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
  Graph graph;
  auto& foo = graph.AddNode<test::Foo>();
  auto& adder = graph.AddNode<FloatAdder>();

  graph.In(FooBar1::kIn).SetName("base") >> foo[Foo::kBase];
  foo[Foo::kOut] >> adder[FloatAdder::kIn][0];
  foo[Foo::kOut] >> adder[FloatAdder::kIn][1];
  adder[FloatAdder::kOut].SetName("out") >> graph.Out(FooBar1::kOut);

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
  Graph graph;
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
  Graph graph;
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

TEST(BuilderTest, StringLikeTags) {
  const char kA[] = "A";
  const std::string kB = "B";
  constexpr absl::string_view kC = "C";

  Graph graph;
  auto& foo = graph.AddNode("Foo");
  graph.In(kA).SetName("a") >> foo.In(kA);
  graph.In(kB).SetName("b") >> foo.In(kB);
  foo.Out(kC).SetName("c") >> graph.Out(kC);

  CalculatorGraphConfig expected =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "A:a"
        input_stream: "B:b"
        output_stream: "C:c"
        node {
          calculator: "Foo"
          input_stream: "A:a"
          input_stream: "B:b"
          output_stream: "C:c"
        }
      )pb");
  EXPECT_THAT(graph.GetConfig(), EqualsProto(expected));
}

TEST(BuilderTest, GraphIndexes) {
  Graph graph;
  auto& foo = graph.AddNode("Foo");
  graph.In(0).SetName("a") >> foo.In("")[0];
  graph.In(1).SetName("c") >> foo.In("")[2];
  graph.In(2).SetName("b") >> foo.In("")[1];
  foo.Out("")[0].SetName("x") >> graph.Out(1);
  foo.Out("")[1].SetName("y") >> graph.Out(0);

  CalculatorGraphConfig expected =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "a"
        input_stream: "c"
        input_stream: "b"
        output_stream: "y"
        output_stream: "x"
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

class AnyAndSameTypeCalculator : public NodeIntf {
 public:
  static constexpr Input<AnyType>::Optional kAnyTypeInput{"INPUT"};
  static constexpr Output<AnyType>::Optional kAnyTypeOutput{"ANY_OUTPUT"};
  static constexpr Output<SameType<kAnyTypeInput>>::Optional kSameTypeOutput{
      "SAME_OUTPUT"};
  static constexpr Output<SameType<kSameTypeOutput>> kRecursiveSameTypeOutput{
      "RECURSIVE_SAME_OUTPUT"};

  static constexpr Input<int>::Optional kIntInput{"INT_INPUT"};
  // `SameType` usage for this output is only for testing purposes.
  //
  // `SameType` is designed to work with inputs of `AnyType` and, normally, you
  // would not use `Output<SameType<kIntInput>>` in a real calculator. You
  // should write `Output<int>` instead, since the type is known.
  static constexpr Output<SameType<kIntInput>>::Optional kSameIntOutput{
      "SAME_INT_OUTPUT"};
  static constexpr Output<SameType<kSameIntOutput>> kRecursiveSameIntOutput{
      "RECURSIVE_SAME_INT_OUTPUT"};

  MEDIAPIPE_NODE_INTERFACE(AnyAndSameTypeCalculator, kAnyTypeInput,
                           kAnyTypeOutput, kSameTypeOutput);
};

TEST(BuilderTest, AnyAndSameTypeHandledProperly) {
  Graph graph;
  Source<AnyType> any_input = graph.In("GRAPH_ANY_INPUT");
  Source<int> int_input = graph.In("GRAPH_INT_INPUT").Cast<int>();

  auto& node = graph.AddNode("AnyAndSameTypeCalculator");
  any_input >> node[AnyAndSameTypeCalculator::kAnyTypeInput];
  int_input >> node[AnyAndSameTypeCalculator::kIntInput];

  Source<AnyType> any_type_output =
      node[AnyAndSameTypeCalculator::kAnyTypeOutput];
  any_type_output.SetName("any_type_output");

  Source<AnyType> same_type_output =
      node[AnyAndSameTypeCalculator::kSameTypeOutput];
  same_type_output.SetName("same_type_output");
  Source<AnyType> recursive_same_type_output =
      node[AnyAndSameTypeCalculator::kRecursiveSameTypeOutput];
  recursive_same_type_output.SetName("recursive_same_type_output");
  Source<int> same_int_output = node[AnyAndSameTypeCalculator::kSameIntOutput];
  same_int_output.SetName("same_int_output");
  Source<int> recursive_same_int_type_output =
      node[AnyAndSameTypeCalculator::kRecursiveSameIntOutput];
  recursive_same_int_type_output.SetName("recursive_same_int_type_output");

  CalculatorGraphConfig expected = mediapipe::ParseTextProtoOrDie<
      CalculatorGraphConfig>(R"pb(
    node {
      calculator: "AnyAndSameTypeCalculator"
      input_stream: "INPUT:__stream_0"
      input_stream: "INT_INPUT:__stream_1"
      output_stream: "ANY_OUTPUT:any_type_output"
      output_stream: "RECURSIVE_SAME_INT_OUTPUT:recursive_same_int_type_output"
      output_stream: "RECURSIVE_SAME_OUTPUT:recursive_same_type_output"
      output_stream: "SAME_INT_OUTPUT:same_int_output"
      output_stream: "SAME_OUTPUT:same_type_output"
    }
    input_stream: "GRAPH_ANY_INPUT:__stream_0"
    input_stream: "GRAPH_INT_INPUT:__stream_1"
  )pb");
  EXPECT_THAT(graph.GetConfig(), EqualsProto(expected));
}

TEST(BuilderTest, AnyTypeCanBeCast) {
  Graph graph;
  Source<std::string> any_input =
      graph.In("GRAPH_ANY_INPUT").Cast<std::string>();

  auto& node = graph.AddNode("AnyAndSameTypeCalculator");
  any_input >> node[AnyAndSameTypeCalculator::kAnyTypeInput];
  Source<double> any_type_output =
      node[AnyAndSameTypeCalculator::kAnyTypeOutput]
          .SetName("any_type_output")
          .Cast<double>();

  any_type_output >> graph.Out("GRAPH_ANY_OUTPUT").Cast<double>();

  CalculatorGraphConfig expected =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "AnyAndSameTypeCalculator"
          input_stream: "INPUT:__stream_0"
          output_stream: "ANY_OUTPUT:any_type_output"
        }
        input_stream: "GRAPH_ANY_INPUT:__stream_0"
        output_stream: "GRAPH_ANY_OUTPUT:any_type_output"
      )pb");
  EXPECT_THAT(graph.GetConfig(), EqualsProto(expected));
}

TEST(BuilderTest, MultiPortIsCastToMultiPort) {
  Graph graph;
  MultiSource<AnyType> any_input = graph.In("ANY_INPUT");
  MultiSource<int> int_input = any_input.Cast<int>();
  MultiDestination<AnyType> any_output = graph.Out("ANY_OUTPUT");
  MultiDestination<int> int_output = any_output.Cast<int>();
  int_input >> int_output;

  CalculatorGraphConfig expected =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "ANY_INPUT:__stream_0"
        output_stream: "ANY_OUTPUT:__stream_0"
      )pb");
  EXPECT_THAT(graph.GetConfig(), EqualsProto(expected));
}

TEST(BuilderTest, MultiPortCanBeSlicedToSinglePort) {
  Graph graph;
  MultiSource<AnyType> any_multi_input = graph.In("ANY_INPUT");
  Source<AnyType> any_input = any_multi_input;
  MultiDestination<AnyType> any_multi_output = graph.Out("ANY_OUTPUT");
  Destination<AnyType> any_output = any_multi_output;
  any_input >> any_output;

  CalculatorGraphConfig expected =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "ANY_INPUT:__stream_0"
        output_stream: "ANY_OUTPUT:__stream_0"
      )pb");
  EXPECT_THAT(graph.GetConfig(), EqualsProto(expected));
}

TEST(BuilderTest, SinglePortAccessWorksThroughSlicing) {
  Graph graph;
  Source<int> int_input = graph.In("INT_INPUT").Cast<int>();
  Source<AnyType> any_input = graph.In("ANY_OUTPUT");
  Destination<int> int_output = graph.Out("INT_OUTPUT").Cast<int>();
  Destination<AnyType> any_output = graph.Out("ANY_OUTPUT");
  int_input >> int_output;
  any_input >> any_output;

  CalculatorGraphConfig expected =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "ANY_OUTPUT:__stream_0"
        input_stream: "INT_INPUT:__stream_1"
        output_stream: "ANY_OUTPUT:__stream_0"
        output_stream: "INT_OUTPUT:__stream_1"
      )pb");
  EXPECT_THAT(graph.GetConfig(), EqualsProto(expected));
}

}  // namespace
}  // namespace mediapipe::api2::builder
