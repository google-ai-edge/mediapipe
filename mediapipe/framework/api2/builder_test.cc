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
#include "mediapipe/framework/stream_handler/fixed_size_input_stream_handler.pb.h"
#include "mediapipe/framework/testdata/night_light_calculator.pb.h"
#include "mediapipe/framework/testdata/sky_light_calculator.pb.h"

namespace mediapipe::api2::builder {
namespace {

using ::mediapipe::api2::test::Bar;
using ::mediapipe::api2::test::FloatAdder;
using ::mediapipe::api2::test::Foo;
using ::mediapipe::api2::test::Foo2;
using ::mediapipe::api2::test::FooBar1;

TEST(BuilderTest, BuildGraph) {
  Graph graph;
  // Graph inputs.
  Stream<AnyType> base = graph.In("IN").SetName("base");
  SidePacket<AnyType> side = graph.SideIn("SIDE").SetName("side");

  auto& foo = graph.AddNode("Foo");
  base >> foo.In("BASE");
  side >> foo.SideIn("SIDE");
  Stream<AnyType> foo_out = foo.Out("OUT");

  auto& bar = graph.AddNode("Bar");
  foo_out >> bar.In("IN");
  Stream<AnyType> bar_out = bar.Out("OUT");

  // Graph outputs.
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

TEST(BuilderTest, BuildGraphDefiningAndSettingExecutors) {
  Graph graph;

  // Graph inputs.
  Stream<AnyType> base = graph.In("IN").SetName("base");
  SidePacket<AnyType> side = graph.SideIn("SIDE").SetName("side");

  // Executors
  auto& executor0 = graph.AddExecutor("ThreadPoolExecutor");

  auto& executor1 = graph.AddExecutor("ThreadPoolExecutor");
  auto& executor1_opts = executor1.GetOptions<ThreadPoolExecutorOptions>();
  executor1_opts.set_num_threads(42);

  // Nodes
  auto& foo1 = graph.AddNode("Foo");
  foo1.SetExecutor(executor0);
  base >> foo1.In("BASE");
  side >> foo1.SideIn("SIDE");
  Stream<AnyType> foo1_out = foo1.Out("OUT");

  auto& foo2 = graph.AddNode("Foo");
  foo2.SetExecutor(executor1);
  base >> foo2.In("BASE");
  side >> foo2.SideIn("SIDE");
  Stream<AnyType> foo2_out = foo2.Out("OUT");

  auto& bar1 = graph.AddNode("Bar");
  bar1.SetExecutor(executor0);
  foo1_out >> bar1.In("IN");
  Stream<AnyType> bar1_out = bar1.Out("OUT");

  auto& bar2 = graph.AddNode("Bar");
  bar2.SetExecutor(executor1);
  foo2_out >> bar2.In("IN");
  Stream<AnyType> bar2_out = bar2.Out("OUT");

  // Graph outputs.
  bar1_out.SetName("out1") >> graph.Out("OUT")[0];
  bar2_out.SetName("out2") >> graph.Out("OUT")[1];

  CalculatorGraphConfig expected =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_side_packet: "SIDE:side"
        input_stream: "IN:base"
        output_stream: "OUT:0:out1"
        output_stream: "OUT:1:out2"

        executor { name: "_b_executor_0" type: "ThreadPoolExecutor" }
        executor {
          name: "_b_executor_1"
          type: "ThreadPoolExecutor"
          options {
            [mediapipe.ThreadPoolExecutorOptions.ext] { num_threads: 42 }
          }
        }

        node {
          calculator: "Foo"
          input_stream: "BASE:base"
          output_stream: "OUT:__stream_0"
          input_side_packet: "SIDE:side"
          executor: "_b_executor_0"
        }
        node {
          calculator: "Foo"
          input_stream: "BASE:base"
          output_stream: "OUT:__stream_1"
          input_side_packet: "SIDE:side"
          executor: "_b_executor_1"
        }
        node {
          calculator: "Bar"
          input_stream: "IN:__stream_0"
          output_stream: "OUT:out1"
          executor: "_b_executor_0"
        }
        node {
          calculator: "Bar"
          input_stream: "IN:__stream_1"
          output_stream: "OUT:out2"
          executor: "_b_executor_1"
        }
      )pb");
  EXPECT_THAT(graph.GetConfig(), EqualsProto(expected));
}

TEST(BuilderTest, BuildGraphSettingInputAndOutputStreamHandlers) {
  Graph graph;
  // Graph inputs.
  Stream<AnyType> base = graph.In("IN").SetName("base");
  SidePacket<AnyType> side = graph.SideIn("SIDE").SetName("side");

  auto& foo = graph.AddNode("Foo");
  auto& foo_ish_opts =
      foo.SetInputStreamHandler("FixedSizeInputStreamHandler")
          .GetOptions<mediapipe::FixedSizeInputStreamHandlerOptions>();
  foo_ish_opts.set_target_queue_size(2);
  foo_ish_opts.set_trigger_queue_size(3);
  foo_ish_opts.set_fixed_min_size(true);
  base >> foo.In("BASE");
  side >> foo.SideIn("SIDE");
  Stream<AnyType> foo_out = foo.Out("OUT");

  auto& bar = graph.AddNode("Bar");
  bar.SetInputStreamHandler("ImmediateInputStreamHandler");
  bar.SetOutputStreamHandler("InOrderOutputStreamHandler");
  foo_out >> bar.In("IN");
  Stream<AnyType> bar_out = bar.Out("OUT");

  // Graph outputs.
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
          input_stream_handler {
            input_stream_handler: "FixedSizeInputStreamHandler"
            options {
              [mediapipe.FixedSizeInputStreamHandlerOptions.ext] {
                trigger_queue_size: 3
                target_queue_size: 2
                fixed_min_size: true
              }
            }
          }
        }
        node {
          calculator: "Bar"
          input_stream: "IN:__stream_0"
          output_stream: "OUT:out"
          input_stream_handler {
            input_stream_handler: "ImmediateInputStreamHandler"
          }
          output_stream_handler {
            output_stream_handler: "InOrderOutputStreamHandler"
          }
        }
      )pb");
  EXPECT_THAT(graph.GetConfig(), EqualsProto(expected));
}

TEST(BuilderTest, CopyableStream) {
  Graph graph;
  Stream<int> a = graph.In("A").SetName("a").Cast<int>();
  Stream<int> b = graph.In("B").SetName("b").Cast<int>();
  SidePacket<float> side_a =
      graph.SideIn("SIDE_A").SetName("side_a").Cast<float>();
  SidePacket<float> side_b =
      graph.SideIn("SIDE_B").SetName("side_b").Cast<float>();
  Destination<int> out = graph.Out("OUT").Cast<int>();
  SideDestination<float> side_out = graph.SideOut("SIDE_OUT").Cast<float>();

  Stream<int> input = a;
  input = b;
  SidePacket<float> side_input = side_b;
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

  // Graph inputs.
  Stream<int> base = graph.In("IN").SetName("base").Cast<int>();
  SidePacket<float> side = graph.SideIn("SIDE").SetName("side").Cast<float>();

  auto foo_fn = [](Stream<int> base, SidePacket<float> side, Graph& graph) {
    auto& foo = graph.AddNode("Foo");
    base >> foo.In("BASE");
    side >> foo.SideIn("SIDE");
    return foo.Out("OUT")[0].Cast<double>();
  };
  Stream<double> foo_out = foo_fn(base, side, graph);

  auto bar_fn = [](Stream<double> in, Graph& graph) {
    auto& bar = graph.AddNode("Bar");
    in >> bar.In("IN");
    return bar.Out("OUT")[0].Cast<double>();
  };
  Stream<double> bar_out = bar_fn(foo_out, graph);

  // Graph outputs.
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
  // Graph inputs.
  Stream<AnyType> base = graph.In("IN").SetName("base");
  SidePacket<AnyType> side = graph.SideIn("SIDE").SetName("side");

  auto& foo = graph.AddNode<FooT>();
  base >> foo.In(MPP_TAG("BASE"));
  side >> foo.SideIn(MPP_TAG("BIAS"));
  Stream<float> foo_out = foo.Out(MPP_TAG("OUT"));

  auto& bar = graph.AddNode<Bar>();
  foo_out >> bar.In(MPP_TAG("IN"));
  Stream<AnyType> bar_out = bar.Out(MPP_TAG("OUT"));

  // Graph outputs.
  bar_out.SetName("out") >> graph.Out("OUT");

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
  // Graph inputs.
  Stream<AnyType> base = graph.In("IN").SetName("base");

  auto& foo = graph.AddNode("Foo");
  base >> foo.In("BASE");
  Stream<AnyType> foo_out = foo.Out("OUT");

  auto& adder = graph.AddNode("FloatAdder");
  foo_out >> adder.In("IN")[0];
  foo_out >> adder.In("IN")[1];
  Stream<AnyType> out = adder.Out("OUT");

  // Graph outputs.
  out.SetName("out") >> graph.Out("OUT");

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
  // Graph inputs.
  Stream<AnyType> base = graph.In("IN").SetName("base");

  auto& foo = graph.AddNode<Foo>();
  base >> foo.In(MPP_TAG("BASE"));
  Stream<float> foo_out = foo.Out(MPP_TAG("OUT"));

  auto& adder = graph.AddNode<FloatAdder>();
  foo_out >> adder.In(MPP_TAG("IN"))[0];
  foo_out >> adder.In(MPP_TAG("IN"))[1];
  Stream<float> out = adder.Out(MPP_TAG("OUT"));

  // Graph outputs.
  out.SetName("out") >> graph.Out("OUT");

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
  // Graph inputs.
  Stream<int> base = graph.In(FooBar1::kIn).SetName("base");

  auto& foo = graph.AddNode<test::Foo>();
  base >> foo[Foo::kBase];
  Stream<float> foo_out = foo[Foo::kOut];

  auto& adder = graph.AddNode<FloatAdder>();
  foo_out >> adder[FloatAdder::kIn][0];
  foo_out >> adder[FloatAdder::kIn][1];
  Stream<float> out = adder[FloatAdder::kOut];

  // Graph outputs.
  out.SetName("out") >> graph.Out(FooBar1::kOut);

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
  // Graph inputs.
  SidePacket<AnyType> side_in = graph.SideIn("IN");

  auto& generator = graph.AddPacketGenerator("FloatGenerator");
  side_in >> generator.SideIn("IN");
  SidePacket<AnyType> side_out = generator.SideOut("OUT");

  // Graph outputs.
  side_out >> graph.SideOut("OUT");

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
  // Graph inputs.
  Stream<AnyType> a = graph.In("A").SetName("a");
  Stream<AnyType> c = graph.In("C").SetName("c");
  Stream<AnyType> b = graph.In("B").SetName("b");

  auto& foo = graph.AddNode("Foo");
  a >> foo.In("")[0];
  c >> foo.In("")[2];
  b >> foo.In("")[1];
  Stream<AnyType> x = foo.Out("")[0];
  Stream<AnyType> y = foo.Out("")[1];

  // Graph outputs.
  x.SetName("x") >> graph.Out("ONE");
  y.SetName("y") >> graph.Out("TWO");

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
  // Graph inputs.
  Stream<AnyType> a = graph.In(kA).SetName("a");
  Stream<AnyType> b = graph.In(kB).SetName("b");

  auto& foo = graph.AddNode("Foo");
  a >> foo.In(kA);
  b >> foo.In(kB);
  Stream<AnyType> c = foo.Out(kC);

  // Graph outputs.
  c.SetName("c") >> graph.Out(kC);

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
  // Graph inputs.
  Stream<AnyType> a = graph.In(0).SetName("a");
  Stream<AnyType> c = graph.In(1).SetName("c");
  Stream<AnyType> b = graph.In(2).SetName("b");

  auto& foo = graph.AddNode("Foo");
  a >> foo.In("")[0];
  c >> foo.In("")[2];
  b >> foo.In("")[1];
  Stream<AnyType> x = foo.Out("")[0];
  Stream<AnyType> y = foo.Out("")[1];

  // Graph outputs.
  x.SetName("x") >> graph.Out(1);
  y.SetName("y") >> graph.Out(0);

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
  Stream<AnyType> any_input = graph.In("GRAPH_ANY_INPUT");
  Stream<int> int_input = graph.In("GRAPH_INT_INPUT").Cast<int>();

  auto& node = graph.AddNode("AnyAndSameTypeCalculator");
  any_input >> node[AnyAndSameTypeCalculator::kAnyTypeInput];
  int_input >> node[AnyAndSameTypeCalculator::kIntInput];
  Stream<AnyType> any_type_output =
      node[AnyAndSameTypeCalculator::kAnyTypeOutput];
  Stream<AnyType> same_type_output =
      node[AnyAndSameTypeCalculator::kSameTypeOutput];
  Stream<AnyType> recursive_same_type_output =
      node[AnyAndSameTypeCalculator::kRecursiveSameTypeOutput];
  Stream<int> same_int_output = node[AnyAndSameTypeCalculator::kSameIntOutput];
  Stream<int> recursive_same_int_type_output =
      node[AnyAndSameTypeCalculator::kRecursiveSameIntOutput];

  any_type_output.SetName("any_type_output");
  same_type_output.SetName("same_type_output");
  recursive_same_type_output.SetName("recursive_same_type_output");
  same_int_output.SetName("same_int_output");
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
  Stream<std::string> any_input =
      graph.In("GRAPH_ANY_INPUT").Cast<std::string>();

  auto& node = graph.AddNode("AnyAndSameTypeCalculator");
  any_input >> node[AnyAndSameTypeCalculator::kAnyTypeInput];
  Stream<double> any_type_output =
      node[AnyAndSameTypeCalculator::kAnyTypeOutput].Cast<double>();

  any_type_output.SetName("any_type_output") >>
      graph.Out("GRAPH_ANY_OUTPUT").Cast<double>();

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
  Stream<AnyType> any_input = any_multi_input;
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
  Stream<int> int_input = graph.In("INT_INPUT").Cast<int>();
  Stream<AnyType> any_input = graph.In("ANY_OUTPUT");
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

TEST(BuilderTest, TestStreamEqualsNotEqualsOperators) {
  Graph graph;
  Stream<AnyType> input0 = graph.In(0);
  EXPECT_TRUE(input0 == input0);
  EXPECT_FALSE(input0 != input0);

  EXPECT_TRUE(input0 == input0.Cast<int>());
  EXPECT_FALSE(input0.Cast<float>() != input0);

  EXPECT_TRUE(input0.Cast<float>() == input0.Cast<int>());
  EXPECT_FALSE(input0.Cast<float>() != input0.Cast<int>());

  Stream<AnyType> input1 = graph.In(1);
  EXPECT_FALSE(input0 == input1);
  EXPECT_TRUE(input0 != input1);

  input1 = input0;
  EXPECT_TRUE(input0 == input1);
  EXPECT_FALSE(input0 != input1);
  EXPECT_TRUE(input0.Cast<int>() == input1.Cast<int>());
  EXPECT_FALSE(input0.Cast<float>() != input1.Cast<float>());
}

TEST(BuilderTest, TestSidePacketEqualsNotEqualsOperators) {
  Graph graph;
  SidePacket<AnyType> side_input0 = graph.SideIn(0);
  EXPECT_TRUE(side_input0 == side_input0);
  EXPECT_FALSE(side_input0 != side_input0);

  EXPECT_TRUE(side_input0 == side_input0.Cast<int>());
  EXPECT_FALSE(side_input0.Cast<float>() != side_input0);

  EXPECT_TRUE(side_input0.Cast<float>() == side_input0.Cast<int>());
  EXPECT_FALSE(side_input0.Cast<float>() != side_input0.Cast<int>());

  SidePacket<AnyType> side_input1 = graph.SideIn(1);
  EXPECT_FALSE(side_input0 == side_input1);
  EXPECT_TRUE(side_input0 != side_input1);

  side_input1 = side_input0;
  EXPECT_TRUE(side_input0 == side_input1);
  EXPECT_FALSE(side_input0 != side_input1);
  EXPECT_TRUE(side_input0.Cast<int>() == side_input1.Cast<int>());
  EXPECT_FALSE(side_input0.Cast<float>() != side_input1.Cast<float>());
}

TEST(GetOptionsTest, AddProto3Options) {
  Graph graph;
  // Graph inputs.
  Stream<AnyType> base = graph.In("IN").SetName("base");
  SidePacket<AnyType> side = graph.SideIn("SIDE").SetName("side");

  auto& foo = graph.AddNode("Foo");
  foo.GetOptions<mediapipe::SkyLightCalculatorOptions>();
  base >> foo.In("BASE");
  side >> foo.SideIn("SIDE");
  Stream<AnyType> foo_out = foo.Out("OUT");

  auto& bar = graph.AddNode("Bar");
  foo_out >> bar.In("IN");
  Stream<AnyType> bar_out = bar.Out("OUT");

  // Graph outputs.
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
          node_options {
            [type.googleapis.com/mediapipe.SkyLightCalculatorOptions] {}
          }
        }
        node {
          calculator: "Bar"
          input_stream: "IN:__stream_0"
          output_stream: "OUT:out"
        }
      )pb");
  EXPECT_THAT(graph.GetConfig(), EqualsProto(expected));
}

TEST(GetOptionsTest, AddProto2Options) {
  Graph graph;
  // Graph inputs.
  Stream<AnyType> base = graph.In("IN").SetName("base");
  SidePacket<AnyType> side = graph.SideIn("SIDE").SetName("side");

  auto& foo = graph.AddNode("Foo");
  foo.GetOptions<mediapipe::NightLightCalculatorOptions>();
  base >> foo.In("BASE");
  side >> foo.SideIn("SIDE");
  Stream<AnyType> foo_out = foo.Out("OUT");

  auto& bar = graph.AddNode("Bar");
  foo_out >> bar.In("IN");
  Stream<AnyType> bar_out = bar.Out("OUT");

  // Graph outputs.
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
          options {
            [mediapipe.NightLightCalculatorOptions.ext] {}
          }
        }
        node {
          calculator: "Bar"
          input_stream: "IN:__stream_0"
          output_stream: "OUT:out"
        }
      )pb");
  EXPECT_THAT(graph.GetConfig(), EqualsProto(expected));
}

TEST(GetOptionsTest, AddBothProto23Options) {
  Graph graph;
  // Graph inputs.
  Stream<AnyType> base = graph.In("IN").SetName("base");
  SidePacket<AnyType> side = graph.SideIn("SIDE").SetName("side");

  auto& foo = graph.AddNode("Foo");
  foo.GetOptions<mediapipe::SkyLightCalculatorOptions>();
  foo.GetOptions<mediapipe::NightLightCalculatorOptions>();
  base >> foo.In("BASE");
  side >> foo.SideIn("SIDE");
  Stream<AnyType> foo_out = foo.Out("OUT");

  auto& bar = graph.AddNode("Bar");
  foo_out >> bar.In("IN");
  Stream<AnyType> bar_out = bar.Out("OUT");

  // Graph outputs.
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
          options {
            [mediapipe.NightLightCalculatorOptions.ext] {}
          }
          node_options {
            [type.googleapis.com/mediapipe.SkyLightCalculatorOptions] {}
          }
        }
        node {
          calculator: "Bar"
          input_stream: "IN:__stream_0"
          output_stream: "OUT:out"
        }
      )pb");
  EXPECT_THAT(graph.GetConfig(), EqualsProto(expected));
}

// Checks casting logic.
TEST(CastTest, FromAnyToAny) {
  Graph graph;
  Stream<int> int_inp = graph.In("A").SetName("a").Cast<int>();
  Stream<AnyType> any_inp = graph.In("B").SetName("b");

  [[maybe_unused]] Stream<AnyType> any_dest = int_inp.Cast<AnyType>();
  [[maybe_unused]] Stream<int> int_dest = any_inp.Cast<int>();
}

}  // namespace
}  // namespace mediapipe::api2::builder
