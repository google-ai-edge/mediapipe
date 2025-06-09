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

#include "mediapipe/framework/api3/graph.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/api3/side_packet.h"
#include "mediapipe/framework/api3/stream.h"
#include "mediapipe/framework/api3/testing/generator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/stream_handler/fixed_size_input_stream_handler.pb.h"
#include "mediapipe/framework/testdata/night_light_calculator.pb.h"
#include "mediapipe/framework/testdata/sky_light_calculator.pb.h"

namespace mediapipe::api3 {
namespace {

struct Image {};
struct Tensor {};

constexpr absl::string_view kFooNodeName = "Foo";
struct FooNode : Node<kFooNodeName> {
  template <typename S>
  struct Contract {
    Input<S, Image> base{"BASE"};
    SideInput<S, float> side{"SIDE"};
    Output<S, Tensor> out{"OUT"};
  };
};

constexpr absl::string_view kBarNodeName = "Bar";
struct BarNode : Node<kBarNodeName> {
  template <typename S>
  struct Contract {
    Input<S, Tensor> in{"IN"};
    Output<S, Image> out{"OUT"};
  };
};

template <typename S>
struct FooBar {
  Input<S, Image> in{"IN"};
  SideInput<S, float> side{"SIDE"};
  Output<S, Image> out{"OUT"};
};

TEST(GenericGraphTest, CanBuildGenericGraph) {
  Graph<FooBar> graph;

  // Graph inputs.
  Stream<Image> base = graph.in.Get().SetName("base");
  SidePacket<float> side = graph.side.Get().SetName("side");

  // Graph body.
  auto& foo = graph.AddNode<FooNode>();
  foo.base.Set(base);
  foo.side.Set(side);
  Stream<Tensor> foo_out = foo.out.Get();

  auto& bar = graph.AddNode<BarNode>();
  bar.in.Set(foo_out);
  Stream<Image> bar_out = bar.out.Get();

  // Graph outputs.
  graph.out.Set(bar_out.SetName("out"));

  CalculatorGraphConfig expected_config =
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
  MP_ASSERT_OK_AND_ASSIGN(CalculatorGraphConfig config, graph.GetConfig());
  EXPECT_THAT(config, EqualsProto(expected_config));
}

template <typename S>
struct FooBarRepeatedOut {
  Input<S, Image> in{"IN"};
  SideInput<S, float> side{"SIDE"};
  Repeated<Output<S, Image>> out{"OUT"};
};

TEST(GenericGraphTest, CanBuildGraphDefiningAndSettingExecutors) {
  Graph<FooBarRepeatedOut> graph;

  // Inputs.
  Stream<Image> base = graph.in.Get().SetName("base");
  SidePacket<float> side = graph.side.Get().SetName("side");

  // Executors.
  auto& executor0 = graph.AddLegacyExecutor("ThreadPoolExecutor");

  auto& executor1 = graph.AddLegacyExecutor("ThreadPoolExecutor");
  auto& executor1_opts = executor1.GetOptions<ThreadPoolExecutorOptions>();
  executor1_opts.set_num_threads(42);

  // Nodes.
  auto& foo1 = graph.AddNode<FooNode>();
  foo1.SetLegacyExecutor(executor0);
  foo1.base.Set(base);
  foo1.side.Set(side);
  Stream<Tensor> foo1_out = foo1.out.Get();

  auto& foo2 = graph.AddNode<FooNode>();
  foo2.SetLegacyExecutor(executor1);
  foo2.base.Set(base);
  foo2.side.Set(side);
  Stream<Tensor> foo2_out = foo2.out.Get();

  auto& bar1 = graph.AddNode<BarNode>();
  bar1.SetLegacyExecutor(executor0);
  bar1.in.Set(foo1_out);
  Stream<Image> bar1_out = bar1.out.Get();

  auto& bar2 = graph.AddNode<BarNode>();
  bar2.SetLegacyExecutor(executor1);
  bar2.in.Set(foo2_out);
  auto bar2_out = bar2.out.Get();

  // Graph outputs.
  graph.out.Add(bar1_out.SetName("out1"));
  graph.out.Add(bar2_out.SetName("out2"));

  CalculatorGraphConfig expected_config =
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
  MP_ASSERT_OK_AND_ASSIGN(CalculatorGraphConfig config, graph.GetConfig());
  EXPECT_THAT(config, EqualsProto(expected_config));
}

TEST(BuilderTest, BuildGraphSettingInputAndOutputStreamHandlers) {
  Graph<FooBar> graph;

  // Graph inputs.
  Stream<Image> base = graph.in.Get().SetName("base");
  SidePacket<float> side = graph.side.Get().SetName("side");

  auto& foo = graph.AddNode<FooNode>();
  auto& foo_ish_opts =
      foo.SetLegacyInputStreamHandler("FixedSizeInputStreamHandler")
          .GetOptions<mediapipe::FixedSizeInputStreamHandlerOptions>();
  foo_ish_opts.set_target_queue_size(2);
  foo_ish_opts.set_trigger_queue_size(3);
  foo_ish_opts.set_fixed_min_size(true);
  foo.base.Set(base);
  foo.side.Set(side);
  Stream<Tensor> foo_out = foo.out.Get();

  auto& bar = graph.AddNode<BarNode>();
  bar.SetLegacyInputStreamHandler("ImmediateInputStreamHandler");
  bar.SetLegacyOutputStreamHandler("InOrderOutputStreamHandler");
  bar.in.Set(foo_out);
  Stream<Image> bar_out = bar.out.Get();

  // Graph outputs.
  graph.out.Set(bar_out.SetName("out"));

  CalculatorGraphConfig expected_config =
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
  MP_ASSERT_OK_AND_ASSIGN(CalculatorGraphConfig config, graph.GetConfig());
  EXPECT_THAT(config, EqualsProto(expected_config));
}

TEST(GenericGraphTest, BuildGraphSettingSourceLayer) {
  Graph<FooBar> graph;
  // Graph inputs.
  Stream<Image> base = graph.in.Get().SetName("base");
  SidePacket<float> side = graph.side.Get().SetName("side");

  auto& foo = graph.AddNode<FooNode>();
  foo.SetSourceLayer(0);
  foo.base.Set(base);
  foo.side.Set(side);
  Stream<Tensor> foo_out = foo.out.Get();

  auto& bar = graph.AddNode<BarNode>();
  bar.SetSourceLayer(1);
  bar.in.Set(foo_out);
  Stream<Image> bar_out = bar.out.Get();

  // Graph outputs.
  graph.out.Set(bar_out.SetName("out"));

  CalculatorGraphConfig expected_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "IN:base"
        input_side_packet: "SIDE:side"
        output_stream: "OUT:out"
        node {
          calculator: "Foo"
          input_stream: "BASE:base"
          input_side_packet: "SIDE:side"
          output_stream: "OUT:__stream_0"
          source_layer: 0
        }
        node {
          calculator: "Bar"
          input_stream: "IN:__stream_0"
          output_stream: "OUT:out"
          source_layer: 1
        }
      )pb");
  MP_ASSERT_OK_AND_ASSIGN(CalculatorGraphConfig config, graph.GetConfig());
  EXPECT_THAT(config, EqualsProto(expected_config));
}

struct Detection {};

constexpr absl::string_view kPreviousLoopbackNodeName =
    "PreviousLoopbackCalculator";
struct PreviousLoopbackNode : Node<kPreviousLoopbackNodeName> {
  template <typename S>
  struct Contract {
    Input<S, Image> main{"MAIN"};
    Input<S, std::vector<Detection>> loop{"LOOP"};
    Output<S, std::vector<Detection>> prev_loop{"PREV_LOOP"};
  };
};

constexpr absl::string_view kObjectDetectionNodeName =
    "ObjectDetectionCalculator";
struct ObjectDetectionNode : Node<kObjectDetectionNodeName> {
  template <typename S>
  struct Contract {
    Input<S, Image> image{"IMAGE"};
    Input<S, std::vector<Detection>> prev_detections{"PREV_DETECTIONS"};
    Output<S, std::vector<Detection>> detections{"DETECTIONS"};
  };
};

template <typename S>
struct ObjectDetection {
  Input<S, Image> image{"IMAGE"};
  Output<S, std::vector<Detection>> out{"OUT"};
};

TEST(GenericGraphTest, CanUseBackEdges) {
  Graph<ObjectDetection> graph;
  // Graph inputs.
  auto image = graph.image.Get().SetName("image");

  // Nodes.
  auto [prev_detections, set_prev_detections_fn] = [&]() {
    auto* loopback_node = &graph.AddNode<PreviousLoopbackNode>();
    loopback_node->main.Set(image);
    auto set_loop_fn = [loopback_node](auto value) {
      loopback_node->loop.Set(value, /*back_edge=*/true);
    };
    auto prev_loop = loopback_node->prev_loop.Get();
    return std::pair(prev_loop, set_loop_fn);
  }();

  Stream<std::vector<Detection>> detections = [&]() {
    auto& detection_node = graph.AddNode<ObjectDetectionNode>();
    detection_node.image.Set(image);
    detection_node.prev_detections.Set(prev_detections);
    return detection_node.detections.Get();
  }();

  set_prev_detections_fn(detections);

  // Graph outputs.
  graph.out.Set(detections.SetName("detections"));

  CalculatorGraphConfig expected_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "PreviousLoopbackCalculator"
          input_stream: "LOOP:detections"
          input_stream: "MAIN:image"
          output_stream: "PREV_LOOP:__stream_0"
          input_stream_info { tag_index: "LOOP" back_edge: true }
        }
        node {
          calculator: "ObjectDetectionCalculator"
          input_stream: "IMAGE:image"
          input_stream: "PREV_DETECTIONS:__stream_0"
          output_stream: "DETECTIONS:detections"
        }
        input_stream: "IMAGE:image"
        output_stream: "OUT:detections"
      )pb");
  MP_ASSERT_OK_AND_ASSIGN(CalculatorGraphConfig config, graph.GetConfig());
  EXPECT_THAT(config, EqualsProto(expected_config));
}

struct Data {};

constexpr absl::string_view kSomeBackEdgeNodeName = "SomeBackEdgeCalculator";
struct SomeBackEdgeNode : Node<kSomeBackEdgeNodeName> {
  template <typename S>
  struct Contract {
    Repeated<Input<S, Data>> data{"DATA"};
    Output<S, Data> processed_data{"PROCESSED_DATA"};
  };
};

constexpr absl::string_view kSomeOutputDataNodeName =
    "SomeOutputDataCalculator";
struct SomeOutputDataNode : Node<kSomeOutputDataNodeName> {
  template <typename S>
  struct Contract {
    Input<S, Data> data{"DATA"};
    Input<S, Data> processed_data{"PROCESSED_DATA"};
    Output<S, Data> output_data{"OUTPUT_DATA"};
  };
};

template <typename S>
struct DataProcessing {
  Input<S, Data> in{"IN"};
  Output<S, Data> out{"OUT"};
};

TEST(GenericGraphTest, CanUseBackEdgesWithRepeated) {
  Graph<DataProcessing> graph;
  // Graph inputs.
  auto in_data = graph.in.Get().SetName("in_data");

  auto [processed_data, add_back_edge_fn] = [&]() {
    auto* back_edge_node = &graph.AddNode<SomeBackEdgeNode>();
    back_edge_node->data.Add(in_data);
    auto set_back_edge_fn = [back_edge_node](auto value) {
      back_edge_node->data.Add(value, /*back_edge=*/true);
    };
    auto processed_data = back_edge_node->processed_data.Get();
    return std::pair(processed_data, set_back_edge_fn);
  }();

  auto output_data = [&]() {
    auto& detection_node = graph.AddNode<SomeOutputDataNode>();
    detection_node.data.Set(in_data);
    detection_node.processed_data.Set(processed_data);
    return detection_node.output_data.Get();
  }();

  add_back_edge_fn(output_data);

  // Graph outputs.
  graph.out.Set(output_data.SetName("out_data"));

  CalculatorGraphConfig expected_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "SomeBackEdgeCalculator"
          input_stream: "DATA:0:in_data"
          input_stream: "DATA:1:out_data"
          output_stream: "PROCESSED_DATA:__stream_0"
          input_stream_info { tag_index: "DATA:1" back_edge: true }
        }
        node {
          calculator: "SomeOutputDataCalculator"
          input_stream: "DATA:in_data"
          input_stream: "PROCESSED_DATA:__stream_0"
          output_stream: "OUTPUT_DATA:out_data"
        }
        input_stream: "IN:in_data"
        output_stream: "OUT:out_data"
      )pb");
  MP_ASSERT_OK_AND_ASSIGN(CalculatorGraphConfig config, graph.GetConfig());
  EXPECT_THAT(config, EqualsProto(expected_config));
}

constexpr absl::string_view kSomeBackEdgeNoInputTagsNodeName =
    "SomeBackEdgeNoInputTagsCalculator";
struct SomeBackEdgeNoInputTagsNode : Node<kSomeBackEdgeNoInputTagsNodeName> {
  template <typename S>
  struct Contract {
    Repeated<Input<S, Data>> data{""};
    Output<S, Data> processed_data{"PROCESSED_DATA"};
  };
};

TEST(GenericGraphTest, CanUseBackEdgesWithRepeatedAndNoTag) {
  Graph<DataProcessing> graph;

  // Graph inputs.
  auto in_data = graph.in.Get().SetName("in_data");

  auto [processed_data, add_back_edge_fn] = [&]() {
    auto* back_edge_node = &graph.AddNode<SomeBackEdgeNoInputTagsNode>();
    back_edge_node->data.Add(in_data);
    auto add_back_edge_fn = [back_edge_node](auto value) {
      back_edge_node->data.Add(value, /*back_edge=*/true);
    };
    auto processed_data = back_edge_node->processed_data.Get();
    return std::pair(processed_data, add_back_edge_fn);
  }();

  Stream<Data> output_data = [&]() {
    auto& detection_node = graph.AddNode<SomeOutputDataNode>();
    detection_node.data.Set(in_data);
    detection_node.processed_data.Set(processed_data);
    return detection_node.output_data.Get();
  }();

  add_back_edge_fn(output_data);

  // Graph outputs.
  graph.out.Set(output_data.SetName("out_data"));

  CalculatorGraphConfig expected_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "SomeBackEdgeNoInputTagsCalculator"
          input_stream: "in_data"
          input_stream: "out_data"
          output_stream: "PROCESSED_DATA:__stream_0"
          input_stream_info { tag_index: ":1" back_edge: true }
        }
        node {
          calculator: "SomeOutputDataCalculator"
          input_stream: "DATA:in_data"
          input_stream: "PROCESSED_DATA:__stream_0"
          output_stream: "OUTPUT_DATA:out_data"
        }
        input_stream: "IN:in_data"
        output_stream: "OUT:out_data"
      )pb");
  MP_ASSERT_OK_AND_ASSIGN(CalculatorGraphConfig config, graph.GetConfig());
  EXPECT_THAT(config, EqualsProto(expected_config));
}

constexpr absl::string_view kFloatFooNodeName = "FloatFoo";
struct FloatFooNode : Node<kFloatFooNodeName> {
  template <typename S>
  struct Contract {
    Input<S, float> base{"BASE"};
    Output<S, float> out{"OUT"};
  };
};

constexpr absl::string_view kFloatAdderNodeName = "FloatAdder";
struct FloatAdderNode : Node<kFloatAdderNodeName> {
  template <typename S>
  struct Contract {
    Repeated<Input<S, float>> in{"IN"};
    Output<S, float> out{"OUT"};
  };
};

template <typename S>
struct FloatProcessing {
  Input<S, float> in{"IN"};
  Output<S, float> out{"OUT"};
};

TEST(GenericGraphTest, FanOut) {
  Graph<FloatProcessing> graph;
  // Graph inputs.
  auto base = graph.in.Get().SetName("base");

  auto& foo = graph.AddNode<FloatFooNode>();
  foo.base.Set(base);
  Stream<float> foo_out = foo.out.Get();

  auto& adder = graph.AddNode<FloatAdderNode>();
  adder.in.Add(foo_out);
  adder.in.Add(foo_out);
  Stream<float> out = adder.out.Get();

  // Graph outputs.
  graph.out.Set(out.SetName("out"));

  CalculatorGraphConfig expected_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "IN:base"
        output_stream: "OUT:out"
        node {
          calculator: "FloatFoo"
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
  MP_ASSERT_OK_AND_ASSIGN(CalculatorGraphConfig config, graph.GetConfig());
  EXPECT_THAT(config, EqualsProto(expected_config));
}

// Still using Node to describe the generator.
constexpr absl::string_view kFloatGeneratorName = "FloatGenerator";
struct FloatGenerator : Node<kFloatGeneratorName> {
  template <typename S>
  struct Contract {
    SideInput<S, float> side_in{"IN"};
    SideOutput<S, float> side_out{"OUT"};
    Options<S, mediapipe::GeneratorOptions> options;
  };
};

template <typename S>
struct FloatGeneration {
  SideInput<S, float> side_in{"IN"};
  SideOutput<S, float> side_out{"OUT"};
};

TEST(GenericGraphTest, CanAddLegacyPacketGenerator) {
  Graph<FloatGeneration> graph;

  // Graph inputs.
  SidePacket<float> side_in = graph.side_in.Get();

  auto& generator = graph.AddLegacyPacketGenerator<FloatGenerator>();
  generator.options.Mutable()->set_value(42);
  generator.side_in.Set(side_in);
  SidePacket<float> side_out = generator.side_out.Get();

  // Graph outputs.
  graph.side_out.Set(side_out);

  CalculatorGraphConfig expected_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_side_packet: "IN:__side_packet_0"
        output_side_packet: "OUT:__side_packet_1"
        packet_generator {
          packet_generator: "FloatGenerator"
          input_side_packet: "IN:__side_packet_0"
          output_side_packet: "OUT:__side_packet_1"
          options {
            [mediapipe.GeneratorOptions.ext] { value: 42 }
          }
        }
      )pb");
  MP_ASSERT_OK_AND_ASSIGN(CalculatorGraphConfig config, graph.GetConfig());
  EXPECT_THAT(config, EqualsProto(expected_config));
}

// Still using Node to describe the generator.
constexpr absl::string_view kRepeatedFloatGeneratorName =
    "RepeatedFloatGenerator";
struct RepeatedFloatGenerator : Node<kRepeatedFloatGeneratorName> {
  template <typename S>
  struct Contract {
    Repeated<SideInput<S, float>> side_in{"IN"};
    Repeated<SideOutput<S, float>> side_out{"OUT"};
  };
};

template <typename S>
struct RepeatedFloatGeneration {
  Repeated<SideInput<S, float>> side_in{"IN"};
  Repeated<SideOutput<S, float>> side_out{"OUT"};
};

TEST(GenericGraphTest, CanAddLegacyPacketGeneratorWithRepeatedFields) {
  Graph<RepeatedFloatGeneration> graph;

  // Graph inputs.
  SidePacket<float> side_0 = graph.side_in.Add();
  SidePacket<float> side_1 = graph.side_in.Add();

  auto& generator = graph.AddLegacyPacketGenerator<RepeatedFloatGenerator>();
  generator.side_in.Add(side_0);
  generator.side_in.Add(side_1);
  SidePacket<float> side_out0 = generator.side_out.Add();
  SidePacket<float> side_out1 = generator.side_out.Add();

  // Graph outputs.
  graph.side_out.Add(side_out0);
  graph.side_out.Add(side_out1);

  CalculatorGraphConfig expected_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_side_packet: "IN:0:__side_packet_0"
        input_side_packet: "IN:1:__side_packet_1"
        output_side_packet: "OUT:0:__side_packet_2"
        output_side_packet: "OUT:1:__side_packet_3"
        packet_generator {
          packet_generator: "RepeatedFloatGenerator"
          input_side_packet: "IN:0:__side_packet_0"
          input_side_packet: "IN:1:__side_packet_1"
          output_side_packet: "OUT:0:__side_packet_2"
          output_side_packet: "OUT:1:__side_packet_3"
        }
      )pb");
  MP_ASSERT_OK_AND_ASSIGN(CalculatorGraphConfig config, graph.GetConfig());
  EXPECT_THAT(config, EqualsProto(expected_config));
}

constexpr absl::string_view kFooEmptyTagsNodeName = "FooEmptyTags";
struct FooEmptyTagsNode : Node<kFooEmptyTagsNodeName> {
  template <typename S>
  struct Contract {
    Repeated<Input<S, int>> in{""};
    Repeated<Output<S, int>> out{""};
  };
};

template <typename S>
struct TestFooEmptyTags {
  Input<S, int> in_a{"A"};
  Input<S, int> in_b{"B"};
  Input<S, int> in_c{"C"};
  Output<S, int> out_one{"ONE"};
  Output<S, int> out_two{"TWO"};
};

TEST(GenericGraphTest, SupportsEmptyTags) {
  Graph<TestFooEmptyTags> graph;
  // Graph inputs.
  Stream<int> a = graph.in_a.Get().SetName("a");
  Stream<int> c = graph.in_c.Get().SetName("c");
  Stream<int> b = graph.in_b.Get().SetName("b");

  auto& foo = graph.AddNode<FooEmptyTagsNode>();
  foo.in.Add(a);
  foo.in.Add(b);
  foo.in.Add(c);
  Stream<int> x = foo.out.Add();
  Stream<int> y = foo.out.Add();

  // Graph outputs.
  graph.out_one.Set(x.SetName("x"));
  graph.out_two.Set(y.SetName("y"));

  CalculatorGraphConfig expected_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "A:a"
        input_stream: "B:b"
        input_stream: "C:c"
        output_stream: "ONE:x"
        output_stream: "TWO:y"
        node {
          calculator: "FooEmptyTags"
          input_stream: "a"
          input_stream: "b"
          input_stream: "c"
          output_stream: "x"
          output_stream: "y"
        }
      )pb");
  MP_ASSERT_OK_AND_ASSIGN(CalculatorGraphConfig config, graph.GetConfig());
  EXPECT_THAT(config, EqualsProto(expected_config));
}

constexpr absl::string_view kSkyLightProto3NodeName = "SkyLightProto3Node";
struct SkyLightProto3Node : Node<kSkyLightProto3NodeName> {
  template <typename S>
  struct Contract {
    Input<S, float> base{"BASE"};
    SideInput<S, float> side{"SIDE"};
    Output<S, float> out{"OUT"};

    Options<S, mediapipe::SkyLightCalculatorOptions> options;
  };
};

template <typename S>
struct OptionsProtoTest {
  Input<S, float> in{"IN"};
  SideInput<S, float> side{"SIDE"};
  Output<S, float> out{"OUT"};
};

TEST(GetOptionsTest, CanAddProto3Options) {
  Graph<OptionsProtoTest> graph;

  // Graph inputs.
  Stream<float> base = graph.in.Get().SetName("base");
  SidePacket<float> side = graph.side.Get().SetName("side");

  // Node.
  auto& foo = graph.AddNode<SkyLightProto3Node>();
  foo.options.Mutable()->set_sky_color("blue");
  foo.base.Set(base);
  foo.side.Set(side);
  Stream<float> foo_out = foo.out.Get();

  // Graph outputs.
  graph.out.Set(foo_out.SetName("out"));

  CalculatorGraphConfig expected_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "IN:base"
        input_side_packet: "SIDE:side"
        output_stream: "OUT:out"
        node {
          calculator: "SkyLightProto3Node"
          input_stream: "BASE:base"
          input_side_packet: "SIDE:side"
          output_stream: "OUT:out"
          node_options {
            [type.googleapis.com/mediapipe.SkyLightCalculatorOptions] {
              sky_color: "blue"
            }
          }
        }
      )pb");
  MP_ASSERT_OK_AND_ASSIGN(CalculatorGraphConfig config, graph.GetConfig());
  EXPECT_THAT(config, EqualsProto(expected_config));
}

constexpr absl::string_view kNightLightProto3NodeName = "NightLightProto2Node";
struct NightLightProto2Node : Node<kNightLightProto3NodeName> {
  template <typename S>
  struct Contract {
    Input<S, float> base{"BASE"};
    SideInput<S, float> side{"SIDE"};
    Output<S, float> out{"OUT"};

    Options<S, mediapipe::NightLightCalculatorOptions> options;
  };
};

TEST(GetOptionsTest, CanAddProto2Options) {
  Graph<OptionsProtoTest> graph;

  // Graph inputs.
  Stream<float> base = graph.in.Get().SetName("base");
  SidePacket<float> side = graph.side.Get().SetName("side");

  // Node.
  auto& foo = graph.AddNode<NightLightProto2Node>();
  foo.options.Mutable()->add_num_lights(1);
  foo.base.Set(base);
  foo.side.Set(side);
  Stream<float> foo_out = foo.out.Get();

  // Graph outputs.
  graph.out.Set(foo_out.SetName("out"));

  CalculatorGraphConfig expected_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "IN:base"
        input_side_packet: "SIDE:side"
        output_stream: "OUT:out"
        node {
          calculator: "NightLightProto2Node"
          input_stream: "BASE:base"
          input_side_packet: "SIDE:side"
          output_stream: "OUT:out"
          options {
            [mediapipe.NightLightCalculatorOptions.ext] { num_lights: 1 }
          }
        }
      )pb");
  MP_ASSERT_OK_AND_ASSIGN(CalculatorGraphConfig config, graph.GetConfig());
  EXPECT_THAT(config, EqualsProto(expected_config));
}

constexpr absl::string_view kProto2And3NodeName = "Proto2And3Node";
struct Proto2And3Node : Node<kProto2And3NodeName> {
  template <typename S>
  struct Contract {
    Input<S, float> base{"BASE"};
    SideInput<S, float> side{"SIDE"};
    Output<S, float> out{"OUT"};

    Options<S, mediapipe::SkyLightCalculatorOptions> proto3_options;
    Options<S, mediapipe::NightLightCalculatorOptions> proto2_options;
  };
};

TEST(GetOptionsTest, AddBothProto23Options) {
  Graph<OptionsProtoTest> graph;

  // Graph inputs.
  Stream<float> base = graph.in.Get().SetName("base");
  SidePacket<float> side = graph.side.Get().SetName("side");

  auto& foo = graph.AddNode<Proto2And3Node>();
  foo.proto2_options.Mutable()->add_num_lights(1);
  foo.proto3_options.Mutable()->set_sky_color("blue");
  foo.base.Set(base);
  foo.side.Set(side);
  Stream<float> foo_out = foo.out.Get();

  // Graph outputs.
  graph.out.Set(foo_out.SetName("out"));

  CalculatorGraphConfig expected_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "IN:base"
        input_side_packet: "SIDE:side"
        output_stream: "OUT:out"
        node {
          calculator: "Proto2And3Node"
          input_stream: "BASE:base"
          input_side_packet: "SIDE:side"
          output_stream: "OUT:out"
          options {
            [mediapipe.NightLightCalculatorOptions.ext] { num_lights: 1 }
          }
          node_options {
            [type.googleapis.com/mediapipe.SkyLightCalculatorOptions] {
              sky_color: "blue"
            }
          }
        }
      )pb");
  MP_ASSERT_OK_AND_ASSIGN(CalculatorGraphConfig config, graph.GetConfig());
  EXPECT_THAT(config, EqualsProto(expected_config));
}

struct Buffer {};

template <typename TensorT>
inline constexpr absl::string_view kImageToTensorName;
template <typename TensorT>
struct ImageToTensorNode : Node<kImageToTensorName<TensorT>> {
  template <typename S>
  struct Contract {
    Input<S, Image> image{"IMAGE"};
    Output<S, TensorT> tensor{"TENSOR"};
  };
};

template <>
inline constexpr absl::string_view kImageToTensorName<Buffer> =
    "ImageToTensorForBuffer";
template <>
inline constexpr absl::string_view kImageToTensorName<Tensor> =
    "ImageToTensorForTensor";

template <typename TensorT>
inline constexpr absl::string_view kInferenceNodeName;
template <typename TensorT>
struct InferenceNode : Node<kInferenceNodeName<TensorT>> {
  template <typename S>
  struct Contract {
    Repeated<Input<S, TensorT>> in_tensor{"REPEATED_TENSOR"};
    Repeated<Output<S, TensorT>> out_tensor{"REPEATED_TENSOR"};
  };
};

template <>
inline constexpr absl::string_view kInferenceNodeName<Buffer> =
    "InferenceForBuffer";
template <>
inline constexpr absl::string_view kInferenceNodeName<Tensor> =
    "InferenceForTensor";

template <typename TensorT>
inline constexpr absl::string_view kTensorToDetectionsName;
template <typename TensorT>
struct TensorToDetectionsNode : Node<kTensorToDetectionsName<TensorT>> {
  template <typename S>
  struct Contract {
    Input<S, TensorT> boxes_tensor{"BOXES"};
    Input<S, TensorT> scores_tensor{"SCORES"};
    Output<S, std::vector<Detection>> detections{"DETECTIONS"};
  };
};

template <>
inline constexpr absl::string_view kTensorToDetectionsName<Buffer> =
    "TensorToDetectionsForBuffer";
template <>
inline constexpr absl::string_view kTensorToDetectionsName<Tensor> =
    "TensorToDetectionsForTensor";

template <typename S>
struct FaceDetection {
  Input<S, Image> image{"IMAGE"};
  Output<S, std::vector<Detection>> detections{"DETECTIONS"};
};

TEST(GraphTest, CanAccessGraphInputsOutputs) {
  Graph<FaceDetection> graph;
  Stream<Image> in = graph.image.Get();

  Stream<Tensor> image_tensor = [&]() {
    auto& node = graph.AddNode<ImageToTensorNode<Tensor>>();
    node.image.Set(in);
    return node.tensor.Get();
  }();

  auto [boxes_tensor, scores_tensor] = [&]() {
    auto& node = graph.AddNode<InferenceNode<Tensor>>();
    node.in_tensor.Add(image_tensor);
    return std::pair(node.out_tensor.Add(), node.out_tensor.Add());
  }();

  Stream<std::vector<Detection>> detections = [&]() {
    auto& node = graph.AddNode<TensorToDetectionsNode<Tensor>>();
    node.boxes_tensor.Set(boxes_tensor);
    node.scores_tensor.Set(scores_tensor);
    return node.detections.Get();
  }();

  graph.detections.Set(detections);

  MP_ASSERT_OK_AND_ASSIGN(auto config, graph.GetConfig());
  EXPECT_THAT(
      config,
      EqualsProto(ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(R"pb(
        node {
          calculator: "ImageToTensorForTensor"
          input_stream: "IMAGE:__stream_0"
          output_stream: "TENSOR:__stream_1"
        }
        node {
          calculator: "InferenceForTensor"
          input_stream: "REPEATED_TENSOR:__stream_1"
          output_stream: "REPEATED_TENSOR:0:__stream_2"
          output_stream: "REPEATED_TENSOR:1:__stream_3"
        }
        node {
          calculator: "TensorToDetectionsForTensor"
          input_stream: "BOXES:__stream_2"
          input_stream: "SCORES:__stream_3"
          output_stream: "DETECTIONS:__stream_4"
        }
        input_stream: "IMAGE:__stream_0"
        output_stream: "DETECTIONS:__stream_4"
      )pb")));
}

TEST(GraphTest, CanAddNodesByContract) {
  Graph<FaceDetection> graph;
  Stream<Image> in = graph.image.Get();

  Stream<Tensor> image_tensor = [&]() {
    auto& node = graph.AddNodeByContract<ImageToTensorNode<Tensor>::Contract>(
        ImageToTensorNode<Tensor>::GetRegistrationName());
    node.image.Set(in);
    return node.tensor.Get();
  }();

  auto [boxes_tensor, scores_tensor] = [&]() {
    auto& node = graph.AddNodeByContract<InferenceNode<Tensor>::Contract>(
        InferenceNode<Tensor>::GetRegistrationName());
    node.in_tensor.Add(image_tensor);
    return std::pair(node.out_tensor.Add(), node.out_tensor.Add());
  }();

  Stream<std::vector<Detection>> detections = [&]() {
    auto& node =
        graph.AddNodeByContract<TensorToDetectionsNode<Tensor>::Contract>(
            TensorToDetectionsNode<Tensor>::GetRegistrationName());
    node.boxes_tensor.Set(boxes_tensor);
    node.scores_tensor.Set(scores_tensor);
    return node.detections.Get();
  }();

  graph.detections.Set(detections);

  MP_ASSERT_OK_AND_ASSIGN(auto config, graph.GetConfig());
  EXPECT_THAT(
      config,
      EqualsProto(ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(R"pb(
        node {
          calculator: "ImageToTensorForTensor"
          input_stream: "IMAGE:__stream_0"
          output_stream: "TENSOR:__stream_1"
        }
        node {
          calculator: "InferenceForTensor"
          input_stream: "REPEATED_TENSOR:__stream_1"
          output_stream: "REPEATED_TENSOR:0:__stream_2"
          output_stream: "REPEATED_TENSOR:1:__stream_3"
        }
        node {
          calculator: "TensorToDetectionsForTensor"
          input_stream: "BOXES:__stream_2"
          input_stream: "SCORES:__stream_3"
          output_stream: "DETECTIONS:__stream_4"
        }
        input_stream: "IMAGE:__stream_0"
        output_stream: "DETECTIONS:__stream_4"
      )pb")));
}

// Start of CanUseUtilityFunctions test case - users should be able to write
// utility functions that can be used across all "specialized" graphs. This is
// achieved by passing a specialized graph as a `GenericGraph`.

template <typename TensorT>
Stream<TensorT> ConvertImageToTensor(GenericGraph& graph, Stream<Image> image) {
  auto& node = graph.AddNode<ImageToTensorNode<TensorT>>();
  node.image.Set(image);
  return node.tensor.Get();
}

template <typename TensorT>
struct DetectionModelOutput {
  Stream<TensorT> boxes;
  Stream<TensorT> scores;
};

template <typename TensorT>
DetectionModelOutput<TensorT> RunDetectionInference(GenericGraph& graph,
                                                    Stream<TensorT> image) {
  auto& node = graph.AddNode<InferenceNode<TensorT>>();
  node.in_tensor.Add(image);
  return DetectionModelOutput<TensorT>{node.out_tensor.Add(),
                                       node.out_tensor.Add()};
}

template <typename TensorT>
Stream<std::vector<Detection>> ConvertTensorToDetections(
    GenericGraph& graph, Stream<TensorT> boxes, Stream<TensorT> scores) {
  auto& node = graph.AddNode<TensorToDetectionsNode<TensorT>>();
  node.boxes_tensor.Set(boxes);
  node.scores_tensor.Set(scores);
  return node.detections.Get();
}

TEST(GraphTest, CanUseUtilityFunctionsAndTemplateType) {
  Graph<FaceDetection> graph;
  Stream<Image> in = graph.image.Get();

  Stream<Buffer> image_tensor = ConvertImageToTensor<Buffer>(graph, in);

  auto [boxes_tensor, scores_tensor] =
      RunDetectionInference(graph, image_tensor);

  Stream<std::vector<Detection>> detections =
      ConvertTensorToDetections(graph, boxes_tensor, scores_tensor);

  graph.detections.Set(detections);

  MP_ASSERT_OK_AND_ASSIGN(auto config, graph.GetConfig());
  EXPECT_THAT(
      config,
      EqualsProto(ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(R"pb(
        node {
          calculator: "ImageToTensorForBuffer"
          input_stream: "IMAGE:__stream_0"
          output_stream: "TENSOR:__stream_1"
        }
        node {
          calculator: "InferenceForBuffer"
          input_stream: "REPEATED_TENSOR:__stream_1"
          output_stream: "REPEATED_TENSOR:0:__stream_2"
          output_stream: "REPEATED_TENSOR:1:__stream_3"
        }
        node {
          calculator: "TensorToDetectionsForBuffer"
          input_stream: "BOXES:__stream_2"
          input_stream: "SCORES:__stream_3"
          output_stream: "DETECTIONS:__stream_4"
        }
        input_stream: "IMAGE:__stream_0"
        output_stream: "DETECTIONS:__stream_4"
      )pb")));
}

template <typename S>
struct EveryFieldContract {
  Input<S, int> in{"IN"};
  Optional<Input<S, int>> optional_in{"OPTIONAL_IN"};
  Repeated<Input<S, int>> repeated_in{"REPEATED_IN"};

  SideInput<S, std::string> side_in{"SIDE_IN"};
  Optional<SideInput<S, std::string>> optional_side_in{"OPTIONAL_SIDE_IN"};
  Repeated<SideInput<S, std::string>> repeated_side_in{"REPEATED_SIDE_IN"};

  Output<S, int> out{"OUT"};
  Optional<Output<S, int>> optional_out{"OPTIONAL_OUT"};
  Repeated<Output<S, int>> repeated_out{"REPEATED_OUT"};

  SideOutput<S, std::string> side_out{"SIDE_OUT"};
  Optional<SideOutput<S, std::string>> optional_side_out{"OPTIONAL_SIDE_OUT"};
  Repeated<SideOutput<S, std::string>> repeated_side_out{"REPEATED_SIDE_OUT"};
};

TEST(GraphTest, CanUseWithEveryFieldContract) {
  Graph<EveryFieldContract> graph;

  auto& node = graph.AddNodeByContract<EveryFieldContract>("EveryFieldNode");
  node.in.Set(graph.in.Get());
  node.optional_in.Set(graph.optional_in.Get());
  node.repeated_in.Add(graph.repeated_in.Add());
  node.repeated_in.Add(graph.repeated_in.Add());

  node.side_in.Set(graph.side_in.Get());
  node.optional_side_in.Set(graph.optional_side_in.Get());
  node.repeated_side_in.Add(graph.repeated_side_in.Add());
  node.repeated_side_in.Add(graph.repeated_side_in.Add());

  graph.out.Set(node.out.Get());
  graph.optional_out.Set(node.optional_out.Get());
  graph.repeated_out.Add(node.repeated_out.Add());
  graph.repeated_out.Add(node.repeated_out.Add());

  graph.side_out.Set(node.side_out.Get());
  graph.optional_side_out.Set(node.optional_side_out.Get());
  graph.repeated_side_out.Add(node.repeated_side_out.Add());
  graph.repeated_side_out.Add(node.repeated_side_out.Add());

  MP_ASSERT_OK_AND_ASSIGN(auto config, graph.GetConfig());
  EXPECT_THAT(
      config,
      EqualsProto(ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(R"pb(
        input_stream: "IN:__stream_0"
        input_stream: "OPTIONAL_IN:__stream_1"
        input_stream: "REPEATED_IN:0:__stream_2"
        input_stream: "REPEATED_IN:1:__stream_3"
        output_stream: "OPTIONAL_OUT:__stream_8"
        output_stream: "OUT:__stream_9"
        output_stream: "REPEATED_OUT:0:__stream_10"
        output_stream: "REPEATED_OUT:1:__stream_11"
        input_side_packet: "OPTIONAL_SIDE_IN:__side_packet_4"
        input_side_packet: "REPEATED_SIDE_IN:0:__side_packet_5"
        input_side_packet: "REPEATED_SIDE_IN:1:__side_packet_6"
        input_side_packet: "SIDE_IN:__side_packet_7"
        output_side_packet: "OPTIONAL_SIDE_OUT:__side_packet_12"
        output_side_packet: "REPEATED_SIDE_OUT:0:__side_packet_13"
        output_side_packet: "REPEATED_SIDE_OUT:1:__side_packet_14"
        output_side_packet: "SIDE_OUT:__side_packet_15"
        node {
          calculator: "EveryFieldNode"
          input_stream: "IN:__stream_0"
          input_stream: "OPTIONAL_IN:__stream_1"
          input_stream: "REPEATED_IN:0:__stream_2"
          input_stream: "REPEATED_IN:1:__stream_3"
          output_stream: "OPTIONAL_OUT:__stream_8"
          output_stream: "OUT:__stream_9"
          output_stream: "REPEATED_OUT:0:__stream_10"
          output_stream: "REPEATED_OUT:1:__stream_11"
          input_side_packet: "OPTIONAL_SIDE_IN:__side_packet_4"
          input_side_packet: "REPEATED_SIDE_IN:0:__side_packet_5"
          input_side_packet: "REPEATED_SIDE_IN:1:__side_packet_6"
          input_side_packet: "SIDE_IN:__side_packet_7"
          output_side_packet: "OPTIONAL_SIDE_OUT:__side_packet_12"
          output_side_packet: "REPEATED_SIDE_OUT:0:__side_packet_13"
          output_side_packet: "REPEATED_SIDE_OUT:1:__side_packet_14"
          output_side_packet: "SIDE_OUT:__side_packet_15"
        }
      )pb")));
}

TEST(GraphTest, CanSetEveryFieldNames) {
  Graph<EveryFieldContract> graph;
  auto in = graph.in.Get().SetName("in");
  auto optional_in = graph.optional_in.Get().SetName("optional_in");
  auto repeated_in0 = graph.repeated_in.Add().SetName("repeated_in0");
  auto repeated_in1 = graph.repeated_in.Add().SetName("repeated_in1");
  auto side_in = graph.side_in.Get().SetName("side_in");
  auto optional_side_in =
      graph.optional_side_in.Get().SetName("optional_side_in");
  auto repeated_side_in0 =
      graph.repeated_side_in.Add().SetName("repeated_side_in0");
  auto repeated_side_in1 =
      graph.repeated_side_in.Add().SetName("repeated_side_in1");

  auto& node = graph.AddNodeByContract<EveryFieldContract>("EveryFieldNode");
  node.in.Set(in);
  node.optional_in.Set(optional_in);
  node.repeated_in.Add(repeated_in0);
  node.repeated_in.Add(repeated_in1);
  node.side_in.Set(side_in);
  node.optional_side_in.Set(optional_side_in);
  node.repeated_side_in.Add(repeated_side_in0);
  node.repeated_side_in.Add(repeated_side_in1);
  auto out = node.out.Get().SetName("out");
  auto optional_out = node.optional_out.Get().SetName("optional_out");
  auto repeated_out0 = node.repeated_out.Add().SetName("repeated_out0");
  auto repeated_out1 = node.repeated_out.Add().SetName("repeated_out1");
  auto side_out = node.side_out.Get().SetName("side_out");
  auto optional_side_out =
      node.optional_side_out.Get().SetName("optional_side_out");
  ;
  auto repeated_side_out0 =
      node.repeated_side_out.Add().SetName("repeated_side_out0");
  auto repeated_side_out1 =
      node.repeated_side_out.Add().SetName("repeated_side_out1");

  graph.out.Set(out);
  graph.optional_out.Set(optional_out);
  graph.repeated_out.Add(repeated_out0);
  graph.repeated_out.Add(repeated_out1);
  graph.side_out.Set(side_out);
  graph.optional_side_out.Set(optional_side_out);
  graph.repeated_side_out.Add(repeated_side_out0);
  graph.repeated_side_out.Add(repeated_side_out1);

  MP_ASSERT_OK_AND_ASSIGN(auto config, graph.GetConfig());
  EXPECT_THAT(
      config,
      EqualsProto(ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(R"pb(
        input_stream: "IN:in"
        input_stream: "OPTIONAL_IN:optional_in"
        input_stream: "REPEATED_IN:0:repeated_in0"
        input_stream: "REPEATED_IN:1:repeated_in1"
        input_side_packet: "OPTIONAL_SIDE_IN:optional_side_in"
        input_side_packet: "REPEATED_SIDE_IN:0:repeated_side_in0"
        input_side_packet: "REPEATED_SIDE_IN:1:repeated_side_in1"
        input_side_packet: "SIDE_IN:side_in"

        node {
          calculator: "EveryFieldNode"
          input_stream: "IN:in"
          input_stream: "OPTIONAL_IN:optional_in"
          input_stream: "REPEATED_IN:0:repeated_in0"
          input_stream: "REPEATED_IN:1:repeated_in1"
          output_stream: "OPTIONAL_OUT:optional_out"
          output_stream: "OUT:out"
          output_stream: "REPEATED_OUT:0:repeated_out0"
          output_stream: "REPEATED_OUT:1:repeated_out1"
          input_side_packet: "OPTIONAL_SIDE_IN:optional_side_in"
          input_side_packet: "REPEATED_SIDE_IN:0:repeated_side_in0"
          input_side_packet: "REPEATED_SIDE_IN:1:repeated_side_in1"
          input_side_packet: "SIDE_IN:side_in"
          output_side_packet: "OPTIONAL_SIDE_OUT:optional_side_out"
          output_side_packet: "REPEATED_SIDE_OUT:0:repeated_side_out0"
          output_side_packet: "REPEATED_SIDE_OUT:1:repeated_side_out1"
          output_side_packet: "SIDE_OUT:side_out"
        }

        output_stream: "OPTIONAL_OUT:optional_out"
        output_stream: "OUT:out"
        output_stream: "REPEATED_OUT:0:repeated_out0"
        output_stream: "REPEATED_OUT:1:repeated_out1"
        output_side_packet: "OPTIONAL_SIDE_OUT:optional_side_out"
        output_side_packet: "REPEATED_SIDE_OUT:0:repeated_side_out0"
        output_side_packet: "REPEATED_SIDE_OUT:1:repeated_side_out1"
        output_side_packet: "SIDE_OUT:side_out"
      )pb")));
}

}  // namespace
}  // namespace mediapipe::api3
