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

#include "mediapipe/framework/api3/internal/graph_builder.h"

#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/stream_handler/fixed_size_input_stream_handler.pb.h"
#include "mediapipe/framework/testdata/night_light_calculator.pb.h"
#include "mediapipe/framework/testdata/sky_light_calculator.pb.h"

namespace mediapipe::api3::builder {
namespace {

TEST(GenericGraphTest, CanBuildGenericGraph) {
  GraphBuilder graph;

  // Graph inputs.
  auto& base = graph.In("IN").At(0).SetName("base");
  auto& side = graph.SideIn("SIDE").At(0).SetName("side");

  // Graph body.
  auto& foo = graph.AddNode("Foo");
  base.ConnectTo(foo.In("BASE").At(0));
  side.ConnectTo(foo.SideIn("SIDE").At(0));
  auto& foo_out = foo.Out("OUT").At(0);

  auto& bar = graph.AddNode("Bar");
  foo_out.ConnectTo(bar.In("IN").At(0));
  auto& bar_out = bar.Out("OUT").At(0);

  // Graph outputs.
  bar_out.SetName("out").ConnectTo(graph.Out("OUT").At(0));

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

TEST(GenericGraphTest, CanBuildGraphDefiningAndSettingExecutors) {
  GraphBuilder graph;

  // Inputs.
  auto& base = graph.In("IN").At(0).SetName("base");
  auto& side = graph.SideIn("SIDE").At(0).SetName("side");

  // Executors.
  auto& executor0 = graph.AddExecutor("ThreadPoolExecutor");

  auto& executor1 = graph.AddExecutor("ThreadPoolExecutor");
  auto& executor1_opts = executor1.GetOptions<ThreadPoolExecutorOptions>();
  executor1_opts.set_num_threads(42);

  // Nodes.
  auto& foo1 = graph.AddNode("Foo");
  foo1.SetExecutor(executor0);
  base.ConnectTo(foo1.In("BASE").At(0));
  side.ConnectTo(foo1.SideIn("SIDE").At(0));
  auto& foo1_out = foo1.Out("OUT").At(0);

  auto& foo2 = graph.AddNode("Foo");
  foo2.SetExecutor(executor1);
  base.ConnectTo(foo2.In("BASE").At(0));
  side.ConnectTo(foo2.SideIn("SIDE").At(0));
  auto& foo2_out = foo2.Out("OUT").At(0);

  auto& bar1 = graph.AddNode("Bar");
  bar1.SetExecutor(executor0);
  foo1_out.ConnectTo(bar1.In("IN").At(0));
  auto& bar1_out = bar1.Out("OUT").At(0);

  auto& bar2 = graph.AddNode("Bar");
  bar2.SetExecutor(executor1);
  foo2_out.ConnectTo(bar2.In("IN").At(0));
  auto& bar2_out = bar2.Out("OUT").At(0);

  // Graph outputs.
  bar1_out.SetName("out1").ConnectTo(graph.Out("OUT").At(0));
  bar2_out.SetName("out2").ConnectTo(graph.Out("OUT").At(1));

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
  GraphBuilder graph;
  // Graph inputs.
  auto& base = graph.In("IN").At(0).SetName("base");
  auto& side = graph.SideIn("SIDE").At(0).SetName("side");

  auto& foo = graph.AddNode("Foo");
  auto& foo_ish_opts =
      foo.SetInputStreamHandler("FixedSizeInputStreamHandler")
          .GetOptions<mediapipe::FixedSizeInputStreamHandlerOptions>();
  foo_ish_opts.set_target_queue_size(2);
  foo_ish_opts.set_trigger_queue_size(3);
  foo_ish_opts.set_fixed_min_size(true);
  base.ConnectTo(foo.In("BASE").At(0));
  side.ConnectTo(foo.SideIn("SIDE").At(0));
  auto& foo_out = foo.Out("OUT").At(0);

  auto& bar = graph.AddNode("Bar");
  bar.SetInputStreamHandler("ImmediateInputStreamHandler");
  bar.SetOutputStreamHandler("InOrderOutputStreamHandler");
  foo_out.ConnectTo(bar.In("IN").At(0));
  auto& bar_out = bar.Out("OUT").At(0);

  // Graph outputs.
  bar_out.SetName("out").ConnectTo(graph.Out("OUT").At(0));

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
  GraphBuilder graph;
  // Graph inputs.
  auto& base = graph.In("IN").At(0).SetName("base");
  auto& side = graph.SideIn("SIDE").At(0).SetName("side");

  auto& foo = graph.AddNode("Foo");
  foo.SetSourceLayer(0);
  base.ConnectTo(foo.In("BASE").At(0));
  side.ConnectTo(foo.SideIn("SIDE").At(0));
  auto& foo_out = foo.Out("OUT").At(0);

  auto& bar = graph.AddNode("Bar");
  bar.SetSourceLayer(1);
  foo_out.ConnectTo(bar.In("IN").At(0));
  auto& bar_out = bar.Out("OUT").At(0);

  // Graph outputs.
  bar_out.SetName("out").ConnectTo(graph.Out("OUT").At(0));

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

TEST(GenericGraphTest, CanUseBackEdges) {
  GraphBuilder graph;
  // Graph inputs.
  auto& image = graph.In("IMAGE").At(0).SetName("image");

  auto& loopback_node = graph.AddNode("PreviousLoopbackCalculator");
  image.ConnectTo(loopback_node.In("MAIN").At(0));
  auto set_prev_detections_fn = [&loopback_node](Source& value) {
    value.ConnectTo(loopback_node.In("LOOP").At(0).AsBackEdge());
  };
  auto& prev_detections = loopback_node.Out("PREV_LOOP").At(0);

  auto& detections = [&]() -> Source& {
    auto& detection_node = graph.AddNode("ObjectDetectionCalculator");
    image.ConnectTo(detection_node.In("IMAGE").At(0));
    prev_detections.ConnectTo(detection_node.In("PREV_DETECTIONS").At(0));
    return detection_node.Out("DETECTIONS").At(0);
  }();

  set_prev_detections_fn(detections);

  // Graph outputs.
  detections.SetName("detections").ConnectTo(graph.Out("OUT").At(0));

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

TEST(GenericGraphTest, CanUseBackEdgesWithIndex) {
  GraphBuilder graph;
  // Graph inputs.
  auto& image = graph.In("IN").At(0).SetName("in_data");

  auto& back_edge_node = graph.AddNode("SomeBackEdgeCalculator");
  image.ConnectTo(back_edge_node.In("DATA").At(0));
  auto set_back_edge_fn = [&back_edge_node](Source& value) {
    auto& loop = back_edge_node.In("DATA").At(1);
    loop.back_edge = true;
    value.ConnectTo(loop);
  };
  auto& processed_data = back_edge_node.Out("PROCESSED_DATA").At(0);

  auto& output_data = [&]() -> Source& {
    auto& detection_node = graph.AddNode("SomeOutputDataCalculator");
    image.ConnectTo(detection_node.In("IMAGE").At(0));
    processed_data.ConnectTo(detection_node.In("PROCESSED_DATA").At(0));
    return detection_node.Out("OUTPUT_DATA").At(0);
  }();

  set_back_edge_fn(output_data);

  // Graph outputs.
  output_data.SetName("out_data").ConnectTo(graph.Out("OUT").At(0));

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
          input_stream: "IMAGE:in_data"
          input_stream: "PROCESSED_DATA:__stream_0"
          output_stream: "OUTPUT_DATA:out_data"
        }
        input_stream: "IN:in_data"
        output_stream: "OUT:out_data"
      )pb");
  MP_ASSERT_OK_AND_ASSIGN(CalculatorGraphConfig config, graph.GetConfig());
  EXPECT_THAT(config, EqualsProto(expected_config));
}

TEST(GenericGraphTest, CanUseBackEdgesWithIndexAndNoTag) {
  GraphBuilder graph;
  // Graph inputs.
  auto& image = graph.In("IN").At(0).SetName("in_data");

  auto& back_edge_node = graph.AddNode("SomeBackEdgeCalculator");
  image.ConnectTo(back_edge_node.In("").At(0));
  auto set_back_edge_fn = [&back_edge_node](Source& loop) {
    loop.ConnectTo(back_edge_node.In("").At(1).AsBackEdge());
  };
  auto& processed_data = back_edge_node.Out("PROCESSED_DATA").At(0);

  auto& output_data = [&]() -> Source& {
    auto& detection_node = graph.AddNode("SomeOutputDataCalculator");
    image.ConnectTo(detection_node.In("IMAGE").At(0));
    processed_data.ConnectTo(detection_node.In("PROCESSED_DATA").At(0));
    return detection_node.Out("OUTPUT_DATA").At(0);
  }();

  set_back_edge_fn(output_data);

  // Graph outputs.
  output_data.SetName("out_data").ConnectTo(graph.Out("OUT").At(0));

  CalculatorGraphConfig expected_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "SomeBackEdgeCalculator"
          input_stream: "in_data"
          input_stream: "out_data"
          output_stream: "PROCESSED_DATA:__stream_0"
          input_stream_info { tag_index: ":1" back_edge: true }
        }
        node {
          calculator: "SomeOutputDataCalculator"
          input_stream: "IMAGE:in_data"
          input_stream: "PROCESSED_DATA:__stream_0"
          output_stream: "OUTPUT_DATA:out_data"
        }
        input_stream: "IN:in_data"
        output_stream: "OUT:out_data"
      )pb");
  MP_ASSERT_OK_AND_ASSIGN(CalculatorGraphConfig config, graph.GetConfig());
  EXPECT_THAT(config, EqualsProto(expected_config));
}

TEST(GenericGraphTest, FanOut) {
  GraphBuilder graph;
  // Graph inputs.
  auto& base = graph.In("IN").At(0).SetName("base");

  auto& foo = graph.AddNode("Foo");
  base.ConnectTo(foo.In("BASE").At(0));
  auto& foo_out = foo.Out("OUT").At(0);

  auto& adder = graph.AddNode("FloatAdder");
  foo_out.ConnectTo(adder.In("IN").At(0));
  foo_out.ConnectTo(adder.In("IN").At(1));
  auto& out = adder.Out("OUT").At(0);

  // Graph outputs.
  out.SetName("out").ConnectTo(graph.Out("OUT").At(0));

  CalculatorGraphConfig expected_config =
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
  MP_ASSERT_OK_AND_ASSIGN(CalculatorGraphConfig config, graph.GetConfig());
  EXPECT_THAT(config, EqualsProto(expected_config));
}

TEST(GenericGraphTest, CanAddPacketGenerator) {
  GraphBuilder graph;
  // Graph inputs.
  auto& side_in = graph.SideIn("IN").At(0);

  auto& generator = graph.AddPacketGenerator("FloatGenerator");
  side_in.ConnectTo(generator.SideIn("IN").At(0));
  auto& side_out = generator.SideOut("OUT").At(0);

  // Graph outputs.
  side_out.ConnectTo(graph.SideOut("OUT").At(0));

  CalculatorGraphConfig expected_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_side_packet: "IN:__side_packet_0"
        output_side_packet: "OUT:__side_packet_1"
        packet_generator {
          packet_generator: "FloatGenerator"
          input_side_packet: "IN:__side_packet_0"
          output_side_packet: "OUT:__side_packet_1"
        }
      )pb");
  MP_ASSERT_OK_AND_ASSIGN(CalculatorGraphConfig config, graph.GetConfig());
  EXPECT_THAT(config, EqualsProto(expected_config));
}

TEST(GenericGraphTest, SupportsEmptyTags) {
  GraphBuilder graph;
  // Graph inputs.
  auto& a = graph.In("A").At(0).SetName("a");
  auto& c = graph.In("C").At(0).SetName("c");
  auto& b = graph.In("B").At(0).SetName("b");

  auto& foo = graph.AddNode("Foo");
  a.ConnectTo(foo.In("").At(0));
  c.ConnectTo(foo.In("").At(2));
  b.ConnectTo(foo.In("").At(1));
  auto& x = foo.Out("").At(0);
  auto& y = foo.Out("").At(1);

  // Graph outputs.
  x.SetName("x").ConnectTo(graph.Out("ONE").At(0));
  y.SetName("y").ConnectTo(graph.Out("TWO").At(0));

  CalculatorGraphConfig expected_config =
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
  MP_ASSERT_OK_AND_ASSIGN(CalculatorGraphConfig config, graph.GetConfig());
  EXPECT_THAT(config, EqualsProto(expected_config));
}

TEST(GenericGraphTest, SupportsStringLikeTags) {
  const char kA[] = "A";
  const std::string kB = "B";
  constexpr absl::string_view kC = "C";

  GraphBuilder graph;
  // Graph inputs.
  auto& a = graph.In(kA).At(0).SetName("a");
  auto& b = graph.In(kB).At(0).SetName("b");

  auto& foo = graph.AddNode("Foo");
  a.ConnectTo(foo.In(kA).At(0));
  b.ConnectTo(foo.In(kB).At(0));
  auto& c = foo.Out(kC).At(0);

  // Graph outputs.
  c.SetName("c").ConnectTo(graph.Out(kC).At(0));

  CalculatorGraphConfig expected_config =
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
  MP_ASSERT_OK_AND_ASSIGN(CalculatorGraphConfig config, graph.GetConfig());
  EXPECT_THAT(config, EqualsProto(expected_config));
}

TEST(GenericGraphTest, SupportsIndexing) {
  GraphBuilder graph;
  // Graph inputs.
  auto& a = graph.In("").At(0).SetName("a");
  auto& c = graph.In("").At(1).SetName("c");
  auto& b = graph.In("").At(2).SetName("b");

  auto& foo = graph.AddNode("Foo");
  a.ConnectTo(foo.In("").At(0));
  c.ConnectTo(foo.In("").At(2));
  b.ConnectTo(foo.In("").At(1));
  auto& x = foo.Out("").At(0);
  auto& y = foo.Out("").At(1);

  // Graph outputs.
  x.SetName("x").ConnectTo(graph.Out("").At(1));
  y.SetName("y").ConnectTo(graph.Out("").At(0));

  CalculatorGraphConfig expected_config =
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
  MP_ASSERT_OK_AND_ASSIGN(CalculatorGraphConfig config, graph.GetConfig());
  EXPECT_THAT(config, EqualsProto(expected_config));
}

TEST(GetOptionsTest, CanAddProto3Options) {
  GraphBuilder graph;
  // Graph inputs.
  auto& base = graph.In("IN").At(0).SetName("base");
  auto& side = graph.SideIn("SIDE").At(0).SetName("side");

  auto& foo = graph.AddNode("Foo");
  foo.GetOptions<mediapipe::SkyLightCalculatorOptions>();
  base.ConnectTo(foo.In("BASE").At(0));
  side.ConnectTo(foo.SideIn("SIDE").At(0));
  auto& foo_out = foo.Out("OUT").At(0);

  auto& bar = graph.AddNode("Bar");
  foo_out.ConnectTo(bar.In("IN").At(0));
  auto& bar_out = bar.Out("OUT").At(0);

  // Graph outputs.
  bar_out.SetName("out").ConnectTo(graph.Out("OUT").At(0));

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
  MP_ASSERT_OK_AND_ASSIGN(CalculatorGraphConfig config, graph.GetConfig());
  EXPECT_THAT(config, EqualsProto(expected_config));
}

TEST(GetOptionsTest, CanAddProto2Options) {
  GraphBuilder graph;
  // Graph inputs.
  auto& base = graph.In("IN").At(0).SetName("base");
  auto& side = graph.SideIn("SIDE").At(0).SetName("side");

  auto& foo = graph.AddNode("Foo");
  foo.GetOptions<mediapipe::NightLightCalculatorOptions>();
  base.ConnectTo(foo.In("BASE").At(0));
  side.ConnectTo(foo.SideIn("SIDE").At(0));
  auto& foo_out = foo.Out("OUT").At(0);

  auto& bar = graph.AddNode("Bar");
  foo_out.ConnectTo(bar.In("IN").At(0));
  auto& bar_out = bar.Out("OUT").At(0);

  // Graph outputs.
  bar_out.SetName("out").ConnectTo(graph.Out("OUT").At(0));

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
  MP_ASSERT_OK_AND_ASSIGN(CalculatorGraphConfig config, graph.GetConfig());
  EXPECT_THAT(config, EqualsProto(expected_config));
}

TEST(GetOptionsTest, AddBothProto23Options) {
  GraphBuilder graph;
  // Graph inputs.
  auto& base = graph.In("IN").At(0).SetName("base");
  auto& side = graph.SideIn("SIDE").At(0).SetName("side");

  auto& foo = graph.AddNode("Foo");
  foo.GetOptions<mediapipe::NightLightCalculatorOptions>();
  foo.GetOptions<mediapipe::SkyLightCalculatorOptions>();
  base.ConnectTo(foo.In("BASE").At(0));
  side.ConnectTo(foo.SideIn("SIDE").At(0));
  auto& foo_out = foo.Out("OUT").At(0);

  auto& bar = graph.AddNode("Bar");
  foo_out.ConnectTo(bar.In("IN").At(0));
  auto& bar_out = bar.Out("OUT").At(0);

  // Graph outputs.
  bar_out.SetName("out").ConnectTo(graph.Out("OUT").At(0));

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
  MP_ASSERT_OK_AND_ASSIGN(CalculatorGraphConfig config, graph.GetConfig());
  EXPECT_THAT(config, EqualsProto(expected_config));
}

TEST(GenericGraphTest, FailsIfSkippingInputSource) {
  GraphBuilder graph;

  auto& multi_inputs_node = graph.AddNode("MultiInputsOutputs");
  auto& base = graph.In("IN").At(0).SetName("base");
  // We only connect to the second input. Missing source for input stream at
  // index 0.
  base.ConnectTo(multi_inputs_node.In("").At(1));

  EXPECT_THAT(
      graph.GetConfig(),
      StatusIs(absl::StatusCode::kInternal,
               testing::HasSubstr("Missing port for tag: \"\", index: 0.")));
}

}  // namespace
}  // namespace mediapipe::api3::builder
