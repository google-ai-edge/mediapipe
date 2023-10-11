/* Copyright 2022 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/tasks/cc/components/utils/gate.h"

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"

namespace mediapipe {
namespace tasks {
namespace components {
namespace utils {
namespace {

using ::mediapipe::api2::builder::SideSource;
using ::mediapipe::api2::builder::Source;

TEST(DisallowGate, VerifyConfig) {
  mediapipe::api2::builder::Graph graph;

  Source<bool> condition = graph.In("CONDITION").Cast<bool>();
  Source<int> value1 = graph.In("VALUE_1").Cast<int>();
  Source<int> value2 = graph.In("VALUE_2").Cast<int>();
  Source<int> value3 = graph.In("VALUE_3").Cast<int>();

  DisallowGate gate(condition, graph);
  gate.Disallow(value1).SetName("gated_stream1");
  gate.Disallow(value2).SetName("gated_stream2");
  gate.Disallow(value3).SetName("gated_stream3");

  EXPECT_THAT(graph.GetConfig(),
              testing::EqualsProto(
                  mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
                    node {
                      calculator: "GateCalculator"
                      input_stream: "__stream_1"
                      input_stream: "__stream_2"
                      input_stream: "__stream_3"
                      input_stream: "DISALLOW:__stream_0"
                      output_stream: "gated_stream1"
                      output_stream: "gated_stream2"
                      output_stream: "gated_stream3"
                      options {
                        [mediapipe.GateCalculatorOptions.ext] {
                          empty_packets_as_allow: true
                        }
                      }
                    }
                    input_stream: "CONDITION:__stream_0"
                    input_stream: "VALUE_1:__stream_1"
                    input_stream: "VALUE_2:__stream_2"
                    input_stream: "VALUE_3:__stream_3"
                  )pb")));

  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

TEST(DisallowIf, VerifyConfig) {
  mediapipe::api2::builder::Graph graph;

  Source<int> value = graph.In("VALUE").Cast<int>();
  Source<bool> condition = graph.In("CONDITION").Cast<bool>();

  auto gated_stream = DisallowIf(value, condition, graph);
  gated_stream.SetName("gated_stream");

  EXPECT_THAT(graph.GetConfig(),
              testing::EqualsProto(
                  mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
                    node {
                      calculator: "GateCalculator"
                      input_stream: "__stream_1"
                      input_stream: "DISALLOW:__stream_0"
                      output_stream: "gated_stream"
                      options {
                        [mediapipe.GateCalculatorOptions.ext] {
                          empty_packets_as_allow: true
                        }
                      }
                    }
                    input_stream: "CONDITION:__stream_0"
                    input_stream: "VALUE:__stream_1"
                  )pb")));

  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

TEST(DisallowIf, VerifyConfigWithSideCondition) {
  mediapipe::api2::builder::Graph graph;

  Source<int> value = graph.In("VALUE").Cast<int>();
  SideSource<bool> condition = graph.SideIn("CONDITION").Cast<bool>();

  auto gated_stream = DisallowIf(value, condition, graph);
  gated_stream.SetName("gated_stream");

  EXPECT_THAT(graph.GetConfig(),
              testing::EqualsProto(
                  mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
                    node {
                      calculator: "GateCalculator"
                      input_stream: "__stream_0"
                      output_stream: "gated_stream"
                      input_side_packet: "DISALLOW:__side_packet_1"
                      options {
                        [mediapipe.GateCalculatorOptions.ext] {
                          empty_packets_as_allow: true
                        }
                      }
                    }
                    input_stream: "VALUE:__stream_0"
                    input_side_packet: "CONDITION:__side_packet_1"
                  )pb")));

  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

TEST(AllowGate, VerifyConfig) {
  mediapipe::api2::builder::Graph graph;

  Source<bool> condition = graph.In("CONDITION").Cast<bool>();
  Source<int> value1 = graph.In("VALUE_1").Cast<int>();
  Source<int> value2 = graph.In("VALUE_2").Cast<int>();
  Source<int> value3 = graph.In("VALUE_3").Cast<int>();

  AllowGate gate(condition, graph);
  gate.Allow(value1).SetName("gated_stream1");
  gate.Allow(value2).SetName("gated_stream2");
  gate.Allow(value3).SetName("gated_stream3");

  EXPECT_THAT(graph.GetConfig(),
              testing::EqualsProto(
                  mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
                    node {
                      calculator: "GateCalculator"
                      input_stream: "__stream_1"
                      input_stream: "__stream_2"
                      input_stream: "__stream_3"
                      input_stream: "ALLOW:__stream_0"
                      output_stream: "gated_stream1"
                      output_stream: "gated_stream2"
                      output_stream: "gated_stream3"
                    }
                    input_stream: "CONDITION:__stream_0"
                    input_stream: "VALUE_1:__stream_1"
                    input_stream: "VALUE_2:__stream_2"
                    input_stream: "VALUE_3:__stream_3"
                  )pb")));

  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

TEST(AllowIf, VerifyConfig) {
  mediapipe::api2::builder::Graph graph;

  Source<int> value = graph.In("VALUE").Cast<int>();
  Source<bool> condition = graph.In("CONDITION").Cast<bool>();

  auto gated_stream = AllowIf(value, condition, graph);
  gated_stream.SetName("gated_stream");

  EXPECT_THAT(graph.GetConfig(),
              testing::EqualsProto(
                  mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
                    node {
                      calculator: "GateCalculator"
                      input_stream: "__stream_1"
                      input_stream: "ALLOW:__stream_0"
                      output_stream: "gated_stream"
                    }
                    input_stream: "CONDITION:__stream_0"
                    input_stream: "VALUE:__stream_1"
                  )pb")));

  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

TEST(AllowIf, VerifyConfigWithSideConition) {
  mediapipe::api2::builder::Graph graph;

  Source<int> value = graph.In("VALUE").Cast<int>();
  SideSource<bool> condition = graph.SideIn("CONDITION").Cast<bool>();

  auto gated_stream = AllowIf(value, condition, graph);
  gated_stream.SetName("gated_stream");

  EXPECT_THAT(graph.GetConfig(),
              testing::EqualsProto(
                  mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
                    node {
                      calculator: "GateCalculator"
                      input_stream: "__stream_0"
                      output_stream: "gated_stream"
                      input_side_packet: "ALLOW:__side_packet_1"
                    }
                    input_stream: "VALUE:__stream_0"
                    input_side_packet: "CONDITION:__side_packet_1"
                  )pb")));

  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(graph.GetConfig()));
}

}  // namespace
}  // namespace utils
}  // namespace components
}  // namespace tasks
}  // namespace mediapipe
