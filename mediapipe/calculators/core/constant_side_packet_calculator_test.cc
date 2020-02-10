// Copyright 2020 The MediaPipe Authors.
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

#include <string>

#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

template <typename T>
void DoTestSingleSidePacket(absl::string_view packet_spec,
                            const T& expected_value) {
  static constexpr absl::string_view graph_config_template = R"(
        node {
          calculator: "ConstantSidePacketCalculator"
          output_side_packet: "PACKET:packet"
          options: {
            [mediapipe.ConstantSidePacketCalculatorOptions.ext]: {
              packet $0
            }
          }
        }
      )";
  CalculatorGraphConfig graph_config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
          absl::Substitute(graph_config_template, packet_spec));
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  MP_ASSERT_OK(graph.GetOutputSidePacket("packet"));
  auto actual_value =
      graph.GetOutputSidePacket("packet").ValueOrDie().template Get<T>();
  EXPECT_EQ(actual_value, expected_value);
}

TEST(ConstantSidePacketCalculatorTest, EveryPossibleType) {
  DoTestSingleSidePacket("{ int_value: 2 }", 2);
  DoTestSingleSidePacket("{ float_value: 6.5f }", 6.5f);
  DoTestSingleSidePacket("{ bool_value: true }", true);
  DoTestSingleSidePacket<std::string>(R"({ string_value: "str" })", "str");
}

TEST(ConstantSidePacketCalculatorTest, MultiplePackets) {
  CalculatorGraphConfig graph_config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        node {
          calculator: "ConstantSidePacketCalculator"
          output_side_packet: "PACKET:0:int_packet"
          output_side_packet: "PACKET:1:float_packet"
          output_side_packet: "PACKET:2:bool_packet"
          output_side_packet: "PACKET:3:string_packet"
          output_side_packet: "PACKET:4:another_string_packet"
          output_side_packet: "PACKET:5:another_int_packet"
          options: {
            [mediapipe.ConstantSidePacketCalculatorOptions.ext]: {
              packet { int_value: 256 }
              packet { float_value: 0.5f }
              packet { bool_value: false }
              packet { string_value: "string" }
              packet { string_value: "another string" }
              packet { int_value: 128 }
            }
          }
        }
      )");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  MP_ASSERT_OK(graph.GetOutputSidePacket("int_packet"));
  EXPECT_EQ(graph.GetOutputSidePacket("int_packet").ValueOrDie().Get<int>(),
            256);
  MP_ASSERT_OK(graph.GetOutputSidePacket("float_packet"));
  EXPECT_EQ(graph.GetOutputSidePacket("float_packet").ValueOrDie().Get<float>(),
            0.5f);
  MP_ASSERT_OK(graph.GetOutputSidePacket("bool_packet"));
  EXPECT_FALSE(
      graph.GetOutputSidePacket("bool_packet").ValueOrDie().Get<bool>());
  MP_ASSERT_OK(graph.GetOutputSidePacket("string_packet"));
  EXPECT_EQ(graph.GetOutputSidePacket("string_packet")
                .ValueOrDie()
                .Get<std::string>(),
            "string");
  MP_ASSERT_OK(graph.GetOutputSidePacket("another_string_packet"));
  EXPECT_EQ(graph.GetOutputSidePacket("another_string_packet")
                .ValueOrDie()
                .Get<std::string>(),
            "another string");
  MP_ASSERT_OK(graph.GetOutputSidePacket("another_int_packet"));
  EXPECT_EQ(
      graph.GetOutputSidePacket("another_int_packet").ValueOrDie().Get<int>(),
      128);
}

TEST(ConstantSidePacketCalculatorTest, ProcessingPacketsWithCorrectTagOnly) {
  CalculatorGraphConfig graph_config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        node {
          calculator: "ConstantSidePacketCalculator"
          output_side_packet: "PACKET:0:int_packet"
          output_side_packet: "no_tag0"
          output_side_packet: "PACKET:1:float_packet"
          output_side_packet: "INCORRECT_TAG:0:name1"
          output_side_packet: "PACKET:2:bool_packet"
          output_side_packet: "PACKET:3:string_packet"
          output_side_packet: "no_tag2"
          output_side_packet: "INCORRECT_TAG:1:name2"
          options: {
            [mediapipe.ConstantSidePacketCalculatorOptions.ext]: {
              packet { int_value: 256 }
              packet { float_value: 0.5f }
              packet { bool_value: false }
              packet { string_value: "string" }
            }
          }
        }
      )");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  MP_ASSERT_OK(graph.GetOutputSidePacket("int_packet"));
  EXPECT_EQ(graph.GetOutputSidePacket("int_packet").ValueOrDie().Get<int>(),
            256);
  MP_ASSERT_OK(graph.GetOutputSidePacket("float_packet"));
  EXPECT_EQ(graph.GetOutputSidePacket("float_packet").ValueOrDie().Get<float>(),
            0.5f);
  MP_ASSERT_OK(graph.GetOutputSidePacket("bool_packet"));
  EXPECT_FALSE(
      graph.GetOutputSidePacket("bool_packet").ValueOrDie().Get<bool>());
  MP_ASSERT_OK(graph.GetOutputSidePacket("string_packet"));
  EXPECT_EQ(graph.GetOutputSidePacket("string_packet")
                .ValueOrDie()
                .Get<std::string>(),
            "string");
}

TEST(ConstantSidePacketCalculatorTest, IncorrectConfig_MoreOptionsThanPackets) {
  CalculatorGraphConfig graph_config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        node {
          calculator: "ConstantSidePacketCalculator"
          output_side_packet: "PACKET:int_packet"
          options: {
            [mediapipe.ConstantSidePacketCalculatorOptions.ext]: {
              packet { int_value: 256 }
              packet { float_value: 0.5f }
            }
          }
        }
      )");
  CalculatorGraph graph;
  EXPECT_FALSE(graph.Initialize(graph_config).ok());
}

TEST(ConstantSidePacketCalculatorTest, IncorrectConfig_MorePacketsThanOptions) {
  CalculatorGraphConfig graph_config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        node {
          calculator: "ConstantSidePacketCalculator"
          output_side_packet: "PACKET:0:int_packet"
          output_side_packet: "PACKET:1:float_packet"
          options: {
            [mediapipe.ConstantSidePacketCalculatorOptions.ext]: {
              packet { int_value: 256 }
            }
          }
        }
      )");
  CalculatorGraph graph;
  EXPECT_FALSE(graph.Initialize(graph_config).ok());
}

}  // namespace mediapipe
