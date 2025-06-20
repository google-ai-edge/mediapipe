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

#include <map>
#include <optional>
#include <string>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

enum class ValueType { kString, kInt, kFloat };

struct ProducesExpectedOutputParam {
  std::string test_name;
  ValueType value_type;
  Packet default_packet;
  std::optional<Packet> optional_packet;
  Packet expected_output;
};

using DefaultSidePacketCalculator =
    testing::TestWithParam<ProducesExpectedOutputParam>;

TEST_P(DefaultSidePacketCalculator, ProducesExpectedOutput) {
  auto& [test_name, value_type, default_packet, optional_packet,
         expected_output] = GetParam();
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_side_packet: "optional_input"
        input_side_packet: "default_input"
        output_side_packet: "output_value"
        node {
          calculator: "DefaultSidePacketCalculator"
          input_side_packet: "OPTIONAL_VALUE:optional_input"
          input_side_packet: "DEFAULT_VALUE:default_input"
          output_side_packet: "VALUE:output_value"
        }
      )pb");

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));

  std::map<std::string, Packet> input_side_packets_map;
  input_side_packets_map["default_input"] = default_packet;

  if (optional_packet.has_value()) {
    input_side_packets_map["optional_input"] = optional_packet.value();
  }

  MP_ASSERT_OK(graph.StartRun(input_side_packets_map));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  MP_ASSERT_OK_AND_ASSIGN(Packet output_packet,
                          graph.GetOutputSidePacket("output_value"));

  if (value_type == ValueType::kString) {
    MP_EXPECT_OK(output_packet.ValidateAsType<std::string>());
    EXPECT_EQ(output_packet.Get<std::string>(),
              expected_output.Get<std::string>());
  } else if (value_type == ValueType::kInt) {
    MP_EXPECT_OK(output_packet.ValidateAsType<int>());
    EXPECT_EQ(output_packet.Get<int>(), expected_output.Get<int>());
  } else if (value_type == ValueType::kFloat) {
    MP_EXPECT_OK(output_packet.ValidateAsType<float>());
    EXPECT_EQ(output_packet.Get<float>(), expected_output.Get<float>());
  }

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

INSTANTIATE_TEST_SUITE_P(
    DefaultSidePacketCalculator, DefaultSidePacketCalculator,
    testing::ValuesIn<ProducesExpectedOutputParam>({
        {
            .test_name = "OutputsOptionalStringWhenSpecified",
            .value_type = ValueType::kString,
            .default_packet = MakePacket<std::string>("default string"),
            .optional_packet = MakePacket<std::string>("optional string"),
            .expected_output = MakePacket<std::string>("optional string"),
        },
        {
            .test_name = "OutputsDefaultStringWhenOnlyDefaultSpecified",
            .value_type = ValueType::kString,
            .default_packet = MakePacket<std::string>("default string"),
            .optional_packet = std::nullopt,
            .expected_output = MakePacket<std::string>("default string"),
        },
        {
            .test_name = "OutputsOptionalIntWhenSpecified",
            .value_type = ValueType::kInt,
            .default_packet = MakePacket<int>(123),
            .optional_packet = MakePacket<int>(456),
            .expected_output = MakePacket<int>(456),
        },
        {
            .test_name = "OutputsDefaultIntWhenOnlyDefaultSpecified",
            .value_type = ValueType::kInt,
            .default_packet = MakePacket<int>(123),
            .optional_packet = std::nullopt,
            .expected_output = MakePacket<int>(123),
        },
        {
            .test_name = "OutputsOptionalFloatWhenSpecified",
            .value_type = ValueType::kFloat,
            .default_packet = MakePacket<float>(1.1),
            .optional_packet = MakePacket<float>(2.2),
            .expected_output = MakePacket<float>(2.2),
        },
        {
            .test_name = "OutputsDefaultFloatWhenOnlyDefaultSpecified",
            .value_type = ValueType::kFloat,
            .default_packet = MakePacket<float>(1.1),
            .optional_packet = std::nullopt,
            .expected_output = MakePacket<float>(1.1),
        },
    }),
    [](const testing::TestParamInfo<ProducesExpectedOutputParam>& info) {
      return info.param.test_name;
    });

TEST(DefaultSidePacketCalculatorTest, NoDefaultValueFails) {
  CalculatorGraphConfig graph_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "DefaultSidePacketCalculator"
          input_side_packet: "OPTIONAL_VALUE:segmentation_mask_optional"
          output_side_packet: "VALUE:segmentation_mask_enabled"
        }
      )pb");
  CalculatorGraph graph;
  auto status = graph.Initialize(graph_config);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              testing::HasSubstr("Default value must be provided"));
}

}  // namespace mediapipe
