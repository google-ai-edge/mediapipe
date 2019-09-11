// Copyright 2019 The MediaPipe Authors.
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
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"  // NOLINT

namespace mediapipe {

TEST(QuantizeFloatVectorCalculatorTest, WrongConfig) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
        calculator: "QuantizeFloatVectorCalculator"
        input_stream: "FLOAT_VECTOR:float_vector"
        output_stream: "ENCODED:encoded"
        options {
          [mediapipe.QuantizeFloatVectorCalculatorOptions.ext]: {
            min_quantized_value: 1
          }
        }
      )");
  CalculatorRunner runner(node_config);
  std::vector<float> empty_vector;
  runner.MutableInputs()
      ->Tag("FLOAT_VECTOR")
      .packets.push_back(
          MakePacket<std::vector<float>>(empty_vector).At(Timestamp(0)));
  auto status = runner.Run();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      testing::HasSubstr(
          "Both max_quantized_value and min_quantized_value must be provided"));
}

TEST(QuantizeFloatVectorCalculatorTest, WrongConfig2) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
        calculator: "QuantizeFloatVectorCalculator"
        input_stream: "FLOAT_VECTOR:float_vector"
        output_stream: "ENCODED:encoded"
        options {
          [mediapipe.QuantizeFloatVectorCalculatorOptions.ext]: {
            max_quantized_value: -1
            min_quantized_value: 1
          }
        }
      )");
  CalculatorRunner runner(node_config);
  std::vector<float> empty_vector;
  runner.MutableInputs()
      ->Tag("FLOAT_VECTOR")
      .packets.push_back(
          MakePacket<std::vector<float>>(empty_vector).At(Timestamp(0)));
  auto status = runner.Run();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      testing::HasSubstr(
          "max_quantized_value must be greater than min_quantized_value"));
}

TEST(QuantizeFloatVectorCalculatorTest, WrongConfig3) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
        calculator: "QuantizeFloatVectorCalculator"
        input_stream: "FLOAT_VECTOR:float_vector"
        output_stream: "ENCODED:encoded"
        options {
          [mediapipe.QuantizeFloatVectorCalculatorOptions.ext]: {
            max_quantized_value: 1
            min_quantized_value: 1
          }
        }
      )");
  CalculatorRunner runner(node_config);
  std::vector<float> empty_vector;
  runner.MutableInputs()
      ->Tag("FLOAT_VECTOR")
      .packets.push_back(
          MakePacket<std::vector<float>>(empty_vector).At(Timestamp(0)));
  auto status = runner.Run();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      testing::HasSubstr(
          "max_quantized_value must be greater than min_quantized_value"));
}

TEST(QuantizeFloatVectorCalculatorTest, TestEmptyVector) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
        calculator: "QuantizeFloatVectorCalculator"
        input_stream: "FLOAT_VECTOR:float_vector"
        output_stream: "ENCODED:encoded"
        options {
          [mediapipe.QuantizeFloatVectorCalculatorOptions.ext]: {
            max_quantized_value: 1
            min_quantized_value: -1
          }
        }
      )");
  CalculatorRunner runner(node_config);
  std::vector<float> empty_vector;
  runner.MutableInputs()
      ->Tag("FLOAT_VECTOR")
      .packets.push_back(
          MakePacket<std::vector<float>>(empty_vector).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const std::vector<Packet>& outputs = runner.Outputs().Tag("ENCODED").packets;
  EXPECT_EQ(1, outputs.size());
  EXPECT_TRUE(outputs[0].Get<std::string>().empty());
  EXPECT_EQ(Timestamp(0), outputs[0].Timestamp());
}

TEST(QuantizeFloatVectorCalculatorTest, TestNonEmptyVector) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
        calculator: "QuantizeFloatVectorCalculator"
        input_stream: "FLOAT_VECTOR:float_vector"
        output_stream: "ENCODED:encoded"
        options {
          [mediapipe.QuantizeFloatVectorCalculatorOptions.ext]: {
            max_quantized_value: 64
            min_quantized_value: -64
          }
        }
      )");
  CalculatorRunner runner(node_config);
  std::vector<float> vector = {0.0f, -64.0f, 64.0f, -32.0f, 32.0f};
  runner.MutableInputs()
      ->Tag("FLOAT_VECTOR")
      .packets.push_back(
          MakePacket<std::vector<float>>(vector).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const std::vector<Packet>& outputs = runner.Outputs().Tag("ENCODED").packets;
  EXPECT_EQ(1, outputs.size());
  const std::string& result = outputs[0].Get<std::string>();
  ASSERT_FALSE(result.empty());
  EXPECT_EQ(5, result.size());
  // 127
  EXPECT_EQ('\x7F', result.c_str()[0]);
  // 0
  EXPECT_EQ('\0', result.c_str()[1]);
  // 255
  EXPECT_EQ('\xFF', result.c_str()[2]);
  // 63
  EXPECT_EQ('\x3F', result.c_str()[3]);
  // 191
  EXPECT_EQ('\xBF', result.c_str()[4]);
  EXPECT_EQ(Timestamp(0), outputs[0].Timestamp());
}

TEST(QuantizeFloatVectorCalculatorTest, TestSaturation) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
        calculator: "QuantizeFloatVectorCalculator"
        input_stream: "FLOAT_VECTOR:float_vector"
        output_stream: "ENCODED:encoded"
        options {
          [mediapipe.QuantizeFloatVectorCalculatorOptions.ext]: {
            max_quantized_value: 64
            min_quantized_value: -64
          }
        }
      )");
  CalculatorRunner runner(node_config);
  std::vector<float> vector = {-65.0f, 65.0f};
  runner.MutableInputs()
      ->Tag("FLOAT_VECTOR")
      .packets.push_back(
          MakePacket<std::vector<float>>(vector).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const std::vector<Packet>& outputs = runner.Outputs().Tag("ENCODED").packets;
  EXPECT_EQ(1, outputs.size());
  const std::string& result = outputs[0].Get<std::string>();
  ASSERT_FALSE(result.empty());
  EXPECT_EQ(2, result.size());
  // 0
  EXPECT_EQ('\0', result.c_str()[0]);
  // 255
  EXPECT_EQ('\xFF', result.c_str()[1]);
  EXPECT_EQ(Timestamp(0), outputs[0].Timestamp());
}

}  // namespace mediapipe
