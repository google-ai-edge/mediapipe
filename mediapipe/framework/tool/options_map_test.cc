
// Copyright 2023 The MediaPipe Authors.
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

#include "mediapipe/framework/tool/options_map.h"

#include <unistd.h>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/testdata/night_light_calculator.pb.h"
#include "mediapipe/framework/testdata/proto3_options.pb.h"

namespace mediapipe {
namespace tool {
namespace {

TEST(OptionsMapTest, QueryNotFound) {
  CalculatorGraphConfig::Node node =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "NightLightCalculator"
        input_side_packet: "input_value"
        output_stream: "values"
      )pb");
  OptionsMap options;
  options.Initialize(node);
  EXPECT_FALSE(options.Has<mediapipe::NightLightCalculatorOptions>());
  EXPECT_FALSE(options.Has<mediapipe::Proto3Options>());
}

TEST(OptionsMapTest, Proto2QueryFound) {
  CalculatorGraphConfig::Node node =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "NightLightCalculator"
        input_side_packet: "input_value"
        output_stream: "values"
        options {
          [mediapipe.NightLightCalculatorOptions.ext] {
            base_timestamp: 123
            output_header: PASS_HEADER
            jitter: 0.123
          }
        }
      )pb");
  OptionsMap options;
  options.Initialize(node);
  EXPECT_TRUE(options.Has<mediapipe::NightLightCalculatorOptions>());
  EXPECT_EQ(
      options.Get<mediapipe::NightLightCalculatorOptions>().base_timestamp()[0],
      123);
}

TEST(MutableOptionsMapTest, InsertProto2AndQueryFound) {
  CalculatorGraphConfig::Node node =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "NightLightCalculator"
        input_side_packet: "input_value"
        output_stream: "values"
      )pb");
  MutableOptionsMap options;
  options.Initialize(node);
  EXPECT_FALSE(options.Has<mediapipe::NightLightCalculatorOptions>());
  mediapipe::NightLightCalculatorOptions night_light_options;
  night_light_options.add_base_timestamp(123);
  options.Set(night_light_options);
  EXPECT_TRUE(options.Has<mediapipe::NightLightCalculatorOptions>());
  EXPECT_EQ(
      options.Get<mediapipe::NightLightCalculatorOptions>().base_timestamp()[0],
      123);
}

TEST(OptionsMapTest, Proto3QueryFound) {
  CalculatorGraphConfig::Node node =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "NightLightCalculator"
        input_side_packet: "input_value"
        output_stream: "values"
        node_options {
          [type.googleapis.com/mediapipe.Proto3Options] { test_value: 123 }
        }
      )pb");
  OptionsMap options;
  options.Initialize(node);
  EXPECT_TRUE(options.Has<mediapipe::Proto3Options>());
  EXPECT_EQ(options.Get<mediapipe::Proto3Options>().test_value(), 123);
}

TEST(MutableOptionsMapTest, InsertProto3AndQueryFound) {
  CalculatorGraphConfig::Node node =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "NightLightCalculator"
        input_side_packet: "input_value"
        output_stream: "values"
      )pb");
  MutableOptionsMap options;
  options.Initialize(node);
  EXPECT_FALSE(options.Has<mediapipe::Proto3Options>());
  mediapipe::Proto3Options proto3_options;
  proto3_options.set_test_value(123);
  options.Set(proto3_options);
  EXPECT_TRUE(options.Has<mediapipe::Proto3Options>());
  EXPECT_EQ(options.Get<mediapipe::Proto3Options>().test_value(), 123);
}

TEST(OptionsMapTest, BothProto2AndProto3QueriesFound) {
  CalculatorGraphConfig::Node node =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "NightLightCalculator"
        input_side_packet: "input_value"
        output_stream: "values"
        options {
          [mediapipe.NightLightCalculatorOptions.ext] { jitter: 321 }
        }
        node_options {
          [type.googleapis.com/mediapipe.Proto3Options] { test_value: 123 }
        }
      )pb");
  OptionsMap options;
  options.Initialize(node);
  EXPECT_TRUE(options.Has<mediapipe::Proto3Options>());
  EXPECT_EQ(options.Get<mediapipe::Proto3Options>().test_value(), 123);
  EXPECT_TRUE(options.Has<mediapipe::NightLightCalculatorOptions>());
  EXPECT_EQ(options.Get<mediapipe::NightLightCalculatorOptions>().jitter(),
            321);
}

TEST(OptionsMapTest, PrefersOptionsOverNodeOptions) {
  CalculatorGraphConfig::Node node =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "NightLightCalculator"
        input_side_packet: "input_value"
        output_stream: "values"
        options {
          [mediapipe.NightLightCalculatorOptions.ext] { jitter: 111 }
        }
        node_options {
          [type.googleapis.com/mediapipe.NightLightCalculatorOptions] {
            jitter: 222
          }
        }
      )pb");
  OptionsMap options;
  options.Initialize(node);
  EXPECT_TRUE(options.Has<mediapipe::NightLightCalculatorOptions>());
  EXPECT_EQ(options.Get<mediapipe::NightLightCalculatorOptions>().jitter(),
            111);
}

}  // namespace
}  // namespace tool
}  // namespace mediapipe
