
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

#include <memory>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/testdata/night_light_calculator.pb.h"

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
}

TEST(OptionsMapTest, QueryFound) {
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

TEST(MutableOptionsMapTest, InsertAndQueryFound) {
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

}  // namespace
}  // namespace tool
}  // namespace mediapipe
