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

#include "mediapipe/framework/calculator_context.h"

// TODO: Move protos in another CL after the C++ code migration.
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_context_manager.h"
#include "mediapipe/framework/calculator_state.h"
#include "mediapipe/framework/output_stream_manager.h"
#include "mediapipe/framework/output_stream_shard.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/testdata/night_light_calculator.pb.h"
#include "mediapipe/framework/testdata/sky_light_calculator.pb.h"
#include "mediapipe/framework/tool/tag_map_helper.h"

namespace mediapipe {

namespace test_ns {

std::string Proto3GraphStr() {
  static std::string kProto3GraphStr = R"(
      node {
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
      }
      node {
        calculator: "NightLightCalculator"
        input_side_packet: "input_value"
        output_stream: "values_also"
        node_options: {
          [type.googleapis.com/mediapipe.NightLightCalculatorOptions] {
            base_timestamp: 123
            output_header: PASS_HEADER
            jitter: 0.123
          }
        }
      }
      node {
        calculator: "SkyLightCalculator"
        node_options: {
          [type.googleapis.com/mediapipe.SkyLightCalculatorOptions] {
            sky_color: "sky_blue"
          }
        }
      }
      node {
        calculator: "SkyLightCalculator"
        input_side_packet: "label"
        input_stream: "values"
        output_stream: "labelled_timestamps"
        node_options: {
          [type.googleapis.com/mediapipe.SkyLightCalculatorOptions] {
            sky_color: "light_blue"
            sky_grid: 2
            sky_grid: 4
            sky_grid: 8
          }
        }
      }
      node {
        calculator: "MakeVectorCalculator"
        input_stream: "labelled_timestamps"
        output_stream: "timestamp_vectors"
      }
  )";
  return kProto3GraphStr;
}

std::unique_ptr<CalculatorState> MakeCalculatorState(
    const CalculatorGraphConfig::Node& node_config, int node_id) {
  auto result = absl::make_unique<CalculatorState>(
      "Node", node_id, "Calculator", node_config, nullptr);
  return result;
}

std::unique_ptr<CalculatorContext> MakeCalculatorContext(
    CalculatorState* calculator_state) {
  return absl::make_unique<CalculatorContext>(calculator_state,
                                              tool::CreateTagMap({}).value(),
                                              tool::CreateTagMap({}).value());
}

TEST(CalculatorTest, NodeId) {
  mediapipe::CalculatorGraphConfig config =
      ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(Proto3GraphStr());

  auto calculator_state_0 = MakeCalculatorState(config.node(0), 0);
  auto cc_0 = MakeCalculatorContext(&*calculator_state_0);
  auto calculator_state_1 = MakeCalculatorState(config.node(1), 1);
  auto cc_1 = MakeCalculatorContext(&*calculator_state_1);
  auto calculator_state_3 = MakeCalculatorState(config.node(3), 3);
  auto cc_3 = MakeCalculatorContext(&*calculator_state_3);

  EXPECT_EQ(cc_0->NodeId(), calculator_state_0->NodeId());
  EXPECT_EQ(cc_1->NodeId(), calculator_state_1->NodeId());
  EXPECT_EQ(cc_3->NodeId(), calculator_state_3->NodeId());
}

TEST(CalculatorTest, GetOptions) {
  mediapipe::CalculatorGraphConfig config =
      ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(Proto3GraphStr());

  auto calculator_state_0 = MakeCalculatorState(config.node(0), 0);
  auto cc_0 = MakeCalculatorContext(&*calculator_state_0);
  auto calculator_state_1 = MakeCalculatorState(config.node(1), 1);
  auto cc_1 = MakeCalculatorContext(&*calculator_state_1);
  auto calculator_state_3 = MakeCalculatorState(config.node(3), 3);
  auto cc_3 = MakeCalculatorContext(&*calculator_state_3);

  // Get a proto2 options extension from Node::options.
  EXPECT_EQ(cc_0->Options<NightLightCalculatorOptions>().jitter(), 0.123);

  // Get a proto2 options extension from Node::node_options.
  EXPECT_EQ(cc_1->Options<NightLightCalculatorOptions>().jitter(), 0.123);

  // Get a proto3 options protobuf::Any from Node::node_options.
  EXPECT_EQ(cc_3->Options<SkyLightCalculatorOptions>().sky_color(),
            "light_blue");
}

}  // namespace test_ns
}  // namespace mediapipe
