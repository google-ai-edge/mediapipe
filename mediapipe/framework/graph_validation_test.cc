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
//
// Declares CalculatorGraph, which links Calculators into a directed acyclic
// graph, and allows its evaluation.

#include "mediapipe/framework/graph_validation.h"

#include <functional>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/message_matchers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/template_parser.h"

namespace mediapipe {

namespace {

// Shows validation success for a graph and a subgraph.
TEST(ValidatedGraphConfigTest, InitializeGraphFromProtos) {
  auto config_1 = ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
    type: "PassThroughGraph"
    input_stream: "INPUT:stream_1"
    output_stream: "OUTPUT:stream_2"
    node {
      calculator: "PassThroughCalculator"
      input_stream: "stream_1"   # Any Type.
      output_stream: "stream_2"  # Same as input.
    }
  )");
  auto config_2 = ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
    input_stream: "INPUT:stream_1"
    output_stream: "OUTPUT:stream_2"
    node {
      calculator: "PassThroughCalculator"
      input_stream: "stream_1"   # Any Type.
      output_stream: "stream_2"  # Same as input.
    }
    node {
      calculator: "PassThroughGraph"
      input_stream: "INPUT:stream_2"    # Any Type.
      output_stream: "OUTPUT:stream_3"  # Same as input.
    }
  )");

  GraphValidation validation_1;
  MP_EXPECT_OK(
      validation_1.Validate({config_1, config_2}, {}, {}, "PassThroughGraph"));
  CalculatorGraph graph_1;
  MP_EXPECT_OK(
      graph_1.Initialize({config_1, config_2}, {}, {}, "PassThroughGraph"));
  EXPECT_THAT(
      graph_1.Config(),
      EqualsProto(::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        type: "PassThroughGraph"
        input_stream: "INPUT:stream_1"
        output_stream: "OUTPUT:stream_2"
        node {
          calculator: "PassThroughCalculator"
          input_stream: "stream_1"
          output_stream: "stream_2"
        }
        executor {}
      )")));

  GraphValidation validation_2;
  MP_EXPECT_OK(validation_2.Validate({config_1, config_2}, {}));
  CalculatorGraph graph_2;
  MP_EXPECT_OK(graph_2.Initialize({config_1, config_2}, {}));
  EXPECT_THAT(
      graph_2.Config(),
      EqualsProto(::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: "INPUT:stream_1"
        output_stream: "OUTPUT:stream_2"
        node {
          calculator: "PassThroughCalculator"
          input_stream: "stream_1"
          output_stream: "stream_2"
        }
        node {
          calculator: "PassThroughCalculator"
          input_stream: "stream_2"
          output_stream: "stream_3"
        }
        executor {}
      )")));
}

// Shows validation failure due to an unregistered subgraph.
TEST(ValidatedGraphConfigTest, InitializeGraphFromLinker) {
  EXPECT_FALSE(SubgraphRegistry::IsRegistered("DubQuadTestSubgraph"));
  ValidatedGraphConfig builder_1;
  ::mediapipe::Status status_1 =
      builder_1.Initialize({}, {}, "DubQuadTestSubgraph");
  EXPECT_EQ(status_1.code(), ::mediapipe::StatusCode::kNotFound);
  EXPECT_THAT(status_1.message(),
              testing::HasSubstr(
                  R"(No registered object with name: DubQuadTestSubgraph)"));
}

// Shows validation success for a graph and a template subgraph.
TEST(ValidatedGraphConfigTest, InitializeTemplateFromProtos) {
  mediapipe::tool::TemplateParser::Parser parser;
  CalculatorGraphTemplate config_1;
  CHECK(parser.ParseFromString(R"(
    type: "PassThroughGraph"
    input_stream: % "INPUT:" + in_name %
    output_stream: "OUTPUT:stream_2"
    node {
      name: %in_name%
      calculator: "PassThroughCalculator"
      input_stream: %in_name%   # Any Type.
      output_stream: "stream_2"  # Same as input.
    }
  )",
                               &config_1));
  auto config_2 = ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
    input_stream: "INPUT:stream_1"
    output_stream: "OUTPUT:stream_2"
    node {
      calculator: "PassThroughCalculator"
      input_stream: "stream_1"   # Any Type.
      output_stream: "stream_2"  # Same as input.
    }
    node {
      calculator: "PassThroughGraph"
      options: {
        [mediapipe.TemplateSubgraphOptions.ext]: {
          dict: {
            arg: {
              key: "in_name"
              value: { str: "stream_8" }
            }
          }
        }
      }
      input_stream: "INPUT:stream_2"    # Any Type.
      output_stream: "OUTPUT:stream_3"  # Same as input.
    }
  )");
  auto options = ParseTextProtoOrDie<Subgraph::SubgraphOptions>(R"(
    [mediapipe.TemplateSubgraphOptions.ext]: {
      dict: {
        arg: {
          key: "in_name"
          value: { str: "stream_9" }
        }
      }
    })");

  GraphValidation validation_1;
  MP_EXPECT_OK(validation_1.Validate({config_2}, {config_1}, {},
                                     "PassThroughGraph", &options));
  CalculatorGraph graph_1;
  MP_EXPECT_OK(graph_1.Initialize({config_2}, {config_1}, {},
                                  "PassThroughGraph", &options));
  EXPECT_THAT(
      graph_1.Config(),
      EqualsProto(::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        type: "PassThroughGraph"
        input_stream: "INPUT:stream_9"
        output_stream: "OUTPUT:stream_2"
        node {
          name: "stream_9"
          calculator: "PassThroughCalculator"
          input_stream: "stream_9"
          output_stream: "stream_2"
        }
        executor {}
      )")));

  GraphValidation validation_2;
  MP_EXPECT_OK(validation_2.Validate({config_2}, {config_1}));
  CalculatorGraph graph_2;
  MP_EXPECT_OK(graph_2.Initialize({config_2}, {config_1}));
  EXPECT_THAT(
      graph_2.Config(),
      EqualsProto(::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: "INPUT:stream_1"
        output_stream: "OUTPUT:stream_2"
        node {
          calculator: "PassThroughCalculator"
          input_stream: "stream_1"
          output_stream: "stream_2"
        }
        node {
          name: "__sg0_stream_8"
          calculator: "PassThroughCalculator"
          input_stream: "stream_2"
          output_stream: "stream_3"
        }
        executor {}
      )")));
}

}  // namespace
}  // namespace mediapipe
