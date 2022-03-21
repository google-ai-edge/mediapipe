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
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/template_parser.h"

namespace mediapipe {

namespace {

constexpr char kOutputTag[] = "OUTPUT";
constexpr char kEnableTag[] = "ENABLE";
constexpr char kSelectTag[] = "SELECT";
constexpr char kSideinputTag[] = "SIDEINPUT";

// Shows validation success for a graph and a subgraph.
TEST(GraphValidationTest, InitializeGraphFromProtos) {
  auto config_1 = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    type: "PassThroughGraph"
    input_stream: "INPUT:stream_1"
    output_stream: "OUTPUT:stream_2"
    node {
      calculator: "PassThroughCalculator"
      input_stream: "stream_1"   # Any Type.
      output_stream: "stream_2"  # Same as input.
    }
  )pb");
  auto config_2 = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
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
  )pb");

  GraphValidation validation_1;
  MP_EXPECT_OK(
      validation_1.Validate({config_1, config_2}, {}, {}, "PassThroughGraph"));
  CalculatorGraph graph_1;
  MP_EXPECT_OK(
      graph_1.Initialize({config_1, config_2}, {}, {}, "PassThroughGraph"));
  EXPECT_THAT(
      graph_1.Config(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        type: "PassThroughGraph"
        input_stream: "INPUT:stream_1"
        output_stream: "OUTPUT:stream_2"
        node {
          calculator: "PassThroughCalculator"
          input_stream: "stream_1"
          output_stream: "stream_2"
        }
        executor {}
      )pb")));

  GraphValidation validation_2;
  MP_EXPECT_OK(validation_2.Validate({config_1, config_2}, {}));
  CalculatorGraph graph_2;
  MP_EXPECT_OK(graph_2.Initialize({config_1, config_2}, {}));
  EXPECT_THAT(
      graph_2.Config(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "INPUT:stream_1"
        output_stream: "OUTPUT:stream_2"
        node {
          calculator: "PassThroughCalculator"
          input_stream: "stream_1"
          output_stream: "stream_2"
        }
        node {
          calculator: "PassThroughCalculator"
          name: "passthroughgraph__PassThroughCalculator"
          input_stream: "stream_2"
          output_stream: "stream_3"
        }
        executor {}
      )pb")));
}

// Shows validation failure due to an unregistered subgraph.
TEST(GraphValidationTest, InitializeGraphFromLinker) {
  EXPECT_FALSE(SubgraphRegistry::IsRegistered("DubQuadTestSubgraph"));
  ValidatedGraphConfig builder_1;
  absl::Status status_1 = builder_1.Initialize({}, {}, "DubQuadTestSubgraph");
  EXPECT_EQ(status_1.code(), absl::StatusCode::kNotFound);
  EXPECT_THAT(status_1.message(),
              testing::HasSubstr(
                  R"(No registered object with name: DubQuadTestSubgraph)"));
}

// Shows validation success for a graph and a template subgraph.
TEST(GraphValidationTest, InitializeTemplateFromProtos) {
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
  auto config_2 = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
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
  )pb");
  auto options = ParseTextProtoOrDie<Subgraph::SubgraphOptions>(R"pb(
    options: {
      [mediapipe.TemplateSubgraphOptions.ext]: {
        dict: {
          arg: {
            key: "in_name"
            value: { str: "stream_9" }
          }
        }
      }
    })pb");

  GraphValidation validation_1;
  MP_EXPECT_OK(validation_1.Validate({config_2}, {config_1}, {},
                                     "PassThroughGraph", &options));
  CalculatorGraph graph_1;
  MP_EXPECT_OK(graph_1.Initialize({config_2}, {config_1}, {},
                                  "PassThroughGraph", &options));
  EXPECT_THAT(
      graph_1.Config(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
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
      )pb")));

  GraphValidation validation_2;
  MP_EXPECT_OK(validation_2.Validate({config_2}, {config_1}));
  CalculatorGraph graph_2;
  MP_EXPECT_OK(graph_2.Initialize({config_2}, {config_1}));
  EXPECT_THAT(
      graph_2.Config(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "INPUT:stream_1"
        output_stream: "OUTPUT:stream_2"
        node {
          calculator: "PassThroughCalculator"
          input_stream: "stream_1"
          output_stream: "stream_2"
        }
        node {
          name: "passthroughgraph__stream_8"
          calculator: "PassThroughCalculator"
          input_stream: "stream_2"
          output_stream: "stream_3"
        }
        executor {}
      )pb")));
}

// Shows passing validation of optional subgraph inputs and output streams.
TEST(GraphValidationTest, OptionalSubgraphStreams) {
  // A subgraph defining two optional input streams
  // and two optional output streams.
  auto config_1 = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    type: "PassThroughGraph"
    input_stream: "INPUT:input_0"
    input_stream: "INPUT:1:input_1"
    output_stream: "OUTPUT:output_0"
    output_stream: "OUTPUT:1:output_1"
    node {
      calculator: "PassThroughCalculator"
      input_stream: "input_0"    # Any Type.
      input_stream: "input_1"    # Any Type.
      output_stream: "output_0"  # Same as input.
    }
  )pb");

  // An enclosing graph that specifies one of the two optional input streams
  // and one of the two optional output streams.
  auto config_2 = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "INPUT:foo_in"
    output_stream: "OUTPUT:foo_out"
    node {
      calculator: "PassThroughCalculator"
      input_stream: "foo_in"    # Any Type.
      output_stream: "foo_bar"  # Same as input.
    }
    node {
      calculator: "PassThroughGraph"
      input_stream: "INPUT:foo_bar"    # Any Type.
      output_stream: "OUTPUT:foo_out"  # Same as input.
    }
  )pb");

  GraphValidation validation_1;
  MP_EXPECT_OK(validation_1.Validate({config_1, config_2}, {}));
  CalculatorGraph graph_1;
  MP_EXPECT_OK(graph_1.Initialize({config_1, config_2}, {}));
  EXPECT_THAT(
      graph_1.Config(),

      // The result includes only the requested input and output streams.
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "INPUT:foo_in"
        output_stream: "OUTPUT:foo_out"
        node {
          calculator: "PassThroughCalculator"
          input_stream: "foo_in"
          output_stream: "foo_bar"
        }
        node {
          calculator: "PassThroughCalculator"
          name: "passthroughgraph__PassThroughCalculator"
          input_stream: "foo_bar"
          output_stream: "foo_out"
        }
        executor {}
      )pb")));

  MP_EXPECT_OK(graph_1.StartRun({}));
  MP_EXPECT_OK(graph_1.CloseAllPacketSources());
  MP_EXPECT_OK(graph_1.WaitUntilDone());
}

// Shows failing validation of optional subgraph inputs and output streams.
TEST(GraphValidationTest, OptionalSubgraphStreamsMismatched) {
  // A subgraph defining two optional input streams
  // and two optional output streams.
  auto config_1 = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    type: "PassThroughGraph"
    input_stream: "INPUT:input_0"
    input_stream: "INPUT:1:input_1"
    output_stream: "OUTPUT:output_0"
    output_stream: "OUTPUT:1:output_1"
    node {
      calculator: "PassThroughCalculator"
      input_stream: "input_0"    # Any Type.
      input_stream: "input_1"    # Any Type.
      output_stream: "output_0"  # Same as input.
    }
  )pb");

  // An enclosing graph that specifies one of the two optional input streams
  // and both of the two optional output streams.
  auto config_2 = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "INPUT:foo_in"
    output_stream: "OUTPUT:foo_out"
    node {
      calculator: "PassThroughCalculator"
      input_stream: "foo_in"    # Any Type.
      output_stream: "foo_bar"  # Same as input.
    }
    node {
      calculator: "PassThroughGraph"
      input_stream: "INPUT:foo_bar"    # Any Type.
      input_stream: "INPUT:1:foo_bar"  # Any Type.
      output_stream: "OUTPUT:foo_out"  # Same as input.
    }
  )pb");

  GraphValidation validation_1;
  absl::Status status = validation_1.Validate({config_1, config_2}, {});
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(status.ToString(),
              testing::HasSubstr(
                  "PassThroughCalculator must use matching tags and indexes"));
}

// A calculator that optionally accepts an input-side-packet.
class OptionalSideInputTestCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->InputSidePackets().Tag(kSideinputTag).Set<std::string>().Optional();
    cc->Inputs().Tag(kSelectTag).Set<int>().Optional();
    cc->Inputs().Tag(kEnableTag).Set<bool>().Optional();
    cc->Outputs().Tag(kOutputTag).Set<std::string>();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    std::string value("default");
    if (cc->InputSidePackets().HasTag(kSideinputTag)) {
      value = cc->InputSidePackets().Tag(kSideinputTag).Get<std::string>();
    }
    cc->Outputs()
        .Tag(kOutputTag)
        .Add(new std::string(value), cc->InputTimestamp());
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(OptionalSideInputTestCalculator);

TEST(GraphValidationTest, OptionalInputNotProvidedForSubgraphCalculator) {
  // A subgraph defining one optional input-side-packet.
  auto config_1 = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    type: "PassThroughGraph"
    input_side_packet: "INPUT:input_0"
    output_stream: "OUTPUT:output_0"
    node {
      calculator: "OptionalSideInputTestCalculator"
      input_side_packet: "SIDEINPUT:input_0"  # string
      output_stream: "OUTPUT:output_0"        # string
    }
  )pb");

  // An enclosing graph that omits the optional input-side-packet.
  auto config_2 = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_side_packet: "INPUT:foo_in"
    output_stream: "OUTPUT:foo_out"
    node {
      calculator: "PassThroughGraph"
      output_stream: "OUTPUT:foo_out"  # string
    }
  )pb");

  GraphValidation validation_1;
  MP_EXPECT_OK(validation_1.Validate({config_1, config_2}, {}));
  CalculatorGraph graph_1;
  MP_EXPECT_OK(graph_1.Initialize({config_1, config_2}, {}));
  EXPECT_THAT(
      graph_1.Config(),

      // The expanded graph omits the optional input-side-packet.
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_side_packet: "INPUT:foo_in"
        output_stream: "OUTPUT:foo_out"
        node {
          calculator: "OptionalSideInputTestCalculator"
          name: "passthroughgraph__OptionalSideInputTestCalculator"
          output_stream: "OUTPUT:foo_out"
        }
        executor {}
      )pb")));

  std::map<std::string, Packet> side_packets;
  side_packets.insert({"foo_in", mediapipe::Adopt(new std::string("input"))});
  MP_EXPECT_OK(graph_1.StartRun(side_packets));
  MP_EXPECT_OK(graph_1.CloseAllPacketSources());
  MP_EXPECT_OK(graph_1.WaitUntilDone());
}

TEST(GraphValidationTest, MultipleOptionalInputsForSubgraph) {
  // A subgraph defining one optional side-packet and two optional inputs.
  auto config_1 = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    type: "PassThroughGraph"
    input_side_packet: "INPUT:input_0"
    input_stream: "SELECT:select"
    input_stream: "ENABLE:enable"
    output_stream: "OUTPUT:output_0"
    node {
      calculator: "OptionalSideInputTestCalculator"
      input_side_packet: "SIDEINPUT:input_0"  # string
      input_stream: "SELECT:select"
      input_stream: "ENABLE:enable"
      output_stream: "OUTPUT:output_0"  # string
    }
  )pb");

  // An enclosing graph that specifies just one optional input.
  auto config_2 = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_side_packet: "INPUT:foo_in"
    input_stream: "SELECT:foo_select"
    output_stream: "OUTPUT:foo_out"
    node {
      calculator: "PassThroughGraph"
      input_stream: "SELECT:foo_select"
      output_stream: "OUTPUT:foo_out"  # string
    }
  )pb");

  GraphValidation validation_1;
  MP_ASSERT_OK(validation_1.Validate({config_1, config_2}, {}));
  CalculatorGraph graph_1;
  MP_ASSERT_OK(graph_1.Initialize({config_1, config_2}, {}));
  EXPECT_THAT(
      graph_1.Config(),

      // The expanded graph includes only the specified input, "SELECT".
      // Without the fix to RemoveIgnoredStreams(), the expanded graph
      // includes the wrong input.
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_side_packet: "INPUT:foo_in"
        input_stream: "SELECT:foo_select"
        output_stream: "OUTPUT:foo_out"
        node {
          calculator: "OptionalSideInputTestCalculator"
          name: "passthroughgraph__OptionalSideInputTestCalculator"
          input_stream: "SELECT:foo_select"
          output_stream: "OUTPUT:foo_out"
        }
        executor {}
      )pb")));

  std::map<std::string, Packet> side_packets;
  side_packets.insert({"foo_in", mediapipe::Adopt(new std::string("input"))});
  MP_EXPECT_OK(graph_1.StartRun(side_packets));
  MP_EXPECT_OK(graph_1.CloseAllPacketSources());
  MP_EXPECT_OK(graph_1.WaitUntilDone());
}

// Shows a calculator graph running with and without one optional side packet.
TEST(GraphValidationTest, OptionalInputsForGraph) {
  // A subgraph defining one optional input-side-packet.
  auto config_1 = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    type: "PassThroughGraph"
    input_side_packet: "side_input_0"
    input_stream: "stream_input_0"
    input_stream: "stream_input_1"
    output_stream: "OUTPUT:output_0"
    node {
      calculator: "OptionalSideInputTestCalculator"
      input_side_packet: "SIDEINPUT:side_input_0"
      input_stream: "SELECT:stream_input_0"
      input_stream: "ENABLE:stream_input_1"
      output_stream: "OUTPUT:output_0"
    }
  )pb");
  GraphValidation validation_1;
  MP_EXPECT_OK(validation_1.Validate({config_1}, {}));
  CalculatorGraph graph_1;
  MP_EXPECT_OK(graph_1.Initialize({config_1}, {}));
  auto out_poller = graph_1.AddOutputStreamPoller("output_0");
  MP_ASSERT_OK(out_poller);

  // Run the graph specifying the optional side packet.
  std::map<std::string, Packet> side_packets;
  side_packets.insert({"side_input_0", MakePacket<std::string>("side_in")});
  MP_EXPECT_OK(graph_1.StartRun(side_packets));
  MP_EXPECT_OK(graph_1.AddPacketToInputStream(
      "stream_input_0", MakePacket<int>(22).At(Timestamp(3000))));
  MP_EXPECT_OK(graph_1.AddPacketToInputStream(
      "stream_input_1", MakePacket<bool>(true).At(Timestamp(3000))));
  Packet out_packet, options_packet;
  EXPECT_TRUE(out_poller->Next(&out_packet));
  EXPECT_EQ(out_packet.Get<std::string>(), "side_in");
  MP_EXPECT_OK(graph_1.CloseAllPacketSources());
  MP_EXPECT_OK(graph_1.WaitUntilDone());

  // Run the graph omitting the optional inputs.
  MP_EXPECT_OK(graph_1.StartRun({}));
  MP_EXPECT_OK(graph_1.CloseInputStream("stream_input_1"));
  MP_EXPECT_OK(graph_1.AddPacketToInputStream(
      "stream_input_0", MakePacket<int>(22).At(Timestamp(3000))));
  EXPECT_TRUE(out_poller->Next(&out_packet));
  EXPECT_EQ(out_packet.Get<std::string>(), "default");
  MP_EXPECT_OK(graph_1.CloseAllPacketSources());
  MP_EXPECT_OK(graph_1.WaitUntilDone());
}

// Shows a calculator graph and DefaultSidePacketCalculator running with and
// without one optional side packet.
TEST(GraphValidationTest, DefaultOptionalInputsForGraph) {
  // A subgraph defining one optional input-side-packet.
  auto config_1 = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    type: "PassThroughGraph"
    input_side_packet: "side_input_0"
    output_side_packet: "OUTPUT:output_0"
    node {
      calculator: "ConstantSidePacketCalculator"
      options: {
        [mediapipe.ConstantSidePacketCalculatorOptions.ext]: {
          packet { int_value: 2 }
        }
      }
      output_side_packet: "PACKET:int_packet"
    }
    node {
      calculator: "DefaultSidePacketCalculator"
      input_side_packet: "OPTIONAL_VALUE:side_input_0"
      input_side_packet: "DEFAULT_VALUE:int_packet"
      output_side_packet: "VALUE:side_output_0"
    }
  )pb");
  GraphValidation validation_1;
  MP_EXPECT_OK(validation_1.Validate({config_1}, {}));
  CalculatorGraph graph_1;
  MP_EXPECT_OK(graph_1.Initialize({config_1}, {}));

  // Run the graph specifying the optional side packet.
  std::map<std::string, Packet> side_packets;
  side_packets.insert({"side_input_0", MakePacket<int>(33)});
  MP_EXPECT_OK(graph_1.StartRun(side_packets));
  MP_EXPECT_OK(graph_1.CloseAllPacketSources());
  MP_EXPECT_OK(graph_1.WaitUntilDone());

  // The specified side packet value is used.
  auto side_packet_0 = graph_1.GetOutputSidePacket("side_output_0");
  EXPECT_EQ(side_packet_0->Get<int>(), 33);

  // Run the graph omitting the optional inputs.
  MP_EXPECT_OK(graph_1.StartRun({}));
  MP_EXPECT_OK(graph_1.CloseAllPacketSources());
  MP_EXPECT_OK(graph_1.WaitUntilDone());

  // The default side packet value is used.
  side_packet_0 = graph_1.GetOutputSidePacket("side_output_0");
  EXPECT_EQ(side_packet_0->Get<int>(), 2);
}

}  // namespace
}  // namespace mediapipe
