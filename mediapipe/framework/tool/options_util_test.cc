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

#include <memory>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/message_matchers.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/testdata/night_light_calculator.pb.h"
#include "mediapipe/framework/tool/node_chain_subgraph.pb.h"
#include "mediapipe/framework/tool/options_registry.h"
#include "mediapipe/framework/tool/options_syntax_util.h"

namespace mediapipe {
namespace {

using ::mediapipe::proto_ns::FieldDescriptorProto;
using FieldType = ::mediapipe::proto_ns::FieldDescriptorProto::Type;

// A test Calculator using DeclareOptions and DefineOptions.
class NightLightCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    return mediapipe::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final {
    return mediapipe::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    return mediapipe::OkStatus();
  }

 private:
  NightLightCalculatorOptions options_;
};
REGISTER_CALCULATOR(NightLightCalculator);

// Tests for calculator and graph options.
//
class OptionsUtilTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Retrieves the description of a protobuf.
TEST_F(OptionsUtilTest, GetProtobufDescriptor) {
  const tool::Descriptor* descriptor =
      tool::OptionsRegistry::GetProtobufDescriptor(
          "mediapipe.CalculatorGraphConfig");
  EXPECT_NE(nullptr, descriptor);
}

// Shows a calculator node deriving options from graph options.
// The subgraph specifies "graph_options" as "NodeChainSubgraphOptions".
// The calculator specifies "node_options as "NightLightCalculatorOptions".
TEST_F(OptionsUtilTest, CopyLiteralOptions) {
  CalculatorGraphConfig subgraph_config;

  auto node = subgraph_config.add_node();
  *node->mutable_calculator() = "NightLightCalculator";
  *node->add_option_value() = "num_lights:options/chain_length";

  // The options framework requires at least an empty options protobuf
  // as an indication the options protobuf type expected by the node.
  NightLightCalculatorOptions node_options;
  node->add_node_options()->PackFrom(node_options);

  NodeChainSubgraphOptions options;
  options.set_chain_length(8);
  subgraph_config.add_graph_options()->PackFrom(options);
  subgraph_config.set_type("NightSubgraph");

  CalculatorGraphConfig graph_config;
  node = graph_config.add_node();
  *node->mutable_calculator() = "NightSubgraph";

  CalculatorGraph graph;
  graph_config.set_num_threads(4);
  MP_EXPECT_OK(graph.Initialize({subgraph_config, graph_config}, {}, {}));

  CalculatorGraphConfig expanded_config = graph.Config();
  expanded_config.clear_executor();
  CalculatorGraphConfig::Node actual_node;
  actual_node = expanded_config.node(0);

  CalculatorGraphConfig::Node expected_node;
  expected_node.set_name("nightsubgraph__NightLightCalculator");
  expected_node.set_calculator("NightLightCalculator");
  NightLightCalculatorOptions expected_node_options;
  expected_node_options.add_num_lights(8);
  expected_node.add_node_options()->PackFrom(expected_node_options);
  *expected_node.add_option_value() = "num_lights:options/chain_length";
  EXPECT_THAT(actual_node, EqualsProto(expected_node));

  MP_EXPECT_OK(graph.StartRun({}));
  MP_EXPECT_OK(graph.CloseAllPacketSources());
  MP_EXPECT_OK(graph.WaitUntilDone());

  // Ensure static protobuf packet registration.
  MakePacket<NodeChainSubgraphOptions>();
  MakePacket<NightLightCalculatorOptions>();
}

// Retrieves the description of a protobuf message and a nested protobuf message
// from the OptionsRegistry.
TEST_F(OptionsUtilTest, GetProtobufDescriptorRegistered) {
  const tool::Descriptor* options_descriptor =
      tool::OptionsRegistry::GetProtobufDescriptor(
          "mediapipe.NightLightCalculatorOptions");
  EXPECT_NE(nullptr, options_descriptor);
  const tool::Descriptor* bundle_descriptor =
      tool::OptionsRegistry::GetProtobufDescriptor(
          "mediapipe.NightLightCalculatorOptions.LightBundle");
  EXPECT_NE(nullptr, bundle_descriptor);
  EXPECT_EQ(options_descriptor->full_name(),
            "mediapipe.NightLightCalculatorOptions");
  const tool::FieldDescriptor* bundle_field =
      options_descriptor->FindFieldByName("bundle");
  EXPECT_EQ(bundle_field->message_type(), bundle_descriptor);
}

// Constructs the FieldPath for a nested node-option.
TEST_F(OptionsUtilTest, OptionsSyntaxUtil) {
  const tool::Descriptor* descriptor =
      tool::OptionsRegistry::GetProtobufDescriptor(
          "mediapipe.NightLightCalculatorOptions");
  std::string tag;
  tool::OptionsSyntaxUtil::FieldPath field_path;
  {
    // The default tag syntax.
    tool::OptionsSyntaxUtil syntax_util;
    tag = syntax_util.OptionFieldsTag("options/sub_options/num_lights");
    EXPECT_EQ(tag, "OPTIONS/sub_options/num_lights");
    field_path = syntax_util.OptionFieldPath(tag, descriptor);
    EXPECT_EQ(field_path.size(), 2);
    EXPECT_EQ(field_path[0].first->name(), "sub_options");
    EXPECT_EQ(field_path[1].first->name(), "num_lights");
  }
  {
    // A tag syntax with a text-coded separator.
    tool::OptionsSyntaxUtil syntax_util("OPTIONS", "options", "_Z0Z_");
    tag = syntax_util.OptionFieldsTag("options/sub_options/num_lights");
    EXPECT_EQ(tag, "OPTIONS_Z0Z_sub_options_Z0Z_num_lights");
    field_path = syntax_util.OptionFieldPath(tag, descriptor);
    EXPECT_EQ(field_path.size(), 2);
    EXPECT_EQ(field_path[0].first->name(), "sub_options");
    EXPECT_EQ(field_path[1].first->name(), "num_lights");
  }
}

}  // namespace
}  // namespace mediapipe
