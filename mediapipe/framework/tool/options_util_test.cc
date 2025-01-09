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
#include <sstream>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/testdata/night_light_calculator.pb.h"
#include "mediapipe/framework/tool/node_chain_subgraph.pb.h"
#include "mediapipe/framework/tool/options_field_util.h"
#include "mediapipe/framework/tool/options_registry.h"
#include "mediapipe/framework/tool/options_syntax_util.h"

namespace mediapipe {
namespace {

using FieldType = ::mediapipe::proto_ns::FieldDescriptorProto::Type;
using ::testing::HasSubstr;

// Assigns the value from a StatusOr if avialable.
#define ASSERT_AND_ASSIGN(lhs, rexpr) \
  {                                   \
    auto statusor = (rexpr);          \
    MP_ASSERT_OK(statusor);           \
    lhs = statusor.value();           \
  }

// A test Calculator using DeclareOptions and DefineOptions.
class NightLightCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final { return absl::OkStatus(); }

  absl::Status Process(CalculatorContext* cc) final { return absl::OkStatus(); }

 private:
  NightLightCalculatorOptions options_;
};
REGISTER_CALCULATOR(NightLightCalculator);

using tool::options_field_util::FieldPath;

// Validates FieldPathEntry contents.
bool Equals(const tool::options_field_util::FieldPathEntry& entry,
            const std::string& field_name, int index,
            const std::string& extension_type) {
  const std::string& name = entry.field ? entry.field->name() : "";
  return name == field_name && entry.index == index &&
         entry.extension_type == extension_type;
}

// Serializes a MessageLite into FieldData.message_value.
FieldData AsFieldData(const proto_ns::MessageLite& message) {
  FieldData result;
  *result.mutable_message_value()->mutable_value() =
      message.SerializeAsString();
  result.mutable_message_value()->set_type_url(message.GetTypeName());
  return result;
}

// Returns the type for the root options message if specified.
std::string ExtensionType(const std::string& option_fields_tag) {
  tool::OptionsSyntaxUtil syntax_util;
  tool::options_field_util::FieldPath field_path =
      syntax_util.OptionFieldPath(option_fields_tag, nullptr);
  std::string result = !field_path.empty() ? field_path[0].extension_type : "";
  return !result.empty() ? result : "*";
}

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
  MP_ASSERT_OK(graph.Initialize({subgraph_config, graph_config}, {}, {}));

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
    EXPECT_EQ(field_path[0].field->name(), "sub_options");
    EXPECT_EQ(field_path[1].field->name(), "num_lights");
  }
  {
    // A tag syntax with a text-coded separator.
    tool::OptionsSyntaxUtil syntax_util("OPTIONS", "options", "_Z0Z_");
    tag = syntax_util.OptionFieldsTag("options/sub_options/num_lights");
    EXPECT_EQ(tag, "OPTIONS_Z0Z_sub_options_Z0Z_num_lights");
    field_path = syntax_util.OptionFieldPath(tag, descriptor);
    EXPECT_EQ(field_path.size(), 2);
    EXPECT_EQ(field_path[0].field->name(), "sub_options");
    EXPECT_EQ(field_path[1].field->name(), "num_lights");
  }
}

TEST_F(OptionsUtilTest, OptionFieldPath) {
  tool::OptionsSyntaxUtil syntax_util;
  std::vector<absl::string_view> split;
  split = syntax_util.StrSplitTags("a/graph/option:a/node/option");
  EXPECT_EQ(2, split.size());
  EXPECT_EQ(split[0], "a/graph/option");
  EXPECT_EQ(split[1], "a/node/option");
  split = syntax_util.StrSplitTags("Ext::a/graph/option:Ext::a/node/option");
  EXPECT_EQ(2, split.size());
  EXPECT_EQ(split[0], "Ext::a/graph/option");
  EXPECT_EQ(split[1], "Ext::a/node/option");

  split =
      syntax_util.StrSplitTags("chain_length:options/sub_options/num_lights");
  EXPECT_EQ(2, split.size());
  EXPECT_EQ(split[0], "chain_length");
  EXPECT_EQ(split[1], "options/sub_options/num_lights");
  const tool::Descriptor* descriptor =
      tool::OptionsRegistry::GetProtobufDescriptor(
          "mediapipe.NightLightCalculatorOptions");
  tool::options_field_util::FieldPath field_path =
      syntax_util.OptionFieldPath(split[1], descriptor);
  EXPECT_EQ(field_path.size(), 2);
  EXPECT_EQ(field_path[0].field->name(), "sub_options");
  EXPECT_EQ(field_path[1].field->name(), "num_lights");
}

TEST_F(OptionsUtilTest, FindOptionsMessage) {
  tool::OptionsSyntaxUtil syntax_util;
  std::vector<absl::string_view> split;
  split =
      syntax_util.StrSplitTags("chain_length:options/sub_options/num_lights");
  EXPECT_EQ(2, split.size());
  EXPECT_EQ(split[0], "chain_length");
  EXPECT_EQ(split[1], "options/sub_options/num_lights");
  const tool::Descriptor* descriptor =
      tool::OptionsRegistry::GetProtobufDescriptor(
          "mediapipe.NightLightCalculatorOptions");
  tool::options_field_util::FieldPath field_path =
      syntax_util.OptionFieldPath(split[1], descriptor);
  EXPECT_EQ(field_path.size(), 2);
  EXPECT_TRUE(Equals(field_path[0], "sub_options", -1, ""));
  EXPECT_TRUE(Equals(field_path[1], "num_lights", -1, ""));

  {
    // NightLightCalculatorOptions in Node.options.
    CalculatorGraphConfig::Node node;
    NightLightCalculatorOptions* options =
        node.mutable_options()->MutableExtension(
            NightLightCalculatorOptions::ext);
    options->mutable_sub_options()->add_num_lights(33);

    // Retrieve the specified option.
    FieldData node_data = AsFieldData(node);
    auto path = field_path;
    std::string node_extension_type = ExtensionType(std::string(split[1]));
    FieldData node_options;
    ASSERT_AND_ASSIGN(node_options, tool::options_field_util::GetNodeOptions(
                                        node_data, node_extension_type));
    FieldData packet_data;
    ASSERT_AND_ASSIGN(packet_data, tool::options_field_util::GetField(
                                       node_options, field_path));
    EXPECT_EQ(packet_data.value_case(), FieldData::kInt32Value);
    EXPECT_EQ(packet_data.int32_value(), 33);
  }

  {
    // NightLightCalculatorOptions in Node.node_options.
    CalculatorGraphConfig::Node node;
    NightLightCalculatorOptions options;
    options.mutable_sub_options()->add_num_lights(33);
    node.add_node_options()->PackFrom(options);

    // Retrieve the specified option.
    FieldData node_data = AsFieldData(node);
    auto path = field_path;
    std::string node_extension_type = ExtensionType(std::string(split[1]));
    FieldData node_options;
    ASSERT_AND_ASSIGN(node_options, tool::options_field_util::GetNodeOptions(
                                        node_data, node_extension_type));
    FieldData packet_data;
    ASSERT_AND_ASSIGN(packet_data, tool::options_field_util::GetField(
                                       node_options, field_path));
    EXPECT_EQ(packet_data.value_case(), FieldData::kInt32Value);
    EXPECT_EQ(packet_data.int32_value(), 33);
  }

  // TODO: Test with specified extension_type.
}

// Constructs the field path for a string of field names.
FieldPath MakeFieldPath(std::string tag, FieldData message_data) {
  tool::OptionsSyntaxUtil syntax_util;
  const tool::Descriptor* descriptor =
      tool::OptionsRegistry::GetProtobufDescriptor(
          tool::options_field_util::ParseTypeUrl(
              message_data.message_value().type_url()));
  return syntax_util.OptionFieldPath(tag, descriptor);
}

// Returns the field path addressing the entire specified field.
FieldPath EntireField(FieldPath field_path) {
  field_path.back().index = -1;
  return field_path;
}

// Converts an int to a FieldData record.
FieldData AsFieldData(int v) {
  return tool::options_field_util::AsFieldData(MakePacket<int>(v)).value();
}

// Equality comparison for field contents.
template <typename T>
absl::Status Equals(const T& v1, const T& v2) {
  RET_CHECK_EQ(v1, v2);
  return absl::OkStatus();
}

// Equality comparison for protobuf field contents.
// The generic Equals() fails because MessageLite lacks operator==().
// The protobuf comparison is performed using testing::EqualsProto.
using LightBundle = NightLightCalculatorOptions::LightBundle;
template <>
absl::Status Equals<LightBundle>(const LightBundle& v1, const LightBundle& v2) {
  std::string s_1, s_2;
  v1.SerializeToString(&s_1);
  v2.SerializeToString(&s_2);
  RET_CHECK(s_1 == s_2);
  return absl::OkStatus();
}

// Equality comparison for FieldData vectors.
template <typename FieldType>
absl::Status Equals(std::vector<FieldData> b1, std::vector<FieldData> b2) {
  using tool::options_field_util::AsPacket;
  RET_CHECK_EQ(b1.size(), b2.size());
  for (int i = 0; i < b1.size(); ++i) {
    MP_ASSIGN_OR_RETURN(Packet p1, AsPacket(b1.at(i)));
    MP_ASSIGN_OR_RETURN(Packet p2, AsPacket(b2.at(i)));
    MP_RETURN_IF_ERROR(Equals(p1.Get<FieldType>(), p2.Get<FieldType>()));
  }
  return absl::OkStatus();
}

// Unit-tests for graph options field accessors from options_field_util.
class OptionsFieldUtilTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Tests empty FieldPaths applied to empty options.
TEST_F(OptionsFieldUtilTest, EmptyFieldPaths) {
  FieldData graph_options;
  FieldData node_options;
  FieldPath graph_path;
  FieldPath node_path;
  std::vector<FieldData> packet_data;
  ASSERT_AND_ASSIGN(packet_data, GetFieldValues(graph_options, graph_path));
  MP_EXPECT_OK(MergeFieldValues(node_options, node_path, packet_data));
}

// Tests GetFieldValues applied to an int field.
TEST_F(OptionsFieldUtilTest, GetFieldValuesInt) {
  NightLightCalculatorOptions node_proto;
  node_proto.mutable_sub_options();
  node_proto.mutable_sub_options()->add_num_lights(33);
  node_proto.mutable_sub_options()->add_num_lights(44);
  FieldData node_data = tool::options_field_util::AsFieldData(node_proto);

  // Read an entire populated repeated field.
  FieldPath path = MakeFieldPath("OPTIONS/sub_options/num_lights", node_data);
  MP_EXPECT_OK(Equals<int>(GetFieldValues(node_data, path).value(),
                           {AsFieldData(33), AsFieldData(44)}));

  // Read a specific populated repeated field index.
  path = MakeFieldPath("OPTIONS/sub_options/num_lights/1", node_data);
  MP_EXPECT_OK(
      Equals<int>(GetFieldValues(node_data, path).value(), {AsFieldData(44)}));
}

// Tests GetFieldValues applied to a protobuf field.
TEST_F(OptionsFieldUtilTest, GetFieldValuesProtobuf) {
  using tool::options_field_util::AsFieldData;
  using LightBundle = NightLightCalculatorOptions::LightBundle;
  NightLightCalculatorOptions node_proto;
  node_proto.mutable_sub_options();
  node_proto.mutable_sub_options()->add_bundle();
  *node_proto.mutable_sub_options()->mutable_bundle(0)->mutable_room_id() =
      "111";
  node_proto.mutable_sub_options()
      ->mutable_bundle(0)
      ->add_room_lights()
      ->set_frame_rate(11.1);
  node_proto.mutable_sub_options()
      ->mutable_bundle(0)
      ->add_room_lights()
      ->set_frame_rate(22.1);
  FieldData node_data = AsFieldData(node_proto);

  // Read all values from a repeated protobuf field.
  LightBundle expected_proto;
  *expected_proto.mutable_room_id() = "111";
  expected_proto.add_room_lights()->set_frame_rate(11.1);
  expected_proto.add_room_lights()->set_frame_rate(22.1);
  FieldData expected_data = AsFieldData(expected_proto);
  FieldPath path = MakeFieldPath("OPTIONS/sub_options/bundle", node_data);
  MP_EXPECT_OK(Equals<LightBundle>(GetFieldValues(node_data, path).value(),
                                   {expected_data}));

  // Read a specific index from a repeated protobuf field.
  path = MakeFieldPath("OPTIONS/sub_options/bundle/0", node_data);
  MP_EXPECT_OK(Equals<LightBundle>(GetFieldValues(node_data, path).value(),
                                   {expected_data}));
}

// Tests SetFieldValues applied to an int field.
TEST_F(OptionsFieldUtilTest, SetFieldValuesInt) {
  NightLightCalculatorOptions node_proto;
  node_proto.mutable_sub_options();
  FieldData node_data = tool::options_field_util::AsFieldData(node_proto);

  // Replace an entire empty repeated field.
  FieldPath path = MakeFieldPath("OPTIONS/sub_options/num_lights", node_data);
  MP_ASSERT_OK(SetFieldValues(node_data, path, {AsFieldData(33)}));
  MP_EXPECT_OK(
      Equals<int>(GetFieldValues(node_data, path).value(), {AsFieldData(33)}));

  // Replace an entire populated repeated field.
  MP_ASSERT_OK(SetFieldValues(node_data, path, {AsFieldData(44)}));
  MP_EXPECT_OK(
      Equals<int>(GetFieldValues(node_data, path).value(), {AsFieldData(44)}));

  // Replace an entire repeated field with a new list of values.
  MP_ASSERT_OK(
      SetFieldValues(node_data, path, {AsFieldData(33), AsFieldData(44)}));
  MP_EXPECT_OK(Equals<int>(GetFieldValues(node_data, path).value(),
                           {AsFieldData(33), AsFieldData(44)}));

  // Replace a single field index with a new list of values.
  path = MakeFieldPath("OPTIONS/sub_options/num_lights/1", node_data);
  MP_ASSERT_OK(
      SetFieldValues(node_data, path, {AsFieldData(55), AsFieldData(66)}));
  MP_EXPECT_OK(
      Equals<int>(GetFieldValues(node_data, EntireField(path)).value(),
                  {AsFieldData(33), AsFieldData(55), AsFieldData(66)}));

  // Replace a single field middle index with a new list of values.
  path = MakeFieldPath("OPTIONS/sub_options/num_lights/1", node_data);
  MP_ASSERT_OK(
      SetFieldValues(node_data, path, {AsFieldData(11), AsFieldData(12)}));
  MP_EXPECT_OK(Equals<int>(
      GetFieldValues(node_data, EntireField(path)).value(),
      {AsFieldData(33), AsFieldData(11), AsFieldData(12), AsFieldData(66)}));

  // Replace field index 0 with a new value.
  path = MakeFieldPath("OPTIONS/sub_options/num_lights/0", node_data);
  MP_ASSERT_OK(SetFieldValues(node_data, path, {AsFieldData(77)}));
  MP_EXPECT_OK(Equals<int>(
      GetFieldValues(node_data, EntireField(path)).value(),
      {AsFieldData(77), AsFieldData(11), AsFieldData(12), AsFieldData(66)}));

  // Replace field index 0 with an empty list of values.
  MP_ASSERT_OK(SetFieldValues(node_data, path, {}));
  MP_EXPECT_OK(
      Equals<int>(GetFieldValues(node_data, EntireField(path)).value(),
                  {AsFieldData(11), AsFieldData(12), AsFieldData(66)}));

  // Replace an entire populated field with an empty list of values.
  path = MakeFieldPath("OPTIONS/sub_options/num_lights", node_data);
  MP_ASSERT_OK(SetFieldValues(node_data, path, {}));
  MP_ASSERT_OK(
      Equals<int>(GetFieldValues(node_data, EntireField(path)).value(), {}));

  // Replace a missing field index with new values.
  path = MakeFieldPath("OPTIONS/sub_options/num_lights/1", node_data);
  absl::Status status =
      SetFieldValues(node_data, path, {AsFieldData(55), AsFieldData(66)});
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  // TODO: status.message() appears empty on KokoroGCPDocker.
  // EXPECT_THAT(status.message(),
  //     HasSubstr("index >= 0 && index <= v.size()"));
}

// Tests SetFieldValues applied to a protobuf field.
TEST_F(OptionsFieldUtilTest, SetFieldValuesProtobuf) {
  using tool::options_field_util::AsFieldData;
  using LightBundle = NightLightCalculatorOptions::LightBundle;
  NightLightCalculatorOptions node_proto;
  node_proto.mutable_sub_options();
  FieldData node_data = AsFieldData(node_proto);

  // Replace an empty repeated protobuf field.
  LightBundle bundle_proto;
  *bundle_proto.mutable_room_id() = "222";
  bundle_proto.add_room_lights()->set_frame_rate(22.1);
  FieldData bundle_data = AsFieldData(bundle_proto);
  FieldData expected_data = bundle_data;
  FieldPath path = MakeFieldPath("OPTIONS/sub_options/bundle", node_data);
  MP_ASSERT_OK(SetFieldValues(node_data, path, {bundle_data}));
  MP_EXPECT_OK(Equals<LightBundle>(
      GetFieldValues(node_data, EntireField(path)).value(), {expected_data}));

  // Replace a populated repeated protobuf field.
  *bundle_proto.mutable_room_id() = "333";
  bundle_proto.mutable_room_lights(0)->set_frame_rate(33.1);
  bundle_data = AsFieldData(bundle_proto);
  LightBundle expected_proto;
  *expected_proto.mutable_room_id() = "333";
  expected_proto.add_room_lights()->set_frame_rate(33.1);
  expected_data = AsFieldData(expected_proto);
  MP_ASSERT_OK(SetFieldValues(node_data, path, {bundle_data}));
  MP_EXPECT_OK(Equals<LightBundle>(
      GetFieldValues(node_data, EntireField(path)).value(), {expected_data}));
}

// Tests MergeFieldValues applied to an int field.
TEST_F(OptionsFieldUtilTest, MergeFieldValuesInt) {
  NightLightCalculatorOptions node_proto;
  node_proto.mutable_sub_options();
  FieldData node_data = tool::options_field_util::AsFieldData(node_proto);

  // Replace an entire empty repeated field.
  FieldPath path = MakeFieldPath("OPTIONS/sub_options/num_lights", node_data);
  MP_ASSERT_OK(MergeFieldValues(node_data, path, {AsFieldData(33)}));
  MP_EXPECT_OK(
      Equals<int>(GetFieldValues(node_data, path).value(), {AsFieldData(33)}));

  // Replace an entire populated repeated field.
  MP_ASSERT_OK(MergeFieldValues(node_data, path, {AsFieldData(44)}));
  MP_EXPECT_OK(
      Equals<int>(GetFieldValues(node_data, path).value(), {AsFieldData(44)}));

  // Replace an entire repeated field with a new list of values.
  MP_ASSERT_OK(
      MergeFieldValues(node_data, path, {AsFieldData(33), AsFieldData(44)}));
  MP_EXPECT_OK(Equals<int>(GetFieldValues(node_data, path).value(),
                           {AsFieldData(33), AsFieldData(44)}));

  // Replace a singe field index with a new list of values.
  path = MakeFieldPath("OPTIONS/sub_options/num_lights/1", node_data);
  MP_ASSERT_OK(
      MergeFieldValues(node_data, path, {AsFieldData(55), AsFieldData(66)}));
  MP_EXPECT_OK(
      Equals<int>(GetFieldValues(node_data, EntireField(path)).value(),
                  {AsFieldData(33), AsFieldData(55), AsFieldData(66)}));

  // Replace a single field middle index with a new list of values.
  path = MakeFieldPath("OPTIONS/sub_options/num_lights/1", node_data);
  MP_ASSERT_OK(
      MergeFieldValues(node_data, path, {AsFieldData(11), AsFieldData(12)}));
  MP_EXPECT_OK(Equals<int>(
      GetFieldValues(node_data, EntireField(path)).value(),
      {AsFieldData(33), AsFieldData(11), AsFieldData(12), AsFieldData(66)}));

  // Replace field index 0 with a new value.
  path = MakeFieldPath("OPTIONS/sub_options/num_lights/0", node_data);
  MP_ASSERT_OK(MergeFieldValues(node_data, path, {AsFieldData(77)}));
  MP_EXPECT_OK(Equals<int>(
      GetFieldValues(node_data, EntireField(path)).value(),
      {AsFieldData(77), AsFieldData(11), AsFieldData(12), AsFieldData(66)}));

  // Replace field index 0 with an empty list of values.
  MP_ASSERT_OK(MergeFieldValues(node_data, path, {}));
  MP_EXPECT_OK(
      Equals<int>(GetFieldValues(node_data, EntireField(path)).value(),
                  {AsFieldData(11), AsFieldData(12), AsFieldData(66)}));

  // Replace an entire populated field with an empty list of values.
  path = MakeFieldPath("OPTIONS/sub_options/num_lights", node_data);
  MP_ASSERT_OK(MergeFieldValues(node_data, path, {}));
  MP_EXPECT_OK(
      Equals<int>(GetFieldValues(node_data, EntireField(path)).value(), {}));

  // Replace a missing field index with new values.
  path = MakeFieldPath("OPTIONS/sub_options/num_lights/1", node_data);
  absl::Status status =
      MergeFieldValues(node_data, path, {AsFieldData(55), AsFieldData(66)});
  EXPECT_EQ(status.code(), absl::StatusCode::kOutOfRange);
  EXPECT_THAT(status.message(),
              HasSubstr("Missing field value: num_lights at index: 1"));
}

// Tests MergeFieldValues applied to a protobuf field.
TEST_F(OptionsFieldUtilTest, MergeFieldValuesProtobuf) {
  using tool::options_field_util::AsFieldData;
  using LightBundle = NightLightCalculatorOptions::LightBundle;
  NightLightCalculatorOptions node_proto;
  node_proto.mutable_sub_options();
  FieldData node_data = AsFieldData(node_proto);

  // Merge an empty repeated protobuf field.
  LightBundle bundle_proto;
  *bundle_proto.mutable_room_id() = "222";
  bundle_proto.add_room_lights()->set_frame_rate(22.1);
  FieldData bundle_data = AsFieldData(bundle_proto);
  FieldData expected_data = bundle_data;
  FieldPath path = MakeFieldPath("OPTIONS/sub_options/bundle", node_data);
  MP_ASSERT_OK(MergeFieldValues(node_data, path, {bundle_data}));
  MP_EXPECT_OK(Equals<LightBundle>(
      GetFieldValues(node_data, EntireField(path)).value(), {expected_data}));

  // Merge a populated repeated protobuf field.
  // "LightBundle.room_id" merges to "333".
  // "LightBundle.room_lights" merges to {{22.1}, {33.1}}.
  *bundle_proto.mutable_room_id() = "333";
  bundle_proto.mutable_room_lights(0)->set_frame_rate(33.1);
  bundle_data = AsFieldData(bundle_proto);
  LightBundle expected_proto;
  *expected_proto.mutable_room_id() = "333";
  expected_proto.add_room_lights()->set_frame_rate(22.1);
  expected_proto.add_room_lights()->set_frame_rate(33.1);
  expected_data = AsFieldData(expected_proto);
  MP_ASSERT_OK(MergeFieldValues(node_data, path, {bundle_data}));
  MP_EXPECT_OK(Equals<LightBundle>(
      GetFieldValues(node_data, EntireField(path)).value(), {expected_data}));
}

}  // namespace
}  // namespace mediapipe
