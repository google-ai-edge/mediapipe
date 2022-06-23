
#include "mediapipe/framework/tool/options_util.h"

#include <memory>
#include <string>
#include <variant>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/input_stream_shard.h"
#include "mediapipe/framework/output_side_packet.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/packet_set.h"
#include "mediapipe/framework/packet_type.h"
#include "mediapipe/framework/port/advanced_proto_inc.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/name_util.h"
#include "mediapipe/framework/tool/options_field_util.h"
#include "mediapipe/framework/tool/options_registry.h"
#include "mediapipe/framework/tool/options_syntax_util.h"
#include "mediapipe/framework/tool/proto_util_lite.h"

namespace mediapipe {
namespace tool {

using options_field_util::FieldPath;
using options_field_util::GetGraphOptions;
using options_field_util::GetNodeOptions;
using options_field_util::MergeFieldValues;
using options_field_util::MergeMessages;

// Returns the type for the root options message if specified.
std::string ExtensionType(const std::string& option_fields_tag) {
  OptionsSyntaxUtil syntax_util;
  options_field_util::FieldPath field_path =
      syntax_util.OptionFieldPath(option_fields_tag, nullptr);
  std::string result = !field_path.empty() ? field_path[0].extension_type : "";
  return !result.empty() ? result : "*";
}

// Constructs a FieldPath for field names starting at a message type.
FieldPath GetPath(const std::string& path_tag,
                  const std::string& message_type) {
  OptionsSyntaxUtil syntax_util;
  const Descriptor* descriptor =
      OptionsRegistry::GetProtobufDescriptor(message_type);
  return syntax_util.OptionFieldPath(path_tag, descriptor);
}

// Returns the message type for a FieldData.
std::string MessageType(FieldData message) {
  return options_field_util::ParseTypeUrl(
      std::string(message.message_value().type_url()));
}

// Assigns the value from a StatusOr if avialable.
#define ASSIGN_IF_OK(lhs, rexpr) \
  {                              \
    auto statusor = (rexpr);     \
    if (statusor.ok()) {         \
      lhs = statusor.value();    \
    }                            \
  }

// Copy literal options from graph_options to node_options.
absl::Status CopyLiteralOptions(CalculatorGraphConfig::Node parent_node,
                                CalculatorGraphConfig* config) {
  absl::Status status;
  FieldData graph_data = options_field_util::AsFieldData(*config);
  FieldData parent_data = options_field_util::AsFieldData(parent_node);

  OptionsSyntaxUtil syntax_util;
  for (auto& node : *config->mutable_node()) {
    for (const std::string& option_def : node.option_value()) {
      FieldData node_data = options_field_util::AsFieldData(node);

      std::vector<absl::string_view> tag_and_name =
          syntax_util.StrSplitTags(option_def);
      std::string graph_tag = syntax_util.OptionFieldsTag(tag_and_name[1]);
      std::string graph_extension_type = ExtensionType(graph_tag);
      std::string node_tag = syntax_util.OptionFieldsTag(tag_and_name[0]);
      std::string node_extension_type = ExtensionType(node_tag);
      FieldData graph_options;
      ASSIGN_IF_OK(graph_options,
                   GetGraphOptions(graph_data, graph_extension_type));
      FieldData parent_options;
      ASSIGN_IF_OK(parent_options,
                   GetNodeOptions(parent_data, graph_extension_type));
      ASSIGN_OR_RETURN(graph_options,
                       MergeMessages(graph_options, parent_options));
      FieldData node_options;
      ASSIGN_OR_RETURN(node_options,
                       GetNodeOptions(node_data, node_extension_type));
      if (!node_options.has_message_value() ||
          !graph_options.has_message_value()) {
        continue;
      }
      FieldPath graph_path = GetPath(graph_tag, MessageType(graph_options));
      FieldPath node_path = GetPath(node_tag, MessageType(node_options));
      std::vector<FieldData> packet_data;
      ASSIGN_OR_RETURN(packet_data, GetFieldValues(graph_options, graph_path));
      MP_RETURN_IF_ERROR(
          MergeFieldValues(node_options, node_path, packet_data));
      options_field_util::SetOptionsMessage(node_options, &node);
    }
    node.clear_option_value();
  }
  return status;
}

// Makes all configuration modifications needed for graph options.
absl::Status DefineGraphOptions(const CalculatorGraphConfig::Node& parent_node,
                                CalculatorGraphConfig* config) {
  MP_RETURN_IF_ERROR(CopyLiteralOptions(parent_node, config));
  return absl::OkStatus();
}

}  // namespace tool
}  // namespace mediapipe
