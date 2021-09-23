
#include "mediapipe/framework/tool/options_util.h"

#include <memory>
#include <string>
#include <variant>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
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

// Copy literal options from graph_options to node_options.
absl::Status CopyLiteralOptions(CalculatorGraphConfig::Node parent_node,
                                CalculatorGraphConfig* config) {
  Status status;
  FieldData config_options, parent_node_options, graph_options;
  status.Update(
      options_field_util::GetOptionsMessage(*config, &config_options));
  status.Update(
      options_field_util::GetOptionsMessage(parent_node, &parent_node_options));
  status.Update(options_field_util::MergeOptionsMessages(
      config_options, parent_node_options, &graph_options));
  const Descriptor* options_descriptor =
      OptionsRegistry::GetProtobufDescriptor(options_field_util::ParseTypeUrl(
          std::string(graph_options.message_value().type_url())));
  if (!options_descriptor) {
    return status;
  }

  OptionsSyntaxUtil syntax_util;
  for (auto& node : *config->mutable_node()) {
    FieldData node_data;
    status.Update(options_field_util::GetOptionsMessage(node, &node_data));
    if (!node_data.has_message_value() || node.option_value_size() == 0) {
      continue;
    }
    const Descriptor* node_options_descriptor =
        OptionsRegistry::GetProtobufDescriptor(options_field_util::ParseTypeUrl(
            std::string(node_data.message_value().type_url())));
    if (!node_options_descriptor) {
      continue;
    }
    for (const std::string& option_def : node.option_value()) {
      std::vector<std::string> tag_and_name = absl::StrSplit(option_def, ':');
      std::string graph_tag = syntax_util.OptionFieldsTag(tag_and_name[1]);
      std::string node_tag = syntax_util.OptionFieldsTag(tag_and_name[0]);
      FieldData packet_data;
      status.Update(options_field_util::GetField(
          syntax_util.OptionFieldPath(graph_tag, options_descriptor),
          graph_options, &packet_data));
      status.Update(options_field_util::MergeField(
          syntax_util.OptionFieldPath(node_tag, node_options_descriptor),
          packet_data, &node_data));
    }
    options_field_util::SetOptionsMessage(node_data, &node);
  }
  return status;
}

// Makes all configuration modifications needed for graph options.
absl::Status DefineGraphOptions(const CalculatorGraphConfig::Node& parent_node,
                                CalculatorGraphConfig* config) {
  MP_RETURN_IF_ERROR(CopyLiteralOptions(parent_node, config));
  return mediapipe::OkStatus();
}

}  // namespace tool
}  // namespace mediapipe
