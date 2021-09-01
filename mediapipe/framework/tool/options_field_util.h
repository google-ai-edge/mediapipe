#ifndef MEDIAPIPE_FRAMEWORK_TOOL_OPTIONS_FIELD_UTIL_H_
#define MEDIAPIPE_FRAMEWORK_TOOL_OPTIONS_FIELD_UTIL_H_

#include <string>
#include <vector>

#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/packet_type.h"
#include "mediapipe/framework/port/advanced_proto_inc.h"
#include "mediapipe/framework/port/proto_ns.h"
#include "mediapipe/framework/tool/field_data.pb.h"
#include "mediapipe/framework/tool/options_registry.h"

namespace mediapipe {

namespace tool {

// Utility to read and write Packet data from protobuf fields.
namespace options_field_util {

// A chain of nested fields and indexes.
using FieldPath = std::vector<std::pair<const FieldDescriptor*, int>>;

// Writes a field value into protobuf field.
absl::Status SetField(const FieldPath& field_path, const FieldData& value,
                      FieldData* message_data);

// Reads a field value from a protobuf field.
absl::Status GetField(const FieldPath& field_path,
                      const FieldData& message_data, FieldData* result);

// Merges a field value into nested protobuf Message.
absl::Status MergeField(const FieldPath& field_path, const FieldData& value,
                        FieldData* message_data);

// Deserializes a packet containing a MessageLite value.
absl::Status ReadMessage(const std::string& value, const std::string& type_name,
                         Packet* result);

// Merge two options protobuf field values.
absl::Status MergeOptionsMessages(const FieldData& base, const FieldData& over,
                                  FieldData* result);

// Returns the options protobuf for a graph.
absl::Status GetOptionsMessage(const CalculatorGraphConfig& config,
                               FieldData* result);

// Returns the options protobuf for a node.
absl::Status GetOptionsMessage(const CalculatorGraphConfig::Node& node,
                               FieldData* result);

// Sets the node_options field in a Node, and clears the options field.
void SetOptionsMessage(const FieldData& node_options,
                       CalculatorGraphConfig::Node* node);

// Constructs a Packet for a FieldData proto.
absl::Status AsPacket(const FieldData& data, Packet* result);

// Constructs a FieldData proto for a Packet.
absl::Status AsFieldData(Packet packet, FieldData* result);

// Returns the protobuf type-url for a protobuf type-name.
std::string TypeUrl(absl::string_view type_name);

// Returns the protobuf type-name for a protobuf type-url.
std::string ParseTypeUrl(absl::string_view type_url);

}  // namespace options_field_util
}  // namespace tool
}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_TOOL_OPTIONS_FIELD_UTIL_H_
