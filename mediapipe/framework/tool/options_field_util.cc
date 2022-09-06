
#include "mediapipe/framework/tool/options_field_util.h"

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/packet_type.h"
#include "mediapipe/framework/port/advanced_proto_inc.h"
#include "mediapipe/framework/port/any_proto.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/name_util.h"
#include "mediapipe/framework/tool/proto_util_lite.h"
#include "mediapipe/framework/tool/type_util.h"

namespace mediapipe {
namespace tool {
namespace options_field_util {

using ::mediapipe::proto_ns::internal::WireFormatLite;
using FieldType = WireFormatLite::FieldType;
using ::mediapipe::proto_ns::io::ArrayInputStream;
using ::mediapipe::proto_ns::io::CodedInputStream;
using ::mediapipe::proto_ns::io::CodedOutputStream;
using ::mediapipe::proto_ns::io::StringOutputStream;

// Utility functions for OptionsFieldUtil.
namespace {

// The type name for the proto3 "Any" type.
constexpr absl::string_view kGoogleProtobufAny = "google.protobuf.Any";

// Converts a FieldDescriptor::Type to the corresponding FieldType.
FieldType AsFieldType(proto_ns::FieldDescriptorProto::Type type) {
  return static_cast<FieldType>(type);
}

// Serializes a packet value.
absl::Status WriteField(const FieldData& packet, const FieldDescriptor* field,
                        std::string* result) {
  return ProtoUtilLite::WriteValue(packet, field->type(), result);
}

// Deserializes a packet from a protobuf field.
absl::Status ReadField(absl::string_view bytes, const FieldDescriptor& field,
                       FieldData* result) {
  std::string message_type = (field.type() == WireFormatLite::TYPE_MESSAGE)
                                 ? field.message_type()->full_name()
                                 : "";
  return ProtoUtilLite::ReadValue(bytes, field.type(), message_type, result);
}

// Reads all values from a repeated field.
absl::StatusOr<std::vector<FieldData>> GetFieldValues(
    const FieldData& message_data, const FieldDescriptor& field) {
  std::vector<FieldData> result;
  const std::string& message_bytes = message_data.message_value().value();
  ProtoUtilLite::ProtoPath proto_path = {{field.number(), 0}};
  int count;
  MP_RETURN_IF_ERROR(ProtoUtilLite::GetFieldCount(message_bytes, proto_path,
                                                  field.type(), &count));
  std::vector<std::string> field_values;
  MP_RETURN_IF_ERROR(ProtoUtilLite::GetFieldRange(
      message_bytes, proto_path, count, field.type(), &field_values));
  for (int i = 0; i < field_values.size(); ++i) {
    FieldData r;
    MP_RETURN_IF_ERROR(ReadField(field_values[i], field, &r));
    result.push_back(std::move(r));
  }
  return result;
}

// Reads one value from a field.
absl::Status GetFieldValue(const FieldData& message_data,
                           const FieldPathEntry& entry, FieldData* result) {
  RET_CHECK_NE(entry.field, nullptr);
  const std::string& message_bytes = message_data.message_value().value();
  FieldType field_type = entry.field->type();
  int index = std::max(0, entry.index);
  ProtoUtilLite::ProtoPath proto_path = {{entry.field->number(), index}};
  std::vector<std::string> field_values;
  MP_RETURN_IF_ERROR(ProtoUtilLite::GetFieldRange(message_bytes, proto_path, 1,
                                                  field_type, &field_values));
  MP_RETURN_IF_ERROR(ReadField(field_values[0], *entry.field, result));
  return absl::OkStatus();
}

// Writes one value to a field.
absl::Status SetFieldValue(FieldData& result, const FieldPathEntry& entry,
                           const FieldData& value) {
  int index = std::max(0, entry.index);
  ProtoUtilLite::ProtoPath proto_path = {{entry.field->number(), index}};
  std::string* message_bytes = result.mutable_message_value()->mutable_value();
  int field_count;
  MP_RETURN_IF_ERROR(ProtoUtilLite::GetFieldCount(
      *message_bytes, proto_path, entry.field->type(), &field_count));
  if (index > field_count) {
    return absl::OutOfRangeError(
        absl::StrCat("Option field index out of range: ", index));
  }
  int replace_length = index < field_count ? 1 : 0;
  std::string field_value;
  MP_RETURN_IF_ERROR(WriteField(value, entry.field, &field_value));
  MP_RETURN_IF_ERROR(ProtoUtilLite::ReplaceFieldRange(
      message_bytes, proto_path, replace_length, entry.field->type(),
      {field_value}));
  return absl::OkStatus();
}

// Writes several values to a repeated field.
// The specified |values| replace the specified |entry| index,
// or if no index is specified all field values are replaced.
absl::Status SetFieldValues(FieldData& result, const FieldPathEntry& entry,
                            const std::vector<FieldData>& values) {
  if (entry.field == nullptr) {
    return absl::InvalidArgumentError("Field not found.");
  }
  FieldType field_type = entry.field->type();
  ProtoUtilLite::ProtoPath proto_path = {{entry.field->number(), 0}};
  std::string* message_bytes = result.mutable_message_value()->mutable_value();
  int field_count;
  MP_RETURN_IF_ERROR(ProtoUtilLite::GetFieldCount(*message_bytes, proto_path,
                                                  field_type, &field_count));
  int replace_start = 0, replace_length = field_count;
  if (entry.index > -1) {
    replace_start = entry.index;
    replace_length = 1;
  }
  std::vector<std::string> field_values(values.size());
  for (int i = 0; i < values.size(); ++i) {
    MP_RETURN_IF_ERROR(WriteField(values[i], entry.field, &field_values[i]));
  }
  proto_path = {{entry.field->number(), replace_start}};
  MP_RETURN_IF_ERROR(ProtoUtilLite::ReplaceFieldRange(
      message_bytes, proto_path, replace_length, field_type, field_values));
  return absl::OkStatus();
}

// Returns true for a field of type "google.protobuf.Any".
bool IsProtobufAny(const FieldDescriptor* field) {
  return field->type() == FieldType::TYPE_MESSAGE &&
         field->message_type()->full_name() == kGoogleProtobufAny;
}

// Returns the message FieldData from a serialized protobuf.Any.
FieldData ParseProtobufAny(const FieldData& data) {
  protobuf::Any any;
  any.ParseFromString(data.message_value().value());
  FieldData result;
  result.mutable_message_value()->set_value(std::string(any.value()));
  result.mutable_message_value()->set_type_url(any.type_url());
  return result;
}

// Returns the serialized protobuf.Any containing a message FieldData.
FieldData SerializeProtobufAny(const FieldData& data) {
  protobuf::Any any;
  any.set_value(data.message_value().value());
  any.set_type_url(data.message_value().type_url());
  FieldData result;
  result.mutable_message_value()->set_value(any.SerializeAsString());
  result.mutable_message_value()->set_type_url(TypeUrl(kGoogleProtobufAny));
  return result;
}

// Returns the field index of an extension type in a repeated field.
StatusOr<int> FindExtensionIndex(const FieldData& message_data,
                                 FieldPathEntry* entry) {
  if (entry->field == nullptr || !IsProtobufAny(entry->field)) {
    return -1;
  }
  std::string& extension_type = entry->extension_type;
  std::vector<FieldData> field_values;
  ASSIGN_OR_RETURN(field_values, GetFieldValues(message_data, *entry->field));
  for (int i = 0; i < field_values.size(); ++i) {
    FieldData extension = ParseProtobufAny(field_values[i]);
    if (extension_type == "*" ||
        ParseTypeUrl(extension.message_value().type_url()) == extension_type) {
      return i;
    }
  }
  return -1;
}

// Returns true if the value of a field is available.
bool HasField(const FieldPath& field_path, const FieldData& message_data) {
  auto value = GetField(message_data, field_path);
  return value.ok() &&
         value->value_case() != mediapipe::FieldData::VALUE_NOT_SET;
}

// Returns the extension field containing the specified extension-type.
const FieldDescriptor* FindExtensionField(const FieldData& message_data,
                                          absl::string_view extension_type) {
  std::string message_type =
      ParseTypeUrl(message_data.message_value().type_url());
  std::vector<const FieldDescriptor*> extensions;
  OptionsRegistry::FindAllExtensions(message_type, &extensions);
  for (const FieldDescriptor* extension : extensions) {
    if (extension->message_type()->full_name() == extension_type) {
      return extension;
    }
    if (extension_type == "*" && HasField({{extension, 0}}, message_data)) {
      return extension;
    }
  }
  return nullptr;
}

// Sets a protobuf in a repeated protobuf::Any field.
void SetOptionsMessage(
    const FieldData& node_options,
    proto_ns::RepeatedPtrField<mediapipe::protobuf::Any>* result) {
  protobuf::Any* options_any = nullptr;
  for (auto& any : *result) {
    if (any.type_url() == node_options.message_value().type_url()) {
      options_any = &any;
    }
  }
  if (!options_any) {
    options_any = result->Add();
    options_any->set_type_url(node_options.message_value().type_url());
  }
  *options_any->mutable_value() = node_options.message_value().value();
}

}  // anonymous namespace

// Deserializes a packet containing a MessageLite value.
absl::StatusOr<Packet> ReadMessage(const std::string& value,
                                   const std::string& type_name) {
  return packet_internal::PacketFromDynamicProto(type_name, value);
}

// Merge two options FieldData values.
absl::StatusOr<FieldData> MergeMessages(const FieldData& base,
                                        const FieldData& over) {
  FieldData result;
  absl::Status status;
  if (over.value_case() == FieldData::VALUE_NOT_SET) {
    return base;
  }
  if (base.value_case() == FieldData::VALUE_NOT_SET) {
    return over;
  }
  if (over.value_case() != base.value_case()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Cannot merge field data with data types: ", base.value_case(), ", ",
        over.value_case()));
  }
  if (over.message_value().type_url() != base.message_value().type_url()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Cannot merge field data with message types: ",
                     base.message_value().type_url(), ", ",
                     over.message_value().type_url()));
  }
  absl::Cord merged_value;
  merged_value.Append(base.message_value().value());
  merged_value.Append(over.message_value().value());
  result.mutable_message_value()->set_type_url(base.message_value().type_url());
  result.mutable_message_value()->set_value(std::string(merged_value));
  return result;
}

// Returns either the extension field or the repeated protobuf.Any field index
// holding the specified extension-type.
absl::Status FindExtension(const FieldData& message_data,
                           FieldPathEntry* entry) {
  if (entry->extension_type.empty()) {
    return absl::OkStatus();
  }

  // For repeated protobuf::Any, find the index for the extension_type.
  ASSIGN_OR_RETURN(int index, FindExtensionIndex(message_data, entry));
  if (index != -1) {
    entry->index = index;
    return absl::OkStatus();
  }

  // Returns the extension field containing the specified extension-type.
  std::string& extension_type = entry->extension_type;
  const FieldDescriptor* field =
      FindExtensionField(message_data, extension_type);
  if (field != nullptr) {
    entry->field = field;
    entry->index = 0;
    return absl::OkStatus();
  }
  return absl::NotFoundError(
      absl::StrCat("Option extension not found: ", extension_type));
}

// Return the FieldPath referencing an extension message.
FieldPath GetExtensionPath(const std::string& parent_type,
                           const std::string& extension_type,
                           const std::string& field_name,
                           bool is_protobuf_any) {
  FieldPath result;
  const tool::Descriptor* parent_descriptor =
      tool::OptionsRegistry::GetProtobufDescriptor(parent_type);
  FieldPathEntry field_entry;
  field_entry.field = parent_descriptor->FindFieldByName(field_name);
  if (is_protobuf_any) {
    field_entry.extension_type = extension_type;
    result = {std::move(field_entry)};
  } else {
    field_entry.index = 0;
    FieldPathEntry extension_entry;
    extension_entry.extension_type = extension_type;
    result = {std::move(field_entry), std::move(extension_entry)};
  }
  return result;
}

// Returns the requested options protobuf for a graph node.
absl::StatusOr<FieldData> GetNodeOptions(const FieldData& message_data,
                                         const std::string& extension_type) {
  constexpr char kOptionsName[] = "options";
  constexpr char kNodeOptionsName[] = "node_options";
  std::string parent_type = options_field_util::ParseTypeUrl(
      std::string(message_data.message_value().type_url()));
  FieldPath path;
  absl::Status status;
  path = GetExtensionPath(parent_type, extension_type, kOptionsName, false);
  auto result = GetField(message_data, path);
  if (result.ok()) {
    return result;
  }
  path = GetExtensionPath(parent_type, extension_type, kNodeOptionsName, true);
  return GetField(message_data, path);
}

// Returns the requested options protobuf for a graph.
absl::StatusOr<FieldData> GetGraphOptions(const FieldData& message_data,
                                          const std::string& extension_type) {
  constexpr char kOptionsName[] = "options";
  constexpr char kGraphOptionsName[] = "graph_options";
  std::string parent_type = options_field_util::ParseTypeUrl(
      std::string(message_data.message_value().type_url()));
  FieldPath path;
  absl::Status status;
  path = GetExtensionPath(parent_type, extension_type, kOptionsName, false);
  auto result = GetField(message_data, path);
  if (result.ok()) {
    return result;
  }
  path = GetExtensionPath(parent_type, extension_type, kGraphOptionsName, true);
  return GetField(message_data, path);
}

// Reads the FieldData values from a protobuf field.
absl::StatusOr<std::vector<FieldData>> GetFieldValues(
    const FieldData& message_data, const FieldPath& field_path) {
  std::vector<FieldData> results;
  if (field_path.empty()) {
    results.push_back(message_data);
    return results;
  }
  FieldPathEntry head = field_path.front();
  FieldPath tail = field_path;
  tail.erase(tail.begin());
  if (!head.extension_type.empty()) {
    MP_RETURN_IF_ERROR(FindExtension(message_data, &head));
  }
  RET_CHECK_NE(head.field, nullptr);
  ASSIGN_OR_RETURN(results, GetFieldValues(message_data, *head.field));
  if (IsProtobufAny(head.field)) {
    for (int i = 0; i < results.size(); ++i) {
      results[i] = ParseProtobufAny(results[i]);
    }
  }
  int index = tail.empty() ? head.index : std::max(0, head.index);
  if ((int)results.size() <= index) {
    return absl::OutOfRangeError(absl::StrCat(
        "Missing field value: ", head.field ? head.field->name() : "#",
        " at index: ", index));
  }
  if (!tail.empty()) {
    FieldData child = results.at(index);
    ASSIGN_OR_RETURN(results, GetFieldValues(child, tail));
  } else if (index > -1) {
    FieldData child = results.at(index);
    results.clear();
    results.push_back(child);
  }
  return results;
}

// Reads a FieldData value from a protobuf field.
absl::StatusOr<FieldData> GetField(const FieldData& message_data,
                                   const FieldPath& field_path) {
  std::vector<FieldData> results;
  ASSIGN_OR_RETURN(results, GetFieldValues(message_data, field_path));
  if (results.empty()) {
    FieldPathEntry tail = field_path.back();
    return absl::OutOfRangeError(absl::StrCat(
        "Missing field value: ", tail.field ? tail.field->name() : "##",
        " at index: ", tail.index));
  }
  return results[0];
}

// Writes FieldData values into protobuf field.
absl::Status SetFieldValues(FieldData& message_data,
                            const FieldPath& field_path,
                            const std::vector<FieldData>& values) {
  if (field_path.empty()) {
    if (values.empty()) {
      return absl::InvalidArgumentError("Missing field value.");
    }
    message_data = values[0];
    return absl::OkStatus();
  }

  FieldPathEntry head = field_path.front();
  FieldPath tail = field_path;
  tail.erase(tail.begin());
  if (!head.extension_type.empty()) {
    MP_RETURN_IF_ERROR(FindExtension(message_data, &head));
  }
  if (tail.empty()) {
    MP_RETURN_IF_ERROR(SetFieldValues(message_data, head, values));
    return absl::OkStatus();
  }
  FieldData child;
  MP_RETURN_IF_ERROR(GetFieldValue(message_data, head, &child));
  MP_RETURN_IF_ERROR(SetFieldValues(child, tail, values));
  if (IsProtobufAny(head.field)) {
    child = SerializeProtobufAny(child);
  }
  MP_RETURN_IF_ERROR(SetFieldValue(message_data, head, child));
  return absl::OkStatus();
}

// Writes a FieldData value into protobuf field.
absl::Status SetField(FieldData& message_data, const FieldPath& field_path,
                      const FieldData& value) {
  return SetFieldValues(message_data, field_path, {value});
}

// Merges FieldData values into nested protobuf Message.
// For each new field index, any previous value is merged with the new value.
absl::Status MergeFieldValues(FieldData& message_data,
                              const FieldPath& field_path,
                              const std::vector<FieldData>& values) {
  absl::Status status;
  FieldType field_type = field_path.empty() ? FieldType::TYPE_MESSAGE
                                            : field_path.back().field->type();
  std::vector<FieldData> results = values;
  std::vector<FieldData> prevs;
  ASSIGN_OR_RETURN(prevs, GetFieldValues(message_data, field_path));
  if (field_type == FieldType::TYPE_MESSAGE) {
    for (int i = 0; i < std::min(values.size(), prevs.size()); ++i) {
      FieldData& v = results[i];
      FieldData& b = prevs[i];
      ASSIGN_OR_RETURN(v, MergeMessages(b, v));
    }
  }
  status.Update(SetFieldValues(message_data, field_path, results));
  return status;
}

// Sets the node_options field in a Node, and clears the options field.
void SetOptionsMessage(const FieldData& node_options,
                       CalculatorGraphConfig::Node* node) {
  SetOptionsMessage(node_options, node->mutable_node_options());
  node->clear_options();
}

// Serialize a MessageLite to a FieldData.
FieldData AsFieldData(const proto_ns::MessageLite& message) {
  FieldData result;
  *result.mutable_message_value()->mutable_value() =
      message.SerializePartialAsString();
  *result.mutable_message_value()->mutable_type_url() =
      TypeUrl(message.GetTypeName());
  return result;
}

// Represents a protobuf enum value stored in a Packet.
struct ProtoEnum {
  ProtoEnum(int32 v) : value(v) {}
  int32 value;
};

absl::StatusOr<Packet> AsPacket(const FieldData& data) {
  Packet result;
  switch (data.value_case()) {
    case FieldData::ValueCase::kInt32Value:
      result = MakePacket<int32>(data.int32_value());
      break;
    case FieldData::ValueCase::kInt64Value:
      result = MakePacket<int64>(data.int64_value());
      break;
    case FieldData::ValueCase::kUint32Value:
      result = MakePacket<uint32>(data.uint32_value());
      break;
    case FieldData::ValueCase::kUint64Value:
      result = MakePacket<uint64>(data.uint64_value());
      break;
    case FieldData::ValueCase::kDoubleValue:
      result = MakePacket<double>(data.double_value());
      break;
    case FieldData::ValueCase::kFloatValue:
      result = MakePacket<float>(data.float_value());
      break;
    case FieldData::ValueCase::kBoolValue:
      result = MakePacket<bool>(data.bool_value());
      break;
    case FieldData::ValueCase::kEnumValue:
      result = MakePacket<ProtoEnum>(data.enum_value());
      break;
    case FieldData::ValueCase::kStringValue:
      result = MakePacket<std::string>(data.string_value());
      break;
    case FieldData::ValueCase::kMessageValue: {
      auto r = packet_internal::PacketFromDynamicProto(
          ParseTypeUrl(std::string(data.message_value().type_url())),
          std::string(data.message_value().value()));
      if (!r.ok()) {
        return r.status();
      }
      result = r.value();
      break;
    }
    case FieldData::VALUE_NOT_SET:
      result = Packet();
  }
  return result;
}

absl::StatusOr<FieldData> AsFieldData(Packet packet) {
  static const auto* kTypeIds = new std::map<TypeId, int32>{
      {kTypeId<int32>, WireFormatLite::CPPTYPE_INT32},
      {kTypeId<int64>, WireFormatLite::CPPTYPE_INT64},
      {kTypeId<uint32>, WireFormatLite::CPPTYPE_UINT32},
      {kTypeId<uint64>, WireFormatLite::CPPTYPE_UINT64},
      {kTypeId<double>, WireFormatLite::CPPTYPE_DOUBLE},
      {kTypeId<float>, WireFormatLite::CPPTYPE_FLOAT},
      {kTypeId<bool>, WireFormatLite::CPPTYPE_BOOL},
      {kTypeId<ProtoEnum>, WireFormatLite::CPPTYPE_ENUM},
      {kTypeId<std::string>, WireFormatLite::CPPTYPE_STRING},
  };

  FieldData result;
  if (packet.ValidateAsProtoMessageLite().ok()) {
    result.mutable_message_value()->set_value(
        packet.GetProtoMessageLite().SerializeAsString());
    result.mutable_message_value()->set_type_url(
        TypeUrl(packet.GetProtoMessageLite().GetTypeName()));
    return absl::OkStatus();
  }

  if (kTypeIds->count(packet.GetTypeId()) == 0) {
    return absl::UnimplementedError(absl::StrCat(
        "Cannot construct FieldData for: ", packet.DebugTypeName()));
  }

  switch (kTypeIds->at(packet.GetTypeId())) {
    case WireFormatLite::CPPTYPE_INT32:
      result.set_int32_value(packet.Get<int32>());
      break;
    case WireFormatLite::CPPTYPE_INT64:
      result.set_int64_value(packet.Get<int64>());
      break;
    case WireFormatLite::CPPTYPE_UINT32:
      result.set_uint32_value(packet.Get<uint32>());
      break;
    case WireFormatLite::CPPTYPE_UINT64:
      result.set_uint64_value(packet.Get<uint64>());
      break;
    case WireFormatLite::CPPTYPE_DOUBLE:
      result.set_double_value(packet.Get<double>());
      break;
    case WireFormatLite::CPPTYPE_FLOAT:
      result.set_float_value(packet.Get<float>());
      break;
    case WireFormatLite::CPPTYPE_BOOL:
      result.set_bool_value(packet.Get<bool>());
      break;
    case WireFormatLite::CPPTYPE_ENUM:
      result.set_enum_value(packet.Get<ProtoEnum>().value);
      break;
    case WireFormatLite::CPPTYPE_STRING:
      result.set_string_value(packet.Get<std::string>());
      break;
  }
  return result;
}

std::string TypeUrl(absl::string_view type_name) {
  return ProtoUtilLite::TypeUrl(type_name);
}

std::string ParseTypeUrl(absl::string_view type_url) {
  return ProtoUtilLite::ParseTypeUrl(type_url);
}

}  // namespace options_field_util
}  // namespace tool
}  // namespace mediapipe
