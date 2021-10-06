
#include "mediapipe/framework/tool/options_field_util.h"

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
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

absl::Status WriteValue(const FieldData& value, FieldType field_type,
                        std::string* field_bytes) {
  StringOutputStream sos(field_bytes);
  CodedOutputStream out(&sos);
  switch (field_type) {
    case WireFormatLite::TYPE_INT32:
      WireFormatLite::WriteInt32NoTag(value.int32_value(), &out);
      break;
    case WireFormatLite::TYPE_SINT32:
      WireFormatLite::WriteSInt32NoTag(value.int32_value(), &out);
      break;
    case WireFormatLite::TYPE_INT64:
      WireFormatLite::WriteInt64NoTag(value.int64_value(), &out);
      break;
    case WireFormatLite::TYPE_SINT64:
      WireFormatLite::WriteSInt64NoTag(value.int64_value(), &out);
      break;
    case WireFormatLite::TYPE_UINT32:
      WireFormatLite::WriteUInt32NoTag(value.uint32_value(), &out);
      break;
    case WireFormatLite::TYPE_UINT64:
      WireFormatLite::WriteUInt64NoTag(value.uint64_value(), &out);
      break;
    case WireFormatLite::TYPE_DOUBLE:
      WireFormatLite::WriteDoubleNoTag(value.uint64_value(), &out);
      break;
    case WireFormatLite::TYPE_FLOAT:
      WireFormatLite::WriteFloatNoTag(value.float_value(), &out);
      break;
    case WireFormatLite::TYPE_BOOL:
      WireFormatLite::WriteBoolNoTag(value.bool_value(), &out);
      break;
    case WireFormatLite::TYPE_ENUM:
      WireFormatLite::WriteEnumNoTag(value.enum_value(), &out);
      break;
    case WireFormatLite::TYPE_STRING:
      out.WriteString(value.string_value());
      break;
    case WireFormatLite::TYPE_MESSAGE:
      out.WriteString(value.message_value().value());
      break;
    default:
      return absl::UnimplementedError(
          absl::StrCat("Cannot write type: ", field_type));
  }
  return absl::OkStatus();
}

// Serializes a packet value.
absl::Status WriteField(const FieldData& packet, const FieldDescriptor* field,
                        std::string* result) {
  FieldType field_type = AsFieldType(field->type());
  return WriteValue(packet, field_type, result);
}

template <typename ValueT, FieldType kFieldType>
static ValueT ReadValue(absl::string_view field_bytes, absl::Status* status) {
  ArrayInputStream ais(field_bytes.data(), field_bytes.size());
  CodedInputStream input(&ais);
  ValueT result;
  if (!WireFormatLite::ReadPrimitive<ValueT, kFieldType>(&input, &result)) {
    status->Update(mediapipe::InvalidArgumentError(absl::StrCat(
        "Bad serialized value: ", MediaPipeTypeStringOrDemangled<ValueT>(),
        ".")));
  }
  return result;
}

absl::Status ReadValue(absl::string_view field_bytes, FieldType field_type,
                       absl::string_view message_type, FieldData* result) {
  absl::Status status;
  result->Clear();
  switch (field_type) {
    case WireFormatLite::TYPE_INT32:
      result->set_int32_value(
          ReadValue<int32, WireFormatLite::TYPE_INT32>(field_bytes, &status));
      break;
    case WireFormatLite::TYPE_SINT32:
      result->set_int32_value(
          ReadValue<int32, WireFormatLite::TYPE_SINT32>(field_bytes, &status));
      break;
    case WireFormatLite::TYPE_INT64:
      result->set_int64_value(
          ReadValue<int64, WireFormatLite::TYPE_INT64>(field_bytes, &status));
      break;
    case WireFormatLite::TYPE_SINT64:
      result->set_int64_value(
          ReadValue<int64, WireFormatLite::TYPE_SINT64>(field_bytes, &status));
      break;
    case WireFormatLite::TYPE_UINT32:
      result->set_uint32_value(
          ReadValue<uint32, WireFormatLite::TYPE_UINT32>(field_bytes, &status));
      break;
    case WireFormatLite::TYPE_UINT64:
      result->set_uint64_value(
          ReadValue<uint32, WireFormatLite::TYPE_UINT32>(field_bytes, &status));
      break;
    case WireFormatLite::TYPE_DOUBLE:
      result->set_double_value(
          ReadValue<double, WireFormatLite::TYPE_DOUBLE>(field_bytes, &status));
      break;
    case WireFormatLite::TYPE_FLOAT:
      result->set_float_value(
          ReadValue<float, WireFormatLite::TYPE_FLOAT>(field_bytes, &status));
      break;
    case WireFormatLite::TYPE_BOOL:
      result->set_bool_value(
          ReadValue<bool, WireFormatLite::TYPE_BOOL>(field_bytes, &status));
      break;
    case WireFormatLite::TYPE_ENUM:
      result->set_enum_value(
          ReadValue<int32, WireFormatLite::TYPE_ENUM>(field_bytes, &status));
      break;
    case WireFormatLite::TYPE_STRING:
      result->set_string_value(std::string(field_bytes));
      break;
    case WireFormatLite::TYPE_MESSAGE:
      result->mutable_message_value()->set_value(std::string(field_bytes));
      result->mutable_message_value()->set_type_url(TypeUrl(message_type));
      break;
    default:
      status = absl::UnimplementedError(
          absl::StrCat("Cannot read type: ", field_type));
      break;
  }
  return status;
}

// Deserializes a packet from a protobuf field.
absl::Status ReadField(absl::string_view bytes, const FieldDescriptor* field,
                       FieldData* result) {
  RET_CHECK_NE(field, nullptr);
  FieldType field_type = AsFieldType(field->type());
  std::string message_type = (field_type == WireFormatLite::TYPE_MESSAGE)
                                 ? field->message_type()->full_name()
                                 : "";
  return ReadValue(bytes, field_type, message_type, result);
}

// Reads all values from a repeated field.
absl::Status GetFieldValues(const FieldData& message_data,
                            const FieldDescriptor& field,
                            std::vector<FieldData>* result) {
  const std::string& message_bytes = message_data.message_value().value();
  FieldType field_type = AsFieldType(field.type());
  ProtoUtilLite proto_util;
  ProtoUtilLite::ProtoPath proto_path = {{field.number(), 0}};
  int count;
  MP_RETURN_IF_ERROR(
      proto_util.GetFieldCount(message_bytes, proto_path, field_type, &count));
  std::vector<std::string> field_values;
  MP_RETURN_IF_ERROR(proto_util.GetFieldRange(message_bytes, proto_path, count,
                                              field_type, &field_values));
  for (int i = 0; i < count; ++i) {
    FieldData r;
    MP_RETURN_IF_ERROR(ReadField(field_values[i], &field, &r));
    result->push_back(std::move(r));
  }
  return absl::OkStatus();
}

// Reads one value from a field.
absl::Status GetFieldValue(const FieldData& message_data,
                           const FieldPathEntry& entry, FieldData* result) {
  RET_CHECK_NE(entry.field, nullptr);
  const std::string& message_bytes = message_data.message_value().value();
  FieldType field_type = AsFieldType(entry.field->type());
  ProtoUtilLite proto_util;
  ProtoUtilLite::ProtoPath proto_path = {{entry.field->number(), entry.index}};
  std::vector<std::string> field_values;
  MP_RETURN_IF_ERROR(proto_util.GetFieldRange(message_bytes, proto_path, 1,
                                              field_type, &field_values));
  MP_RETURN_IF_ERROR(ReadField(field_values[0], entry.field, result));
  return absl::OkStatus();
}

// Writes one value to a field.
absl::Status SetFieldValue(const FieldPathEntry& entry, const FieldData& value,
                           FieldData* result) {
  std::vector<FieldData> field_values;
  ProtoUtilLite proto_util;
  FieldType field_type = AsFieldType(entry.field->type());
  ProtoUtilLite::ProtoPath proto_path = {{entry.field->number(), entry.index}};
  std::string* message_bytes = result->mutable_message_value()->mutable_value();
  int field_count;
  MP_RETURN_IF_ERROR(proto_util.GetFieldCount(*message_bytes, proto_path,
                                              field_type, &field_count));
  if (entry.index > field_count) {
    return absl::OutOfRangeError(
        absl::StrCat("Option field index out of range: ", entry.index));
  }
  int replace_length = entry.index < field_count ? 1 : 0;
  std::string field_value;
  MP_RETURN_IF_ERROR(WriteField(value, entry.field, &field_value));
  MP_RETURN_IF_ERROR(proto_util.ReplaceFieldRange(
      message_bytes, proto_path, replace_length, field_type, {field_value}));
  return absl::OkStatus();
}

// Returns true for a field of type "google.protobuf.Any".
bool IsProtobufAny(const FieldDescriptor* field) {
  return AsFieldType(field->type()) == FieldType::TYPE_MESSAGE &&
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
  RET_CHECK_NE(entry->field, nullptr);
  MP_RETURN_IF_ERROR(
      GetFieldValues(message_data, *entry->field, &field_values));
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
  FieldData value;
  return GetField(field_path, message_data, &value).ok() &&
         value.value_case() != mediapipe::FieldData::VALUE_NOT_SET;
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

// Returns the count of values in a repeated field.
int FieldCount(const FieldData& message_data, const FieldDescriptor* field) {
  const std::string& message_bytes = message_data.message_value().value();
  FieldType field_type = AsFieldType(field->type());
  ProtoUtilLite proto_util;
  ProtoUtilLite::ProtoPath proto_path = {{field->number(), 0}};
  int count;
  if (proto_util.GetFieldCount(message_bytes, proto_path, field_type, &count)
          .ok()) {
    return count;
  }
  return 0;
}

}  // anonymous namespace

// Deserializes a packet containing a MessageLite value.
absl::Status ReadMessage(const std::string& value, const std::string& type_name,
                         Packet* result) {
  auto packet = packet_internal::PacketFromDynamicProto(type_name, value);
  if (packet.ok()) {
    *result = *packet;
  }
  return packet.status();
}

// Merge two options FieldData values.
absl::Status MergeMessages(const FieldData& base, const FieldData& over,
                           FieldData* result) {
  absl::Status status;
  if (over.value_case() == FieldData::VALUE_NOT_SET) {
    *result = base;
    return status;
  }
  if (base.value_case() == FieldData::VALUE_NOT_SET) {
    *result = over;
    return status;
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
  result->mutable_message_value()->set_type_url(
      base.message_value().type_url());
  result->mutable_message_value()->set_value(std::string(merged_value));
  return status;
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
absl::Status GetNodeOptions(const FieldData& message_data,
                            const std::string& extension_type,
                            FieldData* result) {
  constexpr char kOptionsName[] = "options";
  constexpr char kNodeOptionsName[] = "node_options";
  std::string parent_type = options_field_util::ParseTypeUrl(
      std::string(message_data.message_value().type_url()));
  FieldPath path;
  Status status;
  path = GetExtensionPath(parent_type, extension_type, kOptionsName, false);
  status = GetField(path, message_data, result);
  if (status.ok()) {
    return status;
  }
  path = GetExtensionPath(parent_type, extension_type, kNodeOptionsName, true);
  status = GetField(path, message_data, result);
  return status;
}

// Returns the requested options protobuf for a graph.
absl::Status GetGraphOptions(const FieldData& message_data,
                             const std::string& extension_type,
                             FieldData* result) {
  constexpr char kOptionsName[] = "options";
  constexpr char kGraphOptionsName[] = "graph_options";
  std::string parent_type = options_field_util::ParseTypeUrl(
      std::string(message_data.message_value().type_url()));
  FieldPath path;
  Status status;
  path = GetExtensionPath(parent_type, extension_type, kOptionsName, false);
  status = GetField(path, message_data, result);
  if (status.ok()) {
    return status;
  }
  path = GetExtensionPath(parent_type, extension_type, kGraphOptionsName, true);
  status = GetField(path, message_data, result);
  return status;
}

// Reads a FieldData value from a protobuf field.
absl::Status GetField(const FieldPath& field_path,
                      const FieldData& message_data, FieldData* result) {
  if (field_path.empty()) {
    *result->mutable_message_value() = message_data.message_value();
    return absl::OkStatus();
  }
  FieldPathEntry head = field_path.front();
  FieldPath tail = field_path;
  tail.erase(tail.begin());
  if (!head.extension_type.empty()) {
    MP_RETURN_IF_ERROR(FindExtension(message_data, &head));
  }
  if (tail.empty() && FieldCount(message_data, head.field) == 0) {
    return absl::OkStatus();
  }
  MP_RETURN_IF_ERROR(GetFieldValue(message_data, head, result));
  if (IsProtobufAny(head.field)) {
    *result = ParseProtobufAny(*result);
  }
  if (!tail.empty()) {
    FieldData child = *result;
    MP_RETURN_IF_ERROR(GetField(tail, child, result));
  }
  return absl::OkStatus();
}

// Writes a FieldData value into protobuf field.
absl::Status SetField(const FieldPath& field_path, const FieldData& value,
                      FieldData* message_data) {
  if (field_path.empty()) {
    *message_data->mutable_message_value() = value.message_value();
    return absl::OkStatus();
  }
  FieldPathEntry head = field_path.front();
  FieldPath tail = field_path;
  tail.erase(tail.begin());
  if (!head.extension_type.empty()) {
    MP_RETURN_IF_ERROR(FindExtension(*message_data, &head));
  }
  if (tail.empty()) {
    MP_RETURN_IF_ERROR(SetFieldValue(head, value, message_data));
  } else {
    FieldData child;
    MP_RETURN_IF_ERROR(GetFieldValue(*message_data, head, &child));
    MP_RETURN_IF_ERROR(SetField(tail, value, &child));
    if (IsProtobufAny(head.field)) {
      child = SerializeProtobufAny(child);
    }
    MP_RETURN_IF_ERROR(SetFieldValue(head, child, message_data));
  }
  return absl::OkStatus();
}

// Merges a packet value into nested protobuf Message.
absl::Status MergeField(const FieldPath& field_path, const FieldData& value,
                        FieldData* message_data) {
  absl::Status status;
  FieldType field_type = field_path.empty()
                             ? FieldType::TYPE_MESSAGE
                             : AsFieldType(field_path.back().field->type());
  std::string message_type =
      (value.has_message_value())
          ? ParseTypeUrl(std::string(value.message_value().type_url()))
          : "";
  FieldData v = value;
  if (field_type == FieldType::TYPE_MESSAGE) {
    FieldData b;
    status.Update(GetField(field_path, *message_data, &b));
    status.Update(MergeMessages(b, v, &v));
  }
  status.Update(SetField(field_path, v, message_data));
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

absl::Status AsPacket(const FieldData& data, Packet* result) {
  switch (data.value_case()) {
    case FieldData::ValueCase::kInt32Value:
      *result = MakePacket<int32>(data.int32_value());
      break;
    case FieldData::ValueCase::kInt64Value:
      *result = MakePacket<int64>(data.int64_value());
      break;
    case FieldData::ValueCase::kUint32Value:
      *result = MakePacket<uint32>(data.uint32_value());
      break;
    case FieldData::ValueCase::kUint64Value:
      *result = MakePacket<uint64>(data.uint64_value());
      break;
    case FieldData::ValueCase::kDoubleValue:
      *result = MakePacket<double>(data.double_value());
      break;
    case FieldData::ValueCase::kFloatValue:
      *result = MakePacket<float>(data.float_value());
      break;
    case FieldData::ValueCase::kBoolValue:
      *result = MakePacket<bool>(data.bool_value());
      break;
    case FieldData::ValueCase::kEnumValue:
      *result = MakePacket<ProtoEnum>(data.enum_value());
      break;
    case FieldData::ValueCase::kStringValue:
      *result = MakePacket<std::string>(data.string_value());
      break;
    case FieldData::ValueCase::kMessageValue: {
      auto r = packet_internal::PacketFromDynamicProto(
          ParseTypeUrl(std::string(data.message_value().type_url())),
          std::string(data.message_value().value()));
      if (!r.ok()) {
        return r.status();
      }
      *result = r.value();
      break;
    }
    case FieldData::VALUE_NOT_SET:
      *result = Packet();
  }
  return absl::OkStatus();
}

absl::Status AsFieldData(Packet packet, FieldData* result) {
  static const auto* kTypeIds = new std::map<size_t, int32>{
      {tool::GetTypeHash<int32>(), WireFormatLite::CPPTYPE_INT32},
      {tool::GetTypeHash<int64>(), WireFormatLite::CPPTYPE_INT64},
      {tool::GetTypeHash<uint32>(), WireFormatLite::CPPTYPE_UINT32},
      {tool::GetTypeHash<uint64>(), WireFormatLite::CPPTYPE_UINT64},
      {tool::GetTypeHash<double>(), WireFormatLite::CPPTYPE_DOUBLE},
      {tool::GetTypeHash<float>(), WireFormatLite::CPPTYPE_FLOAT},
      {tool::GetTypeHash<bool>(), WireFormatLite::CPPTYPE_BOOL},
      {tool::GetTypeHash<ProtoEnum>(), WireFormatLite::CPPTYPE_ENUM},
      {tool::GetTypeHash<std::string>(), WireFormatLite::CPPTYPE_STRING},
  };

  if (packet.ValidateAsProtoMessageLite().ok()) {
    result->mutable_message_value()->set_value(
        packet.GetProtoMessageLite().SerializeAsString());
    result->mutable_message_value()->set_type_url(
        TypeUrl(packet.GetProtoMessageLite().GetTypeName()));
    return absl::OkStatus();
  }

  if (kTypeIds->count(packet.GetTypeId()) == 0) {
    return absl::UnimplementedError(absl::StrCat(
        "Cannot construct FieldData for: ", packet.DebugTypeName()));
  }

  switch (kTypeIds->at(packet.GetTypeId())) {
    case WireFormatLite::CPPTYPE_INT32:
      result->set_int32_value(packet.Get<int32>());
      break;
    case WireFormatLite::CPPTYPE_INT64:
      result->set_int64_value(packet.Get<int64>());
      break;
    case WireFormatLite::CPPTYPE_UINT32:
      result->set_uint32_value(packet.Get<uint32>());
      break;
    case WireFormatLite::CPPTYPE_UINT64:
      result->set_uint64_value(packet.Get<uint64>());
      break;
    case WireFormatLite::CPPTYPE_DOUBLE:
      result->set_double_value(packet.Get<double>());
      break;
    case WireFormatLite::CPPTYPE_FLOAT:
      result->set_float_value(packet.Get<float>());
      break;
    case WireFormatLite::CPPTYPE_BOOL:
      result->set_bool_value(packet.Get<bool>());
      break;
    case WireFormatLite::CPPTYPE_ENUM:
      result->set_enum_value(packet.Get<ProtoEnum>().value);
      break;
    case WireFormatLite::CPPTYPE_STRING:
      result->set_string_value(packet.Get<std::string>());
      break;
  }
  return absl::OkStatus();
}

std::string TypeUrl(absl::string_view type_name) {
  constexpr std::string_view kTypeUrlPrefix = "type.googleapis.com/";
  return absl::StrCat(std::string(kTypeUrlPrefix), std::string(type_name));
}

std::string ParseTypeUrl(absl::string_view type_url) {
  constexpr std::string_view kTypeUrlPrefix = "type.googleapis.com/";
  if (std::string(type_url).rfind(kTypeUrlPrefix, 0) == 0) {
    return std::string(
        type_url.substr(kTypeUrlPrefix.length(), std::string::npos));
  }
  return std::string(type_url);
}

}  // namespace options_field_util
}  // namespace tool
}  // namespace mediapipe
