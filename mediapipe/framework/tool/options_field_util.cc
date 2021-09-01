
#include "mediapipe/framework/tool/options_field_util.h"

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/packet_type.h"
#include "mediapipe/framework/port/advanced_proto_inc.h"
#include "mediapipe/framework/port/any_proto.h"
#include "mediapipe/framework/port/canonical_errors.h"
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
  return mediapipe::OkStatus();
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
  FieldType field_type = AsFieldType(field->type());
  std::string message_type = (field_type == WireFormatLite::TYPE_MESSAGE)
                                 ? field->message_type()->full_name()
                                 : "";
  return ReadValue(bytes, field_type, message_type, result);
}

// Converts a chain of fields and indexes into field-numbers and indexes.
ProtoUtilLite::ProtoPath AsProtoPath(const FieldPath& field_path) {
  ProtoUtilLite::ProtoPath result;
  for (auto field : field_path) {
    result.push_back({field.first->number(), field.second});
  }
  return result;
}

// Returns the options protobuf for a subgraph.
// TODO: Ensure that this works with multiple options protobufs.
absl::Status GetOptionsMessage(
    const proto_ns::RepeatedPtrField<mediapipe::protobuf::Any>& options_any,
    const proto_ns::MessageLite& options_ext, FieldData* result) {
  // Read the "graph_options" or "node_options" field.
  for (const auto& options : options_any) {
    if (options.type_url().empty()) {
      continue;
    }
    result->mutable_message_value()->set_type_url(options.type_url());
    result->mutable_message_value()->set_value(std::string(options.value()));
    return mediapipe::OkStatus();
  }

  // Read the "options" field.
  FieldData message_data;
  *message_data.mutable_message_value()->mutable_value() =
      options_ext.SerializeAsString();
  message_data.mutable_message_value()->set_type_url(options_ext.GetTypeName());
  std::vector<const FieldDescriptor*> ext_fields;
  OptionsRegistry::FindAllExtensions(options_ext.GetTypeName(), &ext_fields);
  for (auto ext_field : ext_fields) {
    absl::Status status = GetField({{ext_field, 0}}, message_data, result);
    if (!status.ok()) {
      return status;
    }
    if (result->has_message_value()) {
      return status;
    }
  }
  return mediapipe::OkStatus();
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
absl::Status ReadMessage(const std::string& value, const std::string& type_name,
                         Packet* result) {
  auto packet = packet_internal::PacketFromDynamicProto(type_name, value);
  if (packet.ok()) {
    *result = *packet;
  }
  return packet.status();
}

// Merge two options FieldData values.
absl::Status MergeOptionsMessages(const FieldData& base, const FieldData& over,
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

// Writes a FieldData value into protobuf field.
absl::Status SetField(const FieldPath& field_path, const FieldData& value,
                      FieldData* message_data) {
  if (field_path.empty()) {
    *message_data->mutable_message_value() = value.message_value();
    return mediapipe::OkStatus();
  }
  ProtoUtilLite proto_util;
  const FieldDescriptor* field = field_path.back().first;
  FieldType field_type = AsFieldType(field->type());
  std::string field_value;
  MP_RETURN_IF_ERROR(WriteField(value, field, &field_value));
  ProtoUtilLite::ProtoPath proto_path = AsProtoPath(field_path);
  std::string* message_bytes =
      message_data->mutable_message_value()->mutable_value();
  int field_count;
  MP_RETURN_IF_ERROR(proto_util.GetFieldCount(*message_bytes, proto_path,
                                              field_type, &field_count));
  MP_RETURN_IF_ERROR(
      proto_util.ReplaceFieldRange(message_bytes, AsProtoPath(field_path),
                                   field_count, field_type, {field_value}));
  return mediapipe::OkStatus();
}

// Merges a packet value into nested protobuf Message.
absl::Status MergeField(const FieldPath& field_path, const FieldData& value,
                        FieldData* message_data) {
  absl::Status status;
  FieldType field_type = field_path.empty()
                             ? FieldType::TYPE_MESSAGE
                             : AsFieldType(field_path.back().first->type());
  std::string message_type =
      (value.has_message_value())
          ? ParseTypeUrl(std::string(value.message_value().type_url()))
          : "";
  FieldData v = value;
  if (field_type == FieldType::TYPE_MESSAGE) {
    FieldData b;
    status.Update(GetField(field_path, *message_data, &b));
    status.Update(MergeOptionsMessages(b, v, &v));
  }
  status.Update(SetField(field_path, v, message_data));
  return status;
}

// Reads a packet value from a protobuf field.
absl::Status GetField(const FieldPath& field_path,
                      const FieldData& message_data, FieldData* result) {
  if (field_path.empty()) {
    *result->mutable_message_value() = message_data.message_value();
    return mediapipe::OkStatus();
  }
  ProtoUtilLite proto_util;
  const FieldDescriptor* field = field_path.back().first;
  FieldType field_type = AsFieldType(field->type());
  std::vector<std::string> field_values;
  ProtoUtilLite::ProtoPath proto_path = AsProtoPath(field_path);
  const std::string& message_bytes = message_data.message_value().value();
  int field_count;
  MP_RETURN_IF_ERROR(proto_util.GetFieldCount(message_bytes, proto_path,
                                              field_type, &field_count));
  if (field_count == 0) {
    return mediapipe::OkStatus();
  }
  MP_RETURN_IF_ERROR(proto_util.GetFieldRange(message_bytes, proto_path, 1,
                                              field_type, &field_values));
  MP_RETURN_IF_ERROR(ReadField(field_values.front(), field, result));
  return mediapipe::OkStatus();
}

// Returns the options protobuf for a graph.
absl::Status GetOptionsMessage(const CalculatorGraphConfig& config,
                               FieldData* result) {
  return GetOptionsMessage(config.graph_options(), config.options(), result);
}

// Returns the options protobuf for a node.
absl::Status GetOptionsMessage(const CalculatorGraphConfig::Node& node,
                               FieldData* result) {
  return GetOptionsMessage(node.node_options(), node.options(), result);
}

// Sets the node_options field in a Node, and clears the options field.
void SetOptionsMessage(const FieldData& node_options,
                       CalculatorGraphConfig::Node* node) {
  SetOptionsMessage(node_options, node->mutable_node_options());
  node->clear_options();
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
  return mediapipe::OkStatus();
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
    return mediapipe::OkStatus();
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
  return mediapipe::OkStatus();
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
