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

#include "mediapipe/framework/tool/proto_util_lite.h"

#include <tuple>

#include "absl/log/absl_check.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/framework/tool/field_data.pb.h"
#include "mediapipe/framework/type_map.h"

#define RET_CHECK_NO_LOG(cond) RET_CHECK(cond).SetNoLogging()

namespace mediapipe {
namespace tool {

using proto_ns::io::ArrayInputStream;
using proto_ns::io::CodedInputStream;
using proto_ns::io::CodedOutputStream;
using proto_ns::io::StringOutputStream;
using WireFormatLite = ProtoUtilLite::WireFormatLite;
using FieldAccess = ProtoUtilLite::FieldAccess;
using FieldValue = ProtoUtilLite::FieldValue;
using ProtoPath = ProtoUtilLite::ProtoPath;
using FieldType = ProtoUtilLite::FieldType;
using mediapipe::FieldData;

// Returns true if a wire type includes a length indicator.
bool IsLengthDelimited(WireFormatLite::WireType wire_type) {
  return wire_type == WireFormatLite::WIRETYPE_LENGTH_DELIMITED;
}

// Reads a single data value for a wire type.
absl::Status ReadFieldValue(uint32_t tag, CodedInputStream* in,
                            std::string* result) {
  WireFormatLite::WireType wire_type = WireFormatLite::GetTagWireType(tag);
  if (IsLengthDelimited(wire_type)) {
    uint32_t length;
    RET_CHECK_NO_LOG(in->ReadVarint32(&length));
    RET_CHECK_NO_LOG(in->ReadString(result, length));
  } else {
    std::string field_data;
    StringOutputStream sos(&field_data);
    CodedOutputStream cos(&sos);
    RET_CHECK_NO_LOG(WireFormatLite::SkipField(in, tag, &cos));
    // Skip the tag written by SkipField.
    int tag_size = CodedOutputStream::VarintSize32(tag);
    cos.Trim();
    result->assign(field_data, tag_size, std::string::npos);
  }
  return absl::OkStatus();
}

// Reads the packed sequence of data values for a wire type.
absl::Status ReadPackedValues(WireFormatLite::WireType wire_type,
                              CodedInputStream* in,
                              std::vector<std::string>* field_values) {
  uint32_t data_size;
  RET_CHECK_NO_LOG(in->ReadVarint32(&data_size));
  // fake_tag encodes the wire-type for calls to WireFormatLite::SkipField.
  uint32_t fake_tag = WireFormatLite::MakeTag(1, wire_type);
  while (data_size > 0) {
    std::string number;
    MP_RETURN_IF_ERROR(ReadFieldValue(fake_tag, in, &number));
    RET_CHECK_NO_LOG(number.size() <= data_size);
    field_values->push_back(number);
    data_size -= number.size();
  }
  return absl::OkStatus();
}

// Extracts the data value(s) for one field from a serialized message.
// The message with these field values removed is written to |out|.
absl::Status GetFieldValues(uint32_t field_id, CodedInputStream* in,
                            CodedOutputStream* out,
                            std::vector<std::string>* field_values) {
  uint32_t tag;
  while ((tag = in->ReadTag()) != 0) {
    int field_number = WireFormatLite::GetTagFieldNumber(tag);
    WireFormatLite::WireType wire_type = WireFormatLite::GetTagWireType(tag);
    if (field_number == field_id) {
      if (!IsLengthDelimited(wire_type) &&
          IsLengthDelimited(WireFormatLite::GetTagWireType(tag))) {
        MP_RETURN_IF_ERROR(ReadPackedValues(wire_type, in, field_values));
      } else {
        std::string value;
        MP_RETURN_IF_ERROR(ReadFieldValue(tag, in, &value));
        field_values->push_back(value);
      }
    } else {
      RET_CHECK_NO_LOG(WireFormatLite::SkipField(in, tag, out));
    }
  }
  return absl::OkStatus();
}

// Injects the data value(s) for one field into a serialized message.
void SetFieldValues(uint32_t field_id, WireFormatLite::WireType wire_type,
                    const std::vector<std::string>& field_values,
                    CodedOutputStream* out) {
  uint32_t tag = WireFormatLite::MakeTag(field_id, wire_type);
  for (const std::string& field_value : field_values) {
    out->WriteVarint32(tag);
    if (IsLengthDelimited(wire_type)) {
      out->WriteVarint32(field_value.length());
    }
    out->WriteRaw(field_value.data(), field_value.length());
  }
}

FieldAccess::FieldAccess(uint32_t field_id, FieldType field_type)
    : field_id_(field_id), field_type_(field_type) {}

absl::Status FieldAccess::SetMessage(const std::string& message) {
  ArrayInputStream ais(message.data(), message.size());
  CodedInputStream in(&ais);
  StringOutputStream sos(&message_);
  CodedOutputStream out(&sos);
  return GetFieldValues(field_id_, &in, &out, &field_values_);
}

void FieldAccess::GetMessage(std::string* result) {
  *result = message_;
  StringOutputStream sos(result);
  CodedOutputStream out(&sos);
  WireFormatLite::WireType wire_type =
      WireFormatLite::WireTypeForFieldType(field_type_);
  SetFieldValues(field_id_, wire_type, field_values_, &out);
}

std::vector<FieldValue>* FieldAccess::mutable_field_values() {
  return &field_values_;
}

namespace {
using ProtoPathEntry = ProtoUtilLite::ProtoPathEntry;

// Returns the FieldAccess and index for a field-id or a map-id.
// Returns access to the field-id if the field index is found,
// to the map-id if the map entry is found, and to the field-id otherwise.
absl::StatusOr<std::pair<FieldAccess, int>> AccessField(
    const ProtoPathEntry& entry, FieldType field_type,
    const FieldValue& message) {
  FieldAccess result(entry.field_id, field_type);
  if (entry.field_id >= 0) {
    MP_RETURN_IF_ERROR(result.SetMessage(message));
    if (entry.index < result.mutable_field_values()->size()) {
      return std::pair(result, entry.index);
    }
  }
  if (entry.map_id >= 0) {
    FieldAccess access(entry.map_id, field_type);
    MP_RETURN_IF_ERROR(access.SetMessage(message));
    auto& field_values = *access.mutable_field_values();
    for (int index = 0; index < field_values.size(); ++index) {
      FieldAccess key(entry.key_id, entry.key_type);
      MP_RETURN_IF_ERROR(key.SetMessage(field_values[index]));
      if (key.mutable_field_values()->at(0) == entry.key_value) {
        return std::pair(std::move(access), index);
      }
    }
  }
  if (entry.field_id >= 0) {
    return std::pair(result, entry.index);
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "ProtoPath field missing, field-id: ", entry.field_id, ", map-id: ",
      entry.map_id, ", key: ", entry.key_value, " key_type: ", entry.key_type));
}

}  // namespace

// Replaces a range of field values for one field nested within a protobuf.
absl::Status ProtoUtilLite::ReplaceFieldRange(
    FieldValue* message, ProtoPath proto_path, int length, FieldType field_type,
    const std::vector<FieldValue>& field_values) {
  ProtoPathEntry entry = proto_path.front();
  proto_path.erase(proto_path.begin());
  FieldType type =
      !proto_path.empty() ? WireFormatLite::TYPE_MESSAGE : field_type;
  MP_ASSIGN_OR_RETURN(auto r, AccessField(entry, type, *message));
  FieldAccess& access = r.first;
  int index = r.second;
  std::vector<FieldValue>& v = *access.mutable_field_values();
  if (!proto_path.empty()) {
    RET_CHECK_NO_LOG(index >= 0 && index < v.size());
    MP_RETURN_IF_ERROR(ReplaceFieldRange(&v[index], proto_path, length,
                                         field_type, field_values));
  } else {
    RET_CHECK_NO_LOG(index >= 0 && index <= v.size());
    RET_CHECK_NO_LOG(index + length >= 0 && index + length <= v.size());
    v.erase(v.begin() + index, v.begin() + index + length);
    v.insert(v.begin() + index, field_values.begin(), field_values.end());
  }
  message->clear();
  access.GetMessage(message);
  return absl::OkStatus();
}

// Returns a range of field values from one field nested within a protobuf.
absl::Status ProtoUtilLite::GetFieldRange(
    const FieldValue& message, ProtoPath proto_path, int length,
    FieldType field_type, std::vector<FieldValue>* field_values) {
  ProtoPathEntry entry = proto_path.front();
  proto_path.erase(proto_path.begin());
  FieldType type =
      !proto_path.empty() ? WireFormatLite::TYPE_MESSAGE : field_type;
  MP_ASSIGN_OR_RETURN(auto r, AccessField(entry, type, message));
  FieldAccess& access = r.first;
  int index = r.second;
  std::vector<FieldValue>& v = *access.mutable_field_values();
  if (!proto_path.empty()) {
    RET_CHECK_NO_LOG(index >= 0 && index < v.size());
    MP_RETURN_IF_ERROR(
        GetFieldRange(v[index], proto_path, length, field_type, field_values));
  } else {
    if (length == -1) {
      length = v.size() - index;
    }
    RET_CHECK_NO_LOG(index >= 0 && index <= v.size());
    RET_CHECK_NO_LOG(index + length >= 0 && index + length <= v.size());
    field_values->insert(field_values->begin(), v.begin() + index,
                         v.begin() + index + length);
  }
  return absl::OkStatus();
}

// Returns the number of field values in a repeated protobuf field.
absl::Status ProtoUtilLite::GetFieldCount(const FieldValue& message,
                                          ProtoPath proto_path,
                                          FieldType field_type,
                                          int* field_count) {
  ProtoPathEntry entry = proto_path.front();
  proto_path.erase(proto_path.begin());
  FieldType type =
      !proto_path.empty() ? WireFormatLite::TYPE_MESSAGE : field_type;
  MP_ASSIGN_OR_RETURN(auto r, AccessField(entry, type, message));
  FieldAccess& access = r.first;
  int index = r.second;
  std::vector<FieldValue>& v = *access.mutable_field_values();
  if (!proto_path.empty()) {
    RET_CHECK_NO_LOG(index >= 0 && index < v.size());
    MP_RETURN_IF_ERROR(
        GetFieldCount(v[index], proto_path, field_type, field_count));
  } else {
    *field_count = v.size();
  }
  return absl::OkStatus();
}

// If ok, returns OkStatus, otherwise returns InvalidArgumentError.
template <typename T>
absl::Status SyntaxStatus(bool ok, const std::string& text, T* result) {
  return ok ? absl::OkStatus()
            : absl::InvalidArgumentError(absl::StrCat(
                  "Syntax error: \"", text, "\"",
                  " for type: ", MediaPipeTypeStringOrDemangled<T>(), "."));
}

// Templated parsing of a string value.
template <typename T>
absl::Status ParseValue(const std::string& text, T* result) {
  return SyntaxStatus(absl::SimpleAtoi(text, result), text, result);
}
template <>
absl::Status ParseValue<double>(const std::string& text, double* result) {
  return SyntaxStatus(absl::SimpleAtod(text, result), text, result);
}
template <>
absl::Status ParseValue<float>(const std::string& text, float* result) {
  return SyntaxStatus(absl::SimpleAtof(text, result), text, result);
}
template <>
absl::Status ParseValue<bool>(const std::string& text, bool* result) {
  return SyntaxStatus(absl::SimpleAtob(text, result), text, result);
}
template <>
absl::Status ParseValue<std::string>(const std::string& text,
                                     std::string* result) {
  *result = text;
  return absl::OkStatus();
}

// Templated formatting of a primitive value.
template <typename T>
std::string FormatValue(T v) {
  return FieldValue(absl::StrCat(v));
}

// A helper function to parse and serialize one primtive value.
template <typename T>
absl::Status WritePrimitive(void (*writer)(T, proto_ns::io::CodedOutputStream*),
                            const std::string& text, CodedOutputStream* out) {
  T value;
  MP_RETURN_IF_ERROR(ParseValue<T>(text, &value));
  (*writer)(value, out);
  return absl::OkStatus();
}

// Serializes a protobuf FieldValue.
static absl::Status SerializeValue(const std::string& text,
                                   FieldType field_type,
                                   FieldValue* field_value) {
  absl::Status status;
  StringOutputStream sos(field_value);
  CodedOutputStream out(&sos);

  using W = WireFormatLite;
  switch (field_type) {
    case W::TYPE_DOUBLE:
      return WritePrimitive(W::WriteDoubleNoTag, text, &out);
    case W::TYPE_FLOAT:
      return WritePrimitive(W::WriteFloatNoTag, text, &out);
    case W::TYPE_INT64:
      return WritePrimitive(W::WriteInt64NoTag, text, &out);
    case W::TYPE_UINT64:
      return WritePrimitive(W::WriteUInt64NoTag, text, &out);
    case W::TYPE_INT32:
      return WritePrimitive(W::WriteInt32NoTag, text, &out);
    case W::TYPE_FIXED64:
      return WritePrimitive(W::WriteFixed64NoTag, text, &out);
    case W::TYPE_FIXED32:
      return WritePrimitive(W::WriteFixed32NoTag, text, &out);
    case W::TYPE_BOOL: {
      return WritePrimitive(W::WriteBoolNoTag, text, &out);
    }
    case W::TYPE_BYTES:
    case W::TYPE_STRING: {
      out.WriteRaw(text.data(), text.size());
      return absl::OkStatus();
    }
    case W::TYPE_GROUP:
    case W::TYPE_MESSAGE:
      return absl::UnimplementedError(
          "SerializeValue cannot serialize a Message.");
    case W::TYPE_UINT32:
      return WritePrimitive(W::WriteUInt32NoTag, text, &out);
    case W::TYPE_ENUM:
      return WritePrimitive(W::WriteEnumNoTag, text, &out);
    case W::TYPE_SFIXED32:
      return WritePrimitive(W::WriteSFixed32NoTag, text, &out);
    case W::TYPE_SFIXED64:
      return WritePrimitive(W::WriteSFixed64NoTag, text, &out);
    case W::TYPE_SINT32:
      return WritePrimitive(W::WriteSInt32NoTag, text, &out);
    case W::TYPE_SINT64:
      return WritePrimitive(W::WriteSInt64NoTag, text, &out);
  }
  return absl::UnimplementedError("SerializeValue unimplemented type.");
}

// A helper function for deserializing one text value.
template <typename CType, FieldType DeclaredType>
static absl::Status ReadPrimitive(CodedInputStream* input,
                                  std::string* result) {
  CType value;
  if (!WireFormatLite::ReadPrimitive<CType, DeclaredType>(input, &value)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Bad serialized value: ", MediaPipeTypeStringOrDemangled<CType>(),
        "."));
  }
  *result = FormatValue(value);
  return absl::OkStatus();
}

// Deserializes a protobuf FieldValue.
static absl::Status DeserializeValue(const FieldValue& bytes,
                                     FieldType field_type,
                                     std::string* result) {
  ArrayInputStream ais(bytes.data(), bytes.size());
  CodedInputStream input(&ais);
  typedef WireFormatLite W;
  switch (field_type) {
    case W::TYPE_DOUBLE:
      return ReadPrimitive<double, W::TYPE_DOUBLE>(&input, result);
    case W::TYPE_FLOAT:
      return ReadPrimitive<float, W::TYPE_FLOAT>(&input, result);
    case W::TYPE_INT64:
      return ReadPrimitive<proto_int64, W::TYPE_INT64>(&input, result);
    case W::TYPE_UINT64:
      return ReadPrimitive<proto_uint64, W::TYPE_UINT64>(&input, result);
    case W::TYPE_INT32:
      return ReadPrimitive<int32_t, W::TYPE_INT32>(&input, result);
    case W::TYPE_FIXED64:
      return ReadPrimitive<proto_uint64, W::TYPE_FIXED64>(&input, result);
    case W::TYPE_FIXED32:
      return ReadPrimitive<uint32_t, W::TYPE_FIXED32>(&input, result);
    case W::TYPE_BOOL:
      return ReadPrimitive<bool, W::TYPE_BOOL>(&input, result);
    case W::TYPE_BYTES:
    case W::TYPE_STRING: {
      *result = bytes;
      return absl::OkStatus();
    }
    case W::TYPE_GROUP:
    case W::TYPE_MESSAGE:
      ABSL_CHECK(false) << "DeserializeValue cannot deserialize a Message.";
    case W::TYPE_UINT32:
      return ReadPrimitive<uint32_t, W::TYPE_UINT32>(&input, result);
    case W::TYPE_ENUM:
      return ReadPrimitive<int, W::TYPE_ENUM>(&input, result);
    case W::TYPE_SFIXED32:
      return ReadPrimitive<int32_t, W::TYPE_SFIXED32>(&input, result);
    case W::TYPE_SFIXED64:
      return ReadPrimitive<proto_int64, W::TYPE_SFIXED64>(&input, result);
    case W::TYPE_SINT32:
      return ReadPrimitive<int32_t, W::TYPE_SINT32>(&input, result);
    case W::TYPE_SINT64:
      return ReadPrimitive<proto_int64, W::TYPE_SINT64>(&input, result);
  }
  return absl::UnimplementedError("DeserializeValue unimplemented type.");
}

absl::Status ProtoUtilLite::Serialize(
    const std::vector<std::string>& text_values, FieldType field_type,
    std::vector<FieldValue>* result) {
  result->clear();
  result->reserve(text_values.size());
  for (const std::string& text_value : text_values) {
    FieldValue field_value;
    MP_RETURN_IF_ERROR(SerializeValue(text_value, field_type, &field_value));
    result->push_back(field_value);
  }
  return absl::OkStatus();
}

absl::Status ProtoUtilLite::Deserialize(
    const std::vector<FieldValue>& field_values, FieldType field_type,
    std::vector<std::string>* result) {
  result->clear();
  result->reserve(field_values.size());
  for (const FieldValue& field_value : field_values) {
    std::string text_value;
    MP_RETURN_IF_ERROR(DeserializeValue(field_value, field_type, &text_value));
    result->push_back(text_value);
  }
  return absl::OkStatus();
}

absl::Status ProtoUtilLite::WriteValue(const FieldData& value,
                                       FieldType field_type,
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

template <typename ValueT, FieldType kFieldType>
static ValueT ReadValue(absl::string_view field_bytes, absl::Status* status) {
  ArrayInputStream ais(field_bytes.data(), field_bytes.size());
  CodedInputStream input(&ais);
  ValueT result;
  if (!WireFormatLite::ReadPrimitive<ValueT, kFieldType>(&input, &result)) {
    status->Update(absl::InvalidArgumentError(absl::StrCat(
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
          ReadValue<int32_t, WireFormatLite::TYPE_INT32>(field_bytes, &status));
      break;
    case WireFormatLite::TYPE_SINT32:
      result->set_int32_value(ReadValue<int32_t, WireFormatLite::TYPE_SINT32>(
          field_bytes, &status));
      break;
    case WireFormatLite::TYPE_INT64:
      result->set_int64_value(
          ReadValue<int64_t, WireFormatLite::TYPE_INT64>(field_bytes, &status));
      break;
    case WireFormatLite::TYPE_SINT64:
      result->set_int64_value(ReadValue<int64_t, WireFormatLite::TYPE_SINT64>(
          field_bytes, &status));
      break;
    case WireFormatLite::TYPE_UINT32:
      result->set_uint32_value(ReadValue<uint32_t, WireFormatLite::TYPE_UINT32>(
          field_bytes, &status));
      break;
    case WireFormatLite::TYPE_UINT64:
      result->set_uint64_value(ReadValue<uint32_t, WireFormatLite::TYPE_UINT32>(
          field_bytes, &status));
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
          ReadValue<int32_t, WireFormatLite::TYPE_ENUM>(field_bytes, &status));
      break;
    case WireFormatLite::TYPE_STRING:
      result->set_string_value(std::string(field_bytes));
      break;
    case WireFormatLite::TYPE_MESSAGE:
      result->mutable_message_value()->set_value(std::string(field_bytes));
      result->mutable_message_value()->set_type_url(
          ProtoUtilLite::TypeUrl(message_type));
      break;
    default:
      status = absl::UnimplementedError(
          absl::StrCat("Cannot read type: ", field_type));
      break;
  }
  return status;
}

absl::Status ProtoUtilLite::ReadValue(absl::string_view field_bytes,
                                      FieldType field_type,
                                      absl::string_view message_type,
                                      FieldData* result) {
  return mediapipe::tool::ReadValue(field_bytes, field_type, message_type,
                                    result);
}

std::string ProtoUtilLite::TypeUrl(absl::string_view type_name) {
  constexpr std::string_view kTypeUrlPrefix = "type.googleapis.com/";
  return absl::StrCat(std::string(kTypeUrlPrefix), std::string(type_name));
}

std::string ProtoUtilLite::ParseTypeUrl(absl::string_view type_url) {
  constexpr std::string_view kTypeUrlPrefix = "type.googleapis.com/";
  if (absl::StartsWith(std::string(type_url), std::string(kTypeUrlPrefix))) {
    return std::string(type_url.substr(kTypeUrlPrefix.length()));
  }
  return std::string(type_url);
}

}  // namespace tool
}  // namespace mediapipe
