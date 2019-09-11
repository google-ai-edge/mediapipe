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

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/type_map.h"

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

// Returns true if a wire type includes a length indicator.
bool IsLengthDelimited(WireFormatLite::WireType wire_type) {
  return wire_type == WireFormatLite::WIRETYPE_LENGTH_DELIMITED;
}

// Reads a single data value for a wire type.
::mediapipe::Status ReadFieldValue(uint32 tag, CodedInputStream* in,
                                   std::string* result) {
  WireFormatLite::WireType wire_type = WireFormatLite::GetTagWireType(tag);
  if (IsLengthDelimited(wire_type)) {
    uint32 length;
    RET_CHECK(in->ReadVarint32(&length));
    RET_CHECK(in->ReadString(result, length));
  } else {
    std::string field_data;
    StringOutputStream sos(&field_data);
    CodedOutputStream cos(&sos);
    RET_CHECK(WireFormatLite::SkipField(in, tag, &cos));
    // Skip the tag written by SkipField.
    int tag_size = CodedOutputStream::VarintSize32(tag);
    cos.Trim();
    result->assign(field_data, tag_size, std::string::npos);
  }
  return ::mediapipe::OkStatus();
}

// Reads the packed sequence of data values for a wire type.
::mediapipe::Status ReadPackedValues(WireFormatLite::WireType wire_type,
                                     CodedInputStream* in,
                                     std::vector<std::string>* field_values) {
  uint32 data_size;
  RET_CHECK(in->ReadVarint32(&data_size));
  // fake_tag encodes the wire-type for calls to WireFormatLite::SkipField.
  uint32 fake_tag = WireFormatLite::MakeTag(1, wire_type);
  while (data_size > 0) {
    std::string number;
    MP_RETURN_IF_ERROR(ReadFieldValue(fake_tag, in, &number));
    RET_CHECK_LE(number.size(), data_size);
    field_values->push_back(number);
    data_size -= number.size();
  }
  return ::mediapipe::OkStatus();
}

// Extracts the data value(s) for one field from a serialized message.
// The message with these field values removed is written to |out|.
::mediapipe::Status GetFieldValues(uint32 field_id,
                                   WireFormatLite::WireType wire_type,
                                   CodedInputStream* in, CodedOutputStream* out,
                                   std::vector<std::string>* field_values) {
  uint32 tag;
  while ((tag = in->ReadTag()) != 0) {
    int field_number = WireFormatLite::GetTagFieldNumber(tag);
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
      RET_CHECK(WireFormatLite::SkipField(in, tag, out));
    }
  }
  return ::mediapipe::OkStatus();
}

// Injects the data value(s) for one field into a serialized message.
void SetFieldValues(uint32 field_id, WireFormatLite::WireType wire_type,
                    const std::vector<std::string>& field_values,
                    CodedOutputStream* out) {
  uint32 tag = WireFormatLite::MakeTag(field_id, wire_type);
  for (const std::string& field_value : field_values) {
    out->WriteVarint32(tag);
    if (IsLengthDelimited(wire_type)) {
      out->WriteVarint32(field_value.length());
    }
    out->WriteRaw(field_value.data(), field_value.length());
  }
}

FieldAccess::FieldAccess(uint32 field_id, FieldType field_type)
    : field_id_(field_id), field_type_(field_type) {}

::mediapipe::Status FieldAccess::SetMessage(const std::string& message) {
  ArrayInputStream ais(message.data(), message.size());
  CodedInputStream in(&ais);
  StringOutputStream sos(&message_);
  CodedOutputStream out(&sos);
  WireFormatLite::WireType wire_type =
      WireFormatLite::WireTypeForFieldType(field_type_);
  return GetFieldValues(field_id_, wire_type, &in, &out, &field_values_);
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

// Replaces a range of field values for one field nested within a protobuf.
::mediapipe::Status ProtoUtilLite::ReplaceFieldRange(
    FieldValue* message, ProtoPath proto_path, int length, FieldType field_type,
    const std::vector<FieldValue>& field_values) {
  int field_id, index;
  std::tie(field_id, index) = proto_path.front();
  proto_path.erase(proto_path.begin());
  FieldAccess access(field_id, !proto_path.empty()
                                   ? WireFormatLite::TYPE_MESSAGE
                                   : field_type);
  MP_RETURN_IF_ERROR(access.SetMessage(*message));
  std::vector<std::string>& v = *access.mutable_field_values();
  if (!proto_path.empty()) {
    RET_CHECK(index >= 0 && index < v.size());
    MP_RETURN_IF_ERROR(ReplaceFieldRange(&v[index], proto_path, length,
                                         field_type, field_values));
  } else {
    RET_CHECK(index >= 0 && index <= v.size());
    RET_CHECK(index + length >= 0 && index + length <= v.size());
    v.erase(v.begin() + index, v.begin() + index + length);
    v.insert(v.begin() + index, field_values.begin(), field_values.end());
  }
  message->clear();
  access.GetMessage(message);
  return ::mediapipe::OkStatus();
}

// Returns a range of field values from one field nested within a protobuf.
::mediapipe::Status ProtoUtilLite::GetFieldRange(
    const FieldValue& message, ProtoPath proto_path, int length,
    FieldType field_type, std::vector<FieldValue>* field_values) {
  int field_id, index;
  std::tie(field_id, index) = proto_path.front();
  proto_path.erase(proto_path.begin());
  FieldAccess access(field_id, !proto_path.empty()
                                   ? WireFormatLite::TYPE_MESSAGE
                                   : field_type);
  MP_RETURN_IF_ERROR(access.SetMessage(message));
  std::vector<std::string>& v = *access.mutable_field_values();
  if (!proto_path.empty()) {
    RET_CHECK(index >= 0 && index < v.size());
    MP_RETURN_IF_ERROR(
        GetFieldRange(v[index], proto_path, length, field_type, field_values));
  } else {
    RET_CHECK(index >= 0 && index <= v.size());
    RET_CHECK(index + length >= 0 && index + length <= v.size());
    field_values->insert(field_values->begin(), v.begin() + index,
                         v.begin() + index + length);
  }
  return ::mediapipe::OkStatus();
}

// If ok, returns OkStatus, otherwise returns InvalidArgumentError.
template <typename T>
::mediapipe::Status SyntaxStatus(bool ok, const std::string& text, T* result) {
  return ok ? ::mediapipe::OkStatus()
            : ::mediapipe::InvalidArgumentError(absl::StrCat(
                  "Syntax error: \"", text, "\"",
                  " for type: ", MediaPipeTypeStringOrDemangled<T>(), "."));
}

// Templated parsing of a std::string value.
template <typename T>
::mediapipe::Status ParseValue(const std::string& text, T* result) {
  return SyntaxStatus(absl::SimpleAtoi(text, result), text, result);
}
template <>
::mediapipe::Status ParseValue<double>(const std::string& text,
                                       double* result) {
  return SyntaxStatus(absl::SimpleAtod(text, result), text, result);
}
template <>
::mediapipe::Status ParseValue<float>(const std::string& text, float* result) {
  return SyntaxStatus(absl::SimpleAtof(text, result), text, result);
}
template <>
::mediapipe::Status ParseValue<bool>(const std::string& text, bool* result) {
  return SyntaxStatus(absl::SimpleAtob(text, result), text, result);
}
template <>
::mediapipe::Status ParseValue<std::string>(const std::string& text,
                                            std::string* result) {
  *result = text;
  return ::mediapipe::OkStatus();
}

// Templated formatting of a primitive value.
template <typename T>
std::string FormatValue(T v) {
  return FieldValue(absl::StrCat(v));
}

// A helper function to parse and serialize one primtive value.
template <typename T>
::mediapipe::Status WritePrimitive(
    void (*writer)(T, proto_ns::io::CodedOutputStream*),
    const std::string& text, CodedOutputStream* out) {
  T value;
  MP_RETURN_IF_ERROR(ParseValue<T>(text, &value));
  (*writer)(value, out);
  return ::mediapipe::OkStatus();
}

// Serializes a protobuf FieldValue.
static ::mediapipe::Status SerializeValue(const std::string& text,
                                          FieldType field_type,
                                          FieldValue* field_value) {
  ::mediapipe::Status status;
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
      return ::mediapipe::OkStatus();
    }
    case W::TYPE_GROUP:
    case W::TYPE_MESSAGE:
      return ::mediapipe::UnimplementedError(
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
  return ::mediapipe::UnimplementedError("SerializeValue unimplemented type.");
}

// A helper function for deserializing one text value.
template <typename CType, FieldType DeclaredType>
static ::mediapipe::Status ReadPrimitive(CodedInputStream* input,
                                         std::string* result) {
  CType value;
  if (!WireFormatLite::ReadPrimitive<CType, DeclaredType>(input, &value)) {
    return ::mediapipe::InvalidArgumentError(absl::StrCat(
        "Bad serialized value: ", MediaPipeTypeStringOrDemangled<CType>(),
        "."));
  }
  *result = FormatValue(value);
  return ::mediapipe::OkStatus();
}

// Deserializes a protobuf FieldValue.
static ::mediapipe::Status DeserializeValue(const FieldValue& bytes,
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
      return ReadPrimitive<int32, W::TYPE_INT32>(&input, result);
    case W::TYPE_FIXED64:
      return ReadPrimitive<proto_uint64, W::TYPE_FIXED64>(&input, result);
    case W::TYPE_FIXED32:
      return ReadPrimitive<uint32, W::TYPE_FIXED32>(&input, result);
    case W::TYPE_BOOL:
      return ReadPrimitive<bool, W::TYPE_BOOL>(&input, result);
    case W::TYPE_BYTES:
    case W::TYPE_STRING: {
      *result = bytes;
      return ::mediapipe::OkStatus();
    }
    case W::TYPE_GROUP:
    case W::TYPE_MESSAGE:
      CHECK(false) << "DeserializeValue cannot deserialize a Message.";
    case W::TYPE_UINT32:
      return ReadPrimitive<uint32, W::TYPE_UINT32>(&input, result);
    case W::TYPE_ENUM:
      return ReadPrimitive<int, W::TYPE_ENUM>(&input, result);
    case W::TYPE_SFIXED32:
      return ReadPrimitive<int32, W::TYPE_SFIXED32>(&input, result);
    case W::TYPE_SFIXED64:
      return ReadPrimitive<proto_int64, W::TYPE_SFIXED64>(&input, result);
    case W::TYPE_SINT32:
      return ReadPrimitive<int32, W::TYPE_SINT32>(&input, result);
    case W::TYPE_SINT64:
      return ReadPrimitive<proto_int64, W::TYPE_SINT64>(&input, result);
  }
  return ::mediapipe::UnimplementedError(
      "DeserializeValue unimplemented type.");
}

::mediapipe::Status ProtoUtilLite::Serialize(
    const std::vector<std::string>& text_values, FieldType field_type,
    std::vector<FieldValue>* result) {
  result->clear();
  result->reserve(text_values.size());
  for (const std::string& text_value : text_values) {
    FieldValue field_value;
    MP_RETURN_IF_ERROR(SerializeValue(text_value, field_type, &field_value));
    result->push_back(field_value);
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status ProtoUtilLite::Deserialize(
    const std::vector<FieldValue>& field_values, FieldType field_type,
    std::vector<std::string>* result) {
  result->clear();
  result->reserve(field_values.size());
  for (const FieldValue& field_value : field_values) {
    std::string text_value;
    MP_RETURN_IF_ERROR(DeserializeValue(field_value, field_type, &text_value));
    result->push_back(text_value);
  }
  return ::mediapipe::OkStatus();
}

}  // namespace tool
}  // namespace mediapipe
