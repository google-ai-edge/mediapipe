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

#ifndef MEDIAPIPE_FRAMEWORK_TOOL_PROTO_UTIL_LITE_H_
#define MEDIAPIPE_FRAMEWORK_TOOL_PROTO_UTIL_LITE_H_

#include <string>
#include <utility>
#include <vector>

#include "mediapipe/framework/port/advanced_proto_lite_inc.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/proto_ns.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace tool {

class ProtoUtilLite {
 public:
  // Defines field types and tag formats.
  using WireFormatLite = proto_ns::internal::WireFormatLite;

  // Defines a sequence of nested field-number field-index pairs.
  using ProtoPath = std::vector<std::pair<int, int>>;

  // The serialized value for a protobuf field.
  using FieldValue = std::string;

  // The serialized data type for a protobuf field.
  using FieldType = WireFormatLite::FieldType;

  class FieldAccess {
   public:
    // Provides access to a certain protobuf field.
    FieldAccess(uint32 field_id, FieldType field_type);

    // Specifies the original serialized protobuf message.
    ::mediapipe::Status SetMessage(const FieldValue& message);

    // Returns the serialized protobuf message with updated field values.
    void GetMessage(FieldValue* result);

    // Returns the serialized values of the protobuf field.
    std::vector<FieldValue>* mutable_field_values();

   private:
    const uint32 field_id_;
    const FieldType field_type_;
    std::string message_;
    std::vector<FieldValue> field_values_;
  };

  // Replace a range of field values nested within a protobuf.
  // Starting at the proto_path index, "length" values are replaced.
  static ::mediapipe::Status ReplaceFieldRange(
      FieldValue* message, ProtoPath proto_path, int length,
      FieldType field_type, const std::vector<FieldValue>& field_values);

  // Retrieve a range of field values nested within a protobuf.
  // Starting at the proto_path index, "length" values are retrieved.
  static ::mediapipe::Status GetFieldRange(
      const FieldValue& message, ProtoPath proto_path, int length,
      FieldType field_type, std::vector<FieldValue>* field_values);

  // Serialize one or more protobuf field values from text.
  static ::mediapipe::Status Serialize(
      const std::vector<std::string>& text_values, FieldType field_type,
      std::vector<FieldValue>* result);

  // Deserialize one or more protobuf field values to text.
  static ::mediapipe::Status Deserialize(
      const std::vector<FieldValue>& field_values, FieldType field_type,
      std::vector<std::string>* result);
};

}  // namespace tool
}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_TOOL_PROTO_UTIL_LITE_H_
