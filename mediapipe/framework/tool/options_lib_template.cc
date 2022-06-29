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
//
// This template is used by the mediapipe_simple_subgraph macro in
// //mediapipe/framework/tool/mediapipe_graph.bzl

#include "mediapipe/framework/port/advanced_proto_inc.h"
#include "mediapipe/framework/tool/options_registry.h"
#include "{{MESSAGE_NAME_HEADER}}"
#include "{{MESSAGE_PROTO_HEADER}}"

namespace {
constexpr char kDescriptorContents[] =
#include "{{DESCRIPTOR_INC_FILE_PATH}}"
    ;  // NOLINT(whitespace/semicolon)

mediapipe::FieldData ReadFileDescriptorSet(const std::string& pb) {
  mediapipe::FieldData result;
  *result.mutable_message_value()->mutable_type_url() =
      "proto2.FileDescriptorSet";
  *result.mutable_message_value()->mutable_value() = pb;

  // Force linking of the generated options protobuf.
  mediapipe::proto_ns::LinkMessageReflection<
      MP_OPTION_TYPE_NS::MP_OPTION_TYPE_NAME>();
  return result;
}

}  // namespace

namespace mediapipe {
// The protobuf descriptor for an options message type.
template <>
const RegistrationToken tool::OptionsRegistry::registration_token<
    MP_OPTION_TYPE_NS::MP_OPTION_TYPE_NAME> =
    tool::OptionsRegistry::Register(ReadFileDescriptorSet(
        std::string(kDescriptorContents, sizeof(kDescriptorContents) - 1)));
}  // namespace mediapipe
