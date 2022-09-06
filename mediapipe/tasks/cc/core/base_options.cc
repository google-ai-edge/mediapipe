/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/tasks/cc/core/base_options.h"

#include <memory>
#include <string>

#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"

namespace mediapipe {
namespace tasks {
namespace core {

proto::BaseOptions ConvertBaseOptionsToProto(BaseOptions* base_options) {
  proto::BaseOptions base_options_proto;
  if (!base_options->model_file_name.empty()) {
    base_options_proto.mutable_model_file()->set_file_name(
        base_options->model_file_name);
  }
  if (base_options->model_file_contents) {
    base_options_proto.mutable_model_file()->mutable_file_content()->swap(
        *base_options->model_file_contents.release());
  }
  if (base_options->model_file_descriptor_meta.fd > 0) {
    auto* file_descriptor_meta_proto =
        base_options_proto.mutable_model_file()->mutable_file_descriptor_meta();
    file_descriptor_meta_proto->set_fd(
        base_options->model_file_descriptor_meta.fd);
    if (base_options->model_file_descriptor_meta.length > 0) {
      file_descriptor_meta_proto->set_length(
          base_options->model_file_descriptor_meta.length);
    }
    if (base_options->model_file_descriptor_meta.offset > 0) {
      file_descriptor_meta_proto->set_offset(
          base_options->model_file_descriptor_meta.offset);
    }
  }
  return base_options_proto;
}
}  // namespace core
}  // namespace tasks
}  // namespace mediapipe
