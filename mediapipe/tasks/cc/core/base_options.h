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

#ifndef MEDIAPIPE_TASKS_CC_CORE_BASE_OPTIONS_H_
#define MEDIAPIPE_TASKS_CC_CORE_BASE_OPTIONS_H_

#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/kernels/register.h"

namespace mediapipe {
namespace tasks {
namespace core {

// Base options for MediaPipe C++ Tasks.
struct BaseOptions {
  // The model file contents as a string.
  std::unique_ptr<std::string> model_file_contents;

  // The path to the model file to open and mmap in memory.
  std::string model_file_name = "";

  // The file descriptor to a file opened with open(2), with optional additional
  // offset and length information.
  struct FileDescriptorMeta {
    // File descriptor as returned by open(2).
    int fd = -1;

    // Optional length of the mapped memory. If not specified, the actual file
    // size is used at runtime.
    int length = -1;

    // Optional starting offset in the file referred to by the file descriptor
    // `fd`.
    int offset = -1;
  } model_file_descriptor_meta;

  // A non-default OpResolver to support custom Ops or specify a subset of
  // built-in Ops.
  std::unique_ptr<tflite::OpResolver> op_resolver =
      absl::make_unique<tflite::ops::builtin::BuiltinOpResolver>();
};

// Converts a BaseOptions to a BaseOptionsProto.
proto::BaseOptions ConvertBaseOptionsToProto(BaseOptions* base_options);

}  // namespace core
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_CORE_BASE_OPTIONS_H_
