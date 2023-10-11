/* Copyright 2023 The MediaPipe Authors.

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

#include "mediapipe/tasks/ios/test/vision/utils/sources/parse_proto_utils.h"

#include <fstream>
#include <sstream>

#include "google/protobuf/text_format.h"

namespace mediapipe::tasks::ios::test::vision::utils {

namespace {
using ::google::protobuf::Message;
using ::google::protobuf::TextFormat;
}  // anonymous namespace

::absl::Status get_proto_from_pbtxt(const std::string file_path,
                                    Message& proto) {
  std::ifstream file_input_stream(file_path);
  if (!file_input_stream.is_open())
    return absl::InvalidArgumentError("Cannot read input file.");

  std::stringstream strings_stream;
  strings_stream << file_input_stream.rdbuf();

  return TextFormat::ParseFromString(strings_stream.str(), &proto)
             ? ::absl::OkStatus()
             : ::absl::InvalidArgumentError(
                   "Cannot read a valid proto from the input file.");
}

}  // namespace mediapipe::tasks::ios::test::vision::utils
