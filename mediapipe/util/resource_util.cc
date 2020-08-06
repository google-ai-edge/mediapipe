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

#include "mediapipe/util/resource_util.h"

#include "absl/flags/flag.h"
#include "absl/strings/str_split.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/ret_check.h"

ABSL_FLAG(
    std::string, resource_root_dir, "",
    "The absolute path to the resource directory."
    "If specified, resource_root_dir will be prepended to the original path.");

namespace mediapipe {

::mediapipe::StatusOr<std::string> PathToResourceAsFile(
    const std::string& path) {
  return ::mediapipe::file::JoinPath(FLAGS_resource_root_dir.CurrentValue(),
                                     path);
}

::mediapipe::Status GetResourceContents(const std::string& path,
                                        std::string* output) {
  return mediapipe::file::GetContents(path, output);
}

}  // namespace mediapipe
