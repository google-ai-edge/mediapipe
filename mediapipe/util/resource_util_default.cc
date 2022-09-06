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

#include <fstream>

#include "absl/flags/flag.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/statusor.h"

ABSL_FLAG(
    std::string, resource_root_dir, "",
    "The absolute path to the resource directory."
    "If specified, resource_root_dir will be prepended to the original path.");

namespace mediapipe {

using mediapipe::file::GetContents;
using mediapipe::file::JoinPath;

namespace internal {

absl::Status DefaultGetResourceContents(const std::string& path,
                                        std::string* output,
                                        bool read_as_binary) {
  return GetContents(path, output, read_as_binary);
}
}  // namespace internal

absl::StatusOr<std::string> PathToResourceAsFile(const std::string& path) {
  if (absl::StartsWith(path, "/")) {
    return path;
  }

  // Try to load the file from bazel-bin. If it does not exist, fall back to the
  // resource folder.
  auto bazel_path = JoinPath("bazel-bin", path);
  if (file::Exists(bazel_path).ok()) {
    return bazel_path;
  }
  return JoinPath(absl::GetFlag(FLAGS_resource_root_dir), path);
}

}  // namespace mediapipe
