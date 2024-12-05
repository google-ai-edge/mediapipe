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
#include <iterator>

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/resource_util.h"

namespace mediapipe {

absl::StatusOr<std::string> PathToResourceAsFile(const std::string& path,
                                                 bool /*shadow_copy*/) {
  if (absl::StartsWith(path, "/")) {
    return path;
  }

  // Try the test environment.
  absl::string_view workspace = "mediapipe";
  const char* test_srcdir = std::getenv("TEST_SRCDIR");
  auto test_path =
      file::JoinPath(test_srcdir ? test_srcdir : "", workspace, path);
  if (file::Exists(test_path).ok()) {
    return test_path;
  }

  return path;
}

namespace internal {
absl::Status DefaultGetResourceContents(const std::string& path,
                                        std::string* output,
                                        bool read_as_binary) {
  if (!read_as_binary) {
    ABSL_LOG(WARNING)
        << "Setting \"read_as_binary\" to false is a no-op on Emscripten.";
  }
  MP_ASSIGN_OR_RETURN(std::string full_path, PathToResourceAsFile(path));
  std::ifstream input(full_path);
  output->assign((std::istreambuf_iterator<char>(input)),
                 std::istreambuf_iterator<char>());
  if (!input.is_open() || input.bad()) {
    return absl::Status(absl::StatusCode::kUnknown,
                        absl::StrFormat("Failed to read file %s.", full_path));
  }
  return absl::OkStatus();
}

}  // namespace internal
}  // namespace mediapipe
