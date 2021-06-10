// Copyright 2020 The MediaPipe Authors.
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

#include "absl/flags/flag.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/profiler/profiler_resource_util.h"

ABSL_FLAG(std::string, log_root_dir, "",
          "The absolute path to the logging output directory.  If specified, "
          "log_root_dir will be prepended to each specified log file path.");

namespace mediapipe {

absl::StatusOr<std::string> GetLogDirectory() {
  if (!absl::GetFlag(FLAGS_log_root_dir).empty()) {
    return absl::GetFlag(FLAGS_log_root_dir);
  }
  return GetDefaultTraceLogDirectory();
}

absl::StatusOr<std::string> PathToLogFile(const std::string& path) {
  ASSIGN_OR_RETURN(std::string log_dir, GetLogDirectory());
  std::string result = file::JoinPath(log_dir, path);
  MP_RETURN_IF_ERROR(
      mediapipe::file::RecursivelyCreateDir(file::Dirname(result)));
  return result;
}

}  // namespace mediapipe
