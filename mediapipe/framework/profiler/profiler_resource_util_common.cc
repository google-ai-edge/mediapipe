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
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/profiler/profiler_resource_util.h"

// TODO: Move this Android include to port/file_helpers.
// Also move this from resource_util.cc.
#ifdef __ANDROID__
#include "mediapipe/util/android/file/base/filesystem.h"
#else
#include "mediapipe/framework/port/file_helpers.h"
#endif

ABSL_FLAG(std::string, log_root_dir, "",
          "The absolute path to the logging output directory.  If specified, "
          "log_root_dir will be prepended to each specified log file path.");

#ifdef __ANDROID__
namespace mediapipe {
namespace file {
::mediapipe::Status RecursivelyCreateDir(absl::string_view path) {
  return RecursivelyCreateDir(path, file::Options());
}
}  // namespace file
}  // namespace mediapipe
#endif

namespace mediapipe {

::mediapipe::StatusOr<std::string> GetLogDirectory() {
  if (!FLAGS_log_root_dir.CurrentValue().empty()) {
    return FLAGS_log_root_dir.CurrentValue();
  }
  return GetDefaultTraceLogDirectory();
}

::mediapipe::StatusOr<std::string> PathToLogFile(const std::string& path) {
  ASSIGN_OR_RETURN(std::string log_dir, GetLogDirectory());
  std::string result = file::JoinPath(log_dir, path);
  MP_RETURN_IF_ERROR(
      ::mediapipe::file::RecursivelyCreateDir(file::Dirname(result)));
  return result;
}

}  // namespace mediapipe
