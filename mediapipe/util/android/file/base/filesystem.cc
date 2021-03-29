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

#include "absl/strings/str_split.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/android/file/base/file.h"

#ifdef __APPLE__
static_assert(sizeof(off_t) == 8, "Large file support is required");
#define stat64 stat
#define lstat64 lstat
#endif

namespace mediapipe {
namespace file {

absl::Status RecursivelyCreateDir(absl::string_view path,
                                  const file::Options& options) {
  if (path.empty()) {
    return absl::OkStatus();
  }

  std::vector<std::string> path_comp = absl::StrSplit(path, '/');
  if (path[0] == '/') {
    path_comp[0] = "/" + path_comp[0];
  }
  struct stat stat_buf;
  std::string rpath;
  for (const std::string& ix : path_comp) {
    rpath = rpath.empty() ? ix : rpath + "/" + ix;
    const char* crpath = rpath.c_str();
    int statval = stat(crpath, &stat_buf);
    if (statval == 0) {
      if (S_ISDIR(stat_buf.st_mode)) {
        continue;
      }
      return absl::Status(absl::StatusCode::kInternal,
                          "Could not stat " + std::string(crpath));
    } else {
      int mkval = mkdir(crpath, options.permissions());
      if (mkval == -1) {
        return absl::Status(absl::StatusCode::kInternal,
                            "Could not create " + std::string(crpath));
      }
    }
  }

  return absl::OkStatus();
}

absl::Status Exists(absl::string_view path, const file::Options& ignored) {
  struct stat64 stat_buf;
  int statval = lstat64(std::string(path).c_str(), &stat_buf);
  if (statval == 0) {
    return absl::OkStatus();
  } else {
    return absl::Status(absl::StatusCode::kNotFound, "Could not stat file.");
  }
}

absl::Status IsDirectory(absl::string_view path,
                         const file::Options& /*ignored*/) {
  struct stat64 stat_buf;
  int statval = lstat64(std::string(path).c_str(), &stat_buf);
  bool is_dir = (statval == 0 && S_ISREG(stat_buf.st_mode));
  if (is_dir) {
    return absl::OkStatus();
  } else if (statval != 0) {
    return absl::Status(absl::StatusCode::kNotFound, "File does not exists");
  } else {
    return absl::Status(absl::StatusCode::kNotFound, "Not a directory");
  }
}

}  // namespace file.
}  // namespace mediapipe.
