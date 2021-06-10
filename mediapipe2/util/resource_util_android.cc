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

#include <vector>

#include "absl/strings/match.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/singleton.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/util/android/asset_manager_util.h"
#include "mediapipe/util/android/file/base/helpers.h"

namespace mediapipe {

namespace {
absl::StatusOr<std::string> PathToResourceAsFileInternal(
    const std::string& path) {
  return Singleton<AssetManager>::get()->CachedFileFromAsset(path);
}
}  // namespace

namespace internal {
absl::Status DefaultGetResourceContents(const std::string& path,
                                        std::string* output,
                                        bool read_as_binary) {
  if (!read_as_binary) {
    LOG(WARNING)
        << "Setting \"read_as_binary\" to false is a no-op on Android.";
  }
  if (absl::StartsWith(path, "/")) {
    return file::GetContents(path, output, file::Defaults());
  }

  if (absl::StartsWith(path, "content://")) {
    MP_RETURN_IF_ERROR(
        Singleton<AssetManager>::get()->ReadContentUri(path, output));
    return absl::OkStatus();
  }

  // Try the test environment.
  absl::string_view workspace = "mediapipe";
  const char* test_srcdir = std::getenv("TEST_SRCDIR");
  auto test_path =
      file::JoinPath(test_srcdir ? test_srcdir : "", workspace, path);
  if (file::Exists(test_path).ok()) {
    return file::GetContents(path, output, file::Defaults());
  }

  RET_CHECK(Singleton<AssetManager>::get()->ReadFile(path, output))
      << "could not read asset: " << path;
  return absl::OkStatus();
}
}  // namespace internal

absl::StatusOr<std::string> PathToResourceAsFile(const std::string& path) {
  // Return full path.
  if (absl::StartsWith(path, "/")) {
    return path;
  }

  // Try to load a relative path or a base filename as is.
  {
    auto status_or_path = PathToResourceAsFileInternal(path);
    if (status_or_path.ok()) {
      LOG(INFO) << "Successfully loaded: " << path;
      return status_or_path;
    }
  }

  // If that fails, assume it was a relative path, and try just the base name.
  {
    const size_t last_slash_idx = path.find_last_of("\\/");
    CHECK_NE(last_slash_idx, std::string::npos);  // Make sure it's a path.
    auto base_name = path.substr(last_slash_idx + 1);
    auto status_or_path = PathToResourceAsFileInternal(base_name);
    if (status_or_path.ok()) {
      LOG(INFO) << "Successfully loaded: " << base_name;
      return status_or_path;
    }
  }

  // Try the test environment.
  absl::string_view workspace = "mediapipe";
  auto test_path = file::JoinPath(std::getenv("TEST_SRCDIR"), workspace, path);
  if (file::Exists(test_path).ok()) {
    return test_path;
  }

  return path;
}

}  // namespace mediapipe
