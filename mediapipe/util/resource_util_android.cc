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

#include "absl/strings/match.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/singleton.h"
#include "mediapipe/util/android/asset_manager_util.h"
#include "mediapipe/util/android/file/base/helpers.h"
#include "mediapipe/util/resource_util.h"

namespace mediapipe {

::mediapipe::StatusOr<std::string> PathToResourceAsFile(
    const std::string& path) {
  if (absl::StartsWith(path, "/")) {
    return path;
  }

  return Singleton<AssetManager>::get()->CachedFileFromAsset(path);
}

::mediapipe::Status GetResourceContents(const std::string& path,
                                        std::string* output) {
  if (absl::StartsWith(path, "/")) {
    return file::GetContents(path, output, file::Defaults());
  }

  std::vector<uint8_t> data;
  RET_CHECK(Singleton<AssetManager>::get()->ReadFile(path, &data))
      << "could not read asset: " << path;
  output->assign(reinterpret_cast<char*>(data.data()), data.size());
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
