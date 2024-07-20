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

#include "mediapipe/util/resource_path_manager.h"

#include <iostream>

#include "absl/strings/str_split.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

std::vector<std::string> resource_search_paths_;

void ResourcePathManager::AddSearchPath(const std::string& path)
{
  resource_search_paths_.push_back(path);
}

absl::StatusOr<std::string> ResourcePathManager::ResolveFilePath(const std::string& path)
{
   if (absl::StartsWith(path, "/")) {
    return path;
  }
  for (auto & resource_path: resource_search_paths_) {
    auto file_path = file::JoinPath(resource_path, path);
    if (file::Exists(file_path).ok()) {
      return file_path;
    }
  }
  return absl::NotFoundError("No file " + path + " found in declared search paths");
}

}  // namespace mediapipe
