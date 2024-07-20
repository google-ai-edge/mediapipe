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

#ifndef MEDIAPIPE_UTIL_RESOURCE_PATH_MANAGER_H_
#define MEDIAPIPE_UTIL_RESOURCE_PATH_MANAGER_H_

#include <string>

#include "mediapipe/framework/port/statusor.h"

namespace mediapipe {

// The ResourcePathManager provides additionnal search paths handling for resources (tflite...) locations.
class ResourcePathManager {
 public:
    // Adds a path to search resources for.
    static void AddSearchPath(const std::string& path);
    // Tries to resolve a filepath from path and previously added search path.
    // Either return the filepath when it exists, or absl::NotFoundError when the path doesn't exists.
    static absl::StatusOr<std::string> ResolveFilePath(const std::string& path);
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_UTIL_RESOURCE_UTIL_H_
