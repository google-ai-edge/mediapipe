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

#include "mediapipe/java/com/google/mediapipe/framework/jni/class_registry.h"

#include "absl/strings/str_format.h"

namespace mediapipe {
namespace android {

ClassRegistry::ClassRegistry() {}

ClassRegistry& ClassRegistry::GetInstance() {
  static ClassRegistry* instance_ = new ClassRegistry();
  return *instance_;
}

void ClassRegistry::InstallRenamingMap(
    absl::node_hash_map<std::string, std::string> renaming_map) {
  renaming_map_ = renaming_map;
}

std::string ClassRegistry::GetClassName(std::string cls) {
  auto match = renaming_map_.find(cls);
  if (match != renaming_map_.end()) {
    return match->second;
  }
  return cls;
}

std::string ClassRegistry::GetMethodName(std::string cls, std::string method) {
  std::string key = absl::StrFormat("%s#%s", cls, method);
  auto match = renaming_map_.find(key);
  if (match != renaming_map_.end()) {
    return match->second;
  }
  return method;
}

}  // namespace android
}  // namespace mediapipe
