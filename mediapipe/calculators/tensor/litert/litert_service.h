// Copyright 2025 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_FRAMEWORK_PORT_APK_NATIVE_LIB_SERVICE_H_
#define MEDIAPIPE_FRAMEWORK_PORT_APK_NATIVE_LIB_SERVICE_H_

#include <cstdint>
#include <string>
#include <utility>

#include "mediapipe/framework/graph_service.h"

namespace mediapipe {

class LiteRtService {
 public:
  explicit LiteRtService(std::string dispatch_library_path,
                         uintptr_t system_runtime_handle = 0)
      : dispatch_library_path_(std::move(dispatch_library_path)),
        system_runtime_handle_(system_runtime_handle) {}

  // Returns the dispatch library path.
  const std::string& GetDispatchLibraryPath() const {
    return dispatch_library_path_;
  }

  // Returns the externally provided LiteRT runtime handle, which is acquired
  // through the Google Play Services API, or 0 if not provided.
  //
  // TODO: Link to the public documentation once it is available.
  uintptr_t GetSystemRuntimeHandle() const { return system_runtime_handle_; }

 private:
  std::string dispatch_library_path_;
  uintptr_t system_runtime_handle_;
};

// Service for providing the native library path from the Java side to the
// MediaPipe graph, e.g. for LiteRT inference calculator.
inline constexpr GraphService<LiteRtService> kLiteRtService(
    "kLiteRtService", GraphServiceBase::kDisallowDefaultInitialization);

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_PORT_APK_NATIVE_LIB_SERVICE_H_
