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

#include "absl/base/attributes.h"
#include "mediapipe/framework/graph_service.h"

namespace mediapipe {

// Handles to the LiteRT facilities provided by the system, which are acquired
// through the Google Play Services API. A default-initialized member means the
// corresponding facility was not provided.
//
// TODO: Link to the public documentation once it is available.
struct LiteRtSystemHandles {
  // Handle to the LiteRT runtime.
  uintptr_t runtime = 0;

  // Handle to the GPU accelerator, which enables GPU acceleration for models
  // running on the system LiteRT runtime.
  uintptr_t gpu_accelerator = 0;

  // Context of the system LiteRT runtime, which enables its logging.
  const void* context = nullptr;
};

class LiteRtService {
 public:
  explicit LiteRtService(std::string dispatch_library_path,
                         LiteRtSystemHandles system_handles = {})
      : dispatch_library_path_(std::move(dispatch_library_path)),
        system_handles_(system_handles) {}

  // Returns the dispatch library path.
  const std::string& GetDispatchLibraryPath() const {
    return dispatch_library_path_;
  }

  // Returns the externally provided LiteRT handles, which are acquired
  // through the Google Play Services API, or null pointers if not provide.
  //
  // TODO: Link to the public documentation once it is available.
  const LiteRtSystemHandles& GetSystemHandles() const {
    return system_handles_;
  }

  ABSL_DEPRECATED("Use GetSystemHandles().runtime instead.")
  uintptr_t GetSystemRuntimeHandle() const { return system_handles_.runtime; }

 private:
  std::string dispatch_library_path_;
  LiteRtSystemHandles system_handles_;
};

// Service for providing the native library path from the Java side to the
// MediaPipe graph, e.g. for LiteRT inference calculator.
inline constexpr GraphService<LiteRtService> kLiteRtService(
    "kLiteRtService", GraphServiceBase::kDisallowDefaultInitialization);

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_PORT_APK_NATIVE_LIB_SERVICE_H_
