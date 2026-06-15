// Copyright 2026 The MediaPipe Authors.
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

#include "mediapipe/tasks/cc/core/host_environment.h"

#include "absl/log/absl_log.h"
#include "absl/log/log.h"

namespace mediapipe {
namespace tasks {
namespace core {

HostEnvironment ToHostEnvironment(int host_environment) {
  switch (host_environment) {
    case 0:
      return HostEnvironment::HOST_ENVIRONMENT_UNKNOWN;
    case 1:
      return HostEnvironment::HOST_ENVIRONMENT_ANDROID;
    case 2:
      return HostEnvironment::HOST_ENVIRONMENT_IOS;
    case 3:
      return HostEnvironment::HOST_ENVIRONMENT_PYTHON;
    case 4:
      return HostEnvironment::HOST_ENVIRONMENT_WEB;
    default:
      ABSL_LOG(DFATAL) << "Unknown HostEnvironment int value: "
                       << host_environment;
      return HostEnvironment::HOST_ENVIRONMENT_UNKNOWN;
  }
}

HostSystem ToHostSystem(int host_system) {
  switch (host_system) {
    case 0:
      return HostSystem::HOST_SYSTEM_UNKNOWN;
    case 1:
      return HostSystem::HOST_SYSTEM_LINUX;
    case 2:
      return HostSystem::HOST_SYSTEM_MAC;
    case 3:
      return HostSystem::HOST_SYSTEM_WINDOWS;
    case 4:
      return HostSystem::HOST_SYSTEM_IOS;
    case 5:
      return HostSystem::HOST_SYSTEM_ANDROID;
    default:
      ABSL_LOG(DFATAL) << "Unknown HostSystem int value: " << host_system;
      return HostSystem::HOST_SYSTEM_UNKNOWN;
  }
}

}  // namespace core
}  // namespace tasks
}  // namespace mediapipe
