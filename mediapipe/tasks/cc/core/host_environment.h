/* Copyright 2026 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MEDIAPIPE_TASKS_CC_CORE_HOST_ENVIRONMENT_H
#define MEDIAPIPE_TASKS_CC_CORE_HOST_ENVIRONMENT_H

namespace mediapipe {
namespace tasks {
namespace core {

// The platform that MediaPipe runs on.
enum HostEnvironment {
  HOST_ENVIRONMENT_UNKNOWN = 0,
  HOST_ENVIRONMENT_ANDROID = 1,
  HOST_ENVIRONMENT_IOS = 2,
  HOST_ENVIRONMENT_PYTHON = 3,
  HOST_ENVIRONMENT_WEB = 4,
};

// Host OS that MediaPipe runs on.
enum HostSystem {
  HOST_SYSTEM_UNKNOWN = 0,
  HOST_SYSTEM_LINUX = 1,
  HOST_SYSTEM_MAC = 2,
  HOST_SYSTEM_WINDOWS = 3,
  HOST_SYSTEM_IOS = 4,
  HOST_SYSTEM_ANDROID = 5,
};

}  // namespace core
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_CORE_HOST_ENVIRONMENT_H
