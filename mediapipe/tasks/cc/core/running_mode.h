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

#ifndef MEDIAPIPE_TASKS_CC_CORE_RUNNING_MODE_H_
#define MEDIAPIPE_TASKS_CC_CORE_RUNNING_MODE_H_

namespace mediapipe {
namespace tasks {
namespace core {

// The running mode of a MediaPipe Task.
enum RunningMode {
  kUnspecified,
  kImage,
  kVideo,
  kLiveStream,
  kAudioClips,
  kAudioStream,
};

}  // namespace core
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_CORE_RUNNING_MODE_H_
