/* Copyright 2022 The MediaPipe Authors.

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

#ifndef MEDIAPIPE_TASKS_CC_VISION_CORE_RUNNING_MODE_H_
#define MEDIAPIPE_TASKS_CC_VISION_CORE_RUNNING_MODE_H_

#include <string>

namespace mediapipe {
namespace tasks {
namespace vision {
namespace core {

// The running mode of a MediaPipe vision task.
enum RunningMode {
  // Run the vision task on single image inputs.
  IMAGE = 1,

  // Run the vision task on the decoded frames of an input video.
  VIDEO = 2,

  // Run the vision task on a live stream of input data, such as from camera.
  LIVE_STREAM = 3,
};

inline std::string GetRunningModeName(RunningMode mode) {
  switch (mode) {
    case IMAGE:
      return "image mode";
    case VIDEO:
      return "video mode";
    case LIVE_STREAM:
      return "live stream mode";
    default:
      return "unknown mode";
  }
}

}  // namespace core
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_CORE_RUNNING_MODE_H_
