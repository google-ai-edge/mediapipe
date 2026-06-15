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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/tasks/cc/core/running_mode.h"

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
  }
  return "unknown mode";
}

inline mediapipe::tasks::core::RunningMode GetCoreRunningMode(
    RunningMode mode) {
  switch (mode) {
    case IMAGE:
      return mediapipe::tasks::core::RunningMode::kImage;
    case VIDEO:
      return mediapipe::tasks::core::RunningMode::kVideo;
    case LIVE_STREAM:
      return mediapipe::tasks::core::RunningMode::kLiveStream;
    default:
      return mediapipe::tasks::core::RunningMode::kUnspecified;
  }
}

inline absl::StatusOr<RunningMode> GetVisionRunningMode(
    mediapipe::tasks::core::RunningMode mode) {
  switch (mode) {
    case mediapipe::tasks::core::RunningMode::kImage:
      return IMAGE;
    case mediapipe::tasks::core::RunningMode::kVideo:
      return VIDEO;
    case mediapipe::tasks::core::RunningMode::kLiveStream:
      return LIVE_STREAM;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported running mode: ", static_cast<int>(mode)));
  }
}

}  // namespace core
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_CORE_RUNNING_MODE_H_
