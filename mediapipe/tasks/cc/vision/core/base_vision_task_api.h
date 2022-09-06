/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

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

#ifndef MEDIAPIPE_TASKS_CC_VISION_CORE_BASE_VISION_TASK_API_H_
#define MEDIAPIPE_TASKS_CC_VISION_CORE_BASE_VISION_TASK_API_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/tasks/cc/core/base_task_api.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace core {

// The base class of the user-facing mediapipe vision task api classes.
class BaseVisionTaskApi : public tasks::core::BaseTaskApi {
 public:
  // Constructor.
  explicit BaseVisionTaskApi(std::unique_ptr<tasks::core::TaskRunner> runner,
                             RunningMode running_mode)
      : BaseTaskApi(std::move(runner)), running_mode_(running_mode) {}

 protected:
  // A synchronous method to process single image inputs.
  // The call blocks the current thread until a failure status or a successful
  // result is returned.
  absl::StatusOr<tasks::core::PacketMap> ProcessImageData(
      tasks::core::PacketMap inputs) {
    if (running_mode_ != RunningMode::IMAGE) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          absl::StrCat("Task is not initialized with the image mode. Current "
                       "running mode:",
                       GetRunningModeName(running_mode_)),
          MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError);
    }
    return runner_->Process(std::move(inputs));
  }

  // A synchronous method to process continuous video frames.
  // The call blocks the current thread until a failure status or a successful
  // result is returned.
  absl::StatusOr<tasks::core::PacketMap> ProcessVideoData(
      tasks::core::PacketMap inputs) {
    if (running_mode_ != RunningMode::VIDEO) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          absl::StrCat("Task is not initialized with the video mode. Current "
                       "running mode:",
                       GetRunningModeName(running_mode_)),
          MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError);
    }
    return runner_->Process(std::move(inputs));
  }

  // An asynchronous method to send live stream data to the runner. The results
  // will be available in the user-defined results callback.
  absl::Status SendLiveStreamData(tasks::core::PacketMap inputs) {
    if (running_mode_ != RunningMode::LIVE_STREAM) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          absl::StrCat("Task is not initialized with the live stream mode. "
                       "Current running mode:",
                       GetRunningModeName(running_mode_)),
          MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError);
    }
    return runner_->Send(std::move(inputs));
  }

 private:
  RunningMode running_mode_;
};

}  // namespace core
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_CORE_BASE_VISION_TASK_API_H_
