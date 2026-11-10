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

#ifndef MEDIAPIPE_TASKS_CC_CORE_TASK_RUNNER_TEST_PEER_H_
#define MEDIAPIPE_TASKS_CC_CORE_TASK_RUNNER_TEST_PEER_H_

#include <memory>

#include "mediapipe/framework/graph_service.h"
#include "mediapipe/tasks/cc/core/logging/tasks_logger.h"
#include "mediapipe/tasks/cc/core/task_runner.h"

namespace mediapipe {
namespace tasks {
namespace core {

class TaskRunnerTestPeer {
 public:
  template <typename T>
  static std::shared_ptr<T> GetServiceObject(TaskRunner& runner,
                                             const GraphService<T>& service) {
    return runner.graph_.GetServiceObject(service);
  }

  static const logging::LoggingOptions* GetLoggingOptions(
      const TaskRunner& runner) {
    return runner.tasks_logger_ ? runner.tasks_logger_->logging_options()
                                : nullptr;
  }

  template <typename TaskApi>
  static const logging::LoggingOptions* GetLoggingOptions(const TaskApi& task) {
    return task.runner_ ? GetLoggingOptions(*task.runner_) : nullptr;
  }
};

}  // namespace core
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_CORE_TASK_RUNNER_TEST_PEER_H_
