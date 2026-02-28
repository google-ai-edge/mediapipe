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

#ifndef MEDIAPIPE_TASKS_CC_CORE_LOGGING_TASKS_DUMMY_LOGGER_H_
#define MEDIAPIPE_TASKS_CC_CORE_LOGGING_TASKS_DUMMY_LOGGER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/tasks/cc/core/logging/logging_client.h"
#include "mediapipe/tasks/cc/core/logging/tasks_logger.h"

namespace mediapipe {
namespace tasks {
namespace core {
namespace logging {

// A dummy MediaPipe Tasks stats logger that drops all events.
class TasksDummyLogger : public TasksLogger {
 public:
  // Creates the MediaPipe Tasks stats dummy logger.
  static std::unique_ptr<TasksDummyLogger> Create(
      const std::string& task_name_str,
      const std::string& task_running_mode_str,
      std::unique_ptr<LoggingClient> logging_client,
      logs::proto::mediapipe::Platform platform,
      std::optional<absl::string_view> app_id = std::nullopt,
      std::optional<absl::string_view> app_version = std::nullopt) {
    return std::unique_ptr<TasksDummyLogger>(new TasksDummyLogger());
  }

  void LogSessionStart() override {}

  void LogSessionClone() override {}

  void RecordCpuInputArrival(Timestamp packet_timestamp) override {}

  void RecordGpuInputArrival(Timestamp packet_timestamp) override {}

  void RecordInvocationEnd(Timestamp packet_timestamp) override {}

  void LogInvocationReport(const StatsSnapshot& stats) override {}

  void LogSessionEnd() override {}

  void LogInitError() override {}

 private:
  TasksDummyLogger() = default;
};

}  // namespace logging
}  // namespace core
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_CORE_LOGGING_TASKS_DUMMY_LOGGER_H_
