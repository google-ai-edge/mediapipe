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

#ifndef MEDIAPIPE_TASKS_CC_CORE_LOGGING_TASKS_PROTO_LOGGER_H_
#define MEDIAPIPE_TASKS_CC_CORE_LOGGING_TASKS_PROTO_LOGGER_H_

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/tasks/cc/core/logging/logging_client.h"
#include "mediapipe/tasks/cc/core/logging/tasks_logger.h"
#include "mediapipe/util/analytics/mediapipe_log_extension.pb.h"
#include "mediapipe/util/analytics/mediapipe_logging_enums.pb.h"

namespace mediapipe {
namespace tasks {
namespace core {
namespace logging {

// The logger component that logs MediaPipe Tasks stats events via a
// LoggingClient.
class TasksStatsProtoLogger : public TasksLogger {
 public:
  // Creates the MediaPipe Tasks stats proto logger.
  static std::unique_ptr<TasksStatsProtoLogger> Create(
      const std::string& app_id, const std::string& app_version,
      const std::string& task_name_str,
      const std::string& task_running_mode_str,
      std::unique_ptr<LoggingClient> logging_client,
      logs::proto::mediapipe::Platform platform);

  // Logs the start of a MediaPipe Tasks API session.
  void LogSessionStart() override;

  // Logs the cloning of a MediaPipe Tasks API session.
  void LogSessionClone() override;

  // Records MediaPipe Tasks API receiving CPU input data.
  void RecordCpuInputArrival(int64_t packet_timestamp) override;

  // Records MediaPipe Tasks API receiving GPU input data.
  void RecordGpuInputArrival(int64_t packet_timestamp) override;

  // Records the end of a Mediapipe Tasks API invocation.
  void RecordInvocationEnd(int64_t packet_timestamp) override;

  // Logs the MediaPipe Tasks API periodic invocation report.
  void LogInvocationReport(const StatsSnapshot& stats) override;

  // Logs the Tasks API session end event.
  void LogSessionEnd() override;

  // Logs the MediaPipe Tasks API initialization error.
  void LogInitError() override;

 private:
  TasksStatsProtoLogger(logs::proto::mediapipe::SolutionName task_name,
                        logs::proto::mediapipe::SolutionMode task_running_mode,
                        const logs::proto::mediapipe::SystemInfo& system_info,
                        std::unique_ptr<LoggingClient> logging_client);

  static int64_t GetCurrentTimeMs();

  logs::proto::mediapipe::SolutionInvocationReport ProduceInvocationReport(
      const StatsSnapshot& stats);

  void LogTaskEvent(const logs::proto::mediapipe::SolutionEvent& event);

  static constexpr int64_t kReportIntervalMs = 30000L;
  static constexpr int64_t kLatencyTimeoutThresholdMs = 3000L;
  static constexpr char kTasksNamePrefix[] = "TASKS_";
  static constexpr char kTasksModePrefix[] = "MODE_TASKS_";

  std::unique_ptr<LoggingClient> logging_client_;
  const logs::proto::mediapipe::SolutionName task_name_;
  const logs::proto::mediapipe::SolutionMode task_running_mode_;
  const logs::proto::mediapipe::SystemInfo system_info_;

  const int64_t task_init_time_ms_;
  int64_t report_start_time_ms_ = 0;
  StatsSnapshot stats_snapshot_;

  std::atomic<int> cpu_input_count_{0};
  std::atomic<int> gpu_input_count_{0};
  std::atomic<int> finished_count_{0};
  std::atomic<int> dropped_count_{0};
  std::atomic<int64_t> total_latency_ms_{0};
  std::atomic<int64_t> latest_peak_latency_ms_{0};
  std::atomic<int64_t> lifetime_peak_latency_ms_{0};

  std::map<int64_t, int64_t> start_time_map_
      ABSL_GUARDED_BY(start_time_map_mutex_);
  absl::Mutex start_time_map_mutex_;
};

}  // namespace logging
}  // namespace core
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_CORE_LOGGING_TASKS_PROTO_LOGGER_H_
