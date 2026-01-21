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

#ifndef MEDIAPIPE_TASKS_CORE_LOGGING_TASKS_STATS_TASKS_LOGGER_H_
#define MEDIAPIPE_TASKS_CORE_LOGGING_TASKS_STATS_TASKS_LOGGER_H_

#include <cstdint>

namespace mediapipe {
namespace tasks {
namespace core {
namespace logging {

// The stats logger interface that defines what MediaPipe Tasks events to log
class TasksLogger {
 public:
  // Task stats snapshot. A struct to hold the statistics at a point in time.
  struct StatsSnapshot {
    int cpu_input_count = 0;
    int gpu_input_count = 0;
    int finished_count = 0;
    int dropped_count = 0;
    int64_t total_latency_ms = 0;
    int64_t peak_latency_ms = 0;
    int64_t elapsed_time_ms = 0;
  };

  virtual ~TasksLogger() = default;

  // Logs the start of a MediaPipe Tasks API session.
  virtual void LogSessionStart() = 0;

  // Logs the cloning of a MediaPipe Tasks API session.
  virtual void LogSessionClone() = 0;

  // Records MediaPipe Tasks API receiving CPU input data.
  virtual void RecordCpuInputArrival(int64_t packet_timestamp) = 0;

  // Records MediaPipe Tasks API receiving GPU input data.
  virtual void RecordGpuInputArrival(int64_t packet_timestamp) = 0;

  // Records the end of a Mediapipe Tasks API invocation.
  virtual void RecordInvocationEnd(int64_t packet_timestamp) = 0;

  // Logs the MediaPipe Tasks API periodic invocation report.
  virtual void LogInvocationReport(const StatsSnapshot& stats) = 0;

  // Logs the Tasks API session end event.
  virtual void LogSessionEnd() = 0;

  // Logs the MediaPipe Tasks API initialization error.
  virtual void LogInitError() = 0;
};

}  // namespace logging
}  // namespace core
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CORE_LOGGING_TASKS_STATS_TASKS_LOGGER_H_
