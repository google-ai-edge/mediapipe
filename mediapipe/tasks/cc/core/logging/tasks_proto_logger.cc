// Copyright 2025 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/tasks/cc/core/logging/tasks_proto_logger.h"

#include <atomic>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/ascii.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "mediapipe/tasks/cc/core/logging/logging_client.h"
#include "mediapipe/tasks/cc/core/logging/tasks_logger.h"
#include "mediapipe/util/analytics/mediapipe_log_extension.pb.h"
#include "mediapipe/util/analytics/mediapipe_logging_enums.pb.h"

namespace mediapipe {
namespace tasks {
namespace core {
namespace logging {

namespace {
using logs::proto::mediapipe::ErrorCode;
using logs::proto::mediapipe::EventName;
using logs::proto::mediapipe::InputDataType;
using logs::proto::mediapipe::MediaPipeLogExtension;
using logs::proto::mediapipe::SolutionError;
using logs::proto::mediapipe::SolutionEvent;
using logs::proto::mediapipe::SolutionInvocationReport;
using logs::proto::mediapipe::SolutionMode;
using logs::proto::mediapipe::SolutionName;
using logs::proto::mediapipe::SolutionSessionClone;
using logs::proto::mediapipe::SolutionSessionEnd;
using logs::proto::mediapipe::SolutionSessionStart;
using logs::proto::mediapipe::SystemInfo;
}  // namespace

// static
std::unique_ptr<TasksStatsProtoLogger> TasksStatsProtoLogger::Create(
    const std::string& app_id, const std::string& app_version,
    const std::string& task_name_str, const std::string& task_running_mode_str,
    std::unique_ptr<LoggingClient> logging_client,
    logs::proto::mediapipe::Platform platform) {
  SolutionName task_name;
  if (!SolutionName_Parse(
          std::string(kTasksNamePrefix) + absl::AsciiStrToUpper(task_name_str),
          &task_name)) {
    task_name = SolutionName::SOLUTION_UNKNOWN;
  }

  SolutionMode task_running_mode;
  if (!SolutionMode_Parse(std::string(kTasksModePrefix) +
                              absl::AsciiStrToUpper(task_running_mode_str),
                          &task_running_mode)) {
    task_running_mode = SolutionMode::MODE_TASKS_UNSPECIFIED;
  }

  SystemInfo system_info;
  system_info.set_platform(platform);
  system_info.set_app_id(app_id);
  system_info.set_app_version(app_version);
  return std::unique_ptr<TasksStatsProtoLogger>(new TasksStatsProtoLogger(
      task_name, task_running_mode, system_info, std::move(logging_client)));
}

void TasksStatsProtoLogger::LogSessionStart() {
  SolutionSessionStart session_start;
  session_start.set_mode(task_running_mode_);
  session_start.set_init_latency_ms(GetCurrentTimeMs() - task_init_time_ms_);

  SolutionEvent event;
  event.set_solution_name(task_name_);
  event.set_event_name(EventName::EVENT_START);
  *event.mutable_session_start() = session_start;

  LogTaskEvent(event);
  report_start_time_ms_ = GetCurrentTimeMs();
  stats_snapshot_ = {};
}

void TasksStatsProtoLogger::LogSessionClone() {
  SolutionSessionClone session_clone;
  session_clone.set_mode(task_running_mode_);
  session_clone.set_init_latency_ms(GetCurrentTimeMs() - task_init_time_ms_);

  SolutionEvent event;
  event.set_solution_name(task_name_);
  event.set_event_name(EventName::EVENT_START);
  *event.mutable_session_clone() = session_clone;

  LogTaskEvent(event);
  report_start_time_ms_ = GetCurrentTimeMs();
  stats_snapshot_ = {};
}

void TasksStatsProtoLogger::RecordCpuInputArrival(int64_t packet_timestamp) {
  absl::MutexLock lock(start_time_map_mutex_);
  ++cpu_input_count_;
  start_time_map_[packet_timestamp] = GetCurrentTimeMs();
}

void TasksStatsProtoLogger::RecordGpuInputArrival(int64_t packet_timestamp) {
  absl::MutexLock lock(start_time_map_mutex_);
  ++gpu_input_count_;
  start_time_map_[packet_timestamp] = GetCurrentTimeMs();
}

void TasksStatsProtoLogger::RecordInvocationEnd(int64_t packet_timestamp) {
  int64_t start_time_ms;
  {
    absl::MutexLock lock(start_time_map_mutex_);
    auto it = start_time_map_.find(packet_timestamp);
    if (it == start_time_map_.end()) {
      return;
    }
    start_time_ms = it->second;
    start_time_map_.erase(it);
  }

  int64_t current_latency_ms = GetCurrentTimeMs() - start_time_ms;
  finished_count_++;
  if (current_latency_ms > kLatencyTimeoutThresholdMs) {
    return;
  }
  total_latency_ms_ += current_latency_ms;
  int64_t prev_peak = latest_peak_latency_ms_.load();
  while (prev_peak < current_latency_ms &&
         !latest_peak_latency_ms_.compare_exchange_weak(prev_peak,
                                                        current_latency_ms)) {
  }
  prev_peak = lifetime_peak_latency_ms_.load();
  while (prev_peak < current_latency_ms &&
         !lifetime_peak_latency_ms_.compare_exchange_weak(prev_peak,
                                                          current_latency_ms)) {
  }

  bool should_log = false;
  StatsSnapshot latest_snapshot;
  {
    absl::MutexLock lock(start_time_map_mutex_);
    if (GetCurrentTimeMs() > report_start_time_ms_ + kReportIntervalMs) {
      should_log = true;
      auto it = start_time_map_.lower_bound(packet_timestamp);
      dropped_count_ += std::distance(start_time_map_.begin(), it);
      start_time_map_.erase(start_time_map_.begin(), it);

      latest_snapshot = {
          cpu_input_count_.load(),
          gpu_input_count_.load(),
          finished_count_.load(),
          dropped_count_.load(),
          total_latency_ms_.load(),
          latest_peak_latency_ms_.exchange(0),
          GetCurrentTimeMs() - report_start_time_ms_,
      };
      report_start_time_ms_ = GetCurrentTimeMs();
    }
  }

  if (should_log) {
    StatsSnapshot stats_snapshot_diff = {
        latest_snapshot.cpu_input_count - stats_snapshot_.cpu_input_count,
        latest_snapshot.gpu_input_count - stats_snapshot_.gpu_input_count,
        latest_snapshot.finished_count - stats_snapshot_.finished_count,
        latest_snapshot.dropped_count - stats_snapshot_.dropped_count,
        latest_snapshot.total_latency_ms - stats_snapshot_.total_latency_ms,
        latest_snapshot.peak_latency_ms,
        latest_snapshot.elapsed_time_ms,
    };
    LogInvocationReport(stats_snapshot_diff);
    stats_snapshot_ = latest_snapshot;
  }
}

void TasksStatsProtoLogger::LogInvocationReport(const StatsSnapshot& stats) {
  SolutionEvent event;
  event.set_solution_name(task_name_);
  event.set_event_name(EventName::EVENT_INVOCATONS);
  *event.mutable_invocation_report() = ProduceInvocationReport(stats);
  LogTaskEvent(event);
}

void TasksStatsProtoLogger::LogSessionEnd() {
  absl::MutexLock lock(start_time_map_mutex_);
  const StatsSnapshot final_stats = {
      cpu_input_count_.load(),
      gpu_input_count_.load(),
      finished_count_.load(),
      dropped_count_.load() + static_cast<int>(start_time_map_.size()),
      total_latency_ms_.load(),
      lifetime_peak_latency_ms_.load(),
      GetCurrentTimeMs() - task_init_time_ms_,
  };

  SolutionSessionEnd session_end;
  *session_end.mutable_invocation_report() =
      ProduceInvocationReport(final_stats);

  SolutionEvent event;
  event.set_solution_name(task_name_);
  event.set_event_name(EventName::EVENT_END);
  *event.mutable_session_end() = session_end;
  LogTaskEvent(event);
}

void TasksStatsProtoLogger::LogInitError() {
  SolutionError error;
  error.set_error_code(ErrorCode::ERROR_INIT);

  SolutionEvent event;
  event.set_solution_name(task_name_);
  event.set_event_name(EventName::EVENT_ERROR);
  *event.mutable_error_details() = error;
  LogTaskEvent(event);
}

TasksStatsProtoLogger::TasksStatsProtoLogger(
    SolutionName task_name, SolutionMode task_running_mode,
    const SystemInfo& system_info,
    std::unique_ptr<LoggingClient> logging_client)
    : logging_client_(std::move(logging_client)),
      task_name_(task_name),
      task_running_mode_(task_running_mode),
      system_info_(system_info),
      task_init_time_ms_(GetCurrentTimeMs()) {}

// static
int64_t TasksStatsProtoLogger::GetCurrentTimeMs() {
  return absl::ToUnixMillis(absl::Now());
}

SolutionInvocationReport TasksStatsProtoLogger::ProduceInvocationReport(
    const StatsSnapshot& stats) {
  SolutionInvocationReport report;
  report.set_mode(task_running_mode_);
  report.set_dropped(stats.dropped_count);
  report.set_pipeline_peak_latency_ms(stats.peak_latency_ms);
  if (stats.finished_count > 0) {
    report.set_pipeline_average_latency_ms(stats.total_latency_ms /
                                           stats.finished_count);
  }
  report.set_elapsed_time_ms(stats.elapsed_time_ms);
  if (stats.cpu_input_count != 0) {
    auto* invocation_count = report.add_invocation_count();
    invocation_count->set_input_data_type(InputDataType::INPUT_TYPE_TASKS_CPU);
    invocation_count->set_count(stats.cpu_input_count);
  }
  if (stats.gpu_input_count != 0) {
    auto* invocation_count = report.add_invocation_count();
    invocation_count->set_input_data_type(InputDataType::INPUT_TYPE_TASKS_GPU);
    invocation_count->set_count(stats.gpu_input_count);
  }
  return report;
}

void TasksStatsProtoLogger::LogTaskEvent(const SolutionEvent& event) {
  if (logging_client_) {
    MediaPipeLogExtension log;
    *log.mutable_system_info() = system_info_;
    *log.mutable_solution_event() = event;
    logging_client_->LogEvent(log);
  }
}

}  // namespace logging
}  // namespace core
}  // namespace tasks
}  // namespace mediapipe
