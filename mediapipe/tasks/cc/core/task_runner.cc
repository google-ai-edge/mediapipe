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

#include "mediapipe/tasks/cc/core/task_runner.h"

#include <algorithm>
#include <atomic>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/tool/name_util.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/core/model_resources_cache.h"

namespace mediapipe {
namespace tasks {
namespace core {
namespace {

absl::StatusOr<Timestamp> ValidateAndGetPacketTimestamp(
    const PacketMap& packet_map) {
  if (packet_map.empty()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument, "The provided packet map is empty.",
        MediaPipeTasksStatus::kRunnerInvalidTimestampError);
  }
  Timestamp timestamp = packet_map.begin()->second.Timestamp();
  for (const auto& [name, packet] : packet_map) {
    if (packet.Timestamp() != timestamp) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          absl::Substitute("The packets in the packet map have inconsistent "
                           "timestamps: $0 and $1.",
                           timestamp.Value(), packet.Timestamp().Value()),
          MediaPipeTasksStatus::kRunnerInvalidTimestampError);
    }
  }
  return timestamp;
}

absl::StatusOr<PacketMap> GenerateOutputPacketMap(
    const std::vector<Packet>& packets,
    const std::vector<std::string>& stream_names) {
  if (packets.empty() || packets.size() != stream_names.size()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInternal,
        absl::Substitute("Incomplete output packet vector. Expecting having $0 "
                         "output packets, but got $1 packets instead.",
                         stream_names.size(), packets.size()),
        MediaPipeTasksStatus::kRunnerUnexpectedOutputError);
  }
  PacketMap packet_map;
  std::transform(stream_names.begin(), stream_names.end(), packets.begin(),
                 std::inserter(packet_map, packet_map.end()),
                 [](const std::string& stream_name, Packet packet) {
                   return std::make_pair(stream_name, packet);
                 });
  return packet_map;
}

}  // namespace

/* static */
absl::StatusOr<std::unique_ptr<TaskRunner>> TaskRunner::Create(
    CalculatorGraphConfig config,
    std::unique_ptr<tflite::OpResolver> op_resolver,
    PacketsCallback packets_callback) {
  auto task_runner = absl::WrapUnique(new TaskRunner(packets_callback));
  MP_RETURN_IF_ERROR(
      task_runner->Initialize(std::move(config), std::move(op_resolver)));
  MP_RETURN_IF_ERROR(task_runner->Start());
  return task_runner;
}

absl::Status TaskRunner::Initialize(
    CalculatorGraphConfig config,
    std::unique_ptr<tflite::OpResolver> op_resolver) {
  if (initialized_) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Task runner is already initialized.",
        MediaPipeTasksStatus::kRunnerInitializationError);
  }
  for (const auto& output : config.output_stream()) {
    auto name = mediapipe::tool::ParseNameFromStream(output);
    if (name.empty()) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          "Graph output stream has no stream name.",
          MediaPipeTasksStatus::kRunnerInitializationError);
    }
    output_stream_names_.push_back(name);
  }
  if (output_stream_names_.empty()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Graph has no valid output streams.",
        MediaPipeTasksStatus::kRunnerInitializationError);
  }
  config.clear_output_stream();
  PacketMap input_side_packets;
  if (packets_callback_) {
    tool::AddMultiStreamCallback(
        output_stream_names_,
        [this](const std::vector<Packet>& packets) {
          packets_callback_(
              GenerateOutputPacketMap(packets, output_stream_names_));
          return;
        },
        &config, &input_side_packets,
        /*observe_timestamp_bounds=*/true);
  } else {
    mediapipe::tool::AddMultiStreamCallback(
        output_stream_names_,
        [this](const std::vector<Packet>& packets) {
          status_or_output_packets_ =
              GenerateOutputPacketMap(packets, output_stream_names_);
          return;
        },
        &config, &input_side_packets, /*observe_timestamp_bounds=*/true);
  }
  auto model_resources_cache =
      std::make_shared<ModelResourcesCache>(std::move(op_resolver));
  MP_RETURN_IF_ERROR(
      AddPayload(graph_.SetServiceObject(kModelResourcesCacheService,
                                         model_resources_cache),
                 "ModelResourcesCacheService is not set up successfully.",
                 MediaPipeTasksStatus::kRunnerModelResourcesCacheServiceError));
  MP_RETURN_IF_ERROR(
      AddPayload(graph_.Initialize(std::move(config), input_side_packets),
                 "MediaPipe CalculatorGraph is not successfully initialized.",
                 MediaPipeTasksStatus::kRunnerInitializationError));
  initialized_ = true;
  return absl::OkStatus();
}

absl::Status TaskRunner::Start() {
  if (!initialized_) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument, "Task runner is not initialized.",
        MediaPipeTasksStatus::kRunnerFailsToStartError);
  }
  if (is_running_) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument, "Task runner is already running.",
        MediaPipeTasksStatus::kRunnerFailsToStartError);
  }
  {
    absl::MutexLock lock(&mutex_);
    last_seen_ = Timestamp::Unset();
  }
  MP_RETURN_IF_ERROR(
      AddPayload(graph_.StartRun({}),
                 "MediaPipe CalculatorGraph is not successfully started.",
                 MediaPipeTasksStatus::kRunnerFailsToStartError));
  // Waits until the graph becomes idle to ensure that all calculators are
  // successfully opened.
  MP_RETURN_IF_ERROR(
      AddPayload(graph_.WaitUntilIdle(),
                 "MediaPipe CalculatorGraph is not successfully started.",
                 MediaPipeTasksStatus::kRunnerFailsToStartError));
  is_running_ = true;
  return absl::OkStatus();
}

absl::StatusOr<PacketMap> TaskRunner::Process(PacketMap inputs) {
  if (!is_running_) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Task runner is currently not running.",
        MediaPipeTasksStatus::kRunnerNotStartedError);
  }
  if (packets_callback_) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Calling TaskRunner::Process method is illegal when the result "
        "callback is provided.",
        MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError);
  }
  ASSIGN_OR_RETURN(auto input_timestamp, ValidateAndGetPacketTimestamp(inputs));
  // MediaPipe reports runtime errors through CalculatorGraph::WaitUntilIdle or
  // WaitUntilDone without indicating the exact packet timestamp.
  // To ensure that the TaskRunner::Process reports errors per invocation,
  // the rest of the method is guarded by a lock, which guarantees that only one
  // invocation can be processed in the graph concurrently.
  // TODO: Switches back to the original high performance implementation
  // when the MediaPipe CalculatorGraph can report errors in output streams.
  absl::MutexLock lock(&mutex_);
  // Assigns an internal synthetic timestamp when the input packets has no
  // assigned timestamp (packets are with the default Timestamp::Unset()).
  // Using Timestamp increment one second is to avoid interfering with the other
  // synthetic timestamps, such as those defined by BeginLoopCalculator.
  bool use_synthetic_timestamp = input_timestamp == Timestamp::Unset();
  if (use_synthetic_timestamp) {
    input_timestamp = last_seen_ == Timestamp::Unset()
                          ? Timestamp(0)
                          : last_seen_ + Timestamp::kTimestampUnitsPerSecond;
  } else if (input_timestamp <= last_seen_) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Input timestamp must be monotonically increasing.",
        MediaPipeTasksStatus::kRunnerInvalidTimestampError);
  }
  for (auto& [stream_name, packet] : inputs) {
    MP_RETURN_IF_ERROR(AddPayload(
        graph_.AddPacketToInputStream(stream_name,
                                      std::move(packet).At(input_timestamp)),
        absl::StrCat("Failed to add packet to the graph input stream: ",
                     stream_name),
        MediaPipeTasksStatus::kRunnerUnexpectedInputError));
  }
  last_seen_ = input_timestamp;
  if (!graph_.WaitUntilIdle().ok()) {
    absl::Status graph_status;
    graph_.GetCombinedErrors(&graph_status);
    return graph_status;
  }
  // When a synthetic timestamp is used, uses the timestamp of the first
  // output packet as the last seen timestamp if there is any output packet.
  if (use_synthetic_timestamp && status_or_output_packets_.ok()) {
    for (auto& kv : status_or_output_packets_.value()) {
      last_seen_ = std::max(kv.second.Timestamp(), last_seen_);
    }
  }
  return status_or_output_packets_;
}

absl::Status TaskRunner::Send(PacketMap inputs) {
  if (!is_running_) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Task runner is currently not running.",
        MediaPipeTasksStatus::kRunnerNotStartedError);
  }
  if (!packets_callback_) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Calling TaskRunner::Send method is illegal when the result "
        "callback is not provided.",
        MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError);
  }
  ASSIGN_OR_RETURN(auto input_timestamp, ValidateAndGetPacketTimestamp(inputs));
  if (!input_timestamp.IsAllowedInStream()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Calling TaskRunner::Send method with packets having invalid "
        "timestamp.",
        MediaPipeTasksStatus::kRunnerInvalidTimestampError);
  }
  absl::MutexLock lock(&mutex_);
  if (input_timestamp <= last_seen_) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Input timestamp must be monotonically increasing.",
        MediaPipeTasksStatus::kRunnerInvalidTimestampError);
  }
  for (auto& [stream_name, packet] : inputs) {
    MP_RETURN_IF_ERROR(AddPayload(
        graph_.AddPacketToInputStream(stream_name,
                                      std::move(packet).At(input_timestamp)),
        absl::Substitute("Failed to add packet to the graph input stream: $0 "
                         "at timestamp: $1",
                         stream_name, input_timestamp.Value()),
        MediaPipeTasksStatus::kRunnerUnexpectedInputError));
  }
  last_seen_ = input_timestamp;
  return absl::OkStatus();
}

absl::Status TaskRunner::Close() {
  if (!is_running_) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Task runner is currently not running.",
        MediaPipeTasksStatus::kRunnerFailsToCloseError);
  }
  is_running_ = false;
  MP_RETURN_IF_ERROR(
      AddPayload(graph_.CloseAllInputStreams(), "Fail to close intput streams",
                 MediaPipeTasksStatus::kRunnerFailsToCloseError));
  MP_RETURN_IF_ERROR(AddPayload(
      graph_.WaitUntilDone(), "Fail to shutdown the MediaPipe graph.",
      MediaPipeTasksStatus::kRunnerFailsToCloseError));
  return absl::OkStatus();
}

absl::Status TaskRunner::Restart() {
  MP_RETURN_IF_ERROR(Close());
  return Start();
}

}  // namespace core
}  // namespace tasks
}  // namespace mediapipe
