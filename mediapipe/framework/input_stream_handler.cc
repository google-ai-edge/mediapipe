// Copyright 2019 The MediaPipe Authors.
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

#include "mediapipe/framework/input_stream_handler.h"

#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/collection_item_id.h"
#include "mediapipe/framework/mediapipe_profiling.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

::mediapipe::Status InputStreamHandler::InitializeInputStreamManagers(
    InputStreamManager* flat_input_stream_managers) {
  for (CollectionItemId id = input_stream_managers_.BeginId();
       id < input_stream_managers_.EndId(); ++id) {
    input_stream_managers_.Get(id) = &flat_input_stream_managers[id.value()];
  }
  return ::mediapipe::OkStatus();
}

InputStreamManager* InputStreamHandler::GetInputStreamManager(
    CollectionItemId id) {
  return input_stream_managers_.Get(id);
}

::mediapipe::Status InputStreamHandler::SetupInputShards(
    InputStreamShardSet* input_shards) {
  RET_CHECK(input_shards);
  for (CollectionItemId id = input_stream_managers_.BeginId();
       id < input_stream_managers_.EndId(); ++id) {
    const auto& manager = input_stream_managers_.Get(id);
    // Invokes InputStreamShard's private method to set name and header.
    input_shards->Get(id).SetName(&manager->Name());
    input_shards->Get(id).SetHeader(manager->Header());
  }
  return ::mediapipe::OkStatus();
}

std::vector<std::pair<std::string, int>>
InputStreamHandler::GetMonitoringInfo() {
  std::vector<std::pair<std::string, int>> monitoring_info_vector;
  for (auto& stream : input_stream_managers_) {
    if (!stream) {
      continue;
    }
    monitoring_info_vector.emplace_back(
        std::pair<std::string, int>(stream->Name(), stream->QueueSize()));
  }
  return monitoring_info_vector;
}

void InputStreamHandler::PrepareForRun(
    std::function<void()> headers_ready_callback,
    std::function<void()> notification_callback,
    std::function<void(CalculatorContext*)> schedule_callback,
    std::function<void(::mediapipe::Status)> error_callback) {
  headers_ready_callback_ = std::move(headers_ready_callback);
  notification_ = std::move(notification_callback);
  schedule_callback_ = std::move(schedule_callback);
  error_callback_ = std::move(error_callback);
  int unset_header_count = 0;
  for (const auto& stream : input_stream_managers_) {
    if (!stream->BackEdge()) {
      ++unset_header_count;
    }
    stream->PrepareForRun();
  }
  unset_header_count_.store(unset_header_count, std::memory_order_relaxed);
  prepared_context_for_close_ = false;
}

void InputStreamHandler::SetQueueSizeCallbacks(
    InputStreamManager::QueueSizeCallback becomes_full_callback,
    InputStreamManager::QueueSizeCallback becomes_not_full_callback) {
  for (auto& stream : input_stream_managers_) {
    stream->SetQueueSizeCallbacks(becomes_full_callback,
                                  becomes_not_full_callback);
  }
}

void InputStreamHandler::SetHeader(CollectionItemId id, const Packet& header) {
  ::mediapipe::Status result =
      input_stream_managers_.Get(id)->SetHeader(header);
  if (!result.ok()) {
    error_callback_(result);
    return;
  }
  if (!input_stream_managers_.Get(id)->BackEdge()) {
    CHECK_GT(unset_header_count_, 0);
    if (unset_header_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      headers_ready_callback_();
    }
  }
}

void InputStreamHandler::UpdateInputShardHeaders(
    InputStreamShardSet* input_shards) {
  CHECK(input_shards);
  for (CollectionItemId id = input_stream_managers_.BeginId();
       id < input_stream_managers_.EndId(); ++id) {
    input_shards->Get(id).SetHeader(input_stream_managers_.Get(id)->Header());
  }
}

void InputStreamHandler::SetMaxQueueSize(CollectionItemId id,
                                         int max_queue_size) {
  input_stream_managers_.Get(id)->SetMaxQueueSize(max_queue_size);
}

void InputStreamHandler::SetMaxQueueSize(int max_queue_size) {
  for (auto& stream : input_stream_managers_) {
    stream->SetMaxQueueSize(max_queue_size);
  }
}

std::string InputStreamHandler::DebugStreamNames() const {
  std::vector<absl::string_view> stream_names;
  for (const auto& stream : input_stream_managers_) {
    stream_names.push_back(stream->Name());
  }
  if (stream_names.empty()) {
    return "no input streams";
  }
  if (stream_names.size() == 1) {
    return absl::StrCat("input stream: <", stream_names[0], ">");
  }
  return absl::StrCat("input streams: <", absl::StrJoin(stream_names, ","),
                      ">");
}

bool InputStreamHandler::ScheduleInvocations(int max_allowance,
                                             Timestamp* input_bound) {
  *input_bound = Timestamp::Unset();
  Timestamp min_stream_timestamp = Timestamp::Done();
  if (input_stream_managers_.NumEntries() == 0) {
    // A source node doesn't require any input packets.
    CalculatorContext* default_context =
        calculator_context_manager_->GetDefaultCalculatorContext();
    schedule_callback_(default_context);
    return true;
  }
  int invocations_scheduled = 0;
  while (invocations_scheduled < max_allowance) {
    NodeReadiness node_readiness = GetNodeReadiness(&min_stream_timestamp);
    // Sets *input_bound iff the latest node readiness is kNotReady before the
    // function returns regardless of how many invocations have been scheduled.
    if (node_readiness == NodeReadiness::kNotReady) {
      if (batch_size_ > 1 &&
          calculator_context_manager_->ContextHasInputTimestamp(
              *calculator_context_manager_->GetDefaultCalculatorContext())) {
        // When batching is in progress, input_bound stays equal to the first
        // timestamp in the calculator context. This allows timestamp
        // propagation to be performed only for the first timestamp, and
        // prevents propagation for the subsequent inputs.
        *input_bound =
            calculator_context_manager_->GetDefaultCalculatorContext()
                ->InputTimestamp();
      } else {
        *input_bound = min_stream_timestamp;
      }
      CalculatorContext* default_context =
          calculator_context_manager_->GetDefaultCalculatorContext();
      ::mediapipe::LogEvent(default_context->GetProfilingContext(),
                            TraceEvent(TraceEvent::NOT_READY)
                                .set_node_id(default_context->NodeId()));
      break;
    } else if (node_readiness == NodeReadiness::kReadyForProcess) {
      CalculatorContext* calculator_context =
          calculator_context_manager_->PrepareCalculatorContext(
              min_stream_timestamp);
      calculator_context_manager_->PushInputTimestampToContext(
          calculator_context, min_stream_timestamp);
      if (!late_preparation_) {
        FillInputSet(min_stream_timestamp, &calculator_context->Inputs());
      }
      if (calculator_context_manager_->NumberOfContextTimestamps(
              *calculator_context) == batch_size_) {
        schedule_callback_(calculator_context);
        ++invocations_scheduled;
      }
      ::mediapipe::LogEvent(calculator_context->GetProfilingContext(),
                            TraceEvent(TraceEvent::READY_FOR_PROCESS)
                                .set_node_id(calculator_context->NodeId()));
    } else {
      CHECK(node_readiness == NodeReadiness::kReadyForClose);
      // If any parallel invocations are in progress or a calculator context has
      // been prepared for Close(), we shouldn't prepare another calculator
      // context for Close().
      if (calculator_context_manager_->HasActiveContexts() ||
          prepared_context_for_close_) {
        break;
      }
      // If there is an incomplete batch of input sets in the calculator
      // context, it gets scheduled when the calculator is ready for close.
      CalculatorContext* default_context =
          calculator_context_manager_->GetDefaultCalculatorContext();
      calculator_context_manager_->PushInputTimestampToContext(
          default_context, Timestamp::Done());
      schedule_callback_(default_context);
      ++invocations_scheduled;
      prepared_context_for_close_ = true;
      ::mediapipe::LogEvent(default_context->GetProfilingContext(),
                            TraceEvent(TraceEvent::READY_FOR_CLOSE)
                                .set_node_id(default_context->NodeId()));
      break;
    }
  }
  return invocations_scheduled > 0;
}

void InputStreamHandler::FinalizeInputSet(Timestamp timestamp,
                                          InputStreamShardSet* input_set) {
  if (late_preparation_) {
    FillInputSet(timestamp, input_set);
  }
}

void InputStreamHandler::AddPackets(CollectionItemId id,
                                    const std::list<Packet>& packets) {
  bool notify = false;
  ::mediapipe::Status result =
      input_stream_managers_.Get(id)->AddPackets(packets, &notify);
  if (!result.ok()) {
    error_callback_(result);
  }
  if (notify) {
    notification_();
  }
}

void InputStreamHandler::MovePackets(CollectionItemId id,
                                     std::list<Packet>* packets) {
  bool notify = false;
  ::mediapipe::Status result =
      input_stream_managers_.Get(id)->MovePackets(packets, &notify);
  if (!result.ok()) {
    error_callback_(result);
  }
  if (notify) {
    notification_();
  }
}

void InputStreamHandler::SetNextTimestampBound(CollectionItemId id,
                                               Timestamp bound) {
  bool notify = false;
  ::mediapipe::Status result =
      input_stream_managers_.Get(id)->SetNextTimestampBound(bound, &notify);
  if (!result.ok()) {
    error_callback_(result);
  }
  if (notify) {
    notification_();
  }
}

void InputStreamHandler::ClearCurrentInputs(
    CalculatorContext* calculator_context) {
  CHECK(calculator_context);
  calculator_context_manager_->PopInputTimestampFromContext(calculator_context);
  for (auto& input : calculator_context->Inputs()) {
    // Invokes InputStreamShard's private method to clear packet.
    input.ClearCurrentPacket();
  }
}

void InputStreamHandler::Close() {
  for (auto& stream : input_stream_managers_) {
    stream->Close();
  }
}

void InputStreamHandler::SetBatchSize(int batch_size) {
  CHECK(!calculator_run_in_parallel_ || batch_size == 1)
      << "Batching cannot be combined with parallel execution.";
  CHECK(!late_preparation_ || batch_size == 1)
      << "Batching cannot be combined with late preparation.";
  CHECK_GE(batch_size, 1) << "Batch size has to be greater than or equal to 1.";
  // Source nodes shouldn't specify batch_size even if it's set to 1.
  CHECK_GE(NumInputStreams(), 0) << "Source nodes cannot batch input packets.";
  batch_size_ = batch_size;
}

void InputStreamHandler::SetLatePreparation(bool late_preparation) {
  CHECK(batch_size_ == 1 || !late_preparation_)
      << "Batching cannot be combined with late preparation.";
  late_preparation_ = late_preparation;
}

}  // namespace mediapipe
