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

#include "mediapipe/framework/output_stream_handler.h"

#include <string>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/collection_item_id.h"
#include "mediapipe/framework/output_stream_shard.h"

namespace mediapipe {

absl::Status OutputStreamHandler::InitializeOutputStreamManagers(
    OutputStreamManager* flat_output_stream_managers) {
  for (CollectionItemId id = output_stream_managers_.BeginId();
       id < output_stream_managers_.EndId(); ++id) {
    output_stream_managers_.Get(id) = &flat_output_stream_managers[id.value()];
  }
  return absl::OkStatus();
}

absl::Status OutputStreamHandler::SetupOutputShards(
    OutputStreamShardSet* output_shards) {
  ABSL_CHECK(output_shards);
  for (CollectionItemId id = output_stream_managers_.BeginId();
       id < output_stream_managers_.EndId(); ++id) {
    OutputStreamManager* manager = output_stream_managers_.Get(id);
    output_shards->Get(id).SetSpec(manager->Spec());
  }
  return absl::OkStatus();
}

void OutputStreamHandler::PrepareForRun(
    const std::function<void(absl::Status)>& error_callback) {
  for (auto& manager : output_stream_managers_) {
    manager->PrepareForRun(error_callback);
  }
  absl::MutexLock lock(&timestamp_mutex_);
  completed_input_timestamps_.clear();
  task_timestamp_bound_ = Timestamp::Unset();
  propagation_state_ = kIdle;
}

void OutputStreamHandler::Open(OutputStreamShardSet* output_shards) {
  ABSL_CHECK(output_shards);
  PropagateOutputPackets(Timestamp::Unstarted(), output_shards);
  for (auto& manager : output_stream_managers_) {
    manager->PropagateHeader();
    manager->LockIntroData();
  }
}

void OutputStreamHandler::PrepareOutputs(Timestamp input_timestamp,
                                         OutputStreamShardSet* output_shards) {
  ABSL_CHECK(output_shards);
  for (CollectionItemId id = output_stream_managers_.BeginId();
       id < output_stream_managers_.EndId(); ++id) {
    output_stream_managers_.Get(id)->ResetShard(&output_shards->Get(id));
  }
}

void OutputStreamHandler::UpdateTaskTimestampBound(Timestamp timestamp) {
  if (!calculator_run_in_parallel_) {
    TryPropagateTimestampBound(timestamp);
    return;
  }
  {
    absl::MutexLock lock(&timestamp_mutex_);
    if (task_timestamp_bound_ == timestamp) {
      return;
    }
    ABSL_CHECK_GT(timestamp, task_timestamp_bound_);
    task_timestamp_bound_ = timestamp;
    if (propagation_state_ == kPropagatingBound) {
      propagation_state_ = kPropagationPending;
      return;
    }
    if (propagation_state_ != kIdle) {
      return;
    }
    PropagationLoop();
  }
}

void OutputStreamHandler::PostProcess(Timestamp input_timestamp) {
  if (!calculator_run_in_parallel_) {
    CalculatorContext* default_context =
        calculator_context_manager_->GetDefaultCalculatorContext();
    PropagateOutputPackets(input_timestamp, &default_context->Outputs());
    return;
  }
  {
    absl::MutexLock lock(&timestamp_mutex_);
    completed_input_timestamps_.insert(input_timestamp);
    if (propagation_state_ == kPropagatingBound) {
      propagation_state_ = kPropagationPending;
      return;
    }
    if (propagation_state_ != kIdle) {
      return;
    }
    PropagationLoop();
  }
}

std::string OutputStreamHandler::FirstStreamName() const {
  if (output_stream_managers_.NumEntries() == 0) {
    return std::string();
  }
  return (*output_stream_managers_.begin())->Name();
}

std::string OutputStreamHandler::DebugStreamName(CollectionItemId id) const {
  const auto tag_map = output_stream_managers_.TagMap();
  const std::string& stream_name = tag_map->Names()[id.value()];
  const auto& [stream_tag, stream_idx] = tag_map->TagAndIndexFromId(id);
  return absl::StrCat(stream_tag, ":", stream_idx, ":", stream_name);
}

std::vector<OutputStreamHandler::OutputStreamMonitoringInfo>
OutputStreamHandler::GetMonitoringInfo() {
  std::vector<OutputStreamMonitoringInfo> monitoring_info_vector;
  for (CollectionItemId id = output_stream_managers_.BeginId();
       id < output_stream_managers_.EndId(); ++id) {
    const auto& stream = output_stream_managers_.Get(id);
    if (!stream) {
      continue;
    }
    monitoring_info_vector.emplace_back(OutputStreamMonitoringInfo(
        {.stream_name = DebugStreamName(id),
         .num_packets_added = stream->NumPacketsAdded(),
         .next_timestamp_bound = stream->NextTimestampBound()}));
  }
  return monitoring_info_vector;
}

void OutputStreamHandler::TryPropagateTimestampBound(Timestamp input_bound) {
  // TODO Some non-range values, such as PostStream(), should also be
  // propagated.
  if (!input_bound.IsRangeValue()) {
    return;
  }
  OutputStreamShard empty_output;
  for (OutputStreamManager* manager : output_stream_managers_) {
    if (manager->OffsetEnabled() && !manager->IsClosed() &&
        input_bound + manager->Offset() > manager->NextTimestampBound()) {
      manager->PropagateUpdatesToMirrors(input_bound + manager->Offset(),
                                         &empty_output);
    }
  }
}

void OutputStreamHandler::Close(OutputStreamShardSet* output_shards) {
  for (CollectionItemId id = output_stream_managers_.BeginId();
       id < output_stream_managers_.EndId(); ++id) {
    if (output_shards) {
      output_stream_managers_.Get(id)->PropagateUpdatesToMirrors(
          Timestamp::Done(), &output_shards->Get(id));
    }
    output_stream_managers_.Get(id)->Close();
  }
}

void OutputStreamHandler::PropagateOutputPackets(
    Timestamp input_timestamp, OutputStreamShardSet* output_shards) {
  ABSL_CHECK(output_shards);
  for (CollectionItemId id = output_stream_managers_.BeginId();
       id < output_stream_managers_.EndId(); ++id) {
    OutputStreamManager* manager = output_stream_managers_.Get(id);
    if (manager->IsClosed()) {
      continue;
    }
    OutputStreamShard* output = &output_shards->Get(id);
    const Timestamp output_bound =
        manager->ComputeOutputTimestampBound(*output, input_timestamp);
    manager->PropagateUpdatesToMirrors(output_bound, output);
    if (output->IsClosed()) {
      manager->Close();
    }
  }
}

}  // namespace mediapipe
