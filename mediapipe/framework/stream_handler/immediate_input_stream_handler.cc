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
#include "mediapipe/framework/stream_handler/immediate_input_stream_handler.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/calculator_context_manager.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/collection_item_id.h"
#include "mediapipe/framework/input_stream_handler.h"
#include "mediapipe/framework/mediapipe_options.pb.h"
#include "mediapipe/framework/tool/tag_map.h"

namespace mediapipe {

using SyncSet = InputStreamHandler::SyncSet;

REGISTER_INPUT_STREAM_HANDLER(ImmediateInputStreamHandler);

ImmediateInputStreamHandler::ImmediateInputStreamHandler(
    std::shared_ptr<tool::TagMap> tag_map,
    CalculatorContextManager* calculator_context_manager,
    const mediapipe::MediaPipeOptions& options, bool calculator_run_in_parallel)
    : InputStreamHandler(tag_map, calculator_context_manager, options,
                         calculator_run_in_parallel) {
  for (auto id = tag_map->BeginId(); id < tag_map->EndId(); ++id) {
    sync_sets_.emplace_back(this, std::vector<CollectionItemId>{id});
    ready_timestamps_.push_back(Timestamp::Unset());
  }
}

void ImmediateInputStreamHandler::PrepareForRun(
    std::function<void()> headers_ready_callback,
    std::function<void()> notification_callback,
    std::function<void(CalculatorContext*)> schedule_callback,
    std::function<void(absl::Status)> error_callback) {
  {
    absl::MutexLock lock(&mutex_);
    for (int i = 0; i < sync_sets_.size(); ++i) {
      sync_sets_[i].PrepareForRun();
      ready_timestamps_[i] = Timestamp::Unset();
    }
  }
  InputStreamHandler::PrepareForRun(
      std::move(headers_ready_callback), std::move(notification_callback),
      std::move(schedule_callback), std::move(error_callback));
}

NodeReadiness ImmediateInputStreamHandler::GetNodeReadiness(
    Timestamp* min_stream_timestamp) {
  absl::MutexLock lock(&mutex_);
  Timestamp input_timestamp = Timestamp::Done();
  Timestamp min_bound = Timestamp::Done();
  bool stream_became_done = false;
  for (int i = 0; i < sync_sets_.size(); ++i) {
    if (ready_timestamps_[i] > Timestamp::Unset()) {
      min_bound = std::min(min_bound, ready_timestamps_[i]);
      input_timestamp = std::min(input_timestamp, ready_timestamps_[i]);
      continue;
    }
    Timestamp prev_ts = sync_sets_[i].LastProcessed();
    Timestamp stream_ts;
    NodeReadiness readiness = sync_sets_[i].GetReadiness(&stream_ts);
    min_bound = std::min(min_bound, stream_ts);
    if (readiness == NodeReadiness::kReadyForProcess) {
      ready_timestamps_[i] = stream_ts;
      input_timestamp = std::min(input_timestamp, stream_ts);
    } else if (readiness == NodeReadiness::kReadyForClose) {
      ABSL_CHECK_EQ(stream_ts, Timestamp::Done());
      if (ProcessTimestampBounds()) {
        // With kReadyForClose, the timestamp-bound Done is returned.
        // TODO: Make all InputStreamHandlers process Done() like this.
        static const Timestamp kDonePrecedingTimestamp =
            Timestamp::Done().PreviousAllowedInStream();
        if (prev_ts < kDonePrecedingTimestamp) {
          // When kReadyForClose is received for the first time for a sync set,
          // it is processed using the timestamp preceding Done() to indicate
          // input stream is done, but still needs to be processed.
          min_bound = std::min(min_bound, kDonePrecedingTimestamp);
          input_timestamp = std::min(input_timestamp, kDonePrecedingTimestamp);
          ready_timestamps_[i] = kDonePrecedingTimestamp;
        } else {
          ready_timestamps_[i] = Timestamp::Done();
        }
      } else if (prev_ts < Timestamp::Done()) {
        stream_became_done = true;
        ready_timestamps_[i] = Timestamp::Done();
      }
    }
  }
  *min_stream_timestamp = min_bound;

  if (*min_stream_timestamp == Timestamp::Done()) {
    return NodeReadiness::kReadyForClose;
  }

  if (input_timestamp < Timestamp::Done()) {
    // On kReadyForProcess, the input_timestamp is returned.
    *min_stream_timestamp = input_timestamp;
    return NodeReadiness::kReadyForProcess;
  }

  if (stream_became_done) {
    // The stream_became_done logic is kept for backward compatibility.
    // Note that the minimum bound is returned in min_stream_timestamp.
    return NodeReadiness::kReadyForProcess;
  }

  return NodeReadiness::kNotReady;
}

void ImmediateInputStreamHandler::FillInputSet(Timestamp input_timestamp,
                                               InputStreamShardSet* input_set) {
  absl::MutexLock lock(&mutex_);
  for (int i = 0; i < sync_sets_.size(); ++i) {
    if (ready_timestamps_[i] == input_timestamp) {
      sync_sets_[i].FillInputSet(input_timestamp, input_set);
      ready_timestamps_[i] = Timestamp::Unset();
    } else {
      sync_sets_[i].FillInputBounds(input_set);
    }
  }
}

int ImmediateInputStreamHandler::SyncSetCount() {
  absl::MutexLock lock(&mutex_);
  return sync_sets_.size();
}

}  // namespace mediapipe
