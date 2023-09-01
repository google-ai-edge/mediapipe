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
#include "mediapipe/framework/stream_handler/barrier_input_stream_handler.h"

#include <functional>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/collection_item_id.h"
#include "mediapipe/framework/input_stream_handler.h"

namespace mediapipe {

void BarrierInputStreamHandler::PrepareForRun(
    std::function<void()> headers_ready_callback,
    std::function<void()> notification_callback,
    std::function<void(CalculatorContext*)> schedule_callback,
    std::function<void(absl::Status)> error_callback) {
  InputStreamHandler::PrepareForRun(
      std::move(headers_ready_callback), std::move(notification_callback),
      std::move(schedule_callback), std::move(error_callback));
  for (auto& stream : input_stream_managers_) {
    stream->DisableTimestamps();
  }
}

NodeReadiness BarrierInputStreamHandler::GetNodeReadiness(
    Timestamp* min_stream_timestamp) {
  ABSL_DCHECK(min_stream_timestamp);
  *min_stream_timestamp = Timestamp::Done();
  bool all_available = true;
  for (const auto& stream : input_stream_managers_) {
    bool empty;
    Timestamp stream_timestamp = stream->MinTimestampOrBound(&empty);
    if (empty) {
      if (stream_timestamp == Timestamp::Done()) {
        *min_stream_timestamp = Timestamp::Done();
        return NodeReadiness::kReadyForClose;
      }
      all_available = false;
    }
    *min_stream_timestamp = std::min(*min_stream_timestamp, stream_timestamp);
  }

  ABSL_CHECK_NE(*min_stream_timestamp, Timestamp::Done());
  if (all_available) {
    return NodeReadiness::kReadyForProcess;
  }
  return NodeReadiness::kNotReady;
}

void BarrierInputStreamHandler::FillInputSet(Timestamp input_timestamp,
                                             InputStreamShardSet* input_set) {
  ABSL_CHECK(input_timestamp.IsAllowedInStream());
  ABSL_CHECK(input_set);
  for (CollectionItemId id = input_stream_managers_.BeginId();
       id < input_stream_managers_.EndId(); ++id) {
    auto& stream = input_stream_managers_.Get(id);
    bool stream_is_done = false;
    Packet current_packet = stream->PopQueueHead(&stream_is_done);
    AddPacketToShard(&input_set->Get(id), std::move(current_packet),
                     stream_is_done);
  }
}

REGISTER_INPUT_STREAM_HANDLER(BarrierInputStreamHandler);

}  // namespace mediapipe
