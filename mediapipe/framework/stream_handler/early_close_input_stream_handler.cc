// Copyright 2023 The MediaPipe Authors.
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
#include "mediapipe/framework/stream_handler/early_close_input_stream_handler.h"

#include <algorithm>

#include "absl/log/absl_check.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/collection_item_id.h"
#include "mediapipe/framework/input_stream_handler.h"

namespace mediapipe {

// In EarlyCloseInputStreamHandler, a node is "ready" if:
// - any stream is done (need to call Close() in this case), or
// - the minimum bound (over all empty streams) is greater than the smallest
//   timestamp of any stream, which means we have received all the packets
//   that will be available at the next timestamp.
NodeReadiness EarlyCloseInputStreamHandler::GetNodeReadiness(
    Timestamp* min_stream_timestamp) {
  ABSL_DCHECK(min_stream_timestamp);
  *min_stream_timestamp = Timestamp::Done();
  Timestamp min_bound = Timestamp::Done();
  for (const auto& stream : input_stream_managers_) {
    bool empty;
    Timestamp stream_timestamp = stream->MinTimestampOrBound(&empty);
    if (empty) {
      if (stream_timestamp == Timestamp::Done()) {
        *min_stream_timestamp = Timestamp::Done();
        return NodeReadiness::kReadyForClose;
      }
      min_bound = std::min(min_bound, stream_timestamp);
    }
    *min_stream_timestamp = std::min(*min_stream_timestamp, stream_timestamp);
  }

  ABSL_CHECK_NE(*min_stream_timestamp, Timestamp::Done());

  if (min_bound > *min_stream_timestamp) {
    return NodeReadiness::kReadyForProcess;
  }

  ABSL_CHECK_EQ(min_bound, *min_stream_timestamp);
  return NodeReadiness::kNotReady;
}

// Only invoked when associated GetNodeReadiness() returned kReadyForProcess.
void EarlyCloseInputStreamHandler::FillInputSet(
    Timestamp input_timestamp, InputStreamShardSet* input_set) {
  ABSL_CHECK(input_timestamp.IsAllowedInStream());
  ABSL_CHECK(input_set);
  for (CollectionItemId id = input_stream_managers_.BeginId();
       id < input_stream_managers_.EndId(); ++id) {
    auto& stream = input_stream_managers_.Get(id);
    int num_packets_dropped = 0;
    bool stream_is_done = false;
    Packet current_packet = stream->PopPacketAtTimestamp(
        input_timestamp, &num_packets_dropped, &stream_is_done);
    ABSL_CHECK_EQ(num_packets_dropped, 0)
        << absl::Substitute("Dropped $0 packet(s) on input stream \"$1\".",
                            num_packets_dropped, stream->Name());
    AddPacketToShard(&input_set->Get(id), std::move(current_packet),
                     stream_is_done);
  }
}

REGISTER_INPUT_STREAM_HANDLER(EarlyCloseInputStreamHandler);

}  // namespace mediapipe
