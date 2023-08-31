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
#include "mediapipe/framework/stream_handler/mux_input_stream_handler.h"

#include <utility>

#include "absl/log/absl_check.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/collection_item_id.h"
#include "mediapipe/framework/input_stream_handler.h"

namespace mediapipe {

CollectionItemId MuxInputStreamHandler::GetControlStreamId() const {
  return input_stream_managers_.EndId() - 1;
}
void MuxInputStreamHandler::RemoveOutdatedDataPackets(Timestamp timestamp) {
  const CollectionItemId control_stream_id = GetControlStreamId();
  for (CollectionItemId id = input_stream_managers_.BeginId();
       id < control_stream_id; ++id) {
    input_stream_managers_.Get(id)->ErasePacketsEarlierThan(timestamp);
  }
}

// In MuxInputStreamHandler, a node is "ready" if:
// - the control stream is done (need to call Close() in this case), or
// - we have received the packets on the control stream and the selected data
//   stream at the next timestamp.
NodeReadiness MuxInputStreamHandler::GetNodeReadiness(
    Timestamp* min_stream_timestamp) {
  ABSL_DCHECK(min_stream_timestamp);
  absl::MutexLock lock(&input_streams_mutex_);

  const auto& control_stream = input_stream_managers_.Get(GetControlStreamId());
  bool empty;
  *min_stream_timestamp = control_stream->MinTimestampOrBound(&empty);

  // Data streams may contain some outdated packets which failed to be popped
  // out during "FillInputSet". (This handler doesn't sync input streams,
  // hence "FillInputSet" can be triggered before every input stream is
  // filled with packets corresponding to the same timestamp.)
  RemoveOutdatedDataPackets(*min_stream_timestamp);
  if (empty) {
    if (*min_stream_timestamp == Timestamp::Done()) {
      // Calculator is done if the control input stream is done.
      return NodeReadiness::kReadyForClose;
    }
    // Calculator is not ready to run if the control input stream is empty.
    return NodeReadiness::kNotReady;
  }

  Packet control_packet = control_stream->QueueHead();
  ABSL_CHECK(!control_packet.IsEmpty());
  int control_value = control_packet.Get<int>();
  ABSL_CHECK_LE(0, control_value);
  ABSL_CHECK_LT(control_value, input_stream_managers_.NumEntries() - 1);
  const auto& data_stream = input_stream_managers_.Get(
      input_stream_managers_.BeginId() + control_value);

  Timestamp stream_timestamp = data_stream->MinTimestampOrBound(&empty);
  if (empty) {
    if (stream_timestamp <= *min_stream_timestamp) {
      // "data_stream" didn't receive a packet corresponding to the current
      // "control_stream" packet yet.
      return NodeReadiness::kNotReady;
    }
    // "data_stream" timestamp bound update detected.
    return NodeReadiness::kReadyForProcess;
  }
  if (stream_timestamp > *min_stream_timestamp) {
    // The earliest packet "data_stream" holds corresponds to a control packet
    // yet to arrive, which means there won't be a "data_stream" packet
    // corresponding to the current "control_stream" packet, which should be
    // indicated as timestamp boun update.
    return NodeReadiness::kReadyForProcess;
  }
  ABSL_CHECK_EQ(stream_timestamp, *min_stream_timestamp);
  return NodeReadiness::kReadyForProcess;
}

// Only invoked when associated GetNodeReadiness() returned kReadyForProcess.
void MuxInputStreamHandler::FillInputSet(Timestamp input_timestamp,
                                         InputStreamShardSet* input_set) {
  ABSL_CHECK(input_timestamp.IsAllowedInStream());
  ABSL_CHECK(input_set);
  absl::MutexLock lock(&input_streams_mutex_);

  const CollectionItemId control_stream_id = GetControlStreamId();
  auto& control_stream = input_stream_managers_.Get(control_stream_id);
  int num_packets_dropped = 0;
  bool stream_is_done = false;
  Packet control_packet = control_stream->PopPacketAtTimestamp(
      input_timestamp, &num_packets_dropped, &stream_is_done);
  ABSL_CHECK_EQ(num_packets_dropped, 0)
      << absl::Substitute("Dropped $0 packet(s) on input stream \"$1\".",
                          num_packets_dropped, control_stream->Name());
  ABSL_CHECK(!control_packet.IsEmpty());
  int control_value = control_packet.Get<int>();
  AddPacketToShard(&input_set->Get(control_stream_id),
                   std::move(control_packet), stream_is_done);

  const CollectionItemId data_stream_id =
      input_stream_managers_.BeginId() + control_value;
  ABSL_CHECK_LE(input_stream_managers_.BeginId(), data_stream_id);
  ABSL_CHECK_LT(data_stream_id, control_stream_id);
  auto& data_stream = input_stream_managers_.Get(data_stream_id);
  stream_is_done = false;
  Packet data_packet = data_stream->PopPacketAtTimestamp(
      input_timestamp, &num_packets_dropped, &stream_is_done);
  ABSL_CHECK_EQ(num_packets_dropped, 0)
      << absl::Substitute("Dropped $0 packet(s) on input stream \"$1\".",
                          num_packets_dropped, data_stream->Name());
  AddPacketToShard(&input_set->Get(data_stream_id), std::move(data_packet),
                   stream_is_done);

  // Discard old packets on data streams.
  RemoveOutdatedDataPackets(input_timestamp.NextAllowedInStream());
}

REGISTER_INPUT_STREAM_HANDLER(MuxInputStreamHandler);

}  // namespace mediapipe
