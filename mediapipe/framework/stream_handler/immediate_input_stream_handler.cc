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

#include <memory>
#include <vector>

#include "mediapipe/framework/input_stream_handler.h"

namespace mediapipe {

// An input stream handler that delivers input packets to the Calculator
// immediately, with no dependency between input streams.  It also invokes
// Calculator::Process when any input stream becomes done.
//
// NOTE: If packets arrive successively on different input streams with
// identical or decreasing timestamps, this input stream handler will
// invoke its Calculator with a sequence of InputTimestamps that is
// non-increasing.  Its Calculator is responsible for accumulating packets
// with the required timetamps before processing and delivering output.
//
class ImmediateInputStreamHandler : public InputStreamHandler {
 public:
  ImmediateInputStreamHandler() = delete;
  ImmediateInputStreamHandler(
      std::shared_ptr<tool::TagMap> tag_map,
      CalculatorContextManager* calculator_context_manager,
      const MediaPipeOptions& options, bool calculator_run_in_parallel);

 protected:
  // Returns kReadyForProcess whenever a Packet is available at any of
  // the input streams, or any input stream becomes done.
  NodeReadiness GetNodeReadiness(Timestamp* min_stream_timestamp) override;

  // Selects a packet on each stream with an available packet with the
  // specified timestamp, leaving other input streams unaffected.
  void FillInputSet(Timestamp input_timestamp,
                    InputStreamShardSet* input_set) override;

  // Record of the last reported timestamp bound for each input stream.
  mediapipe::internal::Collection<Timestamp> timestamp_bounds_;
};
REGISTER_INPUT_STREAM_HANDLER(ImmediateInputStreamHandler);

ImmediateInputStreamHandler::ImmediateInputStreamHandler(
    std::shared_ptr<tool::TagMap> tag_map,
    CalculatorContextManager* calculator_context_manager,
    const MediaPipeOptions& options, bool calculator_run_in_parallel)
    : InputStreamHandler(tag_map, calculator_context_manager, options,
                         calculator_run_in_parallel),
      timestamp_bounds_(std::move(tag_map)) {}

NodeReadiness ImmediateInputStreamHandler::GetNodeReadiness(
    Timestamp* min_stream_timestamp) {
  *min_stream_timestamp = Timestamp::Done();
  Timestamp input_timestamp = Timestamp::Done();
  bool stream_became_done = false;

  for (CollectionItemId i = input_stream_managers_.BeginId();
       i < input_stream_managers_.EndId(); ++i) {
    const auto& stream = input_stream_managers_.Get(i);
    bool empty;
    Timestamp stream_timestamp = stream->MinTimestampOrBound(&empty);
    if (!empty) {
      input_timestamp = std::min(input_timestamp, stream_timestamp);
    }
    *min_stream_timestamp = std::min(*min_stream_timestamp, stream_timestamp);
    if (stream_timestamp != timestamp_bounds_.Get(i)) {
      if (stream_timestamp == Timestamp::Done()) {
        stream_became_done = true;
      }
      timestamp_bounds_.Get(i) = stream_timestamp;
    }
  }

  if (*min_stream_timestamp == Timestamp::Done()) {
    return NodeReadiness::kReadyForClose;
  }

  if (input_timestamp < Timestamp::Done()) {
    // On kReadyForProcess, the input_timestamp is returned.
    *min_stream_timestamp = input_timestamp;
    return NodeReadiness::kReadyForProcess;
  }

  if (stream_became_done) {
    return NodeReadiness::kReadyForProcess;
  }

  return NodeReadiness::kNotReady;
}

void ImmediateInputStreamHandler::FillInputSet(Timestamp input_timestamp,
                                               InputStreamShardSet* input_set) {
  CHECK(input_timestamp.IsAllowedInStream());
  CHECK(input_set);
  for (CollectionItemId id = input_stream_managers_.BeginId();
       id < input_stream_managers_.EndId(); ++id) {
    auto& stream = input_stream_managers_.Get(id);
    if (stream->QueueHead().Timestamp() == input_timestamp) {
      int num_packets_dropped = 0;
      bool stream_is_done = false;
      Packet current_packet = stream->PopPacketAtTimestamp(
          input_timestamp, &num_packets_dropped, &stream_is_done);
      AddPacketToShard(&input_set->Get(id), std::move(current_packet),
                       stream_is_done);
    } else {
      bool empty = false;
      bool is_done = stream->MinTimestampOrBound(&empty) == Timestamp::Done();
      AddPacketToShard(&input_set->Get(id), Packet(), is_done);
    }
  }
}

}  // namespace mediapipe
