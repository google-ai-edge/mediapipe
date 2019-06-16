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

#include "absl/strings/substitute.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/input_stream_handler.h"
#include "mediapipe/framework/port/logging.h"

namespace mediapipe {

// Implementation of the input stream handler for the MuxCalculator.
//
// One of the input streams is the control stream; all the other input streams
// are data streams. To make MuxInputStreamHandler work properly, the tag of the
// input streams must obey the following rules:
// Let N be the number of input streams. Data streams must use tag "INPUT" with
// index 0, ..., N - 2; the control stream must use tag "SELECT".
//
// The control stream carries packets of type 'int'. The 'int' value in a
// control stream packet must be a valid index in the range 0, ..., N - 2 and
// select the data stream at that index. The selected data stream must have a
// packet with the same timestamp as the control stream packet.
//
// When the control stream is done, GetNodeReadiness() returns
// NodeReadiness::kReadyForClose.
//
// TODO: pass the input stream tags to the MuxInputStreamHandler
// constructor so that it can refer to input streams by tag. See b/30125118.
class MuxInputStreamHandler : public InputStreamHandler {
 public:
  MuxInputStreamHandler() = delete;
  MuxInputStreamHandler(std::shared_ptr<tool::TagMap> tag_map,
                        CalculatorContextManager* cc_manager,
                        const MediaPipeOptions& options,
                        bool calculator_run_in_parallel)
      : InputStreamHandler(std::move(tag_map), cc_manager, options,
                           calculator_run_in_parallel) {}

 protected:
  // In MuxInputStreamHandler, a node is "ready" if:
  // - the control stream is done (need to call Close() in this case), or
  // - we have received the packets on the control stream and the selected data
  //   stream at the next timestamp.
  NodeReadiness GetNodeReadiness(Timestamp* min_stream_timestamp) override {
    DCHECK(min_stream_timestamp);
    absl::MutexLock lock(&input_streams_mutex_);

    const auto& control_stream =
        input_stream_managers_.Get(input_stream_managers_.EndId() - 1);
    bool empty;
    *min_stream_timestamp = control_stream->MinTimestampOrBound(&empty);
    if (empty) {
      if (*min_stream_timestamp == Timestamp::Done()) {
        // Calculator is done if the control input stream is done.
        return NodeReadiness::kReadyForClose;
      }
      // Calculator is not ready to run if the control input stream is empty.
      return NodeReadiness::kNotReady;
    }

    Packet control_packet = control_stream->QueueHead();
    CHECK(!control_packet.IsEmpty());
    int control_value = control_packet.Get<int>();
    CHECK_LE(0, control_value);
    CHECK_LT(control_value, input_stream_managers_.NumEntries() - 1);

    const auto& data_stream = input_stream_managers_.Get(
        input_stream_managers_.BeginId() + control_value);
    Timestamp stream_timestamp = data_stream->MinTimestampOrBound(&empty);
    if (empty) {
      CHECK_LE(stream_timestamp, *min_stream_timestamp);
      return NodeReadiness::kNotReady;
    }
    CHECK_EQ(stream_timestamp, *min_stream_timestamp);
    return NodeReadiness::kReadyForProcess;
  }

  // Only invoked when associated GetNodeReadiness() returned kReadyForProcess.
  void FillInputSet(Timestamp input_timestamp,
                    InputStreamShardSet* input_set) override {
    CHECK(input_timestamp.IsAllowedInStream());
    CHECK(input_set);
    absl::MutexLock lock(&input_streams_mutex_);

    const CollectionItemId control_stream_id =
        input_stream_managers_.EndId() - 1;
    auto& control_stream = input_stream_managers_.Get(control_stream_id);
    int num_packets_dropped = 0;
    bool stream_is_done = false;
    Packet control_packet = control_stream->PopPacketAtTimestamp(
        input_timestamp, &num_packets_dropped, &stream_is_done);
    CHECK_EQ(num_packets_dropped, 0)
        << absl::Substitute("Dropped $0 packet(s) on input stream \"$1\".",
                            num_packets_dropped, control_stream->Name());
    CHECK(!control_packet.IsEmpty());
    int control_value = control_packet.Get<int>();
    AddPacketToShard(&input_set->Get(control_stream_id),
                     std::move(control_packet), stream_is_done);

    const CollectionItemId data_stream_id =
        input_stream_managers_.BeginId() + control_value;
    CHECK_LE(input_stream_managers_.BeginId(), data_stream_id);
    CHECK_LT(data_stream_id, control_stream_id);
    auto& data_stream = input_stream_managers_.Get(data_stream_id);
    stream_is_done = false;
    Packet data_packet = data_stream->PopPacketAtTimestamp(
        input_timestamp, &num_packets_dropped, &stream_is_done);
    CHECK_EQ(num_packets_dropped, 0)
        << absl::Substitute("Dropped $0 packet(s) on input stream \"$1\".",
                            num_packets_dropped, data_stream->Name());
    AddPacketToShard(&input_set->Get(data_stream_id), std::move(data_packet),
                     stream_is_done);
  }

 private:
  // Must be acquired when manipulating the control and data streams to ensure
  // we have a consistent view of the two streams.
  absl::Mutex input_streams_mutex_;
};

REGISTER_INPUT_STREAM_HANDLER(MuxInputStreamHandler);

}  // namespace mediapipe
