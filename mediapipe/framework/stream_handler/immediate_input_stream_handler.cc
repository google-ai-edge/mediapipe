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

using SyncSet = InputStreamHandler::SyncSet;

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
  // Reinitializes this InputStreamHandler before each CalculatorGraph run.
  void PrepareForRun(
      std::function<void()> headers_ready_callback,
      std::function<void()> notification_callback,
      std::function<void(CalculatorContext*)> schedule_callback,
      std::function<void(::mediapipe::Status)> error_callback) override;

  // Returns kReadyForProcess whenever a Packet is available at any of
  // the input streams, or any input stream becomes done.
  NodeReadiness GetNodeReadiness(Timestamp* min_stream_timestamp) override;

  // Selects a packet on each stream with an available packet with the
  // specified timestamp, leaving other input streams unaffected.
  void FillInputSet(Timestamp input_timestamp,
                    InputStreamShardSet* input_set) override;

  // Returns the number of sync-sets maintained by this input-handler.
  int SyncSetCount() override;

  absl::Mutex mutex_;
  // The packet-set builder for each input stream.
  std::vector<SyncSet> sync_sets_ ABSL_GUARDED_BY(mutex_);
  // The input timestamp for each kReadyForProcess input stream.
  std::vector<Timestamp> ready_timestamps_ ABSL_GUARDED_BY(mutex_);
};
REGISTER_INPUT_STREAM_HANDLER(ImmediateInputStreamHandler);

ImmediateInputStreamHandler::ImmediateInputStreamHandler(
    std::shared_ptr<tool::TagMap> tag_map,
    CalculatorContextManager* calculator_context_manager,
    const MediaPipeOptions& options, bool calculator_run_in_parallel)
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
    std::function<void(::mediapipe::Status)> error_callback) {
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
      CHECK_EQ(stream_ts, Timestamp::Done());
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
