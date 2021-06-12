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

#include "mediapipe/framework/stream_handler/default_input_stream_handler.h"
// TODO: Move protos in another CL after the C++ code migration.
#include "mediapipe/framework/stream_handler/fixed_size_input_stream_handler.pb.h"

namespace mediapipe {

// Input stream handler that limits each input queue to a maximum of
// target_queue_size packets, discarding older packets as needed.  When a
// timestamp is dropped from a stream, it is dropped from all others as well.
//
// For example, a calculator node with one input stream and the following input
// stream handler specs:
//
// node {
//   calculator: "CalculatorRunningAtOneFps"
//   input_stream: "packets_streaming_in_at_ten_fps"
//   input_stream_handler {
//     input_stream_handler: "FixedSizeInputStreamHandler"
//   }
// }
//
// will always try to keep the newest packet in the input stream.
//
// A few details: FixedSizeInputStreamHandler takes action when any stream grows
// to trigger_queue_size or larger.  It then keeps at most target_queue_size
// packets in every InputStreamImpl.  Every stream is truncated at the same
// timestamp, so that each included timestamp delivers the same packets as
// DefaultInputStreamHandler includes.
//
class FixedSizeInputStreamHandler : public DefaultInputStreamHandler {
 public:
  FixedSizeInputStreamHandler() = delete;
  FixedSizeInputStreamHandler(std::shared_ptr<tool::TagMap> tag_map,
                              CalculatorContextManager* cc_manager,
                              const MediaPipeOptions& options,
                              bool calculator_run_in_parallel)
      : DefaultInputStreamHandler(std::move(tag_map), cc_manager, options,
                                  calculator_run_in_parallel) {
    const auto& ext =
        options.GetExtension(FixedSizeInputStreamHandlerOptions::ext);
    trigger_queue_size_ = ext.trigger_queue_size();
    target_queue_size_ = ext.target_queue_size();
    fixed_min_size_ = ext.fixed_min_size();
    pending_ = false;
    kept_timestamp_ = Timestamp::Unset();
    // TODO: Either re-enable SetLatePreparation(true) with
    // CalculatorContext::InputTimestamp set correctly, or remove the
    // implementation of SetLatePreparation.
  }

 private:
  // Drops packets if all input streams exceed trigger_queue_size.
  void EraseAllSurplus() ABSL_EXCLUSIVE_LOCKS_REQUIRED(erase_mutex_) {
    Timestamp min_timestamp_all_streams = Timestamp::Max();
    for (const auto& stream : input_stream_managers_) {
      // Check whether every InputStreamImpl grew beyond trigger_queue_size.
      if (stream->QueueSize() < trigger_queue_size_) {
        return;
      }
      Timestamp min_timestamp =
          stream->GetMinTimestampAmongNLatest(target_queue_size_);

      // Record the min timestamp among the newest target_queue_size_ packets
      // across all InputStreamImpls.
      min_timestamp_all_streams =
          std::min(min_timestamp_all_streams, min_timestamp);
    }
    for (auto& stream : input_stream_managers_) {
      stream->ErasePacketsEarlierThan(min_timestamp_all_streams);
    }
  }

  // Returns the latest timestamp allowed before a bound.
  Timestamp PreviousAllowedInStream(Timestamp bound) {
    return bound.IsRangeValue() ? bound - 1 : bound;
  }

  // Returns the lowest timestamp at which a packet may arrive at any stream.
  Timestamp MinStreamBound() {
    Timestamp min_bound = Timestamp::Done();
    for (const auto& stream : input_stream_managers_) {
      Timestamp stream_bound = stream->GetMinTimestampAmongNLatest(1);
      if (stream_bound > Timestamp::Unset()) {
        stream_bound = stream_bound.NextAllowedInStream();
      } else {
        stream_bound = stream->MinTimestampOrBound(nullptr);
      }
      min_bound = std::min(min_bound, stream_bound);
    }
    return min_bound;
  }

  // Returns the lowest timestamp of a packet ready to process.
  Timestamp MinTimestampToProcess() {
    Timestamp min_bound = Timestamp::Done();
    for (const auto& stream : input_stream_managers_) {
      bool empty;
      Timestamp stream_timestamp = stream->MinTimestampOrBound(&empty);
      // If we're using the stream's *bound*, we only want to process up to the
      // packet *before* the bound, because a packet may still arrive at that
      // time.
      if (empty) {
        stream_timestamp = PreviousAllowedInStream(stream_timestamp);
      }
      min_bound = std::min(min_bound, stream_timestamp);
    }
    return min_bound;
  }

  // Keeps only the most recent target_queue_size packets in each stream
  // exceeding trigger_queue_size.  Also, discards all packets older than the
  // first kept timestamp on any stream.
  void EraseAnySurplus(bool keep_one)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(erase_mutex_) {
    // Record the most recent first kept timestamp on any stream.
    for (const auto& stream : input_stream_managers_) {
      int32 queue_size = (stream->QueueSize() >= trigger_queue_size_)
                             ? target_queue_size_
                             : trigger_queue_size_ - 1;
      if (stream->QueueSize() > queue_size) {
        kept_timestamp_ = std::max(
            kept_timestamp_, stream->GetMinTimestampAmongNLatest(queue_size + 1)
                                 .NextAllowedInStream());
      }
    }
    if (keep_one) {
      // In order to preserve one viable timestamp, do not truncate past
      // the timestamp bound of the least current stream.
      kept_timestamp_ =
          std::min(kept_timestamp_, PreviousAllowedInStream(MinStreamBound()));
    }
    for (auto& stream : input_stream_managers_) {
      stream->ErasePacketsEarlierThan(kept_timestamp_);
    }
  }

  void EraseSurplusPackets(bool keep_one)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(erase_mutex_) {
    return (fixed_min_size_) ? EraseAllSurplus() : EraseAnySurplus(keep_one);
  }

  NodeReadiness GetNodeReadiness(Timestamp* min_stream_timestamp) override {
    DCHECK(min_stream_timestamp);
    absl::MutexLock lock(&erase_mutex_);
    // kReadyForProcess is returned only once until FillInputSet completes.
    // In late_preparation mode, GetNodeReadiness must return kReadyForProcess
    // exactly once for each input-set produced.  Here, GetNodeReadiness
    // releases just one input-set at a time and then disables input queue
    // truncation until that promised input-set is consumed.
    if (pending_) {
      return NodeReadiness::kNotReady;
    }
    EraseSurplusPackets(false);
    NodeReadiness result =
        DefaultInputStreamHandler::GetNodeReadiness(min_stream_timestamp);

    // If a packet has arrived below kept_timestamp_, recalculate.
    while (*min_stream_timestamp < kept_timestamp_ &&
           result == NodeReadiness::kReadyForProcess) {
      EraseSurplusPackets(false);
      result =
          DefaultInputStreamHandler::GetNodeReadiness(min_stream_timestamp);
    }
    pending_ = (result == NodeReadiness::kReadyForProcess);
    return result;
  }

  void AddPackets(CollectionItemId id,
                  const std::list<Packet>& packets) override {
    InputStreamHandler::AddPackets(id, packets);
    absl::MutexLock lock(&erase_mutex_);
    if (!pending_) {
      EraseSurplusPackets(false);
    }
  }

  void MovePackets(CollectionItemId id, std::list<Packet>* packets) override {
    InputStreamHandler::MovePackets(id, packets);
    absl::MutexLock lock(&erase_mutex_);
    if (!pending_) {
      EraseSurplusPackets(false);
    }
  }

  void FillInputSet(Timestamp input_timestamp,
                    InputStreamShardSet* input_set) override {
    CHECK(input_set);
    absl::MutexLock lock(&erase_mutex_);
    if (!pending_) {
      LOG(ERROR) << "FillInputSet called without GetNodeReadiness.";
    }
    // input_timestamp is recalculated here to process the most recent packets.
    EraseSurplusPackets(true);
    input_timestamp = MinTimestampToProcess();
    DefaultInputStreamHandler::FillInputSet(input_timestamp, input_set);
    pending_ = false;
  }

 private:
  int32 trigger_queue_size_;
  int32 target_queue_size_;
  bool fixed_min_size_;
  // Indicates that GetNodeReadiness has returned kReadyForProcess once, and
  // the corresponding call to FillInputSet has not yet completed.
  bool pending_ ABSL_GUARDED_BY(erase_mutex_);
  // The timestamp used to truncate all input streams.
  Timestamp kept_timestamp_ ABSL_GUARDED_BY(erase_mutex_);
  absl::Mutex erase_mutex_;
};

REGISTER_INPUT_STREAM_HANDLER(FixedSizeInputStreamHandler);

}  // namespace mediapipe
