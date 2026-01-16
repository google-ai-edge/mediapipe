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
#include "mediapipe/framework/stream_handler/fixed_size_input_stream_handler.h"

#include <algorithm>
#include <list>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/calculator_context_manager.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/collection_item_id.h"
#include "mediapipe/framework/input_stream_handler.h"
#include "mediapipe/framework/mediapipe_options.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/stream_handler/default_input_stream_handler.h"
#include "mediapipe/framework/stream_handler/fixed_size_input_stream_handler.pb.h"
#include "mediapipe/framework/tool/tag_map.h"

namespace mediapipe {

FixedSizeInputStreamHandler::FixedSizeInputStreamHandler(
    std::shared_ptr<tool::TagMap> tag_map, CalculatorContextManager* cc_manager,
    const mediapipe::MediaPipeOptions& options, bool calculator_run_in_parallel)
    : DefaultInputStreamHandler(std::move(tag_map), cc_manager, options,
                                calculator_run_in_parallel) {
  const auto& ext =
      options.GetExtension(mediapipe::FixedSizeInputStreamHandlerOptions::ext);
  trigger_queue_size_ = ext.trigger_queue_size();
  target_queue_size_ = ext.target_queue_size();
  fixed_min_size_ = ext.fixed_min_size();
  pending_ = false;
  kept_timestamp_ = Timestamp::Unset();
  // TODO: Either re-enable SetLatePreparation(true) with
  // CalculatorContext::InputTimestamp set correctly, or remove the
  // implementation of SetLatePreparation.
}

void FixedSizeInputStreamHandler::EraseAllSurplus() {
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

Timestamp FixedSizeInputStreamHandler::PreviousAllowedInStream(
    Timestamp bound) {
  return bound.IsRangeValue() ? bound - 1 : bound;
}

Timestamp FixedSizeInputStreamHandler::MinStreamBound() {
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

Timestamp FixedSizeInputStreamHandler::MinTimestampToProcess() {
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

void FixedSizeInputStreamHandler::EraseAnySurplus(bool keep_one) {
  // Record the most recent first kept timestamp on any stream.
  for (const auto& stream : input_stream_managers_) {
    int32_t queue_size = (stream->QueueSize() >= trigger_queue_size_)
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

void FixedSizeInputStreamHandler::EraseSurplusPackets(bool keep_one) {
  return (fixed_min_size_) ? EraseAllSurplus() : EraseAnySurplus(keep_one);
}

NodeReadiness FixedSizeInputStreamHandler::GetNodeReadiness(
    Timestamp* min_stream_timestamp) {
  ABSL_DCHECK(min_stream_timestamp);
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
    result = DefaultInputStreamHandler::GetNodeReadiness(min_stream_timestamp);
  }
  pending_ = (result == NodeReadiness::kReadyForProcess);
  return result;
}

void FixedSizeInputStreamHandler::AddPackets(CollectionItemId id,
                                             const std::list<Packet>& packets) {
  InputStreamHandler::AddPackets(id, packets);
  absl::MutexLock lock(&erase_mutex_);
  if (!pending_) {
    EraseSurplusPackets(false);
  }
}

void FixedSizeInputStreamHandler::MovePackets(CollectionItemId id,
                                              std::list<Packet>* packets) {
  InputStreamHandler::MovePackets(id, packets);
  absl::MutexLock lock(&erase_mutex_);
  if (!pending_) {
    EraseSurplusPackets(false);
  }
}

void FixedSizeInputStreamHandler::FillInputSet(Timestamp input_timestamp,
                                               InputStreamShardSet* input_set) {
  ABSL_CHECK(input_set);
  absl::MutexLock lock(&erase_mutex_);
  if (!pending_) {
    ABSL_LOG(ERROR) << "FillInputSet called without GetNodeReadiness.";
  }
  // input_timestamp is recalculated here to process the most recent packets.
  EraseSurplusPackets(true);
  input_timestamp = MinTimestampToProcess();
  DefaultInputStreamHandler::FillInputSet(input_timestamp, input_set);
  pending_ = false;
}

REGISTER_INPUT_STREAM_HANDLER(FixedSizeInputStreamHandler);

}  // namespace mediapipe
