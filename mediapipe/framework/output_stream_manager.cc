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

#include "mediapipe/framework/output_stream_manager.h"

#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/input_stream_handler.h"
#include "mediapipe/framework/port/status_builder.h"

namespace mediapipe {

::mediapipe::Status OutputStreamManager::Initialize(
    const std::string& name, const PacketType* packet_type) {
  output_stream_spec_.name = name;
  output_stream_spec_.packet_type = packet_type;
  PrepareForRun(nullptr);
  return ::mediapipe::OkStatus();
}

void OutputStreamManager::PrepareForRun(
    std::function<void(::mediapipe::Status)> error_callback) {
  output_stream_spec_.error_callback = std::move(error_callback);

  output_stream_spec_.locked_intro_data = false;
  output_stream_spec_.offset_enabled = false;
  output_stream_spec_.header = Packet();
  {
    absl::MutexLock lock(&stream_mutex_);
    next_timestamp_bound_ = Timestamp::PreStream();
    closed_ = false;
  }
}

void OutputStreamManager::Close() {
  {
    absl::MutexLock lock(&stream_mutex_);
    if (closed_) {
      return;
    }
    closed_ = true;
    next_timestamp_bound_ = Timestamp::Done();
  }

  for (const auto& mirror : mirrors_) {
    mirror.input_stream_handler->SetNextTimestampBound(mirror.id,
                                                       Timestamp::Done());
  }
}

bool OutputStreamManager::IsClosed() const {
  absl::MutexLock lock(&stream_mutex_);
  return closed_;
}

void OutputStreamManager::PropagateHeader() {
  if (output_stream_spec_.locked_intro_data) {
    output_stream_spec_.TriggerErrorCallback(
        ::mediapipe::FailedPreconditionErrorBuilder(MEDIAPIPE_LOC)
        << "PropagateHeader must be called in CalculatorNode::OpenNode(). "
           "Stream: \""
        << output_stream_spec_.name << "\".");
    return;
  }
  for (auto& mirror : mirrors_) {
    mirror.input_stream_handler->SetHeader(mirror.id,
                                           output_stream_spec_.header);
  }
}

void OutputStreamManager::AddMirror(InputStreamHandler* input_stream_handler,
                                    CollectionItemId id) {
  CHECK(input_stream_handler);
  mirrors_.emplace_back(input_stream_handler, id);
}

void OutputStreamManager::SetMaxQueueSize(int max_queue_size) {
  for (auto& mirror : mirrors_) {
    mirror.input_stream_handler->SetMaxQueueSize(mirror.id, max_queue_size);
  }
}

Timestamp OutputStreamManager::NextTimestampBound() const {
  absl::MutexLock lock(&stream_mutex_);
  return next_timestamp_bound_;
}

// TODO Consider moving the output bound computing logic to
// OutputStreamHandler.
Timestamp OutputStreamManager::ComputeOutputTimestampBound(
    const OutputStreamShard& output_stream_shard,
    Timestamp input_timestamp) const {
  // This function is called for Calculator::Open() and Calculator::Process().
  // It is not called for Calculator::Close() because the output timestamp bound
  // is always Timestamp::Done().
  if (input_timestamp != Timestamp::Unstarted() &&
      !input_timestamp.IsAllowedInStream()) {
    output_stream_spec_.TriggerErrorCallback(
        ::mediapipe::FailedPreconditionErrorBuilder(MEDIAPIPE_LOC)
        << "Invalid input timestamp to compute the output timestamp bound. "
           "Stream: \""
        << output_stream_spec_.name
        << "\", Timestamp: " << input_timestamp.DebugString() << ".");
    return Timestamp::Unset();
  }
  // new_bound = max(AddOffset(completed_timestamp) + 1,
  //                 MaxOutputTimestamp(completed_timestamp) + 1)
  // Note that "MaxOutputTimestamp()" must consider both output packet
  // timetstamp and SetNextTimestampBound values.
  // See the timestamp mapping section in go/mediapipe-bounds for details.
  Timestamp new_bound = output_stream_shard.NextTimestampBound();
  if (output_stream_spec_.offset_enabled &&
      input_timestamp != Timestamp::Unstarted()) {
    Timestamp input_bound;
    if (input_timestamp == Timestamp::PreStream()) {
      // Timestamp::PreStream() is a special value that we shouldn't apply any
      // offset.
      input_bound = Timestamp::Min();
    } else if (input_timestamp == Timestamp::Max()) {
      // If the offset is positive or zero, we set the input_bound to
      // Timestamp::PostStream() since the calculator might process
      // Timestamp::PostStream() in the next invocation.
      if (output_stream_spec_.offset >= 0) {
        input_bound = Timestamp::PostStream();
      } else {
        input_bound = (input_timestamp + output_stream_spec_.offset)
                          .NextAllowedInStream();
      }
    } else if (input_timestamp == Timestamp::PostStream()) {
      // For Timestamp::PostStream(), it's expected that no futher timestamps
      // will occur.
      input_bound = Timestamp::OneOverPostStream();
    } else {
      input_bound =
          input_timestamp.NextAllowedInStream() + output_stream_spec_.offset;
    }
    new_bound = std::max(new_bound, input_bound);
  }

  if (!output_stream_shard.IsEmpty()) {
    new_bound = std::max(
        new_bound,
        output_stream_shard.LastAddedPacketTimestamp().NextAllowedInStream());
  }
  return new_bound;
}

// TODO Consider moving the propagation logic to OutputStreamHandler.
void OutputStreamManager::PropagateUpdatesToMirrors(
    Timestamp next_timestamp_bound, OutputStreamShard* output_stream_shard) {
  CHECK(output_stream_shard);
  {
    absl::MutexLock lock(&stream_mutex_);
    next_timestamp_bound_ = next_timestamp_bound;
  }
  std::list<Packet>* packets_to_propagate = output_stream_shard->OutputQueue();
  VLOG(3) << "Output stream: " << Name()
          << " queue size: " << packets_to_propagate->size();
  VLOG(3) << "Output stream: " << Name()
          << " next timestamp: " << next_timestamp_bound;
  bool add_packets = !packets_to_propagate->empty();
  bool set_bound =
      !add_packets ||
      packets_to_propagate->back().Timestamp().NextAllowedInStream() !=
          next_timestamp_bound;
  int mirror_count = mirrors_.size();
  for (int idx = 0; idx < mirror_count; ++idx) {
    const Mirror& mirror = mirrors_[idx];
    if (add_packets) {
      // If the stream is the last element in mirrors_, moves packets from
      // output_queue_. Otherwise, copies the packets.
      if (idx == mirror_count - 1) {
        mirror.input_stream_handler->MovePackets(mirror.id,
                                                 packets_to_propagate);
      } else {
        mirror.input_stream_handler->AddPackets(mirror.id,
                                                *packets_to_propagate);
      }
    }
    if (set_bound) {
      mirror.input_stream_handler->SetNextTimestampBound(mirror.id,
                                                         next_timestamp_bound);
    }
  }
  // Clear out the packets.
  packets_to_propagate->clear();
}

void OutputStreamManager::ResetShard(OutputStreamShard* output_stream_shard) {
  Timestamp next_timestamp_bound;
  bool closed = false;
  absl::MutexLock lock(&stream_mutex_);
  {
    next_timestamp_bound = next_timestamp_bound_;
    closed = closed_;
  }
  output_stream_shard->Reset(next_timestamp_bound, closed);
}

}  // namespace mediapipe
