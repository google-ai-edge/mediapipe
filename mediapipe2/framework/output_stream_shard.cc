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

#include "mediapipe/framework/output_stream_shard.h"

#include "mediapipe/framework/port/source_location.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_builder.h"

namespace mediapipe {

OutputStreamShard::OutputStreamShard() : closed_(false) {}

void OutputStreamShard::SetSpec(OutputStreamSpec* output_stream_spec) {
  CHECK(output_stream_spec);
  output_stream_spec_ = output_stream_spec;
}

const std::string& OutputStreamShard::Name() const {
  return output_stream_spec_->name;
}

void OutputStreamShard::SetNextTimestampBound(Timestamp bound) {
  if (!bound.IsAllowedInStream() && bound != Timestamp::OneOverPostStream()) {
    output_stream_spec_->TriggerErrorCallback(
        mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
        << "In stream \"" << Name()
        << "\", timestamp bound set to illegal value: " << bound.DebugString());
    return;
  }
  next_timestamp_bound_ = bound;
  updated_next_timestamp_bound_ = next_timestamp_bound_;
}

void OutputStreamShard::Close() {
  closed_ = true;
  next_timestamp_bound_ = Timestamp::Done();
  updated_next_timestamp_bound_ = next_timestamp_bound_;
}

bool OutputStreamShard::IsClosed() const { return closed_; }

void OutputStreamShard::SetOffset(TimestampDiff offset) {
  if (output_stream_spec_->locked_intro_data) {
    output_stream_spec_->TriggerErrorCallback(
        mediapipe::FailedPreconditionErrorBuilder(MEDIAPIPE_LOC)
        << "SetOffset must be called from Calculator::Open(). Stream: \""
        << output_stream_spec_->name << "\".");
    return;
  }
  output_stream_spec_->offset_enabled = true;
  output_stream_spec_->offset = offset;
}

void OutputStreamShard::SetHeader(const Packet& header) {
  if (closed_) {
    output_stream_spec_->TriggerErrorCallback(
        mediapipe::FailedPreconditionErrorBuilder(MEDIAPIPE_LOC)
        << "SetHeader must be called before the stream is closed. Stream: \""
        << output_stream_spec_->name << "\".");
    return;
  }

  if (output_stream_spec_->locked_intro_data) {
    output_stream_spec_->TriggerErrorCallback(
        mediapipe::FailedPreconditionErrorBuilder(MEDIAPIPE_LOC)
        << "SetHeader must be called from Calculator::Open(). Stream: \""
        << output_stream_spec_->name << "\".");
    return;
  }

  output_stream_spec_->header = header;
}

const Packet& OutputStreamShard::Header() const {
  return output_stream_spec_->header;
}

// In this context, T&& is a forwarding reference, and can be deduced as either
// a const lvalue reference for AddPacket(const Packet&) or an rvalue reference
// for AddPacket(Packet&&). Although those two AddPacket versions share the
// code by calling AddPacketInternal, there are two separate copies in the
// binary.  This function can be defined in the .cc file because only two
// versions are ever instantiated, and all call sites are within this .cc file.
template <typename T>
Status OutputStreamShard::AddPacketInternal(T&& packet) {
  if (IsClosed()) {
    return mediapipe::FailedPreconditionErrorBuilder(MEDIAPIPE_LOC)
           << "Packet sent to closed stream \"" << Name() << "\".";
  }

  if (packet.IsEmpty()) {
    SetNextTimestampBound(packet.Timestamp().NextAllowedInStream());
    return absl::OkStatus();
  }

  const Timestamp timestamp = packet.Timestamp();
  if (!timestamp.IsAllowedInStream()) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "In stream \"" << Name()
           << "\", timestamp not specified or set to illegal value: "
           << timestamp.DebugString();
  }

  Status result = output_stream_spec_->packet_type->Validate(packet);
  if (!result.ok()) {
    return StatusBuilder(result, MEDIAPIPE_LOC).SetPrepend() << absl::StrCat(
               "Packet type mismatch on calculator outputting to stream \"",
               Name(), "\": ");
  }

  // Adds the packet to output_queue_ if it's a const lvalue reference.
  // Otherwise, moves the packet into output_queue_.
  output_queue_.push_back(std::forward<T>(packet));
  next_timestamp_bound_ = timestamp.NextAllowedInStream();
  updated_next_timestamp_bound_ = next_timestamp_bound_;

  // TODO debug log?

  return absl::OkStatus();
}

void OutputStreamShard::AddPacket(const Packet& packet) {
  Status status = AddPacketInternal(packet);
  if (!status.ok()) {
    output_stream_spec_->TriggerErrorCallback(status);
  }
}

void OutputStreamShard::AddPacket(Packet&& packet) {
  Status status = AddPacketInternal(std::move(packet));
  if (!status.ok()) {
    output_stream_spec_->TriggerErrorCallback(status);
  }
}

Timestamp OutputStreamShard::LastAddedPacketTimestamp() const {
  if (output_queue_.empty()) {
    return Timestamp::Unset();
  }
  return output_queue_.back().Timestamp();
}

void OutputStreamShard::Reset(Timestamp next_timestamp_bound, bool close) {
  output_queue_.clear();
  next_timestamp_bound_ = next_timestamp_bound;
  updated_next_timestamp_bound_ = Timestamp::Unset();
  closed_ = close;
}

}  // namespace mediapipe
