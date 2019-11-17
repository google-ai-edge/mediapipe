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

#include "mediapipe/framework/output_side_packet_impl.h"

#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/source_location.h"
#include "mediapipe/framework/port/status_builder.h"

namespace mediapipe {

::mediapipe::Status OutputSidePacketImpl::Initialize(
    const std::string& name, const PacketType* packet_type) {
  name_ = name;
  packet_type_ = packet_type;
  return ::mediapipe::OkStatus();
}

void OutputSidePacketImpl::PrepareForRun(
    std::function<void(::mediapipe::Status)> error_callback) {
  error_callback_ = std::move(error_callback);
  initialized_ = false;
}

void OutputSidePacketImpl::Set(const Packet& packet) {
  ::mediapipe::Status status = SetInternal(packet);
  if (!status.ok()) {
    TriggerErrorCallback(status);
  }
}

void OutputSidePacketImpl::AddMirror(
    InputSidePacketHandler* input_side_packet_handler, CollectionItemId id) {
  CHECK(input_side_packet_handler);
  mirrors_.emplace_back(input_side_packet_handler, id);
}

::mediapipe::Status OutputSidePacketImpl::SetInternal(const Packet& packet) {
  if (initialized_) {
    return ::mediapipe::AlreadyExistsErrorBuilder(MEDIAPIPE_LOC)
           << "Output side packet \"" << name_ << "\" was already set.";
  }

  if (packet.IsEmpty()) {
    return ::mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "Empty packet set on output side packet \"" << name_ << "\".";
  }

  if (packet.Timestamp() != Timestamp::Unset()) {
    return ::mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "Output side packet \"" << name_ << "\" has a timestamp "
           << packet.Timestamp().DebugString() << ".";
  }

  ::mediapipe::Status result = packet_type_->Validate(packet);
  if (!result.ok()) {
    return ::mediapipe::StatusBuilder(result, MEDIAPIPE_LOC).SetPrepend()
           << absl::StrCat(
                  "Packet type mismatch on calculator output side packet \"",
                  name_, "\": ");
  }

  packet_ = packet;
  initialized_ = true;
  for (const auto& mirror : mirrors_) {
    mirror.input_side_packet_handler->Set(mirror.id, packet_);
  }
  return ::mediapipe::OkStatus();
}

void OutputSidePacketImpl::TriggerErrorCallback(
    const ::mediapipe::Status& status) const {
  CHECK(error_callback_);
  error_callback_(status);
}

}  // namespace mediapipe
