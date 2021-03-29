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

#include "mediapipe/framework/input_side_packet_handler.h"

#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_builder.h"
#include "mediapipe/framework/tool/fill_packet_set.h"

namespace mediapipe {

absl::Status InputSidePacketHandler::PrepareForRun(
    const PacketTypeSet* input_side_packet_types,
    const std::map<std::string, Packet>& all_side_packets,
    std::function<void()> input_side_packets_ready_callback,
    std::function<void(absl::Status)> error_callback) {
  int missing_input_side_packet_count;
  prev_input_side_packets_ = std::move(input_side_packets_);
  ASSIGN_OR_RETURN(
      input_side_packets_,
      tool::FillPacketSet(*input_side_packet_types, all_side_packets,
                          &missing_input_side_packet_count));

  input_side_packet_types_ = input_side_packet_types;
  missing_input_side_packet_count_.store(missing_input_side_packet_count,
                                         std::memory_order_relaxed);
  input_side_packets_ready_callback_ =
      std::move(input_side_packets_ready_callback);
  error_callback_ = std::move(error_callback);
  return absl::OkStatus();
}

bool InputSidePacketHandler::InputSidePacketsChanged() {
  return prev_input_side_packets_ == nullptr ||
         input_side_packets_ == nullptr ||
         *input_side_packets_ != *prev_input_side_packets_;
}

void InputSidePacketHandler::Set(CollectionItemId id, const Packet& packet) {
  absl::Status status = SetInternal(id, packet);
  if (!status.ok()) {
    TriggerErrorCallback(status);
  }
}

absl::Status InputSidePacketHandler::SetInternal(CollectionItemId id,
                                                 const Packet& packet) {
  RET_CHECK_GT(missing_input_side_packet_count_, 0);
  Packet& side_packet = input_side_packets_->Get(id);

  if (!side_packet.IsEmpty()) {
    return mediapipe::AlreadyExistsErrorBuilder(MEDIAPIPE_LOC)
           << "Input side packet with id " << id << " was already set.";
  }
  absl::Status result = input_side_packet_types_->Get(id).Validate(packet);
  if (!result.ok()) {
    return mediapipe::StatusBuilder(result, MEDIAPIPE_LOC).SetPrepend()
           << absl::StrCat(
                  "Packet type mismatch on calculator input side packet with "
                  "id ",
                  id.value(), ": ");
  }
  side_packet = packet;
  if (missing_input_side_packet_count_.fetch_sub(
          1, std::memory_order_acq_rel) == 1) {
    input_side_packets_ready_callback_();
  }
  return absl::OkStatus();
}

void InputSidePacketHandler::TriggerErrorCallback(
    const absl::Status& status) const {
  CHECK(error_callback_);
  error_callback_(status);
}

}  // namespace mediapipe
