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

#include "mediapipe/framework/packet.h"

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_builder.h"

namespace mediapipe {
namespace packet_internal {

HolderBase::~HolderBase() {}

Packet Create(HolderBase* holder) {
  Packet result;
  result.holder_.reset(holder);
  return result;
}

Packet Create(HolderBase* holder, Timestamp timestamp) {
  Packet result;
  result.holder_.reset(holder);
  result.timestamp_ = timestamp;
  return result;
}

Packet Create(std::shared_ptr<HolderBase> holder, Timestamp timestamp) {
  Packet result;
  result.holder_ = std::move(holder);
  result.timestamp_ = timestamp;
  return result;
}

const HolderBase* GetHolder(const Packet& packet) {
  return packet.holder_.get();
}

::mediapipe::StatusOr<Packet> PacketFromDynamicProto(
    const std::string& type_name, const std::string& serialized) {
  ASSIGN_OR_RETURN(
      auto message_holder,
      packet_internal::MessageHolderRegistry::CreateByName(type_name));
  auto* message =
      const_cast<proto_ns::MessageLite*>(message_holder->GetProtoMessageLite());
  RET_CHECK_NE(message, nullptr);
  RET_CHECK(message->ParseFromString(serialized));
  return packet_internal::Create(message_holder.release());
}

}  // namespace packet_internal

Packet Packet::At(class Timestamp timestamp) const& {
  Packet result(*this);
  result.timestamp_ = timestamp;
  return result;
}

Packet Packet::At(class Timestamp timestamp) && {
  timestamp_ = timestamp;
  // Note: Below object is of type Packet&& which doesn't match return type
  // of Packet, so a new Packet will be created by move constructor here.
  // Moving everything to a new Packet is done intentionally to preserve the
  // immutability of Timestamps.
  return std::move(*this);
}

std::string Packet::RegisteredTypeName() const {
  if (IsEmpty()) {
    return "";
  }
  return holder_->RegisteredTypeName();
}

std::string Packet::DebugTypeName() const {
  if (IsEmpty()) {
    return "{empty}";
  }
  return holder_->DebugTypeName();
}

std::string Packet::DebugString() const {
  std::string result = absl::StrCat("mediapipe::Packet with timestamp: ",
                                    timestamp_.DebugString());
  if (IsEmpty()) {
    absl::StrAppend(&result, " and no data");
  } else {
    absl::StrAppend(&result, " and type: ", holder_->DebugTypeName());
  }
  return result;
}

::mediapipe::Status Packet::ValidateAsProtoMessageLite() const {
  if (ABSL_PREDICT_FALSE(IsEmpty())) {
    return ::mediapipe::InternalError("Packet is empty.");
  }
  if (ABSL_PREDICT_FALSE(holder_->GetProtoMessageLite() == nullptr)) {
    return ::mediapipe::InvalidArgumentError(
        absl::StrCat("The Packet stores \"", holder_->DebugTypeName(), "\"",
                     "which is not convertible to proto_ns::MessageLite."));
  } else {
    return ::mediapipe::OkStatus();
  }
}

const proto_ns::MessageLite& Packet::GetProtoMessageLite() const {
  CHECK(holder_ != nullptr) << "The packet is empty.";
  const proto_ns::MessageLite* proto = holder_->GetProtoMessageLite();
  CHECK(proto != nullptr) << "The Packet stores '" << holder_->DebugTypeName()
                          << "', it cannot be converted to MessageLite type.";
  return *proto;
}

StatusOr<std::vector<const proto_ns::MessageLite*>>
Packet::GetVectorOfProtoMessageLitePtrs() {
  if (holder_ == nullptr) {
    return ::mediapipe::InternalError("Packet is empty.");
  }
  return holder_->GetVectorOfProtoMessageLite();
}

MEDIAPIPE_REGISTER_TYPE(::mediapipe::Packet, "::mediapipe::Packet", nullptr,
                        nullptr);
MEDIAPIPE_REGISTER_TYPE(::std::vector<::mediapipe::Packet>,
                        "::std::vector<::mediapipe::Packet>", nullptr, nullptr);
MEDIAPIPE_REGISTER_TYPE(::mediapipe::Timestamp, "::mediapipe::Timestamp",
                        nullptr, nullptr);
#define PACKET_PAIR_TYPE ::std::pair<::mediapipe::Packet, ::mediapipe::Packet>
MEDIAPIPE_REGISTER_TYPE(PACKET_PAIR_TYPE,
                        "::std::pair<::mediapipe::Packet,::mediapipe::Packet>",
                        nullptr, nullptr);
#undef PACKET_PAIR_TYPE
#define PACKET_MAP_TYPE ::std::map<std::string, ::mediapipe::Packet>
MEDIAPIPE_REGISTER_TYPE(PACKET_MAP_TYPE,
                        "::std::map<std::string,::mediapipe::Packet>", nullptr,
                        nullptr);
#undef PACKET_MAP_TYPE

}  // namespace mediapipe
