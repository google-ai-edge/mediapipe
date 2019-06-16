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

// Definitions for PacketType and PacketTypeSet.

#include "mediapipe/framework/packet_type.h"

#include <unordered_set>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/map_util.h"
#include "mediapipe/framework/port/source_location.h"
#include "mediapipe/framework/port/status_builder.h"
#include "mediapipe/framework/tool/status_util.h"
#include "mediapipe/framework/tool/validate_name.h"

namespace mediapipe {

PacketType::PacketType()
    : initialized_(false),
      no_packets_allowed_(true),
      validate_method_(nullptr),
      type_name_("[Undefined Type]"),
      same_as_(nullptr) {}

PacketType& PacketType::SetAny() {
  no_packets_allowed_ = false;
  validate_method_ = nullptr;
  same_as_ = nullptr;
  type_name_ = "[Any Type]";
  initialized_ = true;
  return *this;
}

PacketType& PacketType::SetNone() {
  no_packets_allowed_ = true;
  validate_method_ = nullptr;
  same_as_ = nullptr;
  type_name_ = "[No Type]";
  initialized_ = true;
  return *this;
}

PacketType& PacketType::SetSameAs(const PacketType* type) {
  // TODO Union sets together when SetSameAs is called multiple times.
  no_packets_allowed_ = false;
  validate_method_ = nullptr;
  same_as_ = type->GetSameAs();
  type_name_ = "";

  if (same_as_ == this) {
    // We're the root of the union-find tree.  There's a cycle, which
    // means we might as well be an "Any" type.
    same_as_ = nullptr;
  }

  initialized_ = true;
  return *this;
}

PacketType& PacketType::Optional() {
  optional_ = true;
  return *this;
}

bool PacketType::IsInitialized() const { return initialized_; }

PacketType* PacketType::GetSameAs() {
  if (!same_as_) {
    return this;
  }
  // Don't optimize the union-find algorithm, since updating the pointer
  // here would require a mutex lock.
  //   same_as_ = same_as_->GetSameAs();
  // Note, we also don't do the "Union by rank" optimization.  We always
  // make the current set point to the root of the other tree.
  // TODO Remove const_cast by making SetSameAs take a non-const
  // PacketType*.
  return const_cast<PacketType*>(same_as_->GetSameAs());
}

const PacketType* PacketType::GetSameAs() const {
  if (!same_as_) {
    return this;
  }
  // See comments in non-const variant.
  return same_as_->GetSameAs();
}

bool PacketType::IsAny() const {
  return !no_packets_allowed_ && validate_method_ == nullptr &&
         same_as_ == nullptr;
}

bool PacketType::IsNone() const { return no_packets_allowed_; }

const std::string* PacketType::RegisteredTypeName() const {
  if (same_as_) {
    return GetSameAs()->RegisteredTypeName();
  }
  return registered_type_name_ptr_;
}

const std::string PacketType::DebugTypeName() const {
  if (same_as_) {
    // Construct a name based on the current chain of same_as_ links
    // (which may change when the framework expands out Any-type).
    return absl::StrCat("[Same Type As ", GetSameAs()->DebugTypeName(), "]");
  }
  return type_name_;
}

::mediapipe::Status PacketType::Validate(const Packet& packet) const {
  if (!initialized_) {
    return ::mediapipe::InvalidArgumentError(
        "Uninitialized PacketType was used for validation.");
  }
  if (same_as_) {
    // Cycles are impossible at this stage due to being checked for
    // in SetSameAs().
    return GetSameAs()->Validate(packet);
  }
  if (no_packets_allowed_) {
    return ::mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "No packets are allowed for type: " << type_name_;
  }
  if (validate_method_ != nullptr) {
    return (packet.*validate_method_)();
  }
  // The PacketType is the Any Type.
  if (packet.IsEmpty()) {
    return ::mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "Empty packets are not allowed for type: " << type_name_;
  }
  return ::mediapipe::OkStatus();
}

bool PacketType::IsConsistentWith(const PacketType& other) const {
  const PacketType* type1 = GetSameAs();
  const PacketType* type2 = other.GetSameAs();

  if (type1->validate_method_ == nullptr ||
      type2->validate_method_ == nullptr) {
    // type1 or type2 either accepts anything or nothing.
    if (type1->validate_method_ == nullptr && !type1->no_packets_allowed_) {
      // type1 accepts anything.
      return true;
    }
    if (type2->validate_method_ == nullptr && !type2->no_packets_allowed_) {
      // type2 accepts anything.
      return true;
    }
    if (type1->no_packets_allowed_ && type2->no_packets_allowed_) {
      // type1 and type2 both accept nothing.
      return true;
    }
    // The only special case left is that only one of "type1" or "type2"
    // accepts nothing, which means there is no match.
    return false;
  }
  return type1->validate_method_ == type2->validate_method_;
}

::mediapipe::Status ValidatePacketTypeSet(
    const PacketTypeSet& packet_type_set) {
  std::vector<std::string> errors;
  if (packet_type_set.GetErrorHandler().HasError()) {
    errors = packet_type_set.GetErrorHandler().ErrorMessages();
  }
  for (CollectionItemId id = packet_type_set.BeginId();
       id < packet_type_set.EndId(); ++id) {
    if (!packet_type_set.Get(id).IsInitialized()) {
      auto item = packet_type_set.TagAndIndexFromId(id);
      errors.push_back(absl::StrCat("Tag \"", item.first, "\" index ",
                                    item.second, " was not expected."));
    }
  }
  if (!errors.empty()) {
    return ::mediapipe::InvalidArgumentError(absl::StrCat(
        "ValidatePacketTypeSet failed:\n", absl::StrJoin(errors, "\n")));
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status ValidatePacketSet(const PacketTypeSet& packet_type_set,
                                      const PacketSet& packet_set) {
  std::vector<::mediapipe::Status> errors;
  if (!packet_type_set.TagMap()->SameAs(*packet_set.TagMap())) {
    return ::mediapipe::InvalidArgumentError(absl::StrCat(
        "TagMaps do not match.  PacketTypeSet TagMap:\n",
        packet_type_set.TagMap()->DebugString(), "\n\nPacketSet TagMap:\n",
        packet_set.TagMap()->DebugString()));
  }
  for (CollectionItemId id = packet_type_set.BeginId();
       id < packet_type_set.EndId(); ++id) {
    ::mediapipe::Status status =
        packet_type_set.Get(id).Validate(packet_set.Get(id));
    if (!status.ok()) {
      std::pair<std::string, int> tag_index =
          packet_type_set.TagAndIndexFromId(id);
      errors.push_back(
          ::mediapipe::StatusBuilder(status, MEDIAPIPE_LOC).SetPrepend()
          << "Packet \"" << packet_type_set.TagMap()->Names()[id.value()]
          << "\" with tag \"" << tag_index.first << "\" and index "
          << tag_index.second << " failed validation.  ");
    }
  }
  if (!errors.empty()) {
    return tool::CombinedStatus("ValidatePacketSet failed:", errors);
  }
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
