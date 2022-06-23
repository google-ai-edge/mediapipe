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

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/map_util.h"
#include "mediapipe/framework/port/source_location.h"
#include "mediapipe/framework/port/status_builder.h"
#include "mediapipe/framework/tool/status_util.h"
#include "mediapipe/framework/tool/type_util.h"
#include "mediapipe/framework/tool/validate_name.h"
#include "mediapipe/framework/type_map.h"

namespace mediapipe {

absl::Status PacketType::AcceptAny(const TypeSpec& type) {
  return absl::OkStatus();
}

absl::Status PacketType::AcceptNone(const TypeSpec& type) {
  auto* special = absl::get_if<SpecialType>(&type);
  if (special &&
      (special->accept_fn_ == AcceptNone || special->accept_fn_ == AcceptAny))
    return absl::OkStatus();
  return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
         << "No packets are allowed for type: [No Type]";
}

PacketType& PacketType::SetAny() {
  type_spec_ = SpecialType{"[Any Type]", &AcceptAny};
  return *this;
}

PacketType& PacketType::SetNone() {
  type_spec_ = SpecialType{"[No Type]", &AcceptNone};
  return *this;
}

PacketType& PacketType::SetSameAs(const PacketType* type) {
  // TODO Union sets together when SetSameAs is called multiple times.
  auto same_as = type->GetSameAs();
  if (same_as == this) {
    // We're the root of the union-find tree.  There's a cycle, which
    // means we might as well be an "Any" type.
    return SetAny();
  }
  type_spec_ = SameAs{same_as};
  return *this;
}

PacketType& PacketType::Optional() {
  optional_ = true;
  return *this;
}

bool PacketType::IsInitialized() const {
  return !absl::holds_alternative<absl::monostate>(type_spec_);
}

const PacketType* PacketType::SameAsPtr() const {
  auto* same_as = absl::get_if<SameAs>(&type_spec_);
  if (same_as) return same_as->other;
  return nullptr;
}

PacketType* PacketType::GetSameAs() {
  auto* same_as = SameAsPtr();
  if (!same_as) {
    return this;
  }
  // Don't optimize the union-find algorithm, since updating the pointer
  // here would require a mutex lock.
  //   same_as_ = same_as_->GetSameAs();
  // Note, we also don't do the "Union by rank" optimization.  We always
  // make the current set point to the root of the other tree.
  // TODO Remove const_cast by making SetSameAs take a non-const
  // PacketType*.
  return const_cast<PacketType*>(same_as->GetSameAs());
}

const PacketType* PacketType::GetSameAs() const {
  auto* same_as = SameAsPtr();
  if (!same_as) {
    return this;
  }
  // See comments in non-const variant.
  return same_as->GetSameAs();
}

bool PacketType::IsAny() const {
  auto* special = absl::get_if<SpecialType>(&type_spec_);
  return special && special->accept_fn_ == AcceptAny;
}

bool PacketType::IsNone() const {
  auto* special = absl::get_if<SpecialType>(&type_spec_);
  // The tests currently require that an uninitialized PacketType return true
  // for IsNone. TODO: change it?
  return !IsInitialized() || (special && special->accept_fn_ == AcceptNone);
}

bool PacketType::IsOneOf() const {
  return absl::holds_alternative<MultiType>(type_spec_);
}

bool PacketType::IsExactType() const {
  return absl::holds_alternative<TypeId>(type_spec_);
}

const std::string* PacketType::RegisteredTypeName() const {
  if (auto* same_as = SameAsPtr()) return same_as->RegisteredTypeName();
  if (auto* type_id = absl::get_if<TypeId>(&type_spec_))
    return MediaPipeTypeStringFromTypeId(*type_id);
  if (auto* multi_type = absl::get_if<MultiType>(&type_spec_))
    return multi_type->registered_type_name;
  return nullptr;
}

namespace internal {

struct TypeIdFormatter {
  void operator()(std::string* out, TypeId t) const {
    absl::StrAppend(out, MediaPipeTypeStringOrDemangled(t));
  }
};

template <class Formatter>
class QuoteFormatter {
 public:
  explicit QuoteFormatter(Formatter&& f) : f_(std::forward<Formatter>(f)) {}

  template <typename T>
  void operator()(std::string* out, const T& t) const {
    absl::StrAppend(out, "\"");
    f_(out, t);
    absl::StrAppend(out, "\"");
  }

 private:
  Formatter f_;
};
template <class Formatter>
explicit QuoteFormatter(Formatter f) -> QuoteFormatter<Formatter>;

}  // namespace internal

std::string PacketType::TypeNameForOneOf(TypeIdSpan types) {
  return absl::StrCat(
      "OneOf<", absl::StrJoin(types, ", ", internal::TypeIdFormatter()), ">");
}

std::string PacketType::DebugTypeName() const {
  if (auto* same_as = absl::get_if<SameAs>(&type_spec_)) {
    // Construct a name based on the current chain of same_as_ links
    // (which may change when the framework expands out Any-type).
    return absl::StrCat("[Same Type As ",
                        same_as->other->GetSameAs()->DebugTypeName(), "]");
  }
  if (auto* special = absl::get_if<SpecialType>(&type_spec_)) {
    return special->name_;
  }
  if (auto* type_id = absl::get_if<TypeId>(&type_spec_)) {
    return MediaPipeTypeStringOrDemangled(*type_id);
  }
  if (auto* multi_type = absl::get_if<MultiType>(&type_spec_)) {
    return TypeNameForOneOf(multi_type->types);
  }
  return "[Undefined Type]";
}

static bool HaveCommonType(absl::Span<const TypeId> types1,
                           absl::Span<const TypeId> types2) {
  for (const auto& first : types1) {
    for (const auto& second : types2) {
      if (first == second) {
        return true;
      }
    }
  }
  return false;
}

absl::Status PacketType::Validate(const Packet& packet) const {
  if (!IsInitialized()) {
    return absl::InvalidArgumentError(
        "Uninitialized PacketType was used for validation.");
  }
  if (SameAsPtr()) {
    // Cycles are impossible at this stage due to being checked for
    // in SetSameAs().
    return GetSameAs()->Validate(packet);
  }
  if (auto* type_id = absl::get_if<TypeId>(&type_spec_)) {
    return packet.ValidateAsType(*type_id);
  }
  if (packet.IsEmpty()) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "Empty packets are not allowed for type: " << DebugTypeName();
  }
  if (auto* multi_type = absl::get_if<MultiType>(&type_spec_)) {
    auto packet_type = packet.GetTypeId();
    if (HaveCommonType(multi_type->types, absl::MakeSpan(&packet_type, 1))) {
      return absl::OkStatus();
    } else {
      return absl::InvalidArgumentError(absl::StrCat(
          "The Packet stores \"", packet.DebugTypeName(), "\", but one of ",
          absl::StrJoin(multi_type->types, ", ",
                        internal::QuoteFormatter(internal::TypeIdFormatter())),
          " was requested."));
    }
  }
  if (auto* special = absl::get_if<SpecialType>(&type_spec_)) {
    return special->accept_fn_(packet.GetTypeId());
  }
  return absl::OkStatus();
}

PacketType::TypeIdSpan PacketType::GetTypeSpan(const TypeSpec& type_spec) {
  if (auto* type_id = absl::get_if<TypeId>(&type_spec))
    return absl::MakeSpan(type_id, 1);
  if (auto* multi_type = absl::get_if<MultiType>(&type_spec))
    return multi_type->types;
  return {};
}

bool PacketType::IsConsistentWith(const PacketType& other) const {
  const PacketType* type1 = GetSameAs();
  const PacketType* type2 = other.GetSameAs();

  TypeIdSpan types1 = GetTypeSpan(type1->type_spec_);
  TypeIdSpan types2 = GetTypeSpan(type2->type_spec_);
  if (!types1.empty() && !types2.empty()) {
    return HaveCommonType(types1, types2);
  }
  if (auto* special1 = absl::get_if<SpecialType>(&type1->type_spec_)) {
    return special1->accept_fn_(type2->type_spec_).ok();
  }
  if (auto* special2 = absl::get_if<SpecialType>(&type2->type_spec_)) {
    return special2->accept_fn_(type1->type_spec_).ok();
  }
  return false;
}

absl::Status ValidatePacketTypeSet(const PacketTypeSet& packet_type_set) {
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
    return absl::InvalidArgumentError(absl::StrCat(
        "ValidatePacketTypeSet failed:\n", absl::StrJoin(errors, "\n")));
  }
  return absl::OkStatus();
}

absl::Status ValidatePacketSet(const PacketTypeSet& packet_type_set,
                               const PacketSet& packet_set) {
  std::vector<absl::Status> errors;
  if (!packet_type_set.TagMap()->SameAs(*packet_set.TagMap())) {
    return absl::InvalidArgumentError(absl::StrCat(
        "TagMaps do not match.  PacketTypeSet TagMap:\n",
        packet_type_set.TagMap()->DebugString(), "\n\nPacketSet TagMap:\n",
        packet_set.TagMap()->DebugString()));
  }
  for (CollectionItemId id = packet_type_set.BeginId();
       id < packet_type_set.EndId(); ++id) {
    absl::Status status = packet_type_set.Get(id).Validate(packet_set.Get(id));
    if (!status.ok()) {
      std::pair<std::string, int> tag_index =
          packet_type_set.TagAndIndexFromId(id);
      errors.push_back(
          mediapipe::StatusBuilder(status, MEDIAPIPE_LOC).SetPrepend()
          << "Packet \"" << packet_type_set.TagMap()->Names()[id.value()]
          << "\" with tag \"" << tag_index.first << "\" and index "
          << tag_index.second << " failed validation.  ");
    }
  }
  if (!errors.empty()) {
    return tool::CombinedStatus("ValidatePacketSet failed:", errors);
  }
  return absl::OkStatus();
}

}  // namespace mediapipe
