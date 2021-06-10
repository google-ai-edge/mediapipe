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

#include "mediapipe/framework/tool/fill_packet_set.h"

#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "mediapipe/framework/port/status_builder.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/framework/tool/status_util.h"

namespace mediapipe {
namespace tool {

absl::StatusOr<std::unique_ptr<PacketSet>> FillPacketSet(
    const PacketTypeSet& input_side_packet_types,
    const std::map<std::string, Packet>& input_side_packets,
    int* missing_packet_count_ptr) {
  if (missing_packet_count_ptr != nullptr) {
    *missing_packet_count_ptr = 0;
  }
  std::vector<absl::Status> errors;
  auto packet_set =
      absl::make_unique<PacketSet>(input_side_packet_types.TagMap());
  const auto& names = input_side_packet_types.TagMap()->Names();
  for (CollectionItemId id = input_side_packet_types.BeginId();
       id < input_side_packet_types.EndId(); ++id) {
    const std::string& name = names[id.value()];
    const auto iter = input_side_packets.find(name);
    if (iter == input_side_packets.end()) {
      if (missing_packet_count_ptr != nullptr) {
        ++(*missing_packet_count_ptr);
      } else {
        errors.push_back(mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
                         << "Missing input side packet: " << name);
      }
      continue;
    }
    packet_set->Get(id) = iter->second;
    // Check the type.
    absl::Status status =
        input_side_packet_types.Get(id).Validate(iter->second);
    if (!status.ok()) {
      std::pair<std::string, int> tag_index =
          input_side_packet_types.TagAndIndexFromId(id);
      errors.push_back(
          mediapipe::StatusBuilder(status, MEDIAPIPE_LOC).SetPrepend()
          << "Packet \""
          << input_side_packet_types.TagMap()->Names()[id.value()]
          << "\" with tag \"" << tag_index.first << "\" and index "
          << tag_index.second << " failed validation.  ");
    }
  }
  if (!errors.empty()) {
    return tool::CombinedStatus("FillPacketSet failed:", errors);
  }
  return std::move(packet_set);
}

}  // namespace tool
}  // namespace mediapipe
