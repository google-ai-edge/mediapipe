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

#include "mediapipe/framework/tool/tag_map_helper.h"

#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/port/core_proto_inc.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/proto_ns.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/framework/tool/tag_map.h"

namespace mediapipe {
namespace tool {

// Create using a vector of TAG:<index>:name.
absl::StatusOr<std::shared_ptr<TagMap>> CreateTagMap(
    const std::vector<std::string>& tag_index_names) {
  proto_ns::RepeatedPtrField<ProtoString> fields;
  for (const auto& tag_index_name : tag_index_names) {
    *fields.Add() = tag_index_name;
  }
  return TagMap::Create(fields);
}

// Create using an integer number of entries (for tag "").
absl::StatusOr<std::shared_ptr<TagMap>> CreateTagMap(int num_entries) {
  RET_CHECK_LE(0, num_entries);
  proto_ns::RepeatedPtrField<ProtoString> fields;
  for (int i = 0; i < num_entries; ++i) {
    *fields.Add() = absl::StrCat("name", i);
  }
  return TagMap::Create(fields);
}

// Create using a vector of just tag names.
absl::StatusOr<std::shared_ptr<TagMap>> CreateTagMapFromTags(
    const std::vector<std::string>& tags) {
  proto_ns::RepeatedPtrField<ProtoString> fields;
  for (int i = 0; i < tags.size(); ++i) {
    *fields.Add() = absl::StrCat(tags[i], ":name", i);
  }
  return TagMap::Create(fields);
}

}  // namespace tool
}  // namespace mediapipe
