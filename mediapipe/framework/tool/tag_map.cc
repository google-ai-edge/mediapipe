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

#include "mediapipe/framework/tool/tag_map.h"

#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/core_proto_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_builder.h"
#include "mediapipe/framework/tool/validate_name.h"

namespace mediapipe {
namespace tool {

void TagMap::InitializeNames(
    const std::map<std::string, std::vector<std::string>>& tag_to_names) {
  names_.reserve(num_entries_);
  for (const auto& item : tag_to_names) {
    names_.insert(names_.end(), item.second.begin(), item.second.end());
  }
}

absl::Status TagMap::Initialize(
    const proto_ns::RepeatedPtrField<ProtoString>& tag_index_names) {
  std::map<std::string, std::vector<std::string>> tag_to_names;
  for (const auto& tag_index_name : tag_index_names) {
    std::string tag;
    int index;
    std::string name;
    MP_RETURN_IF_ERROR(ParseTagIndexName(tag_index_name, &tag, &index, &name));

    // Get a reference to the tag data (possibly creating it).
    TagData& tag_data = mapping_[tag];

    // If index == -1, then we get the index from the number of times
    // the tag has been used (this is only used for tag "").
    if (index == -1) {
      index = tag_data.count;
    }
    ++tag_data.count;

    // Add to the per tag names, being careful about allowing indexes
    // to be out of order.
    std::vector<std::string>& names = tag_to_names[tag];
    if (names.size() <= index) {
      names.resize(index + 1);
    }
    if (!names[index].empty()) {
      return mediapipe::FailedPreconditionErrorBuilder(MEDIAPIPE_LOC)
             << "tag \"" << tag << "\" index " << index
             << " already had a name \"" << names[index]
             << "\" but is being reassigned a name \"" << name << "\"";
    }
    names[index] = name;
  }

  // Set all the initial indexes to an index in the data vector.
  int current_index = 0;
  for (auto& item : mapping_) {
    TagData& tag_data = item.second;
    // Ensure that a name was assigned for each index of the tag.
    // If the number of indexes used matches the size of names array
    // (and an index couldn't have been reused due to the check in the
    // loop above), this means that all indexes were used exactly once.
    const std::vector<std::string>& names = tag_to_names[item.first];
    if (tag_data.count != names.size()) {
      auto builder = mediapipe::FailedPreconditionErrorBuilder(MEDIAPIPE_LOC)
                     << "Not all indexes were assigned names.  Tag \""
                     << item.first << "\" has the following:\n";
      // Note, names.size() will always be larger than tag_data.count.
      for (int index = 0; index < names.size(); ++index) {
        if (!names[index].empty()) {
          builder << "index " << index << " name \"" << names[index] << "\"\n";
        } else {
          builder << "index " << index << " name <missing>\n";
        }
      }
      return std::move(builder);
    }
    tag_data.id = CollectionItemId(current_index);
    current_index += tag_data.count;
  }
  num_entries_ = current_index;

  InitializeNames(tag_to_names);
  return absl::OkStatus();
}

absl::Status TagMap::Initialize(const TagAndNameInfo& info) {
  if (info.tags.empty()) {
    if (!info.names.empty()) {
      mapping_.emplace(
          std::piecewise_construct, std::forward_as_tuple(""),
          std::forward_as_tuple(CollectionItemId(0), info.names.size()));
      names_ = info.names;
    }
    num_entries_ = info.names.size();
  } else {
    std::map<std::string, std::vector<std::string>> tag_to_names;
    if (info.tags.size() != info.names.size()) {
      return absl::FailedPreconditionError(
          "Expected info.tags.size() == info.names.size()");
    }

    // Add the tags (unsorted).
    for (int i = 0; i < info.tags.size(); ++i) {
      auto item = mapping_.emplace(std::piecewise_construct,
                                   std::forward_as_tuple(info.tags[i]),
                                   std::forward_as_tuple());
      RET_CHECK(item.second) << "Tag was used twice.";
      tag_to_names[info.tags[i]].emplace_back(info.names[i]);
    }
    // Assign descriptor values (sorted).
    int current_index = 0;
    for (auto& item : mapping_) {
      item.second.id = CollectionItemId(current_index);
      item.second.count = 1;
      ++current_index;
    }
    num_entries_ = current_index;

    // Now create the names_ array in the correctly sorted order.
    InitializeNames(tag_to_names);
  }
  return absl::OkStatus();
}

proto_ns::RepeatedPtrField<ProtoString> TagMap::CanonicalEntries() const {
  proto_ns::RepeatedPtrField<ProtoString> fields;
  for (const auto& item : mapping_) {
    const std::string& tag = item.first;
    const TagData& tag_data = item.second;
    if (tag.empty()) {
      // "no_tag1", "no_tag2".
      for (int i = 0; i < tag_data.count; ++i) {
        *fields.Add() = names_[tag_data.id.value() + i];
      }
    } else if (tag_data.count <= 1) {
      // "ONLY_ONE_INDEX:name"
      *fields.Add() = absl::StrCat(tag, ":", names_[tag_data.id.value()]);
    } else {
      // "TAG:0:name0", "TAG:1:name1"
      for (int i = 0; i < tag_data.count; ++i) {
        *fields.Add() =
            absl::StrCat(tag, ":", i, ":", names_[tag_data.id.value() + i]);
      }
    }
  }
  return fields;
}

// Examples:
//   BLAH:0:blah1
//   BLAH:1:blah2
//
//   A:a
//   B:b
//
//   A:0:a0
//   A:1:a1
//   A:2:a2
//   B:0:b0
//   B:1:b1
//   C:c0
std::string TagMap::DebugString() const {
  if (num_entries_ == 0) {
    return "empty";
  }
  return absl::StrJoin(CanonicalEntries(), "\n");
}

// Note, this is also currently used internally to check for equivalence.
//
// Examples:
//   {"BLAH", 2}
//
//   {"A", 1}, {"B", 1}
//
//   {"A", 3}, {"B", 2}, {"C", 1}
//
//   {"", 4}, {"A", 3}, {"B", 2}, {"C", 1}
std::string TagMap::ShortDebugString() const {
  if (num_entries_ == 0) {
    return "empty";
  }
  std::string output;
  for (const auto& item : mapping_) {
    if (!output.empty()) {
      absl::StrAppend(&output, ", ");
    }
    if (item.second.count == 0) {
      absl::StrAppend(&output, "\"", item.first, "\"");
    } else {
      absl::StrAppend(&output, "{\"", item.first, "\", ", item.second.count,
                      "}");
    }
  }
  return output;
}

bool TagMap::HasTag(const absl::string_view tag) const {
  return mapping_.contains(tag);
}

int TagMap::NumEntries(const absl::string_view tag) const {
  const auto it = mapping_.find(tag);
  return it != mapping_.end() ? it->second.count : 0;
}

CollectionItemId TagMap::GetId(const absl::string_view tag, int index) const {
  const auto it = mapping_.find(tag);
  if (it == mapping_.end()) {
    return CollectionItemId::GetInvalid();
  }
  if (index < 0 || index >= it->second.count) {
    return CollectionItemId::GetInvalid();
  }
  return it->second.id + index;
}

std::pair<std::string, int> TagMap::TagAndIndexFromId(
    CollectionItemId id) const {
  for (const auto& item : mapping_) {
    if (id >= item.second.id && id < item.second.id + item.second.count) {
      return std::make_pair(item.first, (id - item.second.id).value());
    }
  }
  return {"", -1};
}

CollectionItemId TagMap::BeginId(const absl::string_view tag) const {
  return GetId(tag, 0);
}

CollectionItemId TagMap::EndId(const absl::string_view tag) const {
  const auto it = mapping_.find(tag);
  if (it == mapping_.end()) {
    return CollectionItemId::GetInvalid();
  }
  return it->second.id + it->second.count;
}

std::set<std::string> TagMap::GetTags() const {
  std::set<std::string> tag_names;
  for (const auto& item : mapping_) {
    tag_names.insert(item.first);
  }
  return tag_names;
}

bool TagMap::SameAs(const TagMap& other) const {
  return &other == this || ShortDebugString() == other.ShortDebugString();
}

}  // namespace tool
}  // namespace mediapipe
