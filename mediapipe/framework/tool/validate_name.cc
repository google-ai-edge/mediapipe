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

// Definition of helper functions.
#include "mediapipe/framework/tool/validate_name.h"

#include <cstdint>

#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/core_proto_inc.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/source_location.h"
#include "mediapipe/framework/port/status_builder.h"
#include "mediapipe/framework/port/status_macros.h"

namespace mediapipe {
namespace tool {
#define MEDIAPIPE_NAME_REGEX "[a-z_][a-z0-9_]*"
#define MEDIAPIPE_NUMBER_REGEX "(0|[1-9][0-9]*)"
#define MEDIAPIPE_TAG_REGEX "[A-Z_][A-Z0-9_]*"
#define MEDIAPIPE_TAG_AND_NAME_REGEX \
  "(" MEDIAPIPE_TAG_REGEX ":)?" MEDIAPIPE_NAME_REGEX
#define MEDIAPIPE_TAG_INDEX_NAME_REGEX                \
  "(" MEDIAPIPE_TAG_REGEX ":(" MEDIAPIPE_NUMBER_REGEX \
  ":)?)?" MEDIAPIPE_NAME_REGEX
#define MEDIAPIPE_TAG_INDEX_REGEX \
  "(" MEDIAPIPE_TAG_REGEX ")?(:" MEDIAPIPE_NUMBER_REGEX ")?"

absl::Status GetTagAndNameInfo(
    const proto_ns::RepeatedPtrField<ProtoString>& tags_and_names,
    TagAndNameInfo* info) {
  RET_CHECK(info);
  info->tags.clear();
  info->names.clear();
  for (const auto& tag_and_name : tags_and_names) {
    std::string tag;
    std::string name;
    MP_RETURN_IF_ERROR(ParseTagAndName(tag_and_name, &tag, &name));
    if (!tag.empty()) {
      info->tags.push_back(tag);
    }
    info->names.push_back(name);
  }
  if (!info->tags.empty() && info->names.size() != info->tags.size()) {
    info->tags.clear();
    info->names.clear();
    return absl::InvalidArgumentError(absl::StrCat(
        "Each set of names must use exclusively either tags or indexes.  "
        "Encountered: \"",
        absl::StrJoin(tags_and_names, "\", \""), "\""));
  }
  return absl::OkStatus();
}

absl::Status SetFromTagAndNameInfo(
    const TagAndNameInfo& info,
    proto_ns::RepeatedPtrField<ProtoString>* tags_and_names) {
  tags_and_names->Clear();
  if (info.tags.empty()) {
    for (const auto& name : info.names) {
      MP_RETURN_IF_ERROR(ValidateName(name));
      *tags_and_names->Add() = name;
    }
  } else {
    if (info.names.size() != info.tags.size()) {
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "Number of tags " << info.names.size()
             << " does not match the number of tags " << info.tags.size();
    }
    for (int i = 0; i < info.tags.size(); ++i) {
      MP_RETURN_IF_ERROR(ValidateTag(info.tags[i]));
      MP_RETURN_IF_ERROR(ValidateName(info.names[i]));
      *tags_and_names->Add() = absl::StrCat(info.tags[i], ":", info.names[i]);
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateName(const std::string& name) {
  return name.length() > 0 && (name[0] == '_' || islower(name[0])) &&
                 std::all_of(name.begin() + 1, name.end(),
                             [](char c) {
                               return c == '_' || isdigit(c) || islower(c);
                             })
             ? absl::OkStatus()
             : absl::InvalidArgumentError(absl::StrCat(
                   "Name \"", absl::CEscape(name),
                   "\" does not match \"" MEDIAPIPE_NAME_REGEX "\"."));
}

absl::Status ValidateNumber(const std::string& number) {
  return (number.length() == 1 && isdigit(number[0])) ||
                 (number.length() > 1 && isdigit(number[0]) &&
                  number[0] != '0' &&
                  std::all_of(number.begin() + 1, number.end(),
                              [](char c) { return isdigit(c); }))
             ? absl::OkStatus()
             : absl::InvalidArgumentError(absl::StrCat(
                   "Number \"", absl::CEscape(number),
                   "\" does not match \"" MEDIAPIPE_NUMBER_REGEX "\"."));
}

absl::Status ValidateTag(const std::string& tag) {
  return tag.length() > 0 && (tag[0] == '_' || isupper(tag[0])) &&
                 std::all_of(tag.begin() + 1, tag.end(),
                             [](char c) {
                               return c == '_' || isdigit(c) || isupper(c);
                             })
             ? absl::OkStatus()
             : absl::InvalidArgumentError(absl::StrCat(
                   "Tag \"", absl::CEscape(tag),
                   "\" does not match \"" MEDIAPIPE_TAG_REGEX "\"."));
}

absl::Status ParseTagAndName(absl::string_view tag_and_name, std::string* tag,
                             std::string* name) {
  // An optional tag and colon, followed by a name.
  RET_CHECK(tag);
  RET_CHECK(name);
  absl::Status tag_status = absl::OkStatus();
  absl::Status name_status = absl::UnknownError("");
  int name_index = -1;
  std::vector<std::string> v = absl::StrSplit(tag_and_name, ':');
  if (v.size() == 1) {
    name_status = ValidateName(v[0]);
    name_index = 0;
  } else if (v.size() == 2) {
    tag_status = ValidateTag(v[0]);
    name_status = ValidateName(v[1]);
    name_index = 1;
  }  // else omitted, name_index == -1, triggering error.
  if (name_index == -1 || tag_status != absl::OkStatus() ||
      name_status != absl::OkStatus()) {
    tag->clear();
    name->clear();
    return absl::InvalidArgumentError(
        absl::StrCat("\"tag and name\" is invalid, \"", tag_and_name,
                     "\" does not match "
                     "\"" MEDIAPIPE_TAG_AND_NAME_REGEX
                     "\" (examples: \"TAG:name\", \"longer_name\")."));
  }
  *tag = name_index == 1 ? v[0] : "";
  *name = v[name_index];
  return absl::OkStatus();
}

absl::Status ParseTagIndexName(const std::string& tag_index_name,
                               std::string* tag, int* index,
                               std::string* name) {
  // An optional tag and colon, an optional index and color, followed by a name.
  RET_CHECK(tag);
  RET_CHECK(index);
  RET_CHECK(name);

  absl::Status tag_status = absl::OkStatus();
  absl::Status number_status = absl::OkStatus();
  absl::Status name_status = absl::UnknownError("");
  int name_index = -1;
  int the_index = 0;
  std::vector<std::string> v = absl::StrSplit(tag_index_name, ':');
  if (v.size() == 1) {
    name_status = ValidateName(v[0]);
    the_index = -1;
    name_index = 0;
  } else if (v.size() == 2) {
    tag_status = ValidateTag(v[0]);
    name_status = ValidateName(v[1]);
    name_index = 1;
  } else if (v.size() == 3) {
    tag_status = ValidateTag(v[0]);
    number_status = ValidateNumber(v[1]);
    if (number_status.ok()) {
      int64_t index64;
      RET_CHECK(absl::SimpleAtoi(v[1], &index64));
      RET_CHECK_LE(index64, internal::kMaxCollectionItemId);
      the_index = index64;
    }
    name_status = ValidateName(v[2]);
    name_index = 2;
  }  // else omitted, name_index == -1, triggering error.
  if (name_index == -1 || !tag_status.ok() || !number_status.ok() ||
      !name_status.ok()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "TAG:index:name is invalid, \"", tag_index_name,
        "\" does not match "
        "\"" MEDIAPIPE_TAG_INDEX_NAME_REGEX
        "\" (examples: \"TAG:name\" \"VIDEO:2:name_b\", \"longer_name\")."));
  }
  *tag = name_index != 0 ? v[0] : "";
  *index = the_index;
  *name = v[name_index];
  return absl::OkStatus();
}

absl::Status ParseTagIndex(const std::string& tag_index, std::string* tag,
                           int* index) {
  RET_CHECK(tag);
  RET_CHECK(index);

  absl::Status tag_status = absl::OkStatus();
  absl::Status number_status = absl::OkStatus();
  int the_index = -1;
  std::vector<std::string> v = absl::StrSplit(tag_index, ':');
  if (v.size() == 1) {
    if (!v[0].empty()) {
      tag_status = ValidateTag(v[0]);
    }
    the_index = 0;
  } else if (v.size() == 2) {
    if (!v[0].empty()) {
      tag_status = ValidateTag(v[0]);
    }
    number_status = ValidateNumber(v[1]);
    if (number_status.ok()) {
      int64_t index64;
      RET_CHECK(absl::SimpleAtoi(v[1], &index64));
      RET_CHECK_LE(index64, internal::kMaxCollectionItemId);
      the_index = index64;
    }
  }  // else omitted, the_index == -1, triggering error.
  if (the_index == -1 || !tag_status.ok() || !number_status.ok()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "TAG:index is invalid, \"", tag_index,
        "\" does not match "
        "\"" MEDIAPIPE_TAG_INDEX_REGEX "\" (examples: \"TAG\" \"VIDEO:2\")."));
  }
  *tag = v[0];
  *index = the_index;
  return absl::OkStatus();
}

#undef MEDIAPIPE_NAME_REGEX
#undef MEDIAPIPE_TAG_REGEX
#undef MEDIAPIPE_TAG_AND_NAME_REGEX
#undef MEDIAPIPE_TAG_INDEX_NAME_REGEX
#undef MEDIAPIPE_TAG_INDEX_REGEX

}  // namespace tool
}  // namespace mediapipe
