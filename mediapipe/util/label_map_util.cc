// Copyright 2022 The MediaPipe Authors.
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

#include "mediapipe/util/label_map_util.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/map.h"
#include "mediapipe/util/label_map.pb.h"
#include "mediapipe/util/str_util.h"

namespace mediapipe {

absl::StatusOr<proto_ns::Map<int64_t, LabelMapItem>> BuildLabelMapFromFiles(
    absl::string_view labels_file_contents,
    absl::string_view display_names_file_contents) {
  if (labels_file_contents.empty()) {
    return absl::InvalidArgumentError("Expected non-empty labels file.");
  }
  std::vector<absl::string_view> labels;
  mediapipe::ForEachLine(
      labels_file_contents,
      [&labels](absl::string_view line) { labels.push_back(line); });
  if (!labels.empty() && labels.back().empty()) {
    labels.pop_back();
  }

  std::vector<LabelMapItem> label_map_items;
  label_map_items.reserve(labels.size());
  for (int i = 0; i < labels.size(); ++i) {
    LabelMapItem item;
    item.set_name(std::string(labels[i]));
    label_map_items.push_back(std::move(item));
  }

  if (!display_names_file_contents.empty()) {
    std::vector<absl::string_view> display_names;
    mediapipe::ForEachLine(display_names_file_contents,
                           [&display_names](absl::string_view line) {
                             display_names.push_back(line);
                           });
    if (!display_names.empty() && display_names.back().empty()) {
      display_names.pop_back();
    }
    if (display_names.size() != labels.size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Mismatch between number of labels (%d) and display names (%d).",
          labels.size(), display_names.size()));
    }
    for (int i = 0; i < display_names.size(); ++i) {
      label_map_items[i].set_display_name(std::string(display_names[i]));
    }
  }
  proto_ns::Map<int64_t, LabelMapItem> label_map;
  for (int i = 0; i < label_map_items.size(); ++i) {
    label_map[i] = std::move(label_map_items[i]);
  }
  return label_map;
}

}  // namespace mediapipe
