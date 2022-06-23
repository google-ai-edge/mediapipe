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

#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/util/label_map.pb.h"

namespace mediapipe {

absl::StatusOr<proto_ns::Map<int64, LabelMapItem>> BuildLabelMapFromFiles(
    absl::string_view labels_file_contents,
    absl::string_view display_names_file) {
  if (labels_file_contents.empty()) {
    return absl::InvalidArgumentError("Expected non-empty labels file.");
  }
  std::vector<absl::string_view> labels =
      absl::StrSplit(labels_file_contents, '\n');
  // In most cases, there is an empty line (i.e. newline character) at the end
  // of the file that needs to be ignored. In such a situation, StrSplit() will
  // produce a vector with an empty string as final element. Also note that in
  // case `labels_file_contents` is entirely empty, StrSplit() will produce a
  // vector with one single empty substring, so there's no out-of-range risk
  // here.
  if (labels[labels.size() - 1].empty()) {
    labels.pop_back();
  }

  std::vector<LabelMapItem> label_map_items;
  label_map_items.reserve(labels.size());
  for (int i = 0; i < labels.size(); ++i) {
    LabelMapItem item;
    item.set_name(std::string(labels[i]));
    label_map_items.emplace_back(item);
  }

  if (!display_names_file.empty()) {
    std::vector<std::string> display_names =
        absl::StrSplit(display_names_file, '\n');
    // In most cases, there is an empty line (i.e. newline character) at the end
    // of the file that needs to be ignored. See above.
    if (display_names[display_names.size() - 1].empty()) {
      display_names.pop_back();
    }
    if (display_names.size() != labels.size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Mismatch between number of labels (%d) and display names (%d).",
          labels.size(), display_names.size()));
    }
    for (int i = 0; i < display_names.size(); ++i) {
      label_map_items[i].set_display_name(display_names[i]);
    }
  }
  proto_ns::Map<int64, LabelMapItem> label_map;
  for (int i = 0; i < label_map_items.size(); ++i) {
    label_map[i] = label_map_items[i];
  }
  return label_map;
}

}  // namespace mediapipe
