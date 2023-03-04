#include "mediapipe/tasks/cc/metadata/metadata_version_utils.h"

#include <string>

#include "absl/strings/str_split.h"

namespace mediapipe {
namespace tasks {
namespace metadata {
namespace {

static int32_t GetValueOrZero(const std::vector<std::string> &list,
                              const int index) {
  int32_t value = 0;
  if (index <= list.size() - 1) {
    value = std::stoi(list[index]);
  }
  return value;
}

}  // namespace

int CompareVersions(absl::string_view version_a, absl::string_view version_b) {
  std::vector<std::string> version_a_components =
      absl::StrSplit(version_a, '.', absl::SkipEmpty());
  std::vector<std::string> version_b_components =
      absl::StrSplit(version_b, '.', absl::SkipEmpty());

  const int a_length = version_a_components.size();
  const int b_length = version_b_components.size();
  const int max_length = std::max(a_length, b_length);

  for (int i = 0; i < max_length; ++i) {
    const int a_val = GetValueOrZero(version_a_components, i);
    const int b_val = GetValueOrZero(version_b_components, i);
    if (a_val > b_val) {
      return 1;
    }
    if (a_val < b_val) {
      return -1;
    }
  }
  return 0;
}

}  // namespace metadata
}  // namespace tasks
}  // namespace mediapipe
