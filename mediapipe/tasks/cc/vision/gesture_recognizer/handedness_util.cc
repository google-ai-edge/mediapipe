/* Copyright 2022 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/tasks/cc/vision/gesture_recognizer/handedness_util.h"

#include <algorithm>

#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace gesture_recognizer {

namespace {}  // namespace

bool IsLeftHand(const Classification& c) {
  return absl::EqualsIgnoreCase(c.label(), "Left");
}

bool IsRightHand(const Classification& c) {
  return absl::EqualsIgnoreCase(c.label(), "Right");
}

absl::StatusOr<float> GetRightHandScore(
    const ClassificationList& classification_list) {
  auto classifications = classification_list.classification();
  auto iter_max =
      std::max_element(classifications.begin(), classifications.end(),
                       [](const Classification& a, const Classification& b) {
                         return a.score() < b.score();
                       });
  RET_CHECK(iter_max != classifications.end());
  const auto& h = *iter_max;
  RET_CHECK_GE(h.score(), 0.5f);
  RET_CHECK_LE(h.score(), 1.0f);
  if (IsLeftHand(h)) {
    return 1.0f - h.score();
  } else if (IsRightHand(h)) {
    return h.score();
  } else {
    // Unrecognized handedness label.
    RET_CHECK_FAIL() << "Unrecognized handedness: " << h.label();
  }
}

}  // namespace gesture_recognizer
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
