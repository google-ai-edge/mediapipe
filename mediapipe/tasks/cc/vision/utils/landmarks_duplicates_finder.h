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

#ifndef MEDIAPIPE_TASKS_CC_VISION_UTILS_LANDMARKS_DUPLICATES_FINDER_H_
#define MEDIAPIPE_TASKS_CC_VISION_UTILS_LANDMARKS_DUPLICATES_FINDER_H_

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/landmark.pb.h"

namespace mediapipe::tasks::vision::utils {

class DuplicatesFinder {
 public:
  virtual ~DuplicatesFinder() = default;
  // Returns indices of landmark lists to remove to make @multi_landmarks
  // contain different enough (depending on the implementation) landmark lists
  // only.
  virtual absl::StatusOr<absl::flat_hash_set<int>> FindDuplicates(
      const std::vector<mediapipe::NormalizedLandmarkList>& multi_landmarks,
      int input_width, int input_height) = 0;
};

}  // namespace mediapipe::tasks::vision::utils

#endif  // MEDIAPIPE_TASKS_CC_VISION_UTILS_LANDMARKS_DUPLICATES_FINDER_H_
