// Copyright 2023 The MediaPipe Authors.
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

#include "mediapipe/calculators/tensor/tensors_to_segmentation_utils.h"

#include <tuple>
#include <vector>

#include "absl/status/statusor.h"
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

int NumGroups(int size, int group_size) {
  return (size + group_size - 1) / group_size;
}

bool CanUseGpu() {
#if !MEDIAPIPE_DISABLE_GPU || MEDIAPIPE_METAL_ENABLED
  // TODO: Configure GPU usage policy in individual calculators.
  constexpr bool kAllowGpuProcessing = true;
  return kAllowGpuProcessing;
#else
  return false;
#endif  // !MEDIAPIPE_DISABLE_GPU || MEDIAPIPE_METAL_ENABLED
}

absl::StatusOr<std::tuple<int, int, int>> GetHwcFromDims(
    const std::vector<int>& dims) {
  if (dims.size() == 3) {
    return std::make_tuple(dims[0], dims[1], dims[2]);
  } else if (dims.size() == 4) {
    // BHWC format check B == 1
    RET_CHECK_EQ(dims[0], 1) << "Expected batch to be 1 for BHWC heatmap";
    return std::make_tuple(dims[1], dims[2], dims[3]);
  } else {
    RET_CHECK(false) << "Invalid shape for segmentation tensor " << dims.size();
  }
}
}  // namespace mediapipe
