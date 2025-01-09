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
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::tensors_to_segmentation_utils {
namespace {

using ::testing::HasSubstr;

TEST(TensorsToSegmentationUtilsTest, NumGroupsWorksProperly) {
  EXPECT_EQ(NumGroups(13, 4), 4);
  EXPECT_EQ(NumGroups(4, 13), 1);
}

TEST(TensorsToSegmentationUtilsTest, GetHwcFromDimsWorksProperly) {
  std::vector<int> dims_3 = {2, 3, 4};
  absl::StatusOr<std::tuple<int, int, int>> result_1 = GetHwcFromDims(dims_3);
  MP_ASSERT_OK(result_1);
  EXPECT_EQ(result_1.value(), (std::make_tuple(2, 3, 4)));
  std::vector<int> dims_4 = {1, 3, 4, 5};
  absl::StatusOr<std::tuple<int, int, int>> result_2 = GetHwcFromDims(dims_4);
  MP_ASSERT_OK(result_2);
  EXPECT_EQ(result_2.value(), (std::make_tuple(3, 4, 5)));
}

TEST(TensorsToSegmentationUtilsTest, GetHwcFromDimsBatchCheckFail) {
  std::vector<int> dims_4 = {2, 3, 4, 5};
  absl::StatusOr<std::tuple<int, int, int>> result = GetHwcFromDims(dims_4);
  EXPECT_FALSE(result.ok());
  EXPECT_THAT(result.status().message(),
              HasSubstr("Expected batch to be 1 for BHWC heatmap"));
}

TEST(TensorsToSegmentationUtilsTest, GetHwcFromDimsInvalidShape) {
  std::vector<int> dims_5 = {1, 2, 3, 4, 5};
  absl::StatusOr<std::tuple<int, int, int>> result = GetHwcFromDims(dims_5);
  EXPECT_FALSE(result.ok());
  EXPECT_THAT(result.status().message(),
              HasSubstr("Invalid shape for segmentation tensor"));
}

}  // namespace
}  // namespace mediapipe::tensors_to_segmentation_utils
