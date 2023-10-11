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

#include "mediapipe/tasks/cc/vision/utils/landmarks_utils.h"

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe::tasks::vision::utils {
namespace {

TEST(LandmarkUtilsTest, CalculateIOU) {
  // Do not intersect
  EXPECT_EQ(0, CalculateIOU({0, 0, 1, 1}, {2, 2, 3, 3}));
  // No x intersection
  EXPECT_EQ(0, CalculateIOU({0, 0, 1, 1}, {2, 0, 3, 1}));
  // No y intersection
  EXPECT_EQ(0, CalculateIOU({0, 0, 1, 1}, {0, 2, 1, 3}));
  // Full intersection
  EXPECT_EQ(1, CalculateIOU({0, 0, 2, 2}, {0, 0, 2, 2}));

  // Union is 4 intersection is 1
  EXPECT_EQ(0.25, CalculateIOU({0, 0, 3, 1}, {2, 0, 4, 1}));

  // Same in by y
  EXPECT_EQ(0.25, CalculateIOU({0, 0, 1, 3}, {0, 2, 1, 4}));
}
}  // namespace
}  // namespace mediapipe::tasks::vision::utils
