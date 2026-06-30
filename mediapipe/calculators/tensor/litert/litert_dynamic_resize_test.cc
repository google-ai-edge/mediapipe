// Copyright 2025 The MediaPipe Authors.
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

#include <vector>

#include "mediapipe/calculators/tensor/litert/litert_utils.h"
#include "testing/base/public/gunit.h"

namespace mediapipe {
namespace {

TEST(LiteRtDynamicResizeTest, IsShapeCompatibleWithDynamicDims) {
  // Test exact match
  EXPECT_TRUE(
      IsShapeCompatibleWithDynamicDims({1, 224, 224, 3}, {1, 224, 224, 3}));

  // Test with dynamic batch dimension
  EXPECT_TRUE(
      IsShapeCompatibleWithDynamicDims({-1, 224, 224, 3}, {5, 224, 224, 3}));

  // Test with multiple dynamic dimensions
  EXPECT_TRUE(
      IsShapeCompatibleWithDynamicDims({1, -1, -1, 3}, {1, 300, 400, 3}));

  // Test with all dynamic dimensions
  EXPECT_TRUE(
      IsShapeCompatibleWithDynamicDims({-1, -1, -1, -1}, {2, 512, 512, 4}));

  // Test incompatible shapes - different rank
  EXPECT_FALSE(
      IsShapeCompatibleWithDynamicDims({1, 224, 224}, {1, 224, 224, 3}));

  // Test incompatible shapes - static dimension mismatch
  EXPECT_FALSE(
      IsShapeCompatibleWithDynamicDims({1, 224, 224, 3}, {1, 300, 300, 3}));

  // Test incompatible shapes - last dimension mismatch
  EXPECT_FALSE(
      IsShapeCompatibleWithDynamicDims({-1, -1, -1, 3}, {2, 512, 512, 4}));
}

}  // namespace
}  // namespace mediapipe
