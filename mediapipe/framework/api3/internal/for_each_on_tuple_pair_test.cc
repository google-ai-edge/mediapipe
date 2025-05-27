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

#include "mediapipe/framework/api3/internal/for_each_on_tuple_pair.h"

#include <cstdint>
#include <tuple>

#include "mediapipe/framework/port/gtest.h"

namespace mediapipe::api3 {
namespace {

TEST(ForEachOnTuplePairTest, WorksForTwoTuplesSameSize) {
  std::tuple<uint8_t, float> a = {10, 5.5f};
  std::tuple<int, double> b = {-5, -4.5};

  float sum = 0.0;
  ForEachOnTuplePair(a, b,
                     [&sum](auto el_a, auto el_b) { sum += el_a + el_b; });
  EXPECT_FLOAT_EQ(sum, 6.0f);
}

}  // namespace
}  // namespace mediapipe::api3
