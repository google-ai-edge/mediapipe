// Copyright 2020 The MediaPipe Authors.
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

#include "mediapipe/util/filtering/low_pass_filter.h"

#include "absl/memory/memory.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {

TEST(LowPassFilterTest, LowPassFilterBasicChecks) {
  auto filter = absl::make_unique<LowPassFilter>(1.0f);
  EXPECT_EQ(2.0f, filter->Apply(2.0f));
  EXPECT_EQ(100.0f, filter->Apply(100.0f));

  filter = absl::make_unique<LowPassFilter>(0.0f);
  EXPECT_EQ(2.0f, filter->Apply(2.0f));
  EXPECT_EQ(2.0f, filter->Apply(100.0f));

  filter = absl::make_unique<LowPassFilter>(0.5f);
  EXPECT_EQ(2.0f, filter->Apply(2.0f));
  EXPECT_EQ(51.0f, filter->Apply(100.0f));
}

}  // namespace mediapipe
