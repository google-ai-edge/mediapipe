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

#include "mediapipe/util/filtering/one_euro_filter.h"

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

TEST(OneEuroFilterLegacyTest, OneEuroFilterInvalidValueChecks) {
  EXPECT_THAT(OneEuroFilter::CreateLegacyFilter(1.0f, 0.0f, 1.0f, 0.0f),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(OneEuroFilterTest, OneEuroFilterInvalidValueChecks) {
  EXPECT_THAT(OneEuroFilter::Create(1.0f, 0.0f, 1.0f, 0.0f),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(OneEuroFilterLegacyTest, OneEuroFilterValidValueChecks) {
  MP_ASSERT_OK_AND_ASSIGN(auto filter, OneEuroFilter::CreateLegacyFilter(
                                           1.0f, 0.001f, 1.0f, 0.001f));
  EXPECT_EQ(filter.Apply(absl::Microseconds(0.0f), /*value=*/2.0f,
                         /*value_scale=*/1.0,
                         /*beta_scale=*/1.0),
            2.0f);
  EXPECT_EQ(filter.Apply(absl::Microseconds(1.0f), /*value=*/100.0f,
                         /*value_scale=*/1.0,
                         /*beta_scale=*/1.0),
            100.0f);
}

TEST(OneEuroFilterTest, OneEuroFilterValidValueChecks) {
  MP_ASSERT_OK_AND_ASSIGN(auto filter,
                          OneEuroFilter::Create(1.0f, 0.001f, 1.0f, 0.001f));
  EXPECT_EQ(filter.Apply(absl::Microseconds(0.0f), /*value=*/2.0f,
                         /*value_scale=*/1.0,
                         /*beta_scale=*/1.0),
            2.0f);
  EXPECT_NEAR(filter.Apply(absl::Microseconds(1.0f), /*value=*/100.0f,
                           /*value_scale=*/1.0,
                           /*beta_scale=*/1.0),
              79.8f, 0.2f);
}

TEST(OneEuroFilterLegacyTest, OneEuroFilterValidValueFilter) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto filter, OneEuroFilter::CreateLegacyFilter(1.0f, 0.1f, 0.0f, 0.1f));
  EXPECT_FLOAT_EQ(filter.Apply(absl::Microseconds(1000000.0f), 2.0f,
                               /*value_scale=*/1.0,
                               /*beta_scale=*/1.0),
                  2.0f);
  EXPECT_NEAR(filter.Apply(absl::Microseconds(2000000.0f), 3.0f,
                           /*value_scale=*/1.0,
                           /*beta_scale=*/1.0),
              2.4f, 0.1f);
  EXPECT_NEAR(filter.Apply(absl::Microseconds(3000000.0f), 4.0f,
                           /*value_scale=*/1.0,
                           /*beta_scale=*/1.0),
              3.0f, 0.1f);
}

TEST(OneEuroFilterTest, OneEuroFilterValidValueFilter) {
  MP_ASSERT_OK_AND_ASSIGN(auto filter,
                          OneEuroFilter::Create(1.0f, 0.1f, 0.0f, 0.1f));
  EXPECT_FLOAT_EQ(filter.Apply(absl::Microseconds(1000000.0f), 2.0f,
                               /*value_scale=*/1.0,
                               /*beta_scale=*/1.0),
                  2.0f);
  EXPECT_NEAR(filter.Apply(absl::Microseconds(2000000.0f), 3.0f,
                           /*value_scale=*/1.0,
                           /*beta_scale=*/1.0),
              2.4f, 0.1f);
  EXPECT_NEAR(filter.Apply(absl::Microseconds(3000000.0f), 4.0f,
                           /*value_scale=*/1.0,
                           /*beta_scale=*/1.0),
              3.0f, 0.1f);
}

TEST(OneEuroFilterTest,
     UseFilteredDerivativeCalculatesDerivativeFromLastFilteredValue) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto legacy_filter,
      OneEuroFilter::Create(/*frequency=*/30.0, /*min_cutoff=*/0.01,
                            /*beta=*/0.001, /*derivate_cutoff=*/0.1,
                            /*use_filtered_derivative=*/false));
  MP_ASSERT_OK_AND_ASSIGN(
      auto corrected_filter,
      OneEuroFilter::Create(/*frequency=*/30.0, /*min_cutoff=*/0.01,
                            /*beta=*/0.001, /*derivate_cutoff=*/0.1,
                            /*use_filtered_derivative=*/true));

  // First sample
  EXPECT_FLOAT_EQ(legacy_filter.Apply(absl::Milliseconds(100), 10.0, 1.0, 1.0),
                  10.0f);
  EXPECT_FLOAT_EQ(
      corrected_filter.Apply(absl::Milliseconds(100), 10.0, 1.0, 1.0), 10.0f);

  // Second sample
  double legacy_val2 =
      legacy_filter.Apply(absl::Milliseconds(133), 20.0, 1.0, 1.0);
  double corrected_val2 =
      corrected_filter.Apply(absl::Milliseconds(133), 20.0, 1.0, 1.0);
  EXPECT_NEAR(legacy_val2, corrected_val2, 1e-4);

  // Third sample
  double legacy_val3 =
      legacy_filter.Apply(absl::Milliseconds(166), 30.0, 1.0, 1.0);
  double corrected_val3 =
      corrected_filter.Apply(absl::Milliseconds(166), 30.0, 1.0, 1.0);

  EXPECT_GT(corrected_filter.GetLastDx(), legacy_filter.GetLastDx());
  EXPECT_GT(corrected_val3, legacy_val3);
}

}  // namespace
}  // namespace mediapipe
