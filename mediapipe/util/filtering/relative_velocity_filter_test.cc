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

#include "mediapipe/util/filtering/relative_velocity_filter.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/memory/memory.h"
#include "absl/time/time.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/logging.h"

namespace mediapipe {

using DistanceEstimationMode =
    mediapipe::RelativeVelocityFilter::DistanceEstimationMode;

absl::Duration DurationFromNanos(int64_t nanos) {
  return absl::FromChrono(std::chrono::nanoseconds{nanos});
}

absl::Duration DurationFromMillis(int64_t millis) {
  return absl::FromChrono(std::chrono::milliseconds{millis});
}

TEST(RelativeVelocityFilterTest, ApplyIncorrectTimestamp) {
  auto filter = absl::make_unique<RelativeVelocityFilter>(1, 1.0);

  absl::Duration timestamp1 = DurationFromNanos(1);

  EXPECT_FLOAT_EQ(95.5f, filter->Apply(timestamp1, 0.5f, 95.5f));
  EXPECT_FLOAT_EQ(200.5f, filter->Apply(timestamp1, 0.5f, 200.5f));
  EXPECT_FLOAT_EQ(1000.5f, filter->Apply(timestamp1, 0.5f, 1000.5f));

  EXPECT_FLOAT_EQ(2000.0f, filter->Apply(DurationFromNanos(1), 0.5f, 2000.0f));
}

void TestSameValueScaleDifferentVelocityScales(
    DistanceEstimationMode distance_mode) {
  // Changing the distance estimation mode has no effect with constant scales.

  // More sensitive filter.
  auto filter1 = absl::make_unique<RelativeVelocityFilter>(
      /*window_size=*/5, /*velocity_scale=*/45.0f,
      /*distance_mode=*/distance_mode);
  // Less sensitive filter.
  auto filter2 = absl::make_unique<RelativeVelocityFilter>(
      /*window_size=*/5, /*velocity_scale=*/0.1f,
      /*distance_mode=*/distance_mode);

  float result1;
  float result2;
  float value;
  float value_scale = 1.0f;

  value = 1.0f;
  result1 = filter1->Apply(DurationFromMillis(1), value_scale, value);
  result2 = filter2->Apply(DurationFromMillis(1), value_scale, value);
  EXPECT_EQ(result1, result2);

  value = 10.0f;
  result1 = filter1->Apply(DurationFromMillis(2), value_scale, value);
  result2 = filter2->Apply(DurationFromMillis(2), value_scale, value);
  EXPECT_GT(result1, result2);

  value = 2.0f;
  result1 = filter1->Apply(DurationFromMillis(3), value_scale, value);
  result2 = filter2->Apply(DurationFromMillis(3), value_scale, value);
  EXPECT_LT(result1, result2);

  value = 20.0f;
  result1 = filter1->Apply(DurationFromMillis(4), value_scale, value);
  result2 = filter2->Apply(DurationFromMillis(4), value_scale, value);
  EXPECT_GT(result1, result2);

  value = 10.0f;
  result1 = filter1->Apply(DurationFromMillis(5), value_scale, value);
  result2 = filter2->Apply(DurationFromMillis(5), value_scale, value);
  EXPECT_LT(result1, result2);

  value = 50.0f;
  result1 = filter1->Apply(DurationFromMillis(6), value_scale, value);
  result2 = filter2->Apply(DurationFromMillis(6), value_scale, value);
  EXPECT_GT(result1, result2);

  value = 30.0f;
  result1 = filter1->Apply(DurationFromMillis(7), value_scale, value);
  result2 = filter2->Apply(DurationFromMillis(7), value_scale, value);
  EXPECT_LT(result1, result2);
}

TEST(RelativeVelocityFilterTest, SameValueScaleDifferentVelocityScalesLegacy) {
  TestSameValueScaleDifferentVelocityScales(
      DistanceEstimationMode::kLegacyTransition);
}

TEST(RelativeVelocityFilterTest,
     SameValueScaleDifferentVelocityScalesForceCurrentScale) {
  TestSameValueScaleDifferentVelocityScales(
      DistanceEstimationMode::kForceCurrentScale);
}

void TestDifferentConstantValueScalesSameVelocityScale(
    DistanceEstimationMode distance_mode) {
  const float same_velocity_scale = 1.0f;
  auto filter1 = absl::make_unique<RelativeVelocityFilter>(
      /*window_size=*/3, /*velocity_scale=*/same_velocity_scale,
      /*distance_mode=*/distance_mode);
  auto filter2 = absl::make_unique<RelativeVelocityFilter>(
      /*window_size=*/3, /*velocity_scale=*/same_velocity_scale,
      /*distance_mode=*/distance_mode);

  float result1;
  float result2;
  float value;
  // smaller value scale will decrease cumulative speed and alpha
  // so with smaller scale and same other params filter will believe
  // new values a little bit less
  float value_scale1 = 0.5f;
  float value_scale2 = 1.0f;

  value = 1.0f;
  result1 = filter1->Apply(DurationFromMillis(1), value_scale1, value);
  result2 = filter2->Apply(DurationFromMillis(1), value_scale2, value);
  EXPECT_EQ(result1, result2);

  value = 10.0f;
  result1 = filter1->Apply(DurationFromMillis(2), value_scale1, value);
  result2 = filter2->Apply(DurationFromMillis(2), value_scale2, value);
  EXPECT_LT(result1, result2);

  value = 2.0f;
  result1 = filter1->Apply(DurationFromMillis(3), value_scale1, value);
  result2 = filter2->Apply(DurationFromMillis(3), value_scale2, value);
  EXPECT_GT(result1, result2);

  value = 20.0f;
  result1 = filter1->Apply(DurationFromMillis(4), value_scale1, value);
  result2 = filter2->Apply(DurationFromMillis(4), value_scale2, value);
  EXPECT_LT(result1, result2);
}

TEST(RelativeVelocityFilterTest,
     DifferentConstantValueScalesSameVelocityScale) {
  TestDifferentConstantValueScalesSameVelocityScale(
      DistanceEstimationMode::kLegacyTransition);
}

TEST(RelativeVelocityFilterTest, ApplyCheckValueScales) {
  TestDifferentConstantValueScalesSameVelocityScale(
      DistanceEstimationMode::kForceCurrentScale);
}

void TestTranslationInvariance(DistanceEstimationMode distance_mode) {
  struct ValueAtScale {
    float value;
    float scale;
  };

  // Note that the scales change over time.
  std::vector<ValueAtScale> original_data_points{
      // clang-format off
      {.value =  1.0f, .scale =  0.5f},
      {.value = 10.0f, .scale =  5.0f},
      {.value = 20.0f, .scale = 10.0f},
      {.value = 30.0f, .scale = 15.0f},
      {.value = 40.0f, .scale =  0.5f},
      {.value = 50.0f, .scale =  0.5f},
      {.value = 60.0f, .scale =  5.0f},
      {.value = 70.0f, .scale = 10.0f},
      {.value = 80.0f, .scale = 15.0f},
      {.value = 90.0f, .scale =  5.0f},
      {.value = 70.0f, .scale = 10.0f},
      {.value = 50.0f, .scale = 15.0f},
      {.value = 80.0f, .scale = 15.0f},
      // clang-format on
  };

  // The amount by which the input values are uniformly translated.
  const float kValueOffset = 100.0f;

  // The uniform time delta.
  const absl::Duration time_delta = DurationFromMillis(1);

  // The filter parameters are the same between the two filters.
  const size_t kWindowSize = 5;
  const float kVelocityScale = 0.1f;

  // Perform the translation.
  std::vector<ValueAtScale> translated_data_points = original_data_points;
  for (auto& point : translated_data_points) {
    point.value += kValueOffset;
  }

  auto original_points_filter = absl::make_unique<RelativeVelocityFilter>(
      /*window_size=*/kWindowSize, /*velocity_scale=*/kVelocityScale,
      /*distance_mode=*/distance_mode);
  auto translated_points_filter = absl::make_unique<RelativeVelocityFilter>(
      /*window_size=*/kWindowSize, /*velocity_scale=*/kVelocityScale,
      /*distance_mode=*/distance_mode);

  // The minimal difference which is considered a divergence.
  const float kDivergenceGap = 0.001f;
  // The amount of the times this gap is achieved with `kLegacyTransition`.
  // Note that on the first iteration the filters should output the unfiltered
  // input values, so no divergence should occur.
  // This amount obviously depends on the values in `original_data_points`,
  // so should be changed accordingly when they are updated.
  const size_t kDivergenceTimes = 5;

  // The minimal difference which is considered a large divergence.
  const float kLargeDivergenceGap = 10.0f;
  // The amount of times it is achieved.
  // This amount obviously depends on the values in `original_data_points`,
  // so should be changed accordingly when they are updated.
  const size_t kLargeDivergenceTimes = 1;

  // In contrast, the new mode delivers this error bound across all the samples.
  const float kForceCurrentScaleAbsoluteError = 1.53e-05f;

  size_t times_diverged = 0;
  size_t times_largely_diverged = 0;
  absl::Duration timestamp;
  for (size_t iteration = 0; iteration < original_data_points.size();
       ++iteration, timestamp += time_delta) {
    const ValueAtScale& original_data_point = original_data_points[iteration];
    const float filtered_original_value =
        original_points_filter->Apply(/*timestamp=*/timestamp,
                                      /*value_scale=*/original_data_point.scale,
                                      /*value=*/original_data_point.value);

    const ValueAtScale& translated_data_point =
        translated_data_points[iteration];
    const float actual_filtered_translated_value =
        translated_points_filter->Apply(
            /*timestamp=*/timestamp,
            /*value_scale=*/translated_data_point.scale,
            /*value=*/translated_data_point.value);

    const float expected_filtered_translated_value =
        filtered_original_value + kValueOffset;

    const float difference = std::fabs(actual_filtered_translated_value -
                                       expected_filtered_translated_value);
    if (iteration == 0) {
      // On the first iteration, the unfiltered values are returned.
      EXPECT_EQ(filtered_original_value, original_data_point.value);
      EXPECT_EQ(actual_filtered_translated_value, translated_data_point.value);
      EXPECT_EQ(difference, 0.0f);
    } else if (distance_mode == DistanceEstimationMode::kLegacyTransition) {
      if (difference >= kDivergenceGap) {
        ++times_diverged;
      }
      if (difference >= kLargeDivergenceGap) {
        ++times_largely_diverged;
      }
    } else {
      ABSL_CHECK(distance_mode == DistanceEstimationMode::kForceCurrentScale);
      EXPECT_NEAR(difference, 0.0f, kForceCurrentScaleAbsoluteError);
    }
  }

  if (distance_mode == DistanceEstimationMode::kLegacyTransition) {
    EXPECT_GE(times_diverged, kDivergenceTimes);
    EXPECT_GE(times_largely_diverged, kLargeDivergenceTimes);
  }
}

// This test showcases an undesired property of the current filter design
// that manifests itself when value scales change in time. It turns out that
// the velocity estimation starts depending on the distance from the origin.
TEST(RelativeVelocityFilterTest,
     TestLegacyFilterModeIsNotTranslationInvariant) {
  TestTranslationInvariance(DistanceEstimationMode::kLegacyTransition);
}

TEST(RelativeVelocityFilterTest, TestOtherFilterModeIsTranslationInvariant) {
  TestTranslationInvariance(DistanceEstimationMode::kForceCurrentScale);
}

}  // namespace mediapipe
