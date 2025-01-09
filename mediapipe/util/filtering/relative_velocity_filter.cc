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

#include <cmath>
#include <deque>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/memory/memory.h"

namespace mediapipe {

float RelativeVelocityFilter::Apply(absl::Duration timestamp, float value_scale,
                                    float value) {
  const int64_t new_timestamp = absl::ToInt64Nanoseconds(timestamp);
  if (last_timestamp_ >= new_timestamp) {
    // Results are unpredictable in this case, so nothing to do but
    // return same value
    ABSL_LOG(WARNING) << "New timestamp is equal or less than the last one.";
    return value;
  }

  float alpha;
  if (last_timestamp_ == -1) {
    alpha = 1.0;
  } else {
    ABSL_DCHECK(distance_mode_ == DistanceEstimationMode::kLegacyTransition ||
                distance_mode_ == DistanceEstimationMode::kForceCurrentScale);
    const float distance =
        distance_mode_ == DistanceEstimationMode::kLegacyTransition
            ? value * value_scale -
                  last_value_ * last_value_scale_   // Original.
            : value_scale * (value - last_value_);  // Translation invariant.

    const int64_t duration = new_timestamp - last_timestamp_;

    float cumulative_distance = distance;
    int64_t cumulative_duration = duration;

    // Define max cumulative duration assuming
    // 30 frames per second is a good frame rate, so assuming 30 values
    // per second or 1 / 30 of a second is a good duration per window element
    constexpr int64_t kAssumedMaxDuration = 1000000000 / 30;
    const int64_t max_cumulative_duration =
        (1 + window_.size()) * kAssumedMaxDuration;
    for (const auto& el : window_) {
      if (cumulative_duration + el.duration > max_cumulative_duration) {
        // This helps in cases when durations are large and outdated
        // window elements have bad impact on filtering results
        break;
      }
      cumulative_distance += el.distance;
      cumulative_duration += el.duration;
    }

    constexpr double kNanoSecondsToSecond = 1e-9;
    const float velocity =
        cumulative_distance / (cumulative_duration * kNanoSecondsToSecond);
    alpha = 1.0f - 1.0f / (1.0f + velocity_scale_ * std::abs(velocity));
    window_.push_front({distance, duration});
    if (window_.size() > max_window_size_) {
      window_.pop_back();
    }
  }

  last_value_ = value;
  last_value_scale_ = value_scale;
  last_timestamp_ = new_timestamp;

  return low_pass_filter_.ApplyWithAlpha(value, alpha);
}

}  // namespace mediapipe
