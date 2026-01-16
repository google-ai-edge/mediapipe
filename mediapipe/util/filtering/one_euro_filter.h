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

#ifndef MEDIAPIPE_UTIL_FILTERING_ONE_EURO_FILTER_H_
#define MEDIAPIPE_UTIL_FILTERING_ONE_EURO_FILTER_H_

#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "mediapipe/util/filtering/low_pass_filter.h"

namespace mediapipe {

class OneEuroFilter {
 public:
  // @frequence Frequency of incoming values defined in value-per-second.
  // (E.g. landmarks detected from camera frame stream at 30fps => frequency =
  // 30)
  // NOTE: must be > 0
  //
  // @min_cutoff - Minimum cutoff frequency. Start by tuning this parameter
  // while keeping `beta = 0` to reduce jittering to the desired level. 1Hz is a
  // good starting point.
  // NOTE: must be > 0
  //
  // @beta - Cutoff slope. After `min_cutoff` is configured, start increasing
  // `beta` value to reduce the lag introduced by the `min_cutoff`. Find the
  // desired balance between jittering and lag.
  //
  // @derivate_cutoff - Cutoff frequency for derivative. 1Hz is a good starting
  // point, but can be tuned to further smooth the speed (i.e. derivative) on
  // the object.
  // NOTE: must be > 0
  // See https://gery.casiez.net/1euro/ for more details.
  static absl::StatusOr<OneEuroFilter> Create(double frequency,
                                              double min_cutoff, double beta,
                                              double derivate_cutoff);

  static absl::StatusOr<OneEuroFilter> CreateLegacyFilter(
      double frequency, double min_cutoff, double beta, double derivate_cutoff);

  double Apply(absl::Duration timestamp, double value, double value_scale,
               double beta_scale);

  float GetLastX() const;
  float GetLastDx() const;

 private:
  OneEuroFilter(double frequency, double min_cutoff, double beta,
                double derivate_cutoff, int64_t initial_last_time);

  static absl::StatusOr<OneEuroFilter> InternalCreate(
      double frequency, double min_cutoff, double beta, double derivate_cutoff,
      int64_t initial_last_time);

  double GetAlpha(double cutoff);

  double frequency_;
  double min_cutoff_;
  double beta_;
  double derivate_cutoff_;
  std::unique_ptr<LowPassFilter> x_;
  std::unique_ptr<LowPassFilter> dx_;
  int64_t last_time_;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_UTIL_FILTERING_ONE_EURO_FILTER_H_
