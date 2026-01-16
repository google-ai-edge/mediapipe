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

#include <cmath>
#include <cstdint>
#include <memory>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "mediapipe/util/filtering/low_pass_filter.h"

namespace mediapipe {

static const double kEpsilon = 0.000001;
static constexpr int kUninitializedTimestamp = -1;

absl::StatusOr<OneEuroFilter> OneEuroFilter::InternalCreate(
    double frequency, double min_cutoff, double beta, double derivate_cutoff,
    int64_t initial_last_time) {
  if (frequency <= kEpsilon) {
    return absl::InvalidArgumentError(
        absl::StrCat("frequency should be > 0, but equals: ", frequency));
  }

  if (min_cutoff <= kEpsilon) {
    return absl::InvalidArgumentError(
        absl::StrCat("min_cutoff should be > 0, but equals: ", min_cutoff));
  }

  if (derivate_cutoff <= kEpsilon) {
    return absl::InvalidArgumentError(absl::StrCat(
        "derivate_cutoff should be > 0, but equals: ", derivate_cutoff));
  }

  return OneEuroFilter(frequency, min_cutoff, beta, derivate_cutoff,
                       initial_last_time);
}

absl::StatusOr<OneEuroFilter> OneEuroFilter::Create(double frequency,
                                                    double min_cutoff,
                                                    double beta,
                                                    double derivate_cutoff) {
  return OneEuroFilter::InternalCreate(
      frequency, min_cutoff, beta, derivate_cutoff, kUninitializedTimestamp);
}

absl::StatusOr<OneEuroFilter> OneEuroFilter::CreateLegacyFilter(
    double frequency, double min_cutoff, double beta, double derivate_cutoff) {
  return OneEuroFilter::InternalCreate(
      frequency, min_cutoff, beta, derivate_cutoff, /*initial_last_time=*/0);
}

// Input values frequency, min_cutoff, and derivate_cutoff must be non zero.
OneEuroFilter::OneEuroFilter(double frequency, double min_cutoff, double beta,
                             double derivate_cutoff,
                             int64_t initial_last_time) {
  frequency_ = frequency;
  min_cutoff_ = min_cutoff;
  beta_ = beta;
  derivate_cutoff_ = derivate_cutoff;

  x_ = std::make_unique<LowPassFilter>(GetAlpha(min_cutoff));
  dx_ = std::make_unique<LowPassFilter>(GetAlpha(derivate_cutoff));
  last_time_ = initial_last_time;
}

double OneEuroFilter::Apply(absl::Duration timestamp, double value,
                            double value_scale, double beta_scale) {
  int64_t new_timestamp = absl::ToInt64Nanoseconds(timestamp);
  if (last_time_ >= new_timestamp) {
    // Results are unpredictable in this case, so nothing to do but
    // return same value
    ABSL_LOG(WARNING) << "New timestamp is equal or less than the last one.";
    return value;
  }

  // update the sampling frequency based on timestamps
  if (last_time_ != 0 && new_timestamp != 0) {
    static constexpr double kNanoSecondsToSecond = 1e-9;
    frequency_ = 1.0 / ((new_timestamp - last_time_) * kNanoSecondsToSecond);
  }
  last_time_ = new_timestamp;

  // estimate the current variation per second
  double dvalue = x_->HasLastRawValue()
                      ? (value - x_->LastRawValue()) * value_scale * frequency_
                      : 0.0;  // FIXME: 0.0 or value?
  double edvalue = dx_->ApplyWithAlpha(dvalue, GetAlpha(derivate_cutoff_));
  // use it to update the cutoff frequency
  double scaled_beta = beta_scale * beta_;
  double cutoff = min_cutoff_ + scaled_beta * std::fabs(edvalue);

  // filter the given value
  return x_->ApplyWithAlpha(value, GetAlpha(cutoff));
}

double OneEuroFilter::GetAlpha(double cutoff) {
  double te = 1.0 / frequency_;
  double tau = 1.0 / (2 * M_PI * cutoff);
  return 1.0 / (1.0 + tau / te);
}

float OneEuroFilter::GetLastX() const { return x_->LastValue(); }

float OneEuroFilter::GetLastDx() const { return dx_->LastValue(); }

}  // namespace mediapipe
