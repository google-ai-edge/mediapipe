#include "mediapipe/util/filtering/one_euro_filter.h"

#include <cmath>
#include <cstdint>
#include <limits>

#include "absl/log/absl_log.h"
#include "absl/memory/memory.h"
#include "mediapipe/util/filtering/low_pass_filter.h"

namespace mediapipe {

static const double kEpsilon = 0.000001;

OneEuroFilter::OneEuroFilter(double frequency, double min_cutoff, double beta,
                             double derivate_cutoff) {
  SetFrequency(frequency);
  SetMinCutoff(min_cutoff);
  SetBeta(beta);
  SetDerivateCutoff(derivate_cutoff);
  x_ = absl::make_unique<LowPassFilter>(GetAlpha(min_cutoff));
  dx_ = absl::make_unique<LowPassFilter>(GetAlpha(derivate_cutoff));
  last_time_ = std::numeric_limits<int64_t>::min();
}

double OneEuroFilter::Apply(absl::Duration timestamp, double value_scale,
                            double value) {
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
  double cutoff = min_cutoff_ + beta_ * std::fabs(edvalue);

  // filter the given value
  return x_->ApplyWithAlpha(value, GetAlpha(cutoff));
}

double OneEuroFilter::GetAlpha(double cutoff) {
  double te = 1.0 / frequency_;
  double tau = 1.0 / (2 * M_PI * cutoff);
  return 1.0 / (1.0 + tau / te);
}

void OneEuroFilter::SetFrequency(double frequency) {
  if (frequency <= kEpsilon) {
    ABSL_LOG(ERROR) << "frequency should be > 0";
    return;
  }
  frequency_ = frequency;
}

void OneEuroFilter::SetMinCutoff(double min_cutoff) {
  if (min_cutoff <= kEpsilon) {
    ABSL_LOG(ERROR) << "min_cutoff should be > 0";
    return;
  }
  min_cutoff_ = min_cutoff;
}

void OneEuroFilter::SetBeta(double beta) { beta_ = beta; }

void OneEuroFilter::SetDerivateCutoff(double derivate_cutoff) {
  if (derivate_cutoff <= kEpsilon) {
    ABSL_LOG(ERROR) << "derivate_cutoff should be > 0";
    return;
  }
  derivate_cutoff_ = derivate_cutoff;
}

}  // namespace mediapipe
