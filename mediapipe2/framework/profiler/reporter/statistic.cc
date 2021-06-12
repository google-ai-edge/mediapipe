#include "mediapipe/framework/profiler/reporter/statistic.h"

#include <cmath>

#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_profile.pb.h"

namespace mediapipe {
namespace reporter {

// Pushes a single value into the statistics, updating mean and stddev.
void Statistic::Push(double x) {
  ++counter_;

  if (counter_ == 1) {
    mean_ = x;
    ssd_ = 0.0;
    total_impl_ = x;
  } else {
    // Implementing Welford’s algorithm for computing variance.
    auto old_mean = mean_;
    mean_ = mean_ + (x - mean_) / counter_;
    ssd_ = ssd_ + (x - mean_) * (x - old_mean);
    total_impl_ += x;
  }
}

// Returns the number of data points used to calculator the mean and
// stddev.
int Statistic::data_count() const { return counter_; }

// Returns the mean of the data pushed into this statistic.
double Statistic::mean() const { return (counter_ > 0) ? mean_ : 0.0; }

// Returns the variance of the data pushed into this statistic.
double Statistic::variance() const {
  return ((counter_ > 1) ? ssd_ / (counter_ - 1) : 0.0);
}

// Returns the standard deviation of the data pushed into this statistic.
double Statistic::stddev() const { return std::sqrt(variance()); }

double Statistic::total() const { return total_impl_; }

}  // namespace reporter
}  // namespace mediapipe
