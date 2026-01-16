// Copyright 2019 The MediaPipe Authors.
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

#include "mediapipe/examples/desktop/autoflip/quality/piecewise_linear_function.h"

#include <stddef.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "absl/log/absl_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace autoflip {

void PiecewiseLinearFunction::AddPoint(double x, double y) {
  if (!points_.empty()) {
    ABSL_CHECK_GE(x, points_.back().x)
        << "Points must be provided in non-decreasing x order.";
  }
  points_.push_back(PiecewiseLinearFunction::Point(x, y));
}

std::vector<PiecewiseLinearFunction::Point>::const_iterator
PiecewiseLinearFunction::GetIntervalIterator(double input) const {
  PiecewiseLinearFunction::Point input_point(input, 0);
  std::vector<PiecewiseLinearFunction::Point>::const_iterator iter =
      std::lower_bound(points_.begin(), points_.end(), input_point,
                       PointCompare());
  return iter;
}

double PiecewiseLinearFunction::Interpolate(
    const PiecewiseLinearFunction::Point& p1,
    const PiecewiseLinearFunction::Point& p2, double input) const {
  ABSL_CHECK_LT(p1.x, input);
  ABSL_CHECK_GE(p2.x, input);

  return p2.y - (p2.x - input) / (p2.x - p1.x) * (p2.y - p1.y);
}

double PiecewiseLinearFunction::Evaluate(double const input) const {
  std::vector<PiecewiseLinearFunction::Point>::const_iterator i =
      GetIntervalIterator(input);
  if (i == points_.begin()) {
    return points_.front().y;
  }
  if (i == points_.end()) {
    return points_.back().y;
  }

  std::vector<PiecewiseLinearFunction::Point>::const_iterator prev = i - 1;
  return Interpolate(*prev, *i, input);
}

}  // namespace autoflip
}  // namespace mediapipe
