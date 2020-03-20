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

#ifndef MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_QUALITY_POLYNOMIAL_REGRESSION_PATH_SOLVER_H_
#define MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_QUALITY_POLYNOMIAL_REGRESSION_PATH_SOLVER_H_

#include "ceres/problem.h"
#include "mediapipe/examples/desktop/autoflip/quality/focus_point.pb.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace autoflip {

class PolynomialRegressionPathSolver {
 public:
  PolynomialRegressionPathSolver()
      : xa_(0.0),
        xb_(0.0),
        xc_(0.0),
        xd_(0.0),
        xk_(0.0),
        ya_(0.0),
        yb_(0.0),
        yc_(0.0),
        yd_(0.0),
        yk_(0.0) {}

  // Given a series of focus points on frames, uses polynomial regression to
  // compute a best guess of a 1D camera movement trajectory along x-axis and
  // y-axis, such that focus points can be preserved as much as possible. The
  // returned |all_transforms| hold the camera location at each timestamp
  // corresponding to each input frame.
  ::mediapipe::Status ComputeCameraPath(
      const std::vector<FocusPointFrame>& focus_point_frames,
      const std::vector<FocusPointFrame>& prior_focus_point_frames,
      const int original_width, const int original_height,
      const int output_width, const int output_height,
      std::vector<cv::Mat>* all_transforms);

 private:
  // Adds a new cost function, constructed using |in| and |out|, into |problem|.
  void AddCostFunctionToProblem(const double in, const double out,
                                ceres::Problem* problem, double* a, double* b,
                                double* c, double* d, double* k);

  // The current implementation fixes the polynomial order at 4, i.e. the
  // equation to estimate is: out = a * in + b * in^2 + c * in^3 + d * in^4 + k.
  // The two sets of parameters below are for estimating trajectories along
  // x-axis and y-axis, respectively.
  double xa_, xb_, xc_, xd_, xk_;
  double ya_, yb_, yc_, yd_, yk_;
};

}  // namespace autoflip
}  // namespace mediapipe

#endif  // MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_QUALITY_POLYNOMIAL_REGRESSION_PATH_SOLVER_H_
