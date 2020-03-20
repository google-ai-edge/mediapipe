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

#include "mediapipe/examples/desktop/autoflip/quality/polynomial_regression_path_solver.h"

#include "ceres/autodiff_cost_function.h"
#include "ceres/cost_function.h"
#include "ceres/loss_function.h"
#include "ceres/solver.h"
#include "mediapipe/examples/desktop/autoflip/quality/focus_point.pb.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace autoflip {

using ceres::AutoDiffCostFunction;
using ceres::CauchyLoss;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

namespace {

// A residual operator that computes the error using polynomial fitting.
struct PolynomialResidual {
  PolynomialResidual(double in, double out) : in_(in), out_(out) {}

  template <typename T>
  bool operator()(const T* const a, const T* const b, const T* const c,
                  const T* const d, const T* const k, T* residual) const {
    residual[0] = out_ - a[0] * in_ - b[0] * in_ * in_ -
                  c[0] * in_ * in_ * in_ - d[0] * in_ * in_ * in_ * in_ - k[0];
    return true;
  }

 private:
  const double in_;
  const double out_;
};

// Computes the amount of delta position change along the fitted polynomial
// curve, translates the delta from being relative to the origin of the original
// dimension to being relative to the center of the original dimension, then
// regulates the delta to avoid moving camera off the frame boundaries.
float ComputeDelta(const float in, const int original_dimension,
                   const int output_dimension, const double a, const double b,
                   const double c, const double d, const double k) {
  // The value `out` here represents a normalized distance between the center of
  // the output window and the origin of the original window.
  float out =
      a * in + b * in * in + c * in * in * in + d * in * in * in * in + k;
  // Translate `out` to a pixel distance between the center of the output window
  // and the center of the original window. This value can be negative, 0, or
  // positive.
  float delta = (out - 0.5) * original_dimension;

  // Make sure delta doesn't move the camera off the frame boundary.
  const float max_delta = (original_dimension - output_dimension) / 2.0f;
  if (delta > max_delta) {
    delta = max_delta;
  } else if (delta < -max_delta) {
    delta = -max_delta;
  }
  return delta;
}
}  // namespace

void PolynomialRegressionPathSolver::AddCostFunctionToProblem(
    const double in, const double out, Problem* problem, double* a, double* b,
    double* c, double* d, double* k) {
  // Creating a cost function, with 1D residual and 5 1D parameter blocks. This
  // is what the "1, 1, 1, 1, 1, 1" std::string below means.
  CostFunction* cost_function =
      new AutoDiffCostFunction<PolynomialResidual, 1, 1, 1, 1, 1, 1>(
          new PolynomialResidual(in, out));
  VLOG(1) << "------- adding " << in << ": " << out;
  problem->AddResidualBlock(cost_function, new CauchyLoss(0.5), a, b, c, d, k);
}

::mediapipe::Status PolynomialRegressionPathSolver::ComputeCameraPath(
    const std::vector<FocusPointFrame>& focus_point_frames,
    const std::vector<FocusPointFrame>& prior_focus_point_frames,
    const int original_width, const int original_height, const int output_width,
    const int output_height, std::vector<cv::Mat>* all_transforms) {
  RET_CHECK_GE(original_width, output_width);
  RET_CHECK_GE(original_height, output_height);
  const bool should_solve_x_problem = original_width != output_width;
  const bool should_solve_y_problem = original_height != output_height;
  RET_CHECK_GT(focus_point_frames.size() + prior_focus_point_frames.size(), 0);
  Problem problem_x, problem_y;
  for (int i = 0; i < prior_focus_point_frames.size(); ++i) {
    const auto& spf = prior_focus_point_frames[i];
    for (const auto& sp : spf.point()) {
      const double center_x = sp.norm_point_x();
      const double center_y = sp.norm_point_y();
      const auto t = i;
      if (should_solve_x_problem) {
        AddCostFunctionToProblem(t, center_x, &problem_x, &xa_, &xb_, &xc_,
                                 &xd_, &xk_);
      }
      if (should_solve_y_problem) {
        AddCostFunctionToProblem(t, center_y, &problem_y, &ya_, &yb_, &yc_,
                                 &yd_, &yk_);
      }
    }
  }
  for (int i = 0; i < focus_point_frames.size(); ++i) {
    const auto& spf = focus_point_frames[i];
    for (const auto& sp : spf.point()) {
      const double center_x = sp.norm_point_x();
      const double center_y = sp.norm_point_y();
      const auto t = i + prior_focus_point_frames.size();
      if (should_solve_x_problem) {
        AddCostFunctionToProblem(t, center_x, &problem_x, &xa_, &xb_, &xc_,
                                 &xd_, &xk_);
      }
      if (should_solve_y_problem) {
        AddCostFunctionToProblem(t, center_y, &problem_y, &ya_, &yb_, &yc_,
                                 &yd_, &yk_);
      }
    }
  }

  Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;

  Solver::Summary summary_x, summary_y;
  Solve(options, &problem_x, &summary_x);
  Solve(options, &problem_y, &summary_y);
  all_transforms->clear();
  for (int i = 0;
       i < focus_point_frames.size() + prior_focus_point_frames.size(); i++) {
    // Code below assigns values into an affine model, defined as:
    //  [1 0 dx]
    //  [0 1 dy]
    // When the camera moves along x axis, we assign delta to dx; otherwise we
    // assign delta to dy.
    cv::Mat transform = cv::Mat::eye(2, 3, CV_32FC1);
    const float in = static_cast<float>(i);
    if (should_solve_x_problem) {
      const float delta = ComputeDelta(in, original_width, output_width, xa_,
                                       xb_, xc_, xd_, xk_);
      transform.at<float>(0, 2) = delta;
    }
    if (should_solve_y_problem) {
      const float delta = ComputeDelta(in, original_height, output_height, ya_,
                                       yb_, yc_, yd_, yk_);
      transform.at<float>(1, 2) = delta;
    }
    all_transforms->push_back(transform);
  }
  return mediapipe::OkStatus();
}

}  // namespace autoflip
}  // namespace mediapipe
