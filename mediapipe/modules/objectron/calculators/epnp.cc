// Copyright 2021 The MediaPipe Authors.
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

#include "mediapipe/modules/objectron/calculators/epnp.h"

namespace mediapipe {

namespace {

// NUmber of keypoints.
constexpr int kNumKeypoints = 9;

using Eigen::Map;
using Eigen::Matrix;
using Eigen::Matrix4f;
using Eigen::Vector2f;
using Eigen::Vector3f;

}  // namespace

absl::Status SolveEpnp(const float focal_x, const float focal_y,
                       const float center_x, const float center_y,
                       const bool portrait,
                       const std::vector<Vector2f>& input_points_2d,
                       std::vector<Vector3f>* output_points_3d) {
  if (input_points_2d.size() != kNumKeypoints) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Input must has %d 2D points.", kNumKeypoints));
  }

  if (output_points_3d == nullptr) {
    return absl::InvalidArgumentError(
        "Output pointer output_points_3d is Null.");
  }

  Matrix<float, (kNumKeypoints - 1) * 2, 12> m =
      Matrix<float, (kNumKeypoints - 1) * 2, 12>::Zero();

  Matrix<float, kNumKeypoints - 1, 4> epnp_alpha;
  // The epnp_alpha is the Nx4 weight matrix from the EPnP paper, which is used
  // to express the N box vertices as the weighted sum of 4 control points. The
  // value of epnp_alpha is depedent on the set of control points been used.
  // In our case we used the 4 control points as below (coordinates are in world
  // coordinate system):
  //     c0 = (0.0, 0.0, 0.0)  // Box center
  //     c1 = (1.0, 0.0, 0.0)  // Right face center
  //     c2 = (0.0, 1.0, 0.0)  // Top face center
  //     c3 = (0.0, 0.0, 1.0)  // Front face center
  //
  //       3 + + + + + + + + 7
  //       +\                +\          UP
  //       + \               + \
  //       +  \              +  \        |
  //       +   4 + + + + + + + + 8       | y
  //       +   +             +   +       |
  //       +   +             +   +       |
  //       +   +     (0)     +   +       .------- x
  //       +   +             +   +        \
  //       1 + + + + + + + + 5   +         \
  //        \  +              \  +          \ z
  //         \ +               \ +           \
  //          \+                \+
  //           2 + + + + + + + + 6
  //
  // For each box vertex shown above, we have the below weighted sum expression:
  //   v1 = c0 - (c1 - c0) - (c2 - c0) - (c3 - c0) = 4*c0 - c1 - c2 - c3;
  //   v2 = c0 - (c1 - c0) - (c2 - c0) + (c3 - c0) = 2*c0 - c1 - c2 + c3;
  //   v3 = c0 - (c1 - c0) + (c2 - c0) - (c3 - c0) = 2*c0 - c1 + c2 - c3;
  //   ...
  // Thus we can determine the value of epnp_alpha as been used below.
  //
  // clang-format off
  epnp_alpha << 4.0f, -1.0f, -1.0f, -1.0f,
                2.0f, -1.0f, -1.0f,  1.0f,
                2.0f, -1.0f,  1.0f, -1.0f,
                0.0f, -1.0f,  1.0f,  1.0f,
                2.0f,  1.0f, -1.0f, -1.0f,
                0.0f,  1.0f, -1.0f,  1.0f,
                0.0f,  1.0f,  1.0f, -1.0f,
               -2.0f,  1.0f,  1.0f,  1.0f;
  // clang-format on

  for (int i = 0; i < input_points_2d.size() - 1; ++i) {
    // Skip 0th landmark which is object center.
    const auto& point_2d = input_points_2d[i + 1];

    // Convert 2d point from `pixel coordinates` to `NDC coordinates`([-1, 1])
    // following to the definitions in:
    // https://google.github.io/mediapipe/solutions/objectron#ndc-space
    // If portrait mode is been used, it's the caller's responsibility to
    // convert the input 2d points' coordinates.
    float x_ndc, y_ndc;
    if (portrait) {
      x_ndc = point_2d.y() * 2 - 1;
      y_ndc = point_2d.x() * 2 - 1;
    } else {
      x_ndc = point_2d.x() * 2 - 1;
      y_ndc = 1 - point_2d.y() * 2;
    }

    for (int j = 0; j < 4; ++j) {
      // For each of the 4 control points, formulate two rows of the
      // m matrix (two equations).
      const float control_alpha = epnp_alpha(i, j);
      m(i * 2, j * 3) = focal_x * control_alpha;
      m(i * 2, j * 3 + 2) = (center_x + x_ndc) * control_alpha;
      m(i * 2 + 1, j * 3 + 1) = focal_y * control_alpha;
      m(i * 2 + 1, j * 3 + 2) = (center_y + y_ndc) * control_alpha;
    }
  }
  // This is a self adjoint matrix. Use SelfAdjointEigenSolver for a fast
  // and stable solution.
  Matrix<float, 12, 12> mt_m = m.transpose() * m;
  Eigen::SelfAdjointEigenSolver<Matrix<float, 12, 12>> eigen_solver(mt_m);
  if (eigen_solver.info() != Eigen::Success) {
    return absl::AbortedError("Eigen decomposition failed.");
  }
  CHECK_EQ(12, eigen_solver.eigenvalues().size());

  // Eigenvalues are sorted in increasing order for SelfAdjointEigenSolver
  // only! If you use other Eigen Solvers, it's not guaranteed to be in
  // increasing order. Here, we just take the eigen vector corresponding
  // to first/smallest eigen value, since we used SelfAdjointEigenSolver.
  Eigen::VectorXf eigen_vec = eigen_solver.eigenvectors().col(0);
  Map<Matrix<float, 4, 3, Eigen::RowMajor>> control_matrix(eigen_vec.data());

  // All 3D points should be in front of camera (z < 0).
  if (control_matrix(0, 2) > 0) {
    control_matrix = -control_matrix;
  }
  Matrix<float, kNumKeypoints - 1, 3> vertices = epnp_alpha * control_matrix;

  // Fill 0th 3D points.
  output_points_3d->emplace_back(control_matrix(0, 0), control_matrix(0, 1),
                                 control_matrix(0, 2));
  // Fill the rest 3D points.
  for (int i = 0; i < kNumKeypoints - 1; ++i) {
    output_points_3d->emplace_back(vertices(i, 0), vertices(i, 1),
                                   vertices(i, 2));
  }
  return absl::OkStatus();
}

absl::Status SolveEpnp(const Eigen::Matrix4f& projection_matrix,
                       const bool portrait,
                       const std::vector<Vector2f>& input_points_2d,
                       std::vector<Vector3f>* output_points_3d) {
  const float focal_x = projection_matrix(0, 0);
  const float focal_y = projection_matrix(1, 1);
  const float center_x = projection_matrix(0, 2);
  const float center_y = projection_matrix(1, 2);
  return SolveEpnp(focal_x, focal_y, center_x, center_y, portrait,
                   input_points_2d, output_points_3d);
}

}  // namespace mediapipe
