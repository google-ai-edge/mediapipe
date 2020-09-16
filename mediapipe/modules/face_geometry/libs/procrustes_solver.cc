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

#include "mediapipe/modules/face_geometry/libs/procrustes_solver.h"

#include <cmath>
#include <memory>

#include "Eigen/Dense"
#include "absl/memory/memory.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/statusor.h"

namespace mediapipe {
namespace face_geometry {
namespace {

class FloatPrecisionProcrustesSolver : public ProcrustesSolver {
 public:
  FloatPrecisionProcrustesSolver() = default;

  mediapipe::Status SolveWeightedOrthogonalProblem(
      const Eigen::Matrix3Xf& source_points,  //
      const Eigen::Matrix3Xf& target_points,  //
      const Eigen::VectorXf& point_weights,
      Eigen::Matrix4f& transform_mat) const override {
    // Validate inputs.
    MP_RETURN_IF_ERROR(ValidateInputPoints(source_points, target_points))
        << "Failed to validate weighted orthogonal problem input points!";
    MP_RETURN_IF_ERROR(
        ValidatePointWeights(source_points.cols(), point_weights))
        << "Failed to validate weighted orthogonal problem point weights!";

    // Extract square root from the point weights.
    Eigen::VectorXf sqrt_weights = ExtractSquareRoot(point_weights);

    // Try to solve the WEOP problem.
    MP_RETURN_IF_ERROR(InternalSolveWeightedOrthogonalProblem(
        source_points, target_points, sqrt_weights, transform_mat))
        << "Failed to solve the WEOP problem!";

    return mediapipe::OkStatus();
  }

 private:
  static constexpr float kAbsoluteErrorEps = 1e-9f;

  static mediapipe::Status ValidateInputPoints(
      const Eigen::Matrix3Xf& source_points,
      const Eigen::Matrix3Xf& target_points) {
    RET_CHECK_GT(source_points.cols(), 0)
        << "The number of source points must be positive!";

    RET_CHECK_EQ(source_points.cols(), target_points.cols())
        << "The number of source and target points must be equal!";

    return mediapipe::OkStatus();
  }

  static mediapipe::Status ValidatePointWeights(
      int num_points, const Eigen::VectorXf& point_weights) {
    RET_CHECK_GT(point_weights.size(), 0)
        << "The number of point weights must be positive!";

    RET_CHECK_EQ(point_weights.size(), num_points)
        << "The number of points and point weights must be equal!";

    float total_weight = 0.f;
    for (int i = 0; i < num_points; ++i) {
      RET_CHECK_GE(point_weights(i), 0.f)
          << "Each point weight must be non-negative!";

      total_weight += point_weights(i);
    }

    RET_CHECK_GT(total_weight, kAbsoluteErrorEps)
        << "The total point weight is too small!";

    return mediapipe::OkStatus();
  }

  static Eigen::VectorXf ExtractSquareRoot(
      const Eigen::VectorXf& point_weights) {
    Eigen::VectorXf sqrt_weights(point_weights);
    for (int i = 0; i < sqrt_weights.size(); ++i) {
      sqrt_weights(i) = std::sqrt(sqrt_weights(i));
    }

    return sqrt_weights;
  }

  // Combines a 3x3 rotation-and-scale matrix and a 3x1 translation vector into
  // a single 4x4 transformation matrix.
  static Eigen::Matrix4f CombineTransformMatrix(const Eigen::Matrix3f& r_and_s,
                                                const Eigen::Vector3f& t) {
    Eigen::Matrix4f result = Eigen::Matrix4f::Identity();
    result.leftCols(3).topRows(3) = r_and_s;
    result.col(3).topRows(3) = t;

    return result;
  }

  // The weighted problem is thoroughly addressed in Section 2.4 of:
  // D. Akca, Generalized Procrustes analysis and its applications
  // in photogrammetry, 2003, https://doi.org/10.3929/ethz-a-004656648
  //
  // Notable differences in the code presented here are:
  //
  //   * In the paper, the weights matrix W_p is Cholesky-decomposed as Q^T Q.
  //     Our W_p is diagonal (equal to diag(sqrt_weights^2)),
  //     so we can just set Q = diag(sqrt_weights) instead.
  //
  //   * In the paper, the problem is presented as
  //     (for W_k = I and W_p = tranposed(Q) Q):
  //     || Q (c A T + j tranposed(t) - B) || -> min.
  //
  //     We reformulate it as an equivalent minimization of the transpose's
  //     norm:
  //     || (c tranposed(T) tranposed(A) - tranposed(B)) tranposed(Q) || -> min,
  //     where tranposed(A) and tranposed(B) are the source and the target point
  //     clouds, respectively, c tranposed(T) is the rotation+scaling R sought
  //     for, and Q is diag(sqrt_weights).
  //
  //     Most of the derivations are therefore transposed.
  //
  // Note: the output `transform_mat` argument is used instead of `StatusOr<>`
  // return type in order to avoid Eigen memory alignment issues. Details:
  // https://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
  static mediapipe::Status InternalSolveWeightedOrthogonalProblem(
      const Eigen::Matrix3Xf& sources, const Eigen::Matrix3Xf& targets,
      const Eigen::VectorXf& sqrt_weights, Eigen::Matrix4f& transform_mat) {
    // tranposed(A_w).
    Eigen::Matrix3Xf weighted_sources =
        sources.array().rowwise() * sqrt_weights.array().transpose();
    // tranposed(B_w).
    Eigen::Matrix3Xf weighted_targets =
        targets.array().rowwise() * sqrt_weights.array().transpose();

    // w = tranposed(j_w) j_w.
    float total_weight = sqrt_weights.cwiseProduct(sqrt_weights).sum();

    // Let C = (j_w tranposed(j_w)) / (tranposed(j_w) j_w).
    // Note that C = tranposed(C), hence (I - C) = tranposed(I - C).
    //
    // tranposed(A_w) C = tranposed(A_w) j_w tranposed(j_w) / w =
    // (tranposed(A_w) j_w) tranposed(j_w) / w = c_w tranposed(j_w),
    //
    // where c_w = tranposed(A_w) j_w / w is a k x 1 vector calculated here:
    Eigen::Matrix3Xf twice_weighted_sources =
        weighted_sources.array().rowwise() * sqrt_weights.array().transpose();
    Eigen::Vector3f source_center_of_mass =
        twice_weighted_sources.rowwise().sum() / total_weight;
    // tranposed((I - C) A_w) = tranposed(A_w) (I - C) =
    // tranposed(A_w) - tranposed(A_w) C = tranposed(A_w) - c_w tranposed(j_w).
    Eigen::Matrix3Xf centered_weighted_sources =
        weighted_sources - source_center_of_mass * sqrt_weights.transpose();

    Eigen::Matrix3f rotation;
    MP_RETURN_IF_ERROR(ComputeOptimalRotation(
        weighted_targets * centered_weighted_sources.transpose(), rotation))
        << "Failed to compute the optimal rotation!";
    ASSIGN_OR_RETURN(
        float scale,
        ComputeOptimalScale(centered_weighted_sources, weighted_sources,
                            weighted_targets, rotation),
        _ << "Failed to compute the optimal scale!");

    // R = c tranposed(T).
    Eigen::Matrix3f rotation_and_scale = scale * rotation;

    // Compute optimal translation for the weighted problem.

    // tranposed(B_w - c A_w T) = tranposed(B_w) - R tranposed(A_w) in (54).
    const auto pointwise_diffs =
        weighted_targets - rotation_and_scale * weighted_sources;
    // Multiplication by j_w is a respectively weighted column sum.
    // (54) from the paper.
    const auto weighted_pointwise_diffs =
        pointwise_diffs.array().rowwise() * sqrt_weights.array().transpose();
    Eigen::Vector3f translation =
        weighted_pointwise_diffs.rowwise().sum() / total_weight;

    transform_mat = CombineTransformMatrix(rotation_and_scale, translation);

    return mediapipe::OkStatus();
  }

  // `design_matrix` is a transposed LHS of (51) in the paper.
  //
  // Note: the output `rotation` argument is used instead of `StatusOr<>`
  // return type in order to avoid Eigen memory alignment issues. Details:
  // https://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
  static mediapipe::Status ComputeOptimalRotation(
      const Eigen::Matrix3f& design_matrix, Eigen::Matrix3f& rotation) {
    RET_CHECK_GT(design_matrix.norm(), kAbsoluteErrorEps)
        << "Design matrix norm is too small!";

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(
        design_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Matrix3f postrotation = svd.matrixU();
    Eigen::Matrix3f prerotation = svd.matrixV().transpose();

    // Disallow reflection by ensuring that det(`rotation`) = +1 (and not -1),
    // see "4.6 Constrained orthogonal Procrustes problems"
    // in the Gower & Dijksterhuis's book "Procrustes Analysis".
    // We flip the sign of the least singular value along with a column in W.
    //
    // Note that now the sum of singular values doesn't work for scale
    // estimation due to this sign flip.
    if (postrotation.determinant() * prerotation.determinant() <
        static_cast<float>(0)) {
      postrotation.col(2) *= static_cast<float>(-1);
    }

    // Transposed (52) from the paper.
    rotation = postrotation * prerotation;
    return mediapipe::OkStatus();
  }

  static mediapipe::StatusOr<float> ComputeOptimalScale(
      const Eigen::Matrix3Xf& centered_weighted_sources,
      const Eigen::Matrix3Xf& weighted_sources,
      const Eigen::Matrix3Xf& weighted_targets,
      const Eigen::Matrix3f& rotation) {
    // tranposed(T) tranposed(A_w) (I - C).
    const auto rotated_centered_weighted_sources =
        rotation * centered_weighted_sources;
    // Use the identity trace(A B) = sum(A * B^T)
    // to avoid building large intermediate matrices (* is Hadamard product).
    // (53) from the paper.
    float numerator =
        rotated_centered_weighted_sources.cwiseProduct(weighted_targets).sum();
    float denominator =
        centered_weighted_sources.cwiseProduct(weighted_sources).sum();

    RET_CHECK_GT(denominator, kAbsoluteErrorEps)
        << "Scale expression denominator is too small!";
    RET_CHECK_GT(numerator / denominator, kAbsoluteErrorEps)
        << "Scale is too small!";

    return numerator / denominator;
  }
};

}  // namespace

std::unique_ptr<ProcrustesSolver> CreateFloatPrecisionProcrustesSolver() {
  return absl::make_unique<FloatPrecisionProcrustesSolver>();
}

}  // namespace face_geometry
}  // namespace mediapipe
