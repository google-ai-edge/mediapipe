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

#include "mediapipe/util/tracking/tracking.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <random>
#include <unordered_set>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/SVD"
#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/opencv_calib3d_inc.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/util/tracking/flow_packager.pb.h"
#include "mediapipe/util/tracking/measure_time.h"
#include "mediapipe/util/tracking/motion_models.h"

namespace mediapipe {

bool MotionBox::print_motion_box_warnings_ = true;

namespace {

static constexpr int kNormalizationGridSize = 10;
constexpr float kShortScale = 16383.0f;
constexpr float kInvShortScale = 1.0f / kShortScale;

// Motion vectors with weights larger than kMinInlierWeight are classified as
// inliers.
constexpr float kMinInlierWeight = 0.5f;
// Motion vectors with weights smaller han kMaxOutlierWeight are classified as
// outliers.
constexpr float kMaxOutlierWeight = 0.1f;

// Lexicographic (first x, then y) comparator for MotionVector::pos.
struct MotionVectorComparator {
  bool operator()(const MotionVector& lhs, const MotionVector& rhs) const {
    return (lhs.pos.x() < rhs.pos.x()) ||
           (lhs.pos.x() == rhs.pos.x() && lhs.pos.y() < rhs.pos.y());
  }
};

void StoreInternalState(const std::vector<const MotionVector*>& vectors,
                        const std::vector<float>& inlier_weights,
                        float aspect_ratio, MotionBoxInternalState* internal) {
  const int num_vectors = vectors.size();
  CHECK_EQ(num_vectors, inlier_weights.size());

  float scale_x = 1.0f;
  float scale_y = 1.0f;
  ScaleFromAspect(aspect_ratio, true, &scale_x, &scale_y);

  internal->Clear();
  for (int k = 0; k < num_vectors; ++k) {
    internal->add_pos_x(vectors[k]->pos.x() * scale_x);
    internal->add_pos_y(vectors[k]->pos.y() * scale_y);
    internal->add_dx(vectors[k]->object.x() * scale_x);
    internal->add_dy(vectors[k]->object.y() * scale_y);
    internal->add_camera_dx(vectors[k]->background.x() * scale_x);
    internal->add_camera_dy(vectors[k]->background.y() * scale_y);
    internal->add_track_id(vectors[k]->track_id);
    internal->add_inlier_score(inlier_weights[k]);
  }
}

// protolite compatible MotionBoxState_TrackStatus_Name
std::string TrackStatusToString(MotionBoxState::TrackStatus status) {
  switch (status) {
    case MotionBoxState::BOX_UNTRACKED:
      return "BOX_UNTRACKED";
    case MotionBoxState::BOX_EMPTY:
      return "BOX_EMPTY";
    case MotionBoxState::BOX_NO_FEATURES:
      return "BOX_NO_FEATURES";
    case MotionBoxState::BOX_TRACKED:
      return "BOX_TRACKED";
    case MotionBoxState::BOX_DUPLICATED:
      return "BOX_DUPLICATED";
    case MotionBoxState::BOX_TRACKED_OUT_OF_BOUND:
      return "BOX_TRACKED_OUT_OF_BOUND";
  }
  LOG(FATAL) << "Should not happen.";
  return "UNKNOWN";
}

void ClearInlierState(MotionBoxState* state) {
  state->clear_inlier_ids();
  state->clear_inlier_length();
  state->clear_inlier_id_match_pos();
  state->clear_outlier_ids();
  state->clear_outlier_id_match_pos();
}

// Returns orthogonal error system from motion_vec scaled by irls_scale.
std::pair<Vector2_f, Vector2_f> ComputeIrlsErrorSystem(
    const Vector2_f& irls_scale, const Vector2_f& motion_vec) {
  Vector2_f irls_vec = motion_vec.Normalize();
  Vector2_f irls_vec_ortho = irls_vec.Ortho();
  return std::make_pair(irls_vec * irls_scale.x(),
                        irls_vec_ortho * irls_scale.y());
}

// Returns error for a given difference vector and error system.
float ErrorDiff(const Vector2_f& diff,
                const std::pair<Vector2_f, Vector2_f>& error_system) {
  // Error system is an orthogonal system of originally unit vectors that were
  // pre-multiplied by the corresponding irls scale.
  // One can think of this function here as L2 norm *after* scaling the whole
  // vector space w.r.t. the error system.
  //
  // In particular, we will project the vector diff onto this system and then
  // scale the magnitude along each direction with the corresponding irls scale.
  // As scalar multiplication is commutative with the dot product of vectors
  // pre-multiplication of the scale with the error system is sufficient.
  return Vector2_f(diff.DotProd(error_system.first),
                   diff.DotProd(error_system.second))
      .Norm();
}

// Returns true if point is within the inlier extent of the passed state (within
// small bound of 5% of frame diameter).
bool PointWithinInlierExtent(const Vector2_f pt, const MotionBoxState& state) {
  // No extent known, assume to be inside.
  if (state.prior_weight() == 0) {
    return true;
  }

  const float width_radius = state.inlier_width() * 0.55f;
  const float height_radius = state.inlier_height() * 0.55f;
  const float left = state.inlier_center_x() - width_radius;
  const float right = state.inlier_center_x() + width_radius;
  const float top = state.inlier_center_y() - height_radius;
  const float bottom = state.inlier_center_y() + height_radius;

  return pt.x() >= left && pt.x() <= right && pt.y() >= top && pt.y() <= bottom;
}

// Taken from MotionEstimation::LinearSimilarityL2SolveSystem.
bool LinearSimilarityL2Solve(
    const std::vector<const MotionVector*>& motion_vectors,
    const std::vector<float>& weights, LinearSimilarityModel* model) {
  CHECK(model);
  if (motion_vectors.size() < 4) {
    LOG(ERROR) << "Requiring at least 4 input vectors for sufficient solve.";
    return false;
  }

  cv::Mat matrix(4, 4, CV_32F);
  cv::Mat solution(4, 1, CV_32F);
  cv::Mat rhs(4, 1, CV_32F);

  matrix.setTo(0);
  rhs.setTo(0);

  CHECK_EQ(motion_vectors.size(), weights.size());
  for (int k = 0; k < motion_vectors.size(); ++k) {
    const float x = motion_vectors[k]->pos.x();
    const float y = motion_vectors[k]->pos.y();
    const float w = weights[k];

    // double J[2 * 4] = {1, 0, x,  -y,
    //                    0, 1, y,   x};
    // Compute J^t * J * w = {1,  0,   x,    -y
    //                        0,  1,   y,     x,
    //                        x,  y,   xx+yy, 0,
    //                       -y,  x,   0,     xx+yy} * w;

    const float x_w = x * w;
    const float y_w = y * w;
    const float xx_yy_w = (x * x + y * y) * w;
    float* matrix_ptr = matrix.ptr<float>(0);
    matrix_ptr[0] += w;
    matrix_ptr[2] += x_w;
    matrix_ptr[3] += -y_w;

    matrix_ptr += 4;
    matrix_ptr[1] += w;
    matrix_ptr[2] += y_w;
    matrix_ptr[3] += x_w;

    matrix_ptr += 4;
    matrix_ptr[0] += x_w;
    matrix_ptr[1] += y_w;
    matrix_ptr[2] += xx_yy_w;

    matrix_ptr += 4;
    matrix_ptr[0] += -y_w;
    matrix_ptr[1] += x_w;
    matrix_ptr[3] += xx_yy_w;

    float* rhs_ptr = rhs.ptr<float>(0);

    const float m_x = motion_vectors[k]->object.x() * w;
    const float m_y = motion_vectors[k]->object.y() * w;

    rhs_ptr[0] += m_x;
    rhs_ptr[1] += m_y;
    rhs_ptr[2] += x * m_x + y * m_y;
    rhs_ptr[3] += -y * m_x + x * m_y;
  }

  // Solution parameters p.
  if (cv::solve(matrix, rhs, solution)) {
    const float* ptr = solution.ptr<float>(0);
    model->set_dx(ptr[0]);
    model->set_dy(ptr[1]);
    model->set_a(ptr[2] + 1.0);  // Identity parametrization.
    model->set_b(ptr[3]);
    return true;
  } else {
    return false;
  }
}

// Taken from MotionEstimation::HomographyL2NormalEquationSolve
bool HomographyL2Solve(const std::vector<const MotionVector*>& motion_vectors,
                       const std::vector<float>& weights, Homography* model) {
  CHECK(model);

  cv::Mat matrix(8, 8, CV_32F);
  cv::Mat solution(8, 1, CV_32F);
  cv::Mat rhs(8, 1, CV_32F);

  matrix.setTo(0);
  rhs.setTo(0);

  // Matrix multiplications are hand-coded for speed improvements vs.
  // opencv's cvGEMM calls.
  CHECK_EQ(motion_vectors.size(), weights.size());
  for (int k = 0; k < motion_vectors.size(); ++k) {
    const float x = motion_vectors[k]->pos.x();
    const float y = motion_vectors[k]->pos.y();
    const float w = weights[k];

    const float xw = x * w;
    const float yw = y * w;
    const float xxw = x * x * w;
    const float yyw = y * y * w;
    const float xyw = x * y * w;
    const float mx = x + motion_vectors[k]->object.x();
    const float my = y + motion_vectors[k]->object.y();

    const float mxxyy = mx * mx + my * my;
    // Jacobian
    // double J[2 * 8] = {x, y, 1,  0,  0,   0, -x * m_x, -y * m_x,
    //                   {0, 0, 0,  x,  y,   1, -x * m_y, -y * m_y}
    //
    // // Compute J^t * J * w =
    // ( xx        xy    x      0       0    0    -xx*mx  -xy*mx    )
    // ( xy        yy    y      0       0    0    -xy*mx  -yy*mx    )
    // ( x         y     1      0       0    0     -x*mx   -y*mx    )
    // ( 0         0     0     xx      xy    x    -xx*my  -xy*my    )
    // ( 0         0     0     xy      yy    y    -xy*my  -yy*my    )
    // ( 0         0     0      x      y     1     -x*my   -y*my    )
    // ( -xx*mx -xy*mx -x*mx -xx*my -xy*my -x*my xx*mxxyy  xy*mxxyy )
    // ( -xy*mx -yy*mx -y*mx -xy*my -yy*my -y*my xy*mxxyy  yy*mxxyy  ) * w

    // 1st row:  xx        xy    x      0       0    0    -xx*mx  -xy*mx
    float* matrix_ptr = matrix.ptr<float>(0);
    matrix_ptr[0] += xxw;
    matrix_ptr[1] += xyw;
    matrix_ptr[2] += xw;
    matrix_ptr[6] += -xxw * mx;
    matrix_ptr[7] += -xyw * mx;

    // 2nd row:  xy       yy   y      0       0    0    -xy*mx  -yy*mx
    matrix_ptr += 8;
    matrix_ptr[0] += xyw;
    matrix_ptr[1] += yyw;
    matrix_ptr[2] += yw;
    matrix_ptr[6] += -xyw * mx;
    matrix_ptr[7] += -yyw * mx;

    // 3rd row: x         y     1      0       0    0     -x*mx   -y*mx
    matrix_ptr += 8;
    matrix_ptr[0] += xw;
    matrix_ptr[1] += yw;
    matrix_ptr[2] += w;
    matrix_ptr[6] += -xw * mx;
    matrix_ptr[7] += -yw * mx;

    // 4th row: 0         0     0     xx      xy    x    -xx*my  -xy*my
    matrix_ptr += 8;
    matrix_ptr[3] += xxw;
    matrix_ptr[4] += xyw;
    matrix_ptr[5] += xw;
    matrix_ptr[6] += -xxw * my;
    matrix_ptr[7] += -xyw * my;

    // 5th row: 0         0     0     xy      yy    y    -xy*my  -yy*my
    matrix_ptr += 8;
    matrix_ptr[3] += xyw;
    matrix_ptr[4] += yyw;
    matrix_ptr[5] += yw;
    matrix_ptr[6] += -xyw * my;
    matrix_ptr[7] += -yyw * my;

    // 6th row:  0         0     0     x     y     1      -x*my    -y*my
    matrix_ptr += 8;
    matrix_ptr[3] += xw;
    matrix_ptr[4] += yw;
    matrix_ptr[5] += w;
    matrix_ptr[6] += -xw * my;
    matrix_ptr[7] += -yw * my;

    // 7th row:  -xx*mx -xy*mx -x*mx -xx*my -xy*my -x*my xx*mxxyy  xy*mxxyy
    matrix_ptr += 8;
    matrix_ptr[0] += -xxw * mx;
    matrix_ptr[1] += -xyw * mx;
    matrix_ptr[2] += -xw * mx;
    matrix_ptr[3] += -xxw * my;
    matrix_ptr[4] += -xyw * my;
    matrix_ptr[5] += -xw * my;
    matrix_ptr[6] += xxw * mxxyy;
    matrix_ptr[7] += xyw * mxxyy;

    // 8th row: -xy*mx -yy*mx -y*mx -xy*my -yy*my -y*my xy*mxxyy  yy*mxxyy
    matrix_ptr += 8;
    matrix_ptr[0] += -xyw * mx;
    matrix_ptr[1] += -yyw * mx;
    matrix_ptr[2] += -yw * mx;
    matrix_ptr[3] += -xyw * my;
    matrix_ptr[4] += -yyw * my;
    matrix_ptr[5] += -yw * my;
    matrix_ptr[6] += xyw * mxxyy;
    matrix_ptr[7] += yyw * mxxyy;

    // Right hand side:
    // b = ( x
    //       y )
    // Compute J^t * b  * w =
    // ( x*mx  y*mx  mx  x*my  y*my  my  -x*mxxyy -y*mxxyy ) * w

    float* rhs_ptr = rhs.ptr<float>(0);
    rhs_ptr[0] += xw * mx;
    rhs_ptr[1] += yw * mx;
    rhs_ptr[2] += mx * w;
    rhs_ptr[3] += xw * my;
    rhs_ptr[4] += yw * my;
    rhs_ptr[5] += my * w;
    rhs_ptr[6] += -xw * mxxyy;
    rhs_ptr[7] += -yw * mxxyy;
  }

  // Solution parameters p.
  if (cv::solve(matrix, rhs, solution)) {
    const float* ptr = solution.ptr<float>(0);
    *model = HomographyAdapter::FromFloatPointer(ptr, false);
    return true;
  }

  return false;
}

void TransformQuadInMotionBoxState(const MotionBoxState& curr_pos,
                                   const Homography& homography,
                                   MotionBoxState* next_pos) {
  CHECK(next_pos != nullptr);
  if (!curr_pos.has_pos_x() || !curr_pos.has_pos_y() || !curr_pos.has_width() ||
      !curr_pos.has_height()) {
    LOG(ERROR) << "Previous box does not exist, cannot transform!";
    return;
  }
  const int kQuadVerticesSize = 8;
  const MotionBoxState::Quad* curr_quad_ptr = nullptr;
  auto quad = absl::make_unique<MotionBoxState::Quad>();
  if (curr_pos.has_quad() &&
      curr_pos.quad().vertices_size() == kQuadVerticesSize) {
    curr_quad_ptr = &curr_pos.quad();
  } else {
    std::array<Vector2_f, 4> corners =
        GetCornersOfRotatedRect(curr_pos, Vector2_f(1.0f, 1.0f));
    for (const auto& vertex : corners) {
      quad->add_vertices(vertex.x());
      quad->add_vertices(vertex.y());
    }
    curr_quad_ptr = quad.get();
  }

  MotionBoxState::Quad* next_pos_quad = next_pos->mutable_quad();
  bool next_pos_quad_existed = true;
  if (next_pos_quad->vertices_size() != kQuadVerticesSize) {
    next_pos_quad_existed = false;
    next_pos_quad->clear_vertices();
  }
  for (int i = 0; i < kQuadVerticesSize / 2; ++i) {
    const auto& curr_pos_quad_vertex = Vector2_f(
        curr_quad_ptr->vertices(i * 2), curr_quad_ptr->vertices(i * 2 + 1));
    const auto& next_pos_quad_vertex_diff =
        HomographyAdapter::TransformPoint(homography, curr_pos_quad_vertex) -
        curr_pos_quad_vertex;
    if (next_pos_quad_existed) {
      next_pos_quad->set_vertices(i * 2, next_pos_quad->vertices(i * 2) +
                                             next_pos_quad_vertex_diff.x());
      next_pos_quad->set_vertices(
          i * 2 + 1,
          next_pos_quad->vertices(i * 2 + 1) + next_pos_quad_vertex_diff.y());
    } else {
      next_pos_quad->add_vertices(curr_pos_quad_vertex.x() +
                                  next_pos_quad_vertex_diff.x());
      next_pos_quad->add_vertices(curr_pos_quad_vertex.y() +
                                  next_pos_quad_vertex_diff.y());
    }
  }
}

void UpdateStatePositionAndSizeFromStateQuad(MotionBoxState* box_state) {
  Vector2_f top_left, bottom_right;
  MotionBoxBoundingBox(*box_state, &top_left, &bottom_right);
  box_state->set_width(bottom_right.x() - top_left.x());
  box_state->set_height(bottom_right.y() - top_left.y());
  box_state->set_pos_x(top_left.x());
  box_state->set_pos_y(top_left.y());
}

void ApplyCameraTrackingDegrees(const MotionBoxState& prev_state,
                                const Homography& background_model,
                                const TrackStepOptions& options,
                                const Vector2_f& domain,
                                MotionBoxState* next_state) {
  // Determine center translation.
  Vector2_f center(MotionBoxCenter(prev_state));
  const Vector2_f background_motion =
      HomographyAdapter::TransformPoint(background_model, center) - center;

  if (options.tracking_degrees() ==
          TrackStepOptions::TRACKING_DEGREE_TRANSLATION ||
      !options.track_object_and_camera()) {
    SetMotionBoxPosition(MotionBoxPosition(*next_state) + background_motion,
                         next_state);
    return;
  }

  // Transform corners and fit similarity.
  // Overall idea:
  // We got corners, x0, x1, x2, x3 of the rect in the previous location
  // transform by background_model H.
  // Assuming H = [A | t], their target location in the next frame
  // is:
  // xi' = A * xi + t for i = 0..3
  // We want to express the location of ci' w.r.t. to the translated center c,
  // to decouple H from the translation of the center.
  // In particular, we are looking for the translation of the center:
  // c* = c + t* and points
  // xi* = xi + t*
  // Express location of xi' w.r.t. c:
  // xi' = A(xi* - c*) + c*
  // Axi + t = A(xi - c) + c + t*
  // Axi + t = Axi - Ac + c + t*
  // t* = Ac - c + t
  std::array<Vector2_f, 4> corners = MotionBoxCorners(prev_state);
  std::vector<MotionVector> corner_vecs(4);
  std::vector<const MotionVector*> corner_vec_ptrs(4);

  for (int k = 0; k < 4; ++k) {
    MotionVector v;
    v.pos = corners[k];
    v.object = HomographyAdapter::TransformPoint(background_model, corners[k]) -
               corners[k];
    corner_vecs[k] = v;
    corner_vec_ptrs[k] = &corner_vecs[k];
  }

  LinearSimilarityModel linear_similarity;
  LinearSimilarityL2Solve(corner_vec_ptrs, std::vector<float>(4, 1.0f),
                          &linear_similarity);

  SimilarityModel similarity =
      LinearSimilarityAdapter::ToSimilarity(linear_similarity);

  // See above derivation, motion of the center is t* = Ac + t - c;
  // Could also make the point that background_model instead of
  // linear_similarity is more accurate here due to the fitting operation above.
  SetMotionBoxPosition(MotionBoxPosition(*next_state) +
                           TransformPoint(linear_similarity, center) - center,
                       next_state);

  switch (options.tracking_degrees()) {
    case TrackStepOptions::TRACKING_DEGREE_TRANSLATION:
      break;
    case TrackStepOptions::TRACKING_DEGREE_CAMERA_SCALE:
    case TrackStepOptions::TRACKING_DEGREE_OBJECT_SCALE:
      next_state->set_scale(next_state->scale() * similarity.scale());
      break;
    case TrackStepOptions::TRACKING_DEGREE_CAMERA_ROTATION:
    case TrackStepOptions::TRACKING_DEGREE_OBJECT_ROTATION:
      next_state->set_rotation(next_state->rotation() + similarity.rotation());
      break;
    case TrackStepOptions::TRACKING_DEGREE_CAMERA_ROTATION_SCALE:
    case TrackStepOptions::TRACKING_DEGREE_OBJECT_ROTATION_SCALE:
      next_state->set_scale(next_state->scale() * similarity.scale());
      next_state->set_rotation(next_state->rotation() + similarity.rotation());
      break;
    case TrackStepOptions::TRACKING_DEGREE_CAMERA_PERSPECTIVE:
    case TrackStepOptions::TRACKING_DEGREE_OBJECT_PERSPECTIVE:
      TransformQuadInMotionBoxState(prev_state, background_model, next_state);
      if (prev_state.has_pnp_homography()) {
        *(next_state->mutable_pnp_homography()) = HomographyAdapter::Compose(
            prev_state.pnp_homography(), background_model);
        UpdateStatePositionAndSizeFromStateQuad(next_state);
      }
      break;
  }
}

void ApplyObjectMotion(const MotionBoxState& curr_pos,
                       const Vector2_f& object_translation,
                       const LinearSimilarityModel& object_similarity,
                       const Homography& object_homography,
                       const TrackStepOptions& options,
                       MotionBoxState* next_pos) {
  switch (options.tracking_degrees()) {
    case TrackStepOptions::TRACKING_DEGREE_OBJECT_ROTATION_SCALE:
    case TrackStepOptions::TRACKING_DEGREE_OBJECT_ROTATION:
    case TrackStepOptions::TRACKING_DEGREE_OBJECT_SCALE: {
      Vector2_f center(MotionBoxCenter(curr_pos));
      // See ApplyCameraTrackingDegrees for derivation.
      SetMotionBoxPosition(MotionBoxPosition(*next_pos) +
                               TransformPoint(object_similarity, center) -
                               center,
                           next_pos);
      SimilarityModel similarity =
          LinearSimilarityAdapter::ToSimilarity(object_similarity);
      if (options.tracking_degrees() !=
          TrackStepOptions::TRACKING_DEGREE_OBJECT_ROTATION) {
        next_pos->set_scale(next_pos->scale() * similarity.scale());
      }
      if (options.tracking_degrees() !=
          TrackStepOptions::TRACKING_DEGREE_OBJECT_SCALE) {
        next_pos->set_rotation(next_pos->rotation() + similarity.rotation());
      }
      break;
    }

    case TrackStepOptions::TRACKING_DEGREE_OBJECT_PERSPECTIVE: {
      Vector2_f center(MotionBoxCenter(curr_pos));
      SetMotionBoxPosition(
          MotionBoxPosition(*next_pos) +
              HomographyAdapter::TransformPoint(object_homography, center) -
              center,
          next_pos);
      TransformQuadInMotionBoxState(curr_pos, object_homography, next_pos);
      break;
    }
    default:
      // Use translation per default.
      SetMotionBoxPosition(MotionBoxPosition(*next_pos) + object_translation,
                           next_pos);
  }
}

bool IsBoxValid(const MotionBoxState& state) {
  const float kMaxBoxHeight =
      10000.0f;  // as relative to normalized [0, 1] space
  const float kMaxBoxWidth =
      10000.0f;  // as relative to normalized [0, 1] space
  if (state.width() > kMaxBoxWidth) {
    LOG(ERROR) << "box width " << state.width() << " too big";
    return false;
  }
  if (state.height() > kMaxBoxHeight) {
    LOG(ERROR) << "box height " << state.height() << " too big";
    return false;
  }

  return true;
}

Homography PnpHomographyFromRotationAndTranslation(const cv::Mat& rvec,
                                                   const cv::Mat& tvec) {
  cv::Mat homography_matrix(3, 3, CV_64F);
  cv::Rodrigues(rvec, homography_matrix);

  for (int c = 0; c < 3; ++c) {
    homography_matrix.at<double>(c, 2) = tvec.at<double>(c);
  }

  // check non zero
  homography_matrix /= homography_matrix.at<double>(2, 2);

  return HomographyAdapter::FromDoublePointer(homography_matrix.ptr<double>(0),
                                              false);
}

// Translate CameraIntrinsics proto to cv format.
void ConvertCameraIntrinsicsToCvMat(
    const TrackStepOptions::CameraIntrinsics& camera_intrinsics,
    cv::Mat* camera_mat, cv::Mat* dist_coef) {
  *camera_mat = cv::Mat::eye(3, 3, CV_64F);
  *dist_coef = cv::Mat::zeros(1, 5, CV_64FC1);
  camera_mat->at<double>(0, 0) = camera_intrinsics.fx();
  camera_mat->at<double>(1, 1) = camera_intrinsics.fy();
  camera_mat->at<double>(0, 2) = camera_intrinsics.cx();
  camera_mat->at<double>(1, 2) = camera_intrinsics.cy();
  dist_coef->at<double>(0) = camera_intrinsics.k0();
  dist_coef->at<double>(1) = camera_intrinsics.k1();
  dist_coef->at<double>(4) = camera_intrinsics.k2();
}

}  // namespace.

void ScaleFromAspect(float aspect, bool invert, float* scale_x,
                     float* scale_y) {
  *scale_x = aspect >= 1.0f ? 1.0f : aspect;
  *scale_y = aspect >= 1.0f ? 1.0f / aspect : 1.0f;
  if (invert) {
    *scale_x = 1.0f / *scale_x;
    *scale_y = 1.0f / *scale_y;
  }
}

std::array<Vector2_f, 4> MotionBoxCorners(const MotionBoxState& state,
                                          const Vector2_f& scaling) {
  std::array<Vector2_f, 4> transformed_corners;
  if (state.has_quad() && state.quad().vertices_size() == 8) {
    for (int k = 0; k < 4; ++k) {
      transformed_corners[k] = Vector2_f(state.quad().vertices(2 * k),
                                         state.quad().vertices(2 * k + 1))
                                   .MulComponents(scaling);
    }
  } else {
    transformed_corners = GetCornersOfRotatedRect(state, scaling);
  }

  return transformed_corners;
}

bool MotionBoxLines(const MotionBoxState& state, const Vector2_f& scaling,
                    std::array<Vector3_f, 4>* box_lines) {
  CHECK(box_lines);
  std::array<Vector2_f, 4> corners = MotionBoxCorners(state, scaling);
  std::array<Vector3_f, 4> lines;
  for (int k = 0; k < 4; ++k) {
    const Vector2_f diff = corners[(k + 1) % 4] - corners[k];
    const Vector2_f normal = diff.Ortho().Normalize();
    box_lines->at(k).Set(normal.x(), normal.y(), -normal.DotProd(corners[k]));
    // Double check that second point is on the computed line.
    if (box_lines->at(k).DotProd(Vector3_f(corners[(k + 1) % 4].x(),
                                           corners[(k + 1) % 4].y(), 1.0f)) >=
        0.02f) {
      LOG(ERROR) << "box is abnormal. Line equations don't satisfy constraint";
      return false;
    }
  }
  return true;
}

void MotionBoxBoundingBox(const MotionBoxState& state, Vector2_f* top_left,
                          Vector2_f* bottom_right) {
  CHECK(top_left);
  CHECK(bottom_right);

  std::array<Vector2_f, 4> corners = MotionBoxCorners(state);

  // Determine min and max across dimension.
  *top_left = Vector2_f(std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::max());

  *bottom_right = Vector2_f(std::numeric_limits<float>::min(),
                            std::numeric_limits<float>::min());

  for (const Vector2_f& c : corners) {
    top_left->x(std::min(top_left->x(), c.x()));
    top_left->y(std::min(top_left->y(), c.y()));
    bottom_right->x(std::max(bottom_right->x(), c.x()));
    bottom_right->y(std::max(bottom_right->y(), c.y()));
  }
}

void MotionBoxInlierLocations(const MotionBoxState& state,
                              std::vector<Vector2_f>* inlier_pos) {
  CHECK(inlier_pos);
  inlier_pos->clear();
  for (int k = 0; k < state.inlier_id_match_pos_size(); k += 2) {
    inlier_pos->push_back(
        Vector2_f(state.inlier_id_match_pos(k) * kInvShortScale,
                  state.inlier_id_match_pos(k + 1) * kInvShortScale));
  }
}

void MotionBoxOutlierLocations(const MotionBoxState& state,
                               std::vector<Vector2_f>* outlier_pos) {
  CHECK(outlier_pos);
  outlier_pos->clear();
  for (int k = 0; k < state.outlier_id_match_pos_size(); k += 2) {
    outlier_pos->push_back(
        Vector2_f(state.outlier_id_match_pos(k) * kInvShortScale,
                  state.outlier_id_match_pos(k + 1) * kInvShortScale));
  }
}

std::array<Vector2_f, 4> GetCornersOfRotatedRect(const MotionBoxState& state,
                                                 const Vector2_f& scaling) {
  std::array<Vector2_f, 4> transformed_corners;
  // Scale and rotate 4 corner w.r.t. center.
  const Vector2_f center = MotionBoxCenter(state).MulComponents(scaling);
  const Vector2_f top_left = MotionBoxPosition(state).MulComponents(scaling);
  const std::array<Vector2_f, 4> corners{{
      top_left,
      top_left + Vector2_f(0, state.height() * scaling.y()),
      top_left +
          Vector2_f(state.width() * scaling.x(), state.height() * scaling.y()),
      top_left + Vector2_f(state.width() * scaling.x(), 0),
  }};

  const float cos_a = std::cos(state.rotation());
  const float sin_a = std::sin(state.rotation());
  for (int k = 0; k < 4; ++k) {
    // Scale and rotate w.r.t. center.
    const Vector2_f rad = corners[k] - center;
    const Vector2_f rot_rad(cos_a * rad.x() - sin_a * rad.y(),
                            sin_a * rad.x() + cos_a * rad.y());
    const Vector2_f transformed_corner = center + rot_rad * state.scale();
    transformed_corners[k] = transformed_corner;
  }

  return transformed_corners;
}

void InitializeQuadInMotionBoxState(MotionBoxState* state) {
  CHECK(state != nullptr);
  // Every quad has 4 vertices. Each vertex has x and y 2 coordinates. So
  // a total of 8 floating point values.
  if (state->quad().vertices_size() != 8) {
    MotionBoxState::Quad* quad_ptr = state->mutable_quad();
    quad_ptr->clear_vertices();
    std::array<Vector2_f, 4> corners =
        GetCornersOfRotatedRect(*state, Vector2_f(1.0f, 1.0f));
    for (const auto& vertex : corners) {
      quad_ptr->add_vertices(vertex.x());
      quad_ptr->add_vertices(vertex.y());
    }
  }
}

void InitializeInliersOutliersInMotionBoxState(const TrackingData& tracking,
                                               MotionBoxState* state) {
  MotionVectorFrame mvf;  // Holds motion from current to previous frame.
  MotionVectorFrameFromTrackingData(tracking, &mvf);

  std::array<Vector3_f, 4> box_lines;
  if (!MotionBoxLines(*state, Vector2_f(1.0f, 1.0f), &box_lines)) {
    LOG(ERROR) << "Error in computing MotionBoxLines.";
    return;
  }

  // scale for motion vectors.
  float scale_x, scale_y;
  ScaleFromAspect(mvf.aspect_ratio, true, &scale_x, &scale_y);

  state->clear_inlier_ids();
  state->clear_inlier_length();
  state->clear_outlier_ids();

  float inlier_center_x = 0.0f;
  float inlier_center_y = 0.0f;
  int cnt_inlier = 0;

  float min_x = std::numeric_limits<float>::max();
  float max_x = -std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float max_y = -std::numeric_limits<float>::max();

  for (const auto& motion_vec : mvf.motion_vectors) {
    const float pos_x = motion_vec.pos.x() * scale_x;
    const float pos_y = motion_vec.pos.y() * scale_y;

    bool insider = true;

    for (const Vector3_f& line : box_lines) {
      if (line.DotProd(Vector3_f(pos_x, pos_y, 1.0f)) > 0.0f) {
        insider = false;
        break;
      }
    }

    if (insider) {
      ++cnt_inlier;
      inlier_center_x += pos_x;
      inlier_center_y += pos_y;

      min_x = std::min(min_x, pos_x);
      max_x = std::max(max_x, pos_x);
      min_y = std::min(min_y, pos_y);
      max_y = std::max(max_y, pos_y);

      state->add_inlier_ids(motion_vec.track_id);
      state->add_inlier_length(1.0f);
    } else {
      state->add_outlier_ids(motion_vec.track_id);
    }
  }

  if (cnt_inlier) {
    state->set_prior_weight(1.0f);
    state->set_inlier_center_x(inlier_center_x / cnt_inlier);
    state->set_inlier_center_y(inlier_center_y / cnt_inlier);
    state->set_inlier_width(max_x - min_x);
    state->set_inlier_height(max_y - min_y);
  }
}

void InitializePnpHomographyInMotionBoxState(
    const TrackingData& tracking, const TrackStepOptions& track_step_options,
    MotionBoxState* state) {
  // Only happen when `quad` and `aspect_ratio` are all specified.
  if (!state->has_quad()) {
    VLOG(1) << "Skip pnp tracking since box does not contain quad info.";
    return;
  }

  const int kQuadCornersSize = 4;
  CHECK_EQ(state->quad().vertices_size(), kQuadCornersSize * 2);
  float scale_x, scale_y;
  ScaleFromAspect(tracking.frame_aspect(), false, &scale_x, &scale_y);
  std::vector<cv::Point2f> corners_2d(kQuadCornersSize);
  if (track_step_options.has_camera_intrinsics()) {
    const auto& camera = track_step_options.camera_intrinsics();
    for (int c = 0; c < kQuadCornersSize; ++c) {
      corners_2d[c].x = state->quad().vertices(c * 2) * camera.w();
      corners_2d[c].y = state->quad().vertices(c * 2 + 1) * camera.h();
    }

    cv::Mat camera_mat, dist_coef;
    ConvertCameraIntrinsicsToCvMat(camera, &camera_mat, &dist_coef);
    cv::undistortPoints(corners_2d, corners_2d, camera_mat, dist_coef);
  } else {
    const float center_x = scale_x * 0.5f;
    const float center_y = scale_y * 0.5f;
    for (int c = 0; c < kQuadCornersSize; ++c) {
      corners_2d[c].x = state->quad().vertices(c * 2) * scale_x - center_x;
      corners_2d[c].y = state->quad().vertices(c * 2 + 1) * scale_y - center_y;
    }
  }

  if (!state->has_aspect_ratio()) {
    if (!track_step_options.forced_pnp_tracking()) {
      VLOG(1) << "Skip pnp tracking since aspect ratio is unknown and "
                 "estimation of it is not forced.";
      return;
    }
    const float u2_u0 = corners_2d[2].x - corners_2d[0].x;
    const float v2_v0 = corners_2d[2].y - corners_2d[0].y;
    const float u3_u1 = corners_2d[3].x - corners_2d[1].x;
    const float v3_v1 = corners_2d[3].y - corners_2d[1].y;

    constexpr float kEpsilon = 1e-6f;
    const float denominator = u2_u0 * v3_v1 - v2_v0 * u3_u1;
    if (std::abs(denominator) < kEpsilon) {
      LOG(WARNING) << "Zero denominator. Failed calculating aspect ratio.";
      return;
    }

    float s[kQuadCornersSize];
    s[0] = ((corners_2d[2].x - corners_2d[3].x) * v3_v1 -
            (corners_2d[2].y - corners_2d[3].y) * u3_u1) *
           2.0f / denominator;
    s[1] = -(u2_u0 * (corners_2d[2].y - corners_2d[3].y) -
             v2_v0 * (corners_2d[2].x - corners_2d[3].x)) *
           2.0f / denominator;
    s[2] = 2.0f - s[0];
    s[3] = 2.0f - s[1];

    std::vector<Vector3_f> corners(kQuadCornersSize);
    for (int i = 0; i < kQuadCornersSize; ++i) {
      if (s[0] <= 0) {
        LOG(WARNING) << "Negative scale. Failed calculating aspect ratio.";
        return;
      }
      corners[i] =
          Vector3_f(corners_2d[i].x * s[i], corners_2d[i].y * s[i], s[i]);
    }

    const Vector3_f width_edge = corners[2] - corners[1];
    const Vector3_f height_edge = corners[0] - corners[1];
    const float height_norm = height_edge.Norm();
    const float width_norm = width_edge.Norm();
    if (height_norm < kEpsilon || width_norm < kEpsilon) {
      LOG(WARNING)
          << "abnormal 3d quadrangle. Failed calculating aspect ratio.";
      return;
    }

    constexpr float kMaxCosAngle = 0.258819;  // which is cos(75 deg)
    if (width_edge.DotProd(height_edge) / height_norm / width_norm >
        kMaxCosAngle) {
      LOG(WARNING)
          << "abnormal 3d quadrangle. Failed calculating aspect ratio.";
      return;
    }

    state->set_aspect_ratio(width_norm / height_norm);
  }

  CHECK_GT(state->aspect_ratio(), 0.0f);

  const float half_width = state->aspect_ratio();
  const float half_height = 1.0f;
  const std::vector<cv::Point3f> corners_3d{
      cv::Point3f(-half_width, -half_height, 0.0f),
      cv::Point3f(-half_width, half_height, 0.0f),
      cv::Point3f(half_width, half_height, 0.0f),
      cv::Point3f(half_width, -half_height, 0.0f),
  };

  std::vector<MotionVector> motion_vectors(kQuadCornersSize);
  std::vector<const MotionVector*> motion_vector_pointers(kQuadCornersSize);

  for (int c = 0; c < kQuadCornersSize; ++c) {
    motion_vectors[c].pos = Vector2_f(corners_3d[c].x, corners_3d[c].y);
    motion_vectors[c].object =
        Vector2_f(corners_2d[c].x, corners_2d[c].y) - motion_vectors[c].pos;

    motion_vector_pointers[c] = &motion_vectors[c];
  }

  const std::vector<float> weights(kQuadCornersSize, 1.0f);
  HomographyL2Solve(motion_vector_pointers, weights,
                    state->mutable_pnp_homography());
}

// Scales velocity and all other velocity dependent fields according to
// temporal_scale.
void ScaleStateTemporally(float temporal_scale, MotionBoxState* state) {
  state->set_dx(state->dx() * temporal_scale);
  state->set_dy(state->dy() * temporal_scale);
  state->set_kinetic_energy(state->kinetic_energy() * temporal_scale);
}

void ScaleStateAspect(float aspect, bool invert, MotionBoxState* state) {
  float scale_x = 1.0f;
  float scale_y = 1.0f;
  ScaleFromAspect(aspect, invert, &scale_x, &scale_y);

  if (state->has_quad() && state->quad().vertices_size() == 8) {
    for (int i = 0; i < 4; ++i) {
      float curr_val = state->quad().vertices(i * 2);
      state->mutable_quad()->set_vertices(i * 2, curr_val * scale_x);
      curr_val = state->quad().vertices(i * 2 + 1);
      state->mutable_quad()->set_vertices(i * 2 + 1, curr_val * scale_y);
    }
  }

  state->set_pos_x(state->pos_x() * scale_x);
  state->set_pos_y(state->pos_y() * scale_y);
  state->set_width(state->width() * scale_x);
  state->set_height(state->height() * scale_y);
  state->set_dx(state->dx() * scale_x);
  state->set_dy(state->dy() * scale_y);
  state->set_inlier_center_x(state->inlier_center_x() * scale_x);
  state->set_inlier_center_y(state->inlier_center_y() * scale_y);
  state->set_inlier_width(state->inlier_width() * scale_x);
  state->set_inlier_height(state->inlier_height() * scale_y);
}

MotionVector MotionVector::FromInternalState(
    const MotionBoxInternalState& internal, int index) {
  CHECK_LT(index, internal.pos_x_size());
  MotionVector v;
  v.pos = Vector2_f(internal.pos_x(index), internal.pos_y(index));
  v.object = Vector2_f(internal.dx(index), internal.dy(index));
  v.background =
      Vector2_f(internal.camera_dx(index), internal.camera_dy(index));
  v.track_id = internal.track_id(index);
  return v;
}

void MotionBox::ResetAtFrame(int frame, const MotionBoxState& state) {
  states_.clear();
  queue_start_ = frame;

  states_.push_back(state);
  states_.back().set_track_status(MotionBoxState::BOX_TRACKED);
  // Init inlier dimensions from state if not set.
  if (states_.back().inlier_width() == 0 ||
      states_.back().inlier_height() == 0) {
    states_.back().set_inlier_width(state.width());
    states_.back().set_inlier_height(state.height());
  }

  initial_state_ = state;
}

bool MotionBox::TrackStep(int from_frame,
                          const MotionVectorFrame& motion_vectors,
                          bool forward) {
  if (!TrackableFromFrame(from_frame)) {
    LOG(WARNING) << "Tracking requested for initial position that is not "
                 << "trackable.";
    return false;
  }
  const int queue_pos = from_frame - queue_start_;

  MotionBoxState new_state;
  if (motion_vectors.is_duplicated) {
    // Do not track or update the state, just copy.
    new_state = states_[queue_pos];
    new_state.set_track_status(MotionBoxState::BOX_DUPLICATED);
  } else {
    // Compile history and perform tracking.
    std::vector<const MotionBoxState*> history;
    constexpr int kHistorySize = 10;
    if (forward) {
      for (int k = queue_pos - 1; k >= std::max(0, queue_pos - kHistorySize);
           --k) {
        history.push_back(&states_[k]);
      }
    } else {
      for (int k = queue_pos + 1;
           k <= std::min<int>(states_.size() - 1, queue_pos + kHistorySize);
           ++k) {
        history.push_back(&states_[k]);
      }
    }

    TrackStepImpl(from_frame, states_[queue_pos], motion_vectors, history,
                  &new_state);
  }

  if (new_state.track_status() < MotionBoxState::BOX_TRACKED) {
    new_state.set_tracking_confidence(0.0f);
  }
  if (!new_state.has_tracking_confidence()) {
    // In this case, track status should be >= MotionBoxState::BOX_TRACKED
    new_state.set_tracking_confidence(1.0f);
  }

  VLOG(1) << "Track status from frame " << from_frame << ": "
          << TrackStatusToString(new_state.track_status())
          << ". Has quad: " << new_state.has_quad();

  constexpr float kFailureDisparity = 0.8f;
  if (new_state.track_status() >= MotionBoxState::BOX_TRACKED) {
    if (forward) {
      const int new_pos = queue_pos + 1;
      if (new_pos < states_.size()) {
        states_[new_pos] = new_state;
      } else {
        states_.push_back(new_state);
      }

      // Check for successive tracking failures of in bound boxes.
      if (new_pos >= options_.max_track_failures()) {
        int num_track_errors = 0;
        // Cancel at the N + 1 tracking failure.
        for (int f = new_pos - options_.max_track_failures(); f <= new_pos;
             ++f) {
          if (states_[f].track_status() !=
              MotionBoxState::BOX_TRACKED_OUT_OF_BOUND) {
            num_track_errors += (fabs(states_[f].motion_disparity()) *
                                     states_[f].prior_weight() >
                                 kFailureDisparity);
          }
        }

        if (num_track_errors >= options_.max_track_failures()) {
          LOG_IF(INFO, print_motion_box_warnings_)
              << "Tracking failed during max track failure "
              << "verification.";
          states_[new_pos].set_track_status(MotionBoxState::BOX_UNTRACKED);
          return false;
        }
      }
    } else {
      // Backward tracking.
      int new_pos = queue_pos - 1;
      if (new_pos >= 0) {
        states_[new_pos] = new_state;
      } else {
        states_.push_front(new_state);
        --queue_start_;
        new_pos = 0;
      }

      // Check for successive tracking failures.
      if (new_pos + options_.max_track_failures() + 1 < states_.size()) {
        int num_track_errors = 0;
        // Cancel at the N + 1 tracking failure.
        for (int f = new_pos; f <= new_pos + options_.max_track_failures();
             ++f) {
          if (states_[f].track_status() !=
              MotionBoxState::BOX_TRACKED_OUT_OF_BOUND) {
            num_track_errors += (fabs(states_[f].motion_disparity()) *
                                     states_[f].prior_weight() >
                                 kFailureDisparity);
          }
        }

        if (num_track_errors >= options_.max_track_failures()) {
          LOG_IF(INFO, print_motion_box_warnings_)
              << "Tracking failed during max track failure "
              << "verification.";
          states_[new_pos].set_track_status(MotionBoxState::BOX_UNTRACKED);
          return false;
        }
      }
    }

    // Signal track success.
    return true;
  } else {
    LOG_IF(WARNING, print_motion_box_warnings_)
        << "Tracking error at " << from_frame
        << " status : " << TrackStatusToString(new_state.track_status());
    return false;
  }
}

namespace {

Vector2_f SpatialPriorPosition(const Vector2_f& location,
                               const MotionBoxState& state) {
  const int grid_size = state.spatial_prior_grid_size();
  return Vector2_f(
      Clamp((location.x() - state.pos_x()) / state.width(), 0, 1) *
          (grid_size - 1),
      Clamp((location.y() - state.pos_y()) / state.height(), 0, 1) *
          (grid_size - 1));
}

// Creates spatial prior for current set of inlier vectors and blends
// it with previous prior (based on blend_prior). If interpolate is set to true,
// uses more accurate interpolation into bins, instead of nearest neighbor
// interpolation. If use_next_position is set to true, the position in
// the next/previous frame is used instead of the current one.
void ComputeSpatialPrior(bool interpolate, bool use_next_position,
                         float blend_prior, MotionBoxState* update_pos) {
  const int grid_size = update_pos->spatial_prior_grid_size();

  std::vector<float> old_prior(update_pos->spatial_prior().begin(),
                               update_pos->spatial_prior().end());
  std::vector<float> old_confidence(update_pos->spatial_confidence().begin(),
                                    update_pos->spatial_confidence().end());

  CHECK_EQ(old_confidence.size(), old_prior.size());
  CHECK(old_confidence.empty() ||
        grid_size * grid_size == old_confidence.size())
      << "Empty or priors of constant size expected";

  update_pos->clear_spatial_prior();
  update_pos->clear_spatial_confidence();
  auto* spatial_prior = update_pos->mutable_spatial_prior();
  auto* spatial_confidence = update_pos->mutable_spatial_confidence();
  spatial_prior->Reserve(grid_size * grid_size);
  spatial_confidence->Reserve(grid_size * grid_size);

  for (int k = 0; k < grid_size * grid_size; ++k) {
    spatial_prior->AddAlreadyReserved(0);
    spatial_confidence->AddAlreadyReserved(0);
  }

  // Aggregate inlier weights (0 = outlier, 1 = total inlier) across grid.
  const MotionBoxInternalState& internal = update_pos->internal();
  const int num_elems = internal.pos_x_size();

  for (int k = 0; k < num_elems; ++k) {
    MotionVector vec = MotionVector::FromInternalState(internal, k);
    const Vector2_f pos =
        use_next_position ? vec.MatchLocation() : vec.Location();
    float weight = internal.inlier_score(k);

    const Vector2_f grid_pos = SpatialPriorPosition(pos, *update_pos);

    if (use_next_position) {
      // Check for out of bound and skip.
      if (grid_pos.x() < 0 || grid_pos.y() < 0 ||
          grid_pos.x() > update_pos->spatial_prior_grid_size() - 1 ||
          grid_pos.y() > update_pos->spatial_prior_grid_size() - 1) {
        continue;
      }
    }

    if (interpolate) {
      const int int_x = static_cast<int>(grid_pos.x());
      const int int_y = static_cast<int>(grid_pos.y());

      CHECK_GE(grid_pos.x(), 0) << pos.x() << ", " << update_pos->pos_x();
      CHECK_GE(grid_pos.y(), 0);
      CHECK_LE(grid_pos.x(), grid_size - 1);
      CHECK_LE(grid_pos.y(), grid_size - 1);

      const float dx = grid_pos.x() - int_x;
      const float dy = grid_pos.y() - int_y;
      const float dx_1 = 1.0f - dx;
      const float dy_1 = 1.0f - dy;
      const int stride = static_cast<int>(dx != 0);

      int grid_pos = int_y * grid_size + int_x;

      // Bilinear interpolation. Total sum of weights across all 4 additions
      // (for prior and confidence each), is one.
      *spatial_prior->Mutable(grid_pos) += dx_1 * dy_1 * weight;
      *spatial_confidence->Mutable(grid_pos) += dx_1 * dy_1;

      *spatial_prior->Mutable(grid_pos + stride) += dx * dy_1 * weight;
      *spatial_confidence->Mutable(grid_pos + stride) += dx * dy_1;

      grid_pos += (dy != 0) * grid_size;
      *spatial_prior->Mutable(grid_pos) += dx_1 * dy * weight;
      *spatial_confidence->Mutable(grid_pos) += dx_1 * dy;

      *spatial_prior->Mutable(grid_pos + stride) += dx * dy * weight;
      *spatial_confidence->Mutable(grid_pos + stride) += dx * dy;
    } else {
      // Nearest neighbor.
      const int grid_bin = static_cast<int>(grid_pos.y() + 0.5f) * grid_size +
                           static_cast<int>(grid_pos.x() + 0.5f);
      *spatial_prior->Mutable(grid_bin) += weight;
      *spatial_confidence->Mutable(grid_bin) += 1;
    }
  }

  // Normalize, i.e. max truncation.
  float total_prior_difference = 0;
  float weight_sum = 0;
  for (int k = 0; k < grid_size * grid_size; ++k) {
    // Convert aggregated inlier weights to grid cell prior.
    // Here we consider a grid cell to be an inlier, if at least two
    // 2 inliers within that cell where found.
    *spatial_prior->Mutable(k) = std::min(1.0f, spatial_prior->Get(k) * 0.5f);
    *spatial_confidence->Mutable(k) =
        std::min(1.0f, spatial_confidence->Get(k) * 0.5f);

    if (!old_prior.empty()) {
      // Truncated error, consider a difference of 0.2 within normal
      // update range.
      const float difference = std::max<float>(
          0.f, fabs(update_pos->spatial_prior(k) - old_prior[k]) - 0.2f);
      // Weight error by confidence.
      total_prior_difference += difference * update_pos->spatial_confidence(k);
      weight_sum += update_pos->spatial_confidence(k);

      // Blend confidence with previous confidence.
      const float curr_confidence =
          update_pos->spatial_confidence(k) * (1.0f - blend_prior);
      const float prev_confidence = old_confidence[k] * blend_prior;

      float summed_confidence = curr_confidence + prev_confidence;
      const float denom =
          summed_confidence > 0 ? 1.0f / summed_confidence : 1.0f;

      // Update prior and confidence as weighted linear combination between
      // current and previous prior.
      *spatial_prior->Mutable(k) =
          (update_pos->spatial_prior(k) * curr_confidence +
           old_prior[k] * prev_confidence) *
          denom;

      *spatial_confidence->Mutable(k) =
          (update_pos->spatial_confidence(k) * curr_confidence +
           prev_confidence * prev_confidence) *
          denom;
    }
  }

  update_pos->set_prior_diff(std::sqrt(
      total_prior_difference * (weight_sum > 0 ? 1.0f / weight_sum : 1.0f)));
}

}  // namespace.

void MotionBox::GetStartPosition(const MotionBoxState& curr_pos,
                                 float aspect_ratio, float* expand_mag,
                                 Vector2_f* top_left,
                                 Vector2_f* bottom_right) const {
  CHECK(top_left);
  CHECK(bottom_right);
  CHECK(expand_mag);

  MotionBoxBoundingBox(curr_pos, top_left, bottom_right);

  if (curr_pos.has_pnp_homography()) {
    *expand_mag = 0.0f;
  } else {
    // Expaned box by the specified minimum expansion_size. For fast moving
    // objects, we ensure that magnitude is twice the box velocity, but not more
    // than 1/4 of the box diameter.
    *expand_mag = std::max(options_.expansion_size(),
                           std::min(MotionBoxSize(curr_pos).Norm() * 0.25f,
                                    MotionBoxVelocity(curr_pos).Norm() * 2.0f));
  }

  // Expansion magnitude is not non-uniformly scaled w.r.t. aspect ratio
  // to ensure inclusion test in GetVectorsAndWeights can assume uniform
  // explansion magnitude.
  const Vector2_f expand = Vector2_f(*expand_mag, *expand_mag);
  *top_left -= expand;
  *bottom_right += expand;
}

void MotionBox::GetSpatialGaussWeights(const MotionBoxState& box_state,
                                       const Vector2_f& inv_box_domain,
                                       float* spatial_gauss_x,
                                       float* spatial_gauss_y) const {
  CHECK(spatial_gauss_x);
  CHECK(spatial_gauss_y);

  // Space sigma depends on how much the tracked object fills the rectangle.
  // We get this information from the inlier extent of the previous
  // estimation.
  // Motivation: Choose sigma s such that the inlier domain equals 90% coverage.
  //             i.e. using z-score one sided of 95% = 1.65
  //             s * 1.65 = domain
  //          ==> s = domain / 1.65f
  const float space_sigma_x = std::max(
      options_.spatial_sigma(), box_state.inlier_width() * inv_box_domain.x() *
                                    0.5f * box_state.prior_weight() / 1.65f);
  const float space_sigma_y = options_.spatial_sigma();
  std::max(options_.spatial_sigma(), box_state.inlier_height() *
                                         inv_box_domain.y() * 0.5f *
                                         box_state.prior_weight() / 1.65f);

  *spatial_gauss_x = -0.5f / (space_sigma_x * space_sigma_x);
  *spatial_gauss_y = -0.5f / (space_sigma_y * space_sigma_y);
}

// Computes for each vector its 2D grid position for a grid spanning
// the domain top_left to bottom_right.
// Note: Passed vectors must lie within the domain or function will return false
// for error.
template <int kGridSize>
bool ComputeGridPositions(const Vector2_f& top_left,
                          const Vector2_f& bottom_right,
                          const std::vector<const MotionVector*>& vectors,
                          std::vector<Vector2_f>* grid_positions) {
  CHECK(grid_positions);

  // Slightly larger domain to avoid boundary issues.
  const Vector2_f inv_grid_domain(
      (1.0f - 1e-3f) / (bottom_right.x() - top_left.x()),
      (1.0f - 1e-3f) / (bottom_right.y() - top_left.y()));

  grid_positions->clear();
  grid_positions->reserve(vectors.size());
  for (const MotionVector* vec : vectors) {
    // Get grid position. Note grid is never rotated, but we only use it for
    // density estimation.
    const Vector2_f grid_pos =
        (vec->pos - top_left).MulComponents(inv_grid_domain) * (kGridSize - 1);
    if (grid_pos.x() < 0 || grid_pos.y() < 0 || grid_pos.x() > kGridSize ||
        grid_pos.y() > kGridSize) {
      return false;
    }

    grid_positions->push_back(grid_pos);
  }

  return true;
}

template <int kGridSize>
void AddToGrid(const Vector2_f& grid_pos, std::vector<float>* grid) {
  const float grid_x = grid_pos.x();
  const float grid_y = grid_pos.y();

  const int int_grid_x = grid_x;
  const int int_grid_y = grid_y;

  const float dx = grid_x - int_grid_x;
  const float dy = grid_y - int_grid_y;
  const float dxdy = dx * dy;
  const float dx_plus_dy = dx + dy;

  const int inc_x = dx != 0;
  const int inc_y = dy != 0;

  int bin_idx = int_grid_y * kGridSize + int_grid_x;
  // (1 - dx)(1 - dy) = 1 - (dx + dy) + dx*dy
  (*grid)[bin_idx] += 1 - dx_plus_dy + dxdy;
  // dx * (1 - dy) = dx - dxdy
  (*grid)[bin_idx + inc_x] += dx - dxdy;

  bin_idx += kGridSize * inc_y;
  // (1 - dx) * dy = dy - dxdy
  (*grid)[bin_idx] += dy - dxdy;
  (*grid)[bin_idx + inc_x] += dxdy;
}

template <int kGridSize>
float SampleFromGrid(const Vector2_f& grid_pos,
                     const std::vector<float>& grid) {
  const float grid_x = grid_pos.x();
  const float grid_y = grid_pos.y();

  const int int_grid_x = grid_x;
  const int int_grid_y = grid_y;

  const float dx = grid_x - int_grid_x;
  const float dy = grid_y - int_grid_y;
  const float dxdy = dx * dy;
  const float dx_plus_dy = dx + dy;
  int inc_x = dx != 0;
  int inc_y = dy != 0;

  float normalizer = 0;
  int bin_idx = int_grid_y * kGridSize + int_grid_x;

  // See above.
  normalizer += grid[bin_idx] * (1 - dx_plus_dy + dxdy);
  normalizer += grid[bin_idx + inc_x] * (dx - dxdy);

  bin_idx += kGridSize * inc_y;
  normalizer += grid[bin_idx] * (dy - dxdy);
  normalizer += grid[bin_idx + inc_x] * dxdy;

  const float inv_normalizer = normalizer > 0 ? 1.0f / normalizer : 0;
  // Density should always decrease weight; never increase.
  return std::min(1.0f, inv_normalizer);
}

MotionBox::DistanceWeightsComputer::DistanceWeightsComputer(
    const MotionBoxState& initial_state, const MotionBoxState& current_state,
    const TrackStepOptions& options) {
  tracking_degrees_ = options.tracking_degrees();
  const Vector2_f box_domain(current_state.width() * current_state.scale(),
                             current_state.height() * current_state.scale());
  CHECK_GT(box_domain.x(), 0.0f);
  CHECK_GT(box_domain.y(), 0.0f);
  inv_box_domain_ = Vector2_f(1.0f / box_domain.x(), 1.0f / box_domain.y());

  // Space sigma depends on how much the tracked object fills the rectangle.
  // We get this information from the inlier extent of the previous
  // estimation.
  // Motivation: Choose sigma s such that the inlier domain equals 90%
  // coverage.
  //             i.e. using z-score one sided of 95% = 1.65
  //             s * 1.65 = domain
  //          ==> s = domain / 1.65f
  const float space_sigma_x =
      std::max(options.spatial_sigma(),
               current_state.inlier_width() * inv_box_domain_.x() * 0.5f *
                   current_state.prior_weight() / 1.65f);

  const float space_sigma_y =
      std::max(options.spatial_sigma(),
               current_state.inlier_height() * inv_box_domain_.y() * 0.5f *
                   current_state.prior_weight() / 1.65f);

  spatial_gauss_x_ = -0.5f / (space_sigma_x * space_sigma_x);
  spatial_gauss_y_ = -0.5f / (space_sigma_y * space_sigma_y);

  if (tracking_degrees_ == TrackStepOptions::TRACKING_DEGREE_OBJECT_ROTATION ||
      tracking_degrees_ ==
          TrackStepOptions::TRACKING_DEGREE_OBJECT_ROTATION_SCALE) {
    cos_neg_a_ = std::cos(-current_state.rotation());
    sin_neg_a_ = std::sin(-current_state.rotation());
    if (std::abs(current_state.rotation()) > 0.01f) {
      is_large_rotation_ = true;
    }
  }

  // Compute box center as blend between geometric center and inlier center.
  constexpr float kMaxBoxCenterBlendWeight = 0.5f;
  box_center_ =
      Lerp(MotionBoxCenter(current_state), InlierCenter(current_state),
           std::min(kMaxBoxCenterBlendWeight, current_state.prior_weight()));
  if (tracking_degrees_ ==
      TrackStepOptions::TRACKING_DEGREE_OBJECT_PERSPECTIVE) {
    CHECK(initial_state.has_quad());
    CHECK(current_state.has_quad());
    homography_ =
        ComputeHomographyFromQuad(current_state.quad(), initial_state.quad());
    box_center_transformed_ =
        HomographyAdapter::TransformPoint(homography_, box_center_);
  }
}

float MotionBox::DistanceWeightsComputer::ComputeDistanceWeight(
    const MotionVector& test_vector) {
  // Distance weighting.
  Vector2_f diff_center;
  if (tracking_degrees_ ==
      TrackStepOptions::TRACKING_DEGREE_OBJECT_PERSPECTIVE) {
    Vector2_f test_vector_transformed =
        HomographyAdapter::TransformPoint(homography_, test_vector.pos);
    diff_center = test_vector_transformed - box_center_transformed_;
  } else {
    diff_center = test_vector.pos - box_center_;
    if (is_large_rotation_) {
      // Rotate difference vector to normalized domain.
      diff_center.Set(
          cos_neg_a_ * diff_center.x() - sin_neg_a_ * diff_center.y(),
          sin_neg_a_ * diff_center.x() + cos_neg_a_ * diff_center.y());
    }
  }

  const Vector2_f diff = diff_center.MulComponents(inv_box_domain_);
  // Regular gaussian with variance in each direction, assuming directions
  // are indpendent.
  float weight = std::exp(diff.x() * diff.x() * spatial_gauss_x_ +
                          diff.y() * diff.y() * spatial_gauss_y_);
  return weight;
}

Homography MotionBox::DistanceWeightsComputer::ComputeHomographyFromQuad(
    const MotionBoxState::Quad& src_quad,
    const MotionBoxState::Quad& dst_quad) {
  std::vector<float> src_quad_vec(8);
  std::vector<float> dst_quad_vec(8);
  for (int i = 0; i < 8; ++i) {
    src_quad_vec[i] = src_quad.vertices(i);
    dst_quad_vec[i] = dst_quad.vertices(i);
  }
  // Construct the matrix
  Eigen::Matrix<float, 8, 8> A = Eigen::Matrix<float, 8, 8>::Zero();
  for (int i = 0; i < 4; ++i) {
    const int r0 = 2 * i;
    const int r1 = 2 * i + 1;
    A(r0, 0) = src_quad_vec[r0];
    A(r0, 1) = src_quad_vec[r1];
    A(r0, 2) = 1.f;
    A(r0, 6) = -src_quad_vec[r0] * dst_quad_vec[r0];
    A(r0, 7) = -src_quad_vec[r1] * dst_quad_vec[r0];
    A(r1, 3) = src_quad_vec[r0];
    A(r1, 4) = src_quad_vec[r1];
    A(r1, 5) = 1.f;
    A(r1, 6) = -src_quad_vec[r0] * dst_quad_vec[r1];
    A(r1, 7) = -src_quad_vec[r1] * dst_quad_vec[r1];
  }

  // Map arrays to Eigen vectors without memcpy
  std::vector<float> homography_vector(8);
  Eigen::Map<Eigen::Matrix<float, 8, 1> > x(homography_vector.data());
  Eigen::Map<const Eigen::Matrix<float, 8, 1> > b(dst_quad_vec.data());

  x = A.fullPivLu().solve(b);
  Homography homography;
  homography.set_h_00(homography_vector[0]);
  homography.set_h_01(homography_vector[1]);
  homography.set_h_02(homography_vector[2]);
  homography.set_h_10(homography_vector[3]);
  homography.set_h_11(homography_vector[4]);
  homography.set_h_12(homography_vector[5]);
  homography.set_h_20(homography_vector[6]);
  homography.set_h_21(homography_vector[7]);
  return homography;
}

bool MotionBox::GetVectorsAndWeights(
    const std::vector<MotionVector>& motion_vectors, int start_idx, int end_idx,
    const Vector2_f& top_left, const Vector2_f& bottom_right,
    const MotionBoxState& box_state, bool valid_background_model,
    bool is_chunk_boundary, float temporal_scale, float expand_mag,
    const std::vector<const MotionBoxState*>& history,
    std::vector<const MotionVector*>* vectors, std::vector<float>* weights,
    int* number_of_good_prior, int* number_of_cont_inliers) const {
  CHECK(weights);
  CHECK(vectors);
  CHECK(number_of_good_prior);
  CHECK(number_of_cont_inliers);

  const int num_max_vectors = end_idx - start_idx;
  weights->clear();
  vectors->clear();
  weights->reserve(num_max_vectors);
  vectors->reserve(num_max_vectors);

  const Vector2_f box_domain(box_state.width() * box_state.scale(),
                             box_state.height() * box_state.scale());

  CHECK_GT(box_domain.x(), 0.0f);
  CHECK_GT(box_domain.y(), 0.0f);
  const Vector2_f inv_box_domain(1.0f / box_domain.x(), 1.0f / box_domain.y());

  // The four lines of the rotated and scaled box.
  std::array<Vector3_f, 4> box_lines;
  if (!MotionBoxLines(box_state, Vector2_f(1.0f, 1.0f), &box_lines)) {
    LOG(ERROR) << "Error in computing MotionBoxLines. Return 0 good inits and "
                  "continued inliers";
    return false;
  }

  // Get list of previous tracking inliers and outliers.
  // Ids are used for non-chunk boundaries (faster matching), locations
  // for chunk boundaries.
  std::unordered_map<int, int> inlier_ids;
  std::unordered_set<int> outlier_ids;
  std::vector<Vector2_f> inlier_locations;
  std::vector<Vector2_f> outlier_locations;

  if (!is_chunk_boundary) {
    MotionBoxInliers(box_state, &inlier_ids);
    MotionBoxOutliers(box_state, &outlier_ids);

    // Never map ids in history across a chunk boundary.
    for (const auto* state_ptr : history) {
      MotionBoxOutliers(*state_ptr, &outlier_ids);
    }
    // Why don't we build inlier map from a history of inliers?
    // It is unlikely we skip a feature as an inlier, it is either
    // consistently part of the motion model or it is not.
  } else {
    MotionBoxInlierLocations(box_state, &inlier_locations);
    MotionBoxOutlierLocations(box_state, &outlier_locations);
  }

  // Indicator for each vector, if inlier or outlier from prev. estimation.
  std::vector<uchar> is_inlier;
  std::vector<uchar> is_outlier;
  is_inlier.reserve(num_max_vectors);
  is_outlier.reserve(num_max_vectors);
  int num_cont_inliers = 0;

  // Approx. 2 pix at SD resolution.
  constexpr float kSqProximity = 2e-3 * 2e-3;

  for (int k = start_idx; k < end_idx; ++k) {
    // x is within bound due to sorting.
    const MotionVector& test_vector = motion_vectors[k];

    if (test_vector.pos.y() < top_left.y() ||
        test_vector.pos.y() > bottom_right.y()) {
      continue;
    }

    if (std::abs(box_state.rotation()) > 0.01f ||
        options_.tracking_degrees() ==
            TrackStepOptions::TRACKING_DEGREE_OBJECT_PERSPECTIVE) {
      // Test also if vector is within transformed convex area.
      bool accepted = true;
      for (const Vector3_f& line : box_lines) {
        if (line.DotProd(Vector3_f(test_vector.pos.x(), test_vector.pos.y(),
                                   1.0f)) > expand_mag) {
          // Outside, reject.
          accepted = false;
          break;
        }
      }
      if (!accepted) {
        continue;
      }
    }

    vectors->push_back(&motion_vectors[k]);

    auto is_close_to_test_vector = [test_vector](const Vector2_f v) -> bool {
      return (v - test_vector.pos).Norm2() < kSqProximity;
    };

    const bool is_inlier_flag =
        inlier_ids.find(test_vector.track_id) != inlier_ids.end() ||
        std::find_if(inlier_locations.begin(), inlier_locations.end(),
                     is_close_to_test_vector) != inlier_locations.end();
    num_cont_inliers += is_inlier_flag;

    const bool is_outlier_flag =
        outlier_ids.find(test_vector.track_id) != outlier_ids.end() ||
        std::find_if(outlier_locations.begin(), outlier_locations.end(),
                     is_close_to_test_vector) != outlier_locations.end();

    is_inlier.push_back(is_inlier_flag);
    is_outlier.push_back(is_outlier_flag);
  }

  CHECK_EQ(vectors->size(), is_inlier.size());
  CHECK_EQ(vectors->size(), is_outlier.size());

  const float prev_motion_mag = MotionBoxVelocity(box_state).Norm();

  // Try to lock on object again, if disparity is high.
  constexpr float kMinPriorMotionWeight = 0.2f;
  const float prior_motion_weight =
      std::max(kMinPriorMotionWeight, std::abs(box_state.motion_disparity())) *
      box_state.prior_weight();

  const float motion_sigma =
      std::max<float>(options_.min_motion_sigma(),
                      prev_motion_mag * options_.relative_motion_sigma());
  const float motion_gaussian_scale = -0.5f / (motion_sigma * motion_sigma);

  // Maps current kinetic energy to [0, 1] quantifying static (0) vs. moving (1)
  // object.
  // Map normalized thresholds to current frame period.
  const float low_kinetic_energy =
      options_.low_kinetic_energy() * temporal_scale;
  const float high_kinetic_energy =
      options_.high_kinetic_energy() * temporal_scale;
  const float kinetic_identity = LinearRamp(
      box_state.kinetic_energy(), low_kinetic_energy, high_kinetic_energy);
  int num_good_inits = 0;

  // Map number of continued inliers to score in [0, 1].
  const float cont_inlier_score = LinearRamp(num_cont_inliers, 10, 30);

  VLOG(1) << "GetVectorsAndWeights, found cont. inliers: " << num_cont_inliers
          << "  score: " << cont_inlier_score;

  DistanceWeightsComputer distance_weights_computer(initial_state_, box_state,
                                                    options_);
  for (int k = 0; k < vectors->size(); ++k) {
    const MotionVector& test_vector = *(*vectors)[k];

    float weight = distance_weights_computer.ComputeDistanceWeight(test_vector);

    if (valid_background_model) {
      const float motion_diff =
          fabs(prev_motion_mag - test_vector.object.Norm());
      const float motion_weight =
          std::exp(motion_gaussian_scale * motion_diff * motion_diff);

      // Blend with spatial weight, that is the higher the disparity
      // (i.e. we lost tracking, the more inclined we are to lock
      //       onto vectors of similar motion magnitude regardless of their
      //       position).
      // Note: One might feel inclined to always bias towards the previous
      // motion, by multiplying weight with motion_weight. This however fails
      // when tracking objects that start at rest and start moving.
      weight = Lerp(weight, motion_weight, prior_motion_weight);
    }

    // There are two kinds of vectors we are trying to balance here:
    // - inliers from previous estimation
    // - similar vectors
    //
    // Current strategy:
    // - For static objects: Boost inliers a lot, discount outliers a lot,
    // do not care about similar vectors.
    // - For moving objects: Boost inliers proportional to number of continued
    // inliers, discount outliers a lot, boost similar vectors and
    // actively downweight dis-similar objects.
    //
    // Motivation: Inliers are usually not very stable, so if not sufficient
    // have been continued prefer velocity over inliers for moving objects.

    // NOTE: Regarding additive vs. multiplicative weights. We need to
    // multiply the weight here. Adding the weight messes
    // with the gaussian spatial weighting which in turn makes it hard to lock
    // onto moving objects in the first place (as center is assumed to be
    // placed over moving objects, this helps distinguish initial foreground
    // and background).

    // Up-weighting of inlier vectors and vectors of similar motion.
    float upweight = 1.0f;
    if (is_inlier[k]) {
      // Previous track, boost weight significantly.
      //
      // NOTE: Regarding the amount of up-weighting: Long features are not
      // very stable on moving objects. Therefore only upweight strongly for
      // static objects.
      constexpr float kWeakMultiplier = 5.0f;
      constexpr float kStrongMultiplier = 20.0f;

      // Map 0 -> 1 and values >= 0.5 -> 0, because long features are not
      // very stable on moving objects. Therefore only upweight strongly for
      // static objects.
      const float kinetic_alpha =
          std::max(0.0f, 1.0f - 2.0f * kinetic_identity);

      // Choose strong multiplier only when kinetic_alpha OR inlier score
      // support it.
      const float multiplier = Lerp(kWeakMultiplier, kStrongMultiplier,
                                    std::max(cont_inlier_score, kinetic_alpha));
      upweight *= multiplier;
    }

    // Scale weight boost for moving objects by prior, i.e. modulate
    // strength of scale w.r.t. confidence.
    const float kin_scale = Lerp(1, 10, box_state.prior_weight());
    // 80% moving object weighted by prior. This weighting is biasing towards
    // moving object when the prior is low.
    if (kinetic_identity >= 0.8f * box_state.prior_weight() &&
        test_vector.object.Norm() > high_kinetic_energy && !is_outlier[k]) {
      // If we track a moving object, long tracks are less likely to be stable
      // due to appearance variations. In that case boost similar vectors.
      upweight *= 5.f * kin_scale;
    }

    float downweight = 1.0f;
    // Down-weighting of outlier vectors and vectors of different motion.
    if (!is_inlier[k]) {
      // Outlier.
      if (is_outlier[k]) {
        // Note: Outlier ids might overlap with inliers as outliers are built
        // from a history of frames.
        // *Always favor inliers over outliers*! Important to keep!!
        downweight *= 20.0f;
      }

      // Vectors of different motion, for 100% moving object downweight
      // vectors with small motion.
      if (kinetic_identity >= 1.0f * box_state.prior_weight() &&
          test_vector.object.Norm() < low_kinetic_energy) {
        downweight *= 2.f * kin_scale;
      }
    }

    // Cap any kind of up or down weighting so that no vector overwhelms
    // all others.
    const float kMaxWeight = 100.f;
    upweight = std::min(kMaxWeight, upweight);
    downweight = std::min(kMaxWeight, downweight);
    weight *= upweight / downweight;

    num_good_inits += (weight >= 0.1f);
    weights->push_back(weight);
  }

  const int num_vectors = vectors->size();
  CHECK_EQ(num_vectors, weights->size());

  const float weight_sum =
      std::accumulate(weights->begin(), weights->end(), 0.0f);

  // Normalize weights.
  if (weight_sum > 0) {
    const float inv_weight_sum = 1.0f / weight_sum;
    for (auto& weight : *weights) {
      weight *= inv_weight_sum;
    }
  }

  *number_of_good_prior = num_good_inits;
  *number_of_cont_inliers = num_cont_inliers;

  return true;
}

void MotionBox::TranslationIrlsInitialization(
    const std::vector<const MotionVector*>& vectors,
    const Vector2_f& irls_scale, std::vector<float>* weights) const {
  const int num_features = vectors.size();

  const auto& irls_options = options_.irls_initialization();
  if (!irls_options.activated() || !num_features) {
    return;
  }

  // Bool indicator which features agree with model in each round.
  // In case no RANSAC rounds are performed considered all features inliers.
  std::vector<uint8> best_features(num_features, 1);
  std::vector<uint8> curr_features(num_features);
  float best_sum = 0;

  unsigned int seed = 900913;
  std::default_random_engine rand_gen(seed);
  std::uniform_int_distribution<> distribution(0, num_features - 1);

  const float cutoff = irls_options.cutoff();
  const float sq_cutoff = cutoff * cutoff;

  for (int rounds = 0; rounds < irls_options.rounds(); ++rounds) {
    float curr_sum = 0;
    // Pick a random vector.
    const int rand_idx = distribution(rand_gen);
    const Vector2_f flow = vectors[rand_idx]->object;
    const auto error_system = ComputeIrlsErrorSystem(irls_scale, flow);

    // curr_features gets set for every feature below; no need to reset.
    for (int i = 0; i < num_features; ++i) {
      const Vector2_f diff = vectors[i]->object - flow;
      const float error = ErrorDiff(diff, error_system);
      curr_features[i] = static_cast<uint8>(error < sq_cutoff);
      if (curr_features[i]) {
        curr_sum += (*weights)[i];
      }
    }

    if (curr_sum > best_sum) {
      best_sum = curr_sum;
      std::swap(best_features, curr_features);
    }
  }

  std::vector<float> inlier_weights;
  inlier_weights.reserve(num_features);

  // Score outliers low.
  int num_inliers = 0;
  for (int i = 0; i < num_features; ++i) {
    if (best_features[i] == 0) {
      (*weights)[i] = 1e-10f;
    } else {
      ++num_inliers;
      inlier_weights.push_back((*weights)[i]);
    }
  }

  if (!inlier_weights.empty()) {
    // Ensure that all selected inlier features have at least median weight.
    auto median = inlier_weights.begin() + inlier_weights.size() * 0.5f;
    std::nth_element(inlier_weights.begin(), median, inlier_weights.end());

    for (int i = 0; i < num_features; ++i) {
      if (best_features[i] != 0) {
        (*weights)[i] = std::max(*median, (*weights)[i]);
      }
    }
  }
}

void MotionBox::EstimateObjectMotion(
    const std::vector<const MotionVector*>& motion_vectors,
    const std::vector<float>& prior_weights, int num_continued_inliers,
    const Vector2_f& irls_scale, std::vector<float>* weights,
    Vector2_f* object_translation, LinearSimilarityModel* object_similarity,
    Homography* object_homography) const {
  CHECK(object_translation);
  CHECK(object_similarity);
  CHECK(object_homography);

  const int num_vectors = motion_vectors.size();
  CHECK_EQ(num_vectors, prior_weights.size());
  CHECK_EQ(num_vectors, weights->size());

  // Create backup of weights if needed.
  std::vector<float> similarity_weights;

  switch (options_.tracking_degrees()) {
    case TrackStepOptions::TRACKING_DEGREE_OBJECT_ROTATION:
    case TrackStepOptions::TRACKING_DEGREE_OBJECT_SCALE:
    case TrackStepOptions::TRACKING_DEGREE_OBJECT_ROTATION_SCALE:
    case TrackStepOptions::TRACKING_DEGREE_OBJECT_PERSPECTIVE:
      similarity_weights = *weights;
      break;

    default:
      // Nothing to do here.
      break;
  }

  EstimateTranslation(motion_vectors, prior_weights, irls_scale, weights,
                      object_translation);

  TranslationModel translation_model = TranslationAdapter::FromArgs(
      object_translation->x(), object_translation->y());

  // For any additional degrees of freedom, require a good set of inliers.
  if (num_continued_inliers < options_.object_similarity_min_contd_inliers()) {
    VLOG_IF(2, options_.tracking_degrees() !=
                   TrackStepOptions::TRACKING_DEGREE_TRANSLATION)
        << "Falling back to translation!!!";
    VLOG(1) << "num_continued_inliers: " << num_continued_inliers << " < "
            << options_.object_similarity_min_contd_inliers()
            << ", fall back to translation";
    *object_similarity = LinearSimilarityAdapter::Embed(translation_model);
    *object_homography = HomographyAdapter::Embed(translation_model);
    return;
  }

  switch (options_.tracking_degrees()) {
    case TrackStepOptions::TRACKING_DEGREE_OBJECT_ROTATION:
    case TrackStepOptions::TRACKING_DEGREE_OBJECT_SCALE:
    case TrackStepOptions::TRACKING_DEGREE_OBJECT_ROTATION_SCALE: {
      if (EstimateSimilarity(motion_vectors, prior_weights, irls_scale,
                             &similarity_weights, object_similarity)) {
        if (!ObjectMotionValidator::IsValidSimilarity(
                *object_similarity, options_.box_similarity_max_scale(),
                options_.box_similarity_max_rotation())) {
          LOG(WARNING) << "Unstable similarity model - falling back to "
                       << "translation.";
          *object_similarity =
              LinearSimilarityAdapter::Embed(translation_model);
        } else {
          // Good estimation, use weights as output.
          weights->swap(similarity_weights);
        }
      } else {
        *object_similarity = LinearSimilarityAdapter::Embed(translation_model);
      }

      break;
    }
    case TrackStepOptions::TRACKING_DEGREE_OBJECT_PERSPECTIVE:
      if (EstimateHomography(motion_vectors, prior_weights, irls_scale,
                             &similarity_weights, object_homography)) {
        if (!ObjectMotionValidator::IsValidHomography(
                *object_homography, options_.quad_homography_max_scale(),
                options_.quad_homography_max_rotation())) {
          LOG(WARNING) << "Unstable homography model - falling back to "
                       << "translation.";
          *object_homography = HomographyAdapter::Embed(translation_model);
        } else {
          weights->swap(similarity_weights);
        }
      } else {
        *object_homography = HomographyAdapter::Embed(translation_model);
      }
      VLOG(1) << "Got homography: "
              << HomographyAdapter::ToString(*object_homography);
      break;
    default:
      // Plenty of CAMERA_ cases are not handled in this function.
      break;
  }
}

void MotionBox::EstimateTranslation(
    const std::vector<const MotionVector*>& motion_vectors,
    const std::vector<float>& prior_weights, const Vector2_f& irls_scale,
    std::vector<float>* weights, Vector2_f* translation) const {
  CHECK(weights);
  CHECK(translation);

  const int iterations = options_.irls_iterations();

  // NOTE: Floating point accuracy is totally sufficient here.
  //                 We tried changing to double now 3 times and it just does
  //                 not matter. Do not do it again.      - Past self
  Vector2_f object_translation;
  const int num_vectors = motion_vectors.size();
  const float kEpsilon = 1e-8f;

  VLOG(1) << "Estimating translation for " << num_vectors << " vectors";

  for (int i = 0; i < iterations; ++i) {
    float flow_sum = 0;
    object_translation = Vector2_f(0.0, 0.0);
    for (int k = 0; k < num_vectors; ++k) {
      const MotionVector& motion_vector = *motion_vectors[k];
      const Vector2_f flow = motion_vector.object;
      object_translation += flow * (*weights)[k];
      flow_sum += (*weights)[k];
    }

    if (flow_sum > 0) {
      object_translation *= (1.0 / flow_sum);

      const auto error_system =
          ComputeIrlsErrorSystem(irls_scale, object_translation);

      // Update irls weights.
      for (int k = 0; k < num_vectors; ++k) {
        const MotionVector& motion_vector = *motion_vectors[k];
        Vector2_f diff(motion_vector.object - object_translation);
        const float error = ErrorDiff(diff, error_system);
        // In last iteration compute weights without any prior bias.
        const float numerator = (i + 1 == iterations) ? 1.0f : prior_weights[k];
        (*weights)[k] = numerator / (error + kEpsilon);
      }
    }
  }
  *translation = object_translation;

  VLOG(1) << "Got translation: " << *translation;
}

bool MotionBox::EstimateSimilarity(
    const std::vector<const MotionVector*>& motion_vectors,
    const std::vector<float>& prior_weights, const Vector2_f& irls_scale,
    std::vector<float>* weights, LinearSimilarityModel* lin_sim) const {
  CHECK(weights);
  CHECK(lin_sim);

  const int iterations = options_.irls_iterations();
  LinearSimilarityModel object_similarity;
  const int num_vectors = motion_vectors.size();
  const float kEpsilon = 1e-8f;

  VLOG(1) << "Estimating similarity for " << num_vectors << " vectors";
  for (int i = 0; i < iterations; ++i) {
    if (LinearSimilarityL2Solve(motion_vectors, *weights, &object_similarity)) {
      // Update irls weights.
      for (int k = 0; k < num_vectors; ++k) {
        const MotionVector& motion_vector = *motion_vectors[k];
        const Vector2_f model_vec =
            TransformPoint(object_similarity, motion_vector.pos) -
            motion_vector.pos;
        const auto error_system = ComputeIrlsErrorSystem(irls_scale, model_vec);

        Vector2_f diff(motion_vector.object - model_vec);
        const float error = ErrorDiff(diff, error_system);
        // In last iteration compute weights without any prior bias.
        const float numerator = (i + 1 == iterations) ? 1.0f : prior_weights[k];
        (*weights)[k] = numerator / (error + kEpsilon);
      }
    } else {
      return false;
    }
  }
  *lin_sim = object_similarity;

  VLOG(1) << "Got similarity: "
          << LinearSimilarityAdapter::ToString(object_similarity);
  return true;
}

bool MotionBox::EstimateHomography(
    const std::vector<const MotionVector*>& motion_vectors,
    const std::vector<float>& prior_weights, const Vector2_f& irls_scale,
    std::vector<float>* weights, Homography* object_homography) const {
  CHECK(weights);

  const int iterations = options_.irls_iterations();
  Homography homography;
  const int num_vectors = motion_vectors.size();
  const float kEpsilon = 1e-8f;

  VLOG(1) << "Estimating homography for " << num_vectors << " vectors";
  for (int i = 0; i < iterations; ++i) {
    if (HomographyL2Solve(motion_vectors, *weights, &homography)) {
      // Update irls weights.
      for (int k = 0; k < num_vectors; ++k) {
        const MotionVector& motion_vector = *motion_vectors[k];
        const Vector2_f model_vec =
            TransformPoint(homography, motion_vector.pos) - motion_vector.pos;
        const auto error_system = ComputeIrlsErrorSystem(irls_scale, model_vec);

        Vector2_f diff(motion_vector.object - model_vec);
        const float error = ErrorDiff(diff, error_system);
        // In last iteration compute weights without any prior bias.
        const float numerator = (i + 1 == iterations) ? 1.0f : prior_weights[k];
        (*weights)[k] = numerator / (error + kEpsilon);
      }
    } else {
      return false;
    }
  }
  *object_homography = homography;

  return true;
}

bool MotionBox::EstimatePnpHomography(
    const MotionBoxState& curr_pos,
    const std::vector<const MotionVector*>& motion_vectors,
    const std::vector<float>& weights, float domain_x, float domain_y,
    Homography* pnp_homography) const {
  constexpr int kMinVectors = 4;
  if (motion_vectors.size() < kMinVectors) return false;

  Homography inv_h = HomographyAdapter::Invert(curr_pos.pnp_homography());

  std::vector<cv::Point3f> vectors_3d;
  vectors_3d.reserve(motion_vectors.size());
  std::vector<cv::Point2f> vectors_2d;
  vectors_2d.reserve(motion_vectors.size());
  if (options_.has_camera_intrinsics()) {
    const auto& camera = options_.camera_intrinsics();
    cv::Mat camera_mat, dist_coef;
    ConvertCameraIntrinsicsToCvMat(camera, &camera_mat, &dist_coef);
    float scale = std::max(camera.w(), camera.h());

    std::vector<cv::Point2f> mv_p;
    mv_p.reserve(motion_vectors.size());
    std::vector<cv::Point2f> mv_q;
    mv_q.reserve(motion_vectors.size());
    for (int j = 0; j < motion_vectors.size(); ++j) {
      if (weights[j] < kMaxOutlierWeight) continue;
      mv_p.emplace_back(motion_vectors[j]->pos.x() * scale,
                        motion_vectors[j]->pos.y() * scale);

      Vector2_f q = motion_vectors[j]->pos + motion_vectors[j]->object +
                    motion_vectors[j]->background;
      mv_q.emplace_back(q.x() * scale, q.y() * scale);
    }

    if (mv_p.size() < kMinVectors) return false;

    cv::undistortPoints(mv_p, mv_p, camera_mat, dist_coef);
    cv::undistortPoints(mv_q, mv_q, camera_mat, dist_coef);

    vectors_3d.resize(mv_p.size());
    vectors_2d.resize(mv_q.size());
    for (int j = 0; j < mv_p.size(); ++j) {
      Vector2_f p = TransformPoint(inv_h, Vector2_f(mv_p[j].x, mv_p[j].y));
      vectors_3d[j] = cv::Point3f(p.x(), p.y(), 0.0f);
      vectors_2d[j] = cv::Point2f(mv_q[j].x, mv_q[j].y);
    }
  } else {
    Vector2_f center(domain_x * 0.5f, domain_y * 0.5f);
    for (int j = 0; j < motion_vectors.size(); ++j) {
      if (weights[j] < kMaxOutlierWeight) continue;
      Vector2_f p = TransformPoint(inv_h, motion_vectors[j]->pos - center);
      vectors_3d.emplace_back(p.x(), p.y(), 0.0f);

      Vector2_f q = motion_vectors[j]->pos + motion_vectors[j]->object +
                    motion_vectors[j]->background - center;
      vectors_2d.emplace_back(q.x(), q.y());
    }

    if (vectors_3d.size() < kMinVectors) return false;
  }

  // TODO: use previous rvec and tvec to initilize the solver.
  cv::Mat rvec, tvec;
  cv::solvePnP(vectors_3d, vectors_2d,
               cv::Mat::eye(3, 3, CV_64F) /* camera_mat */,
               cv::Mat::zeros(1, 5, CV_64FC1) /* dist_coef */, rvec, tvec);
  *pnp_homography = PnpHomographyFromRotationAndTranslation(rvec, tvec);

  return true;
}

void MotionBox::ApplyObjectMotionPerspectively(const MotionBoxState& curr_pos,
                                               const Homography& pnp_homography,
                                               float domain_x, float domain_y,
                                               MotionBoxState* next_pos) const {
  const float half_width = curr_pos.aspect_ratio();
  const float half_height = 1.0f;

  constexpr int kQuadCornersSize = 4;

  // Omitting the 3rd dimension because they are all zeros.
  const std::vector<Vector2_f> corners_3d{
      Vector2_f(-half_width, -half_height),
      Vector2_f(-half_width, half_height),
      Vector2_f(half_width, half_height),
      Vector2_f(half_width, -half_height),
  };

  std::vector<Vector2_f> corners_2d(kQuadCornersSize);
  for (int c = 0; c < kQuadCornersSize; ++c) {
    corners_2d[c] =
        HomographyAdapter::TransformPoint(pnp_homography, corners_3d[c]);
  }

  if (options_.has_camera_intrinsics()) {
    std::vector<cv::Point3f> cv_points(4);
    for (int c = 0; c < kQuadCornersSize; ++c) {
      cv_points[c] = cv::Point3f(corners_2d[c].x(), corners_2d[c].y(), 1.0);
    }

    const auto& camera = options_.camera_intrinsics();
    cv::Mat camera_mat, dist_coef;
    ConvertCameraIntrinsicsToCvMat(camera, &camera_mat, &dist_coef);
    cv::Mat dummy_zeros = cv::Mat::zeros(1, 3, CV_64FC1);
    std::vector<cv::Point2f> cv_points_distorted;
    cv::projectPoints(cv_points, dummy_zeros /* rvec */, dummy_zeros /* tvec */,
                      camera_mat, dist_coef, cv_points_distorted);
    const float scale = 1.0f / std::max(camera.w(), camera.h());
    for (int c = 0; c < kQuadCornersSize; ++c) {
      next_pos->mutable_quad()->set_vertices(c * 2,
                                             cv_points_distorted[c].x * scale);
      next_pos->mutable_quad()->set_vertices(c * 2 + 1,
                                             cv_points_distorted[c].y * scale);
    }
  } else {
    const float center_x = domain_x * 0.5f;
    const float center_y = domain_y * 0.5f;
    for (int c = 0; c < kQuadCornersSize; ++c) {
      next_pos->mutable_quad()->set_vertices(c * 2,
                                             corners_2d[c].x() + center_x);
      next_pos->mutable_quad()->set_vertices(c * 2 + 1,
                                             corners_2d[c].y() + center_y);
    }
  }

  *next_pos->mutable_pnp_homography() = pnp_homography;
  UpdateStatePositionAndSizeFromStateQuad(next_pos);
}

float MotionBox::ComputeMotionDisparity(
    const MotionBoxState& curr_pos, const Vector2_f& irls_scale,
    float continued_inliers, int num_inliers,
    const Vector2_f& object_translation) const {
  // Motion disparity does not take into account change of direction,
  // only use parallel irls scale.
  const float curr_velocity = MotionBoxVelocity(curr_pos).Norm();
  const float sign = object_translation.Norm() < curr_velocity ? -1.0f : 1.0f;
  const float motion_diff = fabs(object_translation.Norm() - curr_velocity);

  // Score difference.
  const float measured_motion_disparity = LinearRamp(
      motion_diff * irls_scale.x(), options_.motion_disparity_low_level(),
      options_.motion_disparity_high_level());

  // Cap disparity measurement by inlier ratio, to account for objects
  // suddenly stopping/accelerating. In this case measured disparity might be
  // high, whereas inliers continue to be tracked.
  const float max_disparity = 1.0f - continued_inliers;

  const float capped_disparity =
      std::min(max_disparity, measured_motion_disparity);

  // Take into account large disparity in previous frames. Score by prior of
  // previous motion.
  const float motion_disparity =
      std::max(curr_pos.motion_disparity() * options_.disparity_decay(),
               capped_disparity) *
      curr_pos.prior_weight();

  // Map number of inliers to score in [0, 1], assuming a lot of inliers
  // indicate lock onto object.
  const float inlier_score = LinearRamp(num_inliers, 20, 40);

  // Decaying motion disparity faster if number of inliers indicate lock onto
  // tracking objects has occurred.
  return std::min(1.0f - inlier_score, motion_disparity) * sign;
}

void MotionBox::ScoreAndRecordInliers(
    const MotionBoxState& curr_pos,
    const std::vector<const MotionVector*>& vectors,
    const std::vector<Vector2_f>& grid_positions,
    const std::vector<float>& pre_estimation_weights,
    const std::vector<float>& post_estimation_weights,
    float background_discrimination, MotionBoxState* next_pos,
    std::vector<float>* inlier_weights, std::vector<float>* inlier_density,
    int* continued_inliers, int* swapped_inliers, float* motion_inliers_out,
    float* kinetic_average_out) const {
  CHECK(inlier_weights);
  CHECK(inlier_density);
  CHECK(continued_inliers);
  CHECK(swapped_inliers);
  CHECK(motion_inliers_out);
  CHECK(kinetic_average_out);

  std::unordered_map<int, int> prev_inliers;
  MotionBoxInliers(curr_pos, &prev_inliers);

  std::unordered_set<int> prev_outliers;
  MotionBoxOutliers(curr_pos, &prev_outliers);

  ClearInlierState(next_pos);

  // Continued inlier fraction denotes amount of spatial occlusion. Very low
  // values indicate that we are in very difficult tracking territory.
  *continued_inliers = 0;
  *swapped_inliers = 0;
  float kinetic_average = 0;
  float kinetic_average_sum = 0;
  float motion_inliers = 0;
  const int num_vectors = vectors.size();
  inlier_weights->resize(num_vectors);
  inlier_density->resize(num_vectors);

  // Inliers normalization grid.
  std::vector<float> grid_count(kNormalizationGridSize *
                                kNormalizationGridSize);
  const float prev_object_motion = MotionBoxVelocity(curr_pos).Norm();
  // Count number of similar moving inliers as previous object motion.
  const float similar_motion_threshold =
      std::max(2e-3f, prev_object_motion * 0.3f);

  // If background discrimination is low, inliers are ambiguous: Hard to
  // distinguish from earlier outliers. In this case do not record inliers
  // outside our current tracking extent, as everything will look like an
  // inlier.
  //
  // TODO: Compute 2nd moment for inliers and describe as ellipse,
  // improve shape here then.
  bool inlier_ambiguity = background_discrimination < 0.5f;
  int rejected = 0;
  int num_inliers = 0;
  for (int k = 0; k < num_vectors; ++k) {
    (*inlier_weights)[k] =
        LinearRamp(post_estimation_weights[k], options_.inlier_low_weight(),
                   options_.inlier_high_weight());
    const int track_id = vectors[k]->track_id;

    bool is_prev_outlier = prev_outliers.find(track_id) != prev_outliers.end();

    const Vector2_f match_loc = vectors[k]->MatchLocation();
    if ((*inlier_weights)[k] > kMinInlierWeight) {  // Inlier.
      if (is_prev_outlier) {
        ++(*swapped_inliers);
      }

      if (inlier_ambiguity &&
          !PointWithinInlierExtent(vectors[k]->Location(), curr_pos)) {
        ++rejected;
        continue;
      }

      ++num_inliers;

      AddToGrid<kNormalizationGridSize>(grid_positions[k], &grid_count);

      if (track_id >= 0) {
        next_pos->add_inlier_ids(track_id);
        next_pos->add_inlier_id_match_pos(match_loc.x() * kShortScale);
        next_pos->add_inlier_id_match_pos(match_loc.y() * kShortScale);
        auto pos = prev_inliers.find(track_id);
        if (pos != prev_inliers.end()) {
          // Count length of observation.
          next_pos->add_inlier_length(pos->second + 1.0f);
          ++(*continued_inliers);
        } else {
          next_pos->add_inlier_length(1);
        }
      }

      // Note: This should be weighted by the pre estimation weights, simply
      // adding a 1 for each inlier leads to lower irls averages.
      kinetic_average += vectors[k]->object.Norm() * pre_estimation_weights[k];
      kinetic_average_sum += pre_estimation_weights[k];

      // Count number of inliers that agree with previous kinetic energy
      // estimate.
      if (std::abs(vectors[k]->object.Norm() - prev_object_motion) *
              curr_pos.prior_weight() <
          similar_motion_threshold) {
        motion_inliers += pre_estimation_weights[k];
      }
    } else if ((*inlier_weights)[k] < kMaxOutlierWeight) {  // Outlier.
      next_pos->add_outlier_ids(track_id);
      next_pos->add_outlier_id_match_pos(match_loc.x() * kShortScale);
      next_pos->add_outlier_id_match_pos(match_loc.y() * kShortScale);
    }
  }

  // Read out density of inliers.
  for (int k = 0; k < num_vectors; ++k) {
    if ((*inlier_weights)[k] > kMinInlierWeight) {
      (*inlier_density)[k] = 2 * SampleFromGrid<kNormalizationGridSize>(
                                     grid_positions[k], grid_count);
    } else {
      (*inlier_density)[k] = 0;
    }
  }

  if (kinetic_average_sum > 0) {
    kinetic_average *= 1.0f / kinetic_average_sum;
  }

  VLOG(1) << "num inliers: " << num_inliers << " rejected: " << rejected;

  *kinetic_average_out = kinetic_average;
  *motion_inliers_out = motion_inliers;
}

void MotionBox::ComputeInlierCenterAndExtent(
    const std::vector<const MotionVector*>& motion_vectors,
    const std::vector<float>& weights, const std::vector<float>& density,
    const MotionBoxState& box_state, float* min_inlier_sum, Vector2_f* center,
    Vector2_f* extent) const {
  CHECK(min_inlier_sum);
  CHECK(center);
  CHECK(extent);

  float weight_sum = 0;
  float inlier_sum = 0;
  const int num_vectors = motion_vectors.size();
  CHECK_EQ(num_vectors, weights.size());
  CHECK_EQ(num_vectors, density.size());

  Vector2_f first_moment(0.0f, 0.0f);
  Vector2_f second_moment(0.0f, 0.0f);

  Vector2_f top_left;
  Vector2_f bottom_right;
  MotionBoxBoundingBox(box_state, &top_left, &bottom_right);

  for (int k = 0; k < num_vectors; ++k) {
    const MotionVector& motion_vector = *motion_vectors[k];
    const Vector2_f match = motion_vector.MatchLocation();
    float space_multiplier = 1.0f;
    // Decrease contribution of out of bound inliers.
    // Note: If all inliers are out of bound this down weighting will have no
    // effect. It is designed to prevent skewing the inlier center towards
    // similar moving inliers outside the tracked box.
    if (match.x() < top_left.x() || match.x() > bottom_right.x() ||
        match.y() < top_left.y() || match.y() > bottom_right.y()) {
      space_multiplier = 0.25f;
    }
    const float w = weights[k] * density[k] * space_multiplier;
    if (w > 0) {
      first_moment += w * match;
      second_moment +=
          w * Vector2_f(match.x() * match.x(), match.y() * match.y());
      weight_sum += w;
      inlier_sum += weights[k];
    }
  }

  // Update center if sufficient inliers present.
  if (inlier_sum > *min_inlier_sum) {
    const float inv_weight_sum = 1.0f / weight_sum;
    first_moment *= inv_weight_sum;
    second_moment *= inv_weight_sum;

    *center = first_moment;
    *extent = second_moment - Vector2_f(first_moment.x() * first_moment.x(),
                                        first_moment.y() * first_moment.y());

    // 1.645 sigmas in each direction = 90% of the data captured.
    *extent =
        Vector2_f(std::sqrt(extent->x()) * 3.29, std::sqrt(extent->y()) * 3.29);
  } else {
    // Gravitate back to box center with inlier center.
    *center = Lerp(MotionBoxCenter(box_state), InlierCenter(box_state), 0.5f);
  }

  // Record number of inliers present.
  *min_inlier_sum = weight_sum;
}

float MotionBox::ScaleEstimate(
    const std::vector<const MotionVector*>& motion_vectors,
    const std::vector<float>& weights, float min_sum) const {
  const int num_vectors = motion_vectors.size();
  CHECK_EQ(num_vectors, weights.size());

  float scale_sum = 0;

  // First moments.
  Vector2_d sum_coords(0, 0);
  Vector2_d match_sum_coords(0, 0);

  // Second moments.
  Vector2_d sum_sq_coords(0, 0);
  Vector2_d match_sum_sq_coords(0, 0);

  for (int k = 0; k < num_vectors; ++k) {
    const MotionVector& motion_vector = *motion_vectors[k];

    const Vector2_d pos(motion_vector.pos.x(), motion_vector.pos.y());
    double weight = weights[k];
    sum_coords += weight * pos;
    sum_sq_coords += weight * Vector2_d(pos.x() * pos.x(), pos.y() * pos.y());

    const Vector2_d match =
        Vector2_d::Cast<float>(motion_vector.MatchLocation());
    match_sum_coords += weight * match;
    match_sum_sq_coords +=
        weight * Vector2_d(match.x() * match.x(), match.y() * match.y());
    scale_sum += weights[k];
  }

  if (scale_sum > min_sum) {
    const double denom = 1.0f / scale_sum;
    sum_coords *= denom;
    match_sum_coords *= denom;
    sum_sq_coords *= denom;
    match_sum_sq_coords *= denom;

    const float curr_scale =
        sqrt(sum_sq_coords.x() - sum_coords.x() * sum_coords.x() +
             sum_sq_coords.y() - sum_coords.y() * sum_coords.y());
    const float next_scale = sqrt(
        match_sum_sq_coords.x() - match_sum_coords.x() * match_sum_coords.x() +
        match_sum_sq_coords.y() - match_sum_coords.y() * match_sum_coords.y());
    return next_scale / curr_scale;
  }

  return 1.0f;
}

void MotionBox::ApplySpringForce(const Vector2_f& center_of_interest,
                                 const float rel_threshold,
                                 const float spring_force,
                                 MotionBoxState* box_state) const {
  // Apply spring force towards center of interest.
  const Vector2_f center = MotionBoxCenter(*box_state);
  const float center_diff_x = center_of_interest.x() - center.x();
  const float center_diff_y = center_of_interest.y() - center.y();

  const float diff_x = fabs(center_diff_x) - box_state->width() * rel_threshold;
  const float diff_y =
      fabs(center_diff_y) - box_state->height() * rel_threshold;

  if (diff_x > 0) {
    const float correction_mag = diff_x * spring_force;
    const float correction =
        center_diff_x < 0 ? -correction_mag : correction_mag;
    box_state->set_pos_x(box_state->pos_x() + correction);
  }

  if (diff_y > 0) {
    const float correction_mag = diff_y * spring_force;
    const float correction =
        center_diff_y < 0 ? -correction_mag : correction_mag;
    box_state->set_pos_y(box_state->pos_y() + correction);
  }
}

void MotionBox::TrackStepImpl(int from_frame, const MotionBoxState& curr_pos,
                              const MotionVectorFrame& motion_frame,
                              const std::vector<const MotionBoxState*>& history,
                              MotionBoxState* next_pos) const {
  // Create new curr pos with velocity scaled to current duration.
  constexpr float kDefaultPeriodMs = 1000.0f / kTrackingDefaultFps;

  // Scale to be applied to velocity related fields in MotionBoxState
  // to transform state from standard frame period to current one.
  float temporal_scale = (motion_frame.duration_ms == 0)
                             ? 1.0f
                             : motion_frame.duration_ms / kDefaultPeriodMs;

  MotionBoxState curr_pos_normalized = curr_pos;
  ScaleStateTemporally(temporal_scale, &curr_pos_normalized);
  ScaleStateAspect(motion_frame.aspect_ratio, false, &curr_pos_normalized);

  TrackStepImplDeNormalized(from_frame, curr_pos_normalized, motion_frame,
                            history, next_pos);

  // Scale back velocity and aspect to normalized domains.
  ScaleStateTemporally(1.0f / temporal_scale, next_pos);
  ScaleStateAspect(motion_frame.aspect_ratio, true, next_pos);

  // Test if out of bound, only for moving objects.
  const float static_motion =
      options_.static_motion_temporal_ratio() * temporal_scale;
  if (MotionBoxVelocity(*next_pos).Norm() > static_motion) {
    // Test if close to boundary and keep moving towards it.
    constexpr float kRatio = 0.3;
    if ((next_pos->pos_x() < -next_pos->width() * kRatio &&
         next_pos->dx() < -static_motion / 2) ||
        (next_pos->pos_y() < -next_pos->height() * kRatio &&
         next_pos->dy() < -static_motion / 2) ||
        (next_pos->pos_x() > 1.0f - next_pos->width() * (1.0f - kRatio) &&
         next_pos->dx() > static_motion / 2) ||
        (next_pos->pos_y() > 1.0f - next_pos->height() * (1.0f - kRatio) &&
         next_pos->dy() > static_motion / 2)) {
      VLOG(1) << "Tracked box went out of bound.";
      next_pos->set_track_status(MotionBoxState::BOX_UNTRACKED);
      return;
    }
  }
}

float MotionBox::ComputeTrackingConfidence(
    const MotionBoxState& motion_box_state) const {
  const float inlier_num_lower_bound = 10.0f;
  const float inlier_num_upper_bound = 30.0f;
  const float confidence =
      LinearRamp(motion_box_state.inlier_ids_size(), inlier_num_lower_bound,
                 inlier_num_upper_bound);
  return confidence;
}

// General tracking algorithm overview:
// Tracking algorithm consists of 6 main stages.
// 1.) Selecting features from passed MotionVectorFrame based on incidence with
//     rectangle defined by curr_pos
// 2.) Assigning each vector a prior weight. Vector are mainly scored by a box
//     centered gaussian, giving more weights to vectors within the center of
//     the box. If current state is deemed unreliable, vectors with similar
//     velocity to previously estimated velocity are favored. If current state
//     indicates tracking of a moving object, high velocity vectors are favored.
// 3.) Estimating a translational model via IRLS enforcing the prior of step 2
//     in every iteration.
// 4.) Score how much the estimate model deviates from the previous motion
//     (termed motion disparity) and how discriminative the motion is
//     from the background motion (termed motion discrimination)
// 5.) Computing the inlier center (position of vectors used for the motion
//     model in the next frame) and center of high velocity vectors. Apply a
//     spring force towards each center based on the motion discrimination.
// 6.) Update velocity and kinetic energy by blending current measurment with
//     previous one.
void MotionBox::TrackStepImplDeNormalized(
    int from_frame, const MotionBoxState& curr_pos,
    const MotionVectorFrame& motion_frame,
    const std::vector<const MotionBoxState*>& history,
    MotionBoxState* next_pos) const {
  CHECK(next_pos);

  constexpr float kDefaultPeriodMs = 1000.0f / kTrackingDefaultFps;
  float temporal_scale = (motion_frame.duration_ms == 0)
                             ? 1.0f
                             : motion_frame.duration_ms / kDefaultPeriodMs;

  // Initialize to current position.
  *next_pos = curr_pos;

  if (!IsBoxValid(curr_pos)) {
    LOG(ERROR) << "curr_pos is not a valid box. Stop tracking!";
    next_pos->set_track_status(MotionBoxState::BOX_UNTRACKED);
    return;
  }

  Vector2_f top_left, bottom_right;
  float expand_mag = 0.0f;
  GetStartPosition(curr_pos, motion_frame.aspect_ratio, &expand_mag, &top_left,
                   &bottom_right);

  const float aspect_ratio = motion_frame.aspect_ratio;
  float domain_x = 1.0f;
  float domain_y = 1.0f;
  ScaleFromAspect(aspect_ratio, false, &domain_x, &domain_y);

  // Binary search for start and end index (lexicographic search, i.e.
  // x indices are guaranteed to be within bounds, but y coordinates could be
  // outside and need to be checked against the domain of the box via
  // GetVectorsAndWeights below).
  MotionVector search_start;
  MotionVector search_end;
  search_start.pos = top_left;
  search_end.pos = bottom_right;

  int start_idx = std::lower_bound(motion_frame.motion_vectors.begin(),
                                   motion_frame.motion_vectors.end(),
                                   search_start, MotionVectorComparator()) -
                  motion_frame.motion_vectors.begin();

  int end_idx = std::lower_bound(motion_frame.motion_vectors.begin(),
                                 motion_frame.motion_vectors.end(), search_end,
                                 MotionVectorComparator()) -
                motion_frame.motion_vectors.begin();

  const float static_motion =
      options_.static_motion_temporal_ratio() * temporal_scale;
  if (start_idx >= end_idx || top_left.x() >= domain_x - expand_mag ||
      top_left.y() >= domain_y - expand_mag || bottom_right.x() <= expand_mag ||
      bottom_right.y() <= expand_mag) {
    // Empty box, no features found. It can happen if box is outside
    // field of view, or there is no features in the box.
    // Move box by background model if it has static motion, else move return
    // tracking error.
    if (MotionBoxVelocity(curr_pos).Norm() > static_motion ||
        (!motion_frame.valid_background_model && from_frame != queue_start_)) {
      next_pos->set_track_status(MotionBoxState::BOX_NO_FEATURES);
    } else {
      // Static object, move by background model.
      next_pos->set_track_status(MotionBoxState::BOX_TRACKED_OUT_OF_BOUND);
      ApplyCameraTrackingDegrees(curr_pos, motion_frame.background_model,
                                 options_, Vector2_f(domain_x, domain_y),
                                 next_pos);

      // The further the quad is away from the FOV (range 0 to 1), the larger
      // scale change will be applied to the quad by homography transform. To
      // some certain extent, the position of vertices will be flipped from
      // positive to negative, or vice versa. Here we reject all the quads with
      // abnormal shape by convexity of the quad..
      if (next_pos->has_quad() &&
          (ObjectMotionValidator::IsQuadOutOfFov(
               next_pos->quad(), Vector2_f(domain_x, domain_y)) ||
           !ObjectMotionValidator::IsValidQuad(next_pos->quad()))) {
        LOG(ERROR) << "Quad is out of fov or not convex. Cancel tracking.";
        next_pos->set_track_status(MotionBoxState::BOX_UNTRACKED);
        return;
      }
    }
    return;
  }

  float start_x = Clamp(top_left.x(), 0.0f, domain_x);
  float start_y = Clamp(top_left.y(), 0.0f, domain_y);
  float end_x = Clamp(bottom_right.x(), 0.0f, domain_x);
  float end_y = Clamp(bottom_right.y(), 0.0f, domain_y);

  const Vector2_f curr_pos_size = MotionBoxSize(curr_pos);
  constexpr float kMinSize = 1e-3f;  // 1 pix for 1080p.
  if (start_x >= end_x || start_y >= end_y || curr_pos_size.x() < kMinSize ||
      curr_pos_size.y() < kMinSize) {
    next_pos->set_track_status(MotionBoxState::BOX_EMPTY);
    return;
  }

  top_left = Vector2_f(start_x, start_y);
  bottom_right = Vector2_f(end_x, end_y);

  // Get indices of features within box, corresponding priors and position
  // in feature grid.
  std::vector<const MotionVector*> vectors;
  std::vector<float> prior_weights;
  bool valid_background_model = motion_frame.valid_background_model;

  int num_good_inits;
  int num_cont_inliers;
  const bool get_vec_weights_status = GetVectorsAndWeights(
      motion_frame.motion_vectors, start_idx, end_idx, top_left, bottom_right,
      curr_pos, valid_background_model, motion_frame.is_chunk_boundary,
      temporal_scale, expand_mag, history, &vectors, &prior_weights,
      &num_good_inits, &num_cont_inliers);
  if (!get_vec_weights_status) {
    LOG(ERROR) << "error in GetVectorsAndWeights. Terminate tracking.";
    next_pos->set_track_status(MotionBoxState::BOX_UNTRACKED);
    return;
  }

  // `num_good_inits` comes from motion vector weights, but pnp solver currently
  // does not depend on weights. So for pnp tracking mode, we don't fall back to
  // camera motion tracking based on num_good_inits.
  if (!curr_pos.has_pnp_homography() &&
      (num_good_inits < 3 &&
       MotionBoxVelocity(curr_pos).Norm() <= static_motion)) {
    // Static object, move by background model.
    next_pos->set_track_status(MotionBoxState::BOX_TRACKED);
    ApplyCameraTrackingDegrees(curr_pos, motion_frame.background_model,
                               options_, Vector2_f(domain_x, domain_y),
                               next_pos);
    VLOG(1) << "No good inits; applying camera motion for static object";

    if (next_pos->has_quad() &&
        !ObjectMotionValidator::IsValidQuad(next_pos->quad())) {
      LOG(ERROR) << "Quad is not convex. Cancel tracking.";
      next_pos->set_track_status(MotionBoxState::BOX_UNTRACKED);
      return;
    }
    return;
  }

  VLOG(1) << "Good inits: " << num_good_inits;

  const int num_vectors = vectors.size();
  CHECK_EQ(num_vectors, prior_weights.size());

  Vector2_f object_translation;

  // Compute a rough estimate of the current motion.
  for (int k = 0; k < num_vectors; ++k) {
    object_translation += vectors[k]->Motion() * prior_weights[k];
  }

  Vector2_f prev_object_motion = MotionBoxVelocity(curr_pos);

  // Estimate expected motion magnitude. In case of low prior, skew more
  // towards rough estimate instead of previous motion.
  const float motion_mag_estimate =
      std::max(object_translation.Norm(),
               prev_object_motion.Norm() * curr_pos.prior_weight());

  // For motivation about this: See MotionEstimation::GetIRLSResidualScale.
  // Assume 1 pixel estimation error for tracked features at 360p video.
  // This serves as absolute minimum of the estimation error, so we do not
  // scale translation fractions below this threshold.
  constexpr float kMinError = 1.25e-3f;

  // We use a combination of absolute and relative error. If a predefined
  // fraction of the motion exceeds the minimum error, we scale the error
  // such that the relative error equals the min error.
  // We use different thresholds parallel and perpendicular to the estimation
  // direction.
  // Motivation is that we allow for more error perpendicular to an estimation
  // (angular difference) than it is direction (magnitude error).

  // Scale in parallel, orthogonal direction.
  Vector2_f irls_scale(1.0f, 1.0f);

  const Vector2_f kMotionPercentage(0.1f, 0.25f);
  const Vector2_f motion_mag_scaled = motion_mag_estimate * kMotionPercentage;

  if (motion_mag_scaled.x() > kMinError) {
    irls_scale.x(kMinError / motion_mag_scaled.x());
  }
  if (motion_mag_scaled.y() > kMinError) {
    irls_scale.y(kMinError / motion_mag_scaled.y());
  }

  // Irls init for translation.
  // TODO: Adjust to object tracking degrees.
  TranslationIrlsInitialization(vectors, irls_scale, &prior_weights);

  LinearSimilarityModel object_similarity;
  Homography object_homography;
  Homography pnp_homography;

  std::vector<float> weights = prior_weights;
  if (num_good_inits > 0) {
    MEASURE_TIME << "Estimate object motion.";
    EstimateObjectMotion(vectors, prior_weights, num_cont_inliers, irls_scale,
                         &weights, &object_translation, &object_similarity,
                         &object_homography);
  } else {
    // There is no hope to estimate a model in a stable manner here.
    object_translation = prev_object_motion;
    VLOG(1) << "No good inits, reusing prev. motion instead of estimation";
  }

  // Multiplier to quanitify how discriminative object motion is
  // (larger motions are more discriminative).
  // Note: Independent from temporal scale.
  float background_discrimination = curr_pos.background_discrimination();
  if (valid_background_model) {
    background_discrimination =
        LinearRamp(object_translation.Norm(),
                   options_.background_discrimination_low_level(),
                   options_.background_discrimination_high_level());
  }

  // Score weights from motion estimation to determine set of inliers.
  std::vector<float> inlier_weights;
  std::vector<float> inlier_density;

  // Compute grid positions for each vector to determine density of inliers.
  std::vector<Vector2_f> grid_positions;
  ComputeGridPositions<kNormalizationGridSize>(top_left, bottom_right, vectors,
                                               &grid_positions);

  // Continued inlier fraction denotes amount of spatial occlusion. Very low
  // values indicate that we are in very difficult tracking territory.
  int continued_inliers = 0;
  int swapped_inliers = 0;
  float kinetic_average = 0;
  float motion_inliers = 0;
  ScoreAndRecordInliers(curr_pos, vectors, grid_positions, prior_weights,
                        weights, background_discrimination, next_pos,
                        &inlier_weights, &inlier_density, &continued_inliers,
                        &swapped_inliers, &motion_inliers, &kinetic_average);

  const int num_prev_inliers = curr_pos.inlier_ids_size();
  int num_prev_inliers_not_actively_discarded = num_prev_inliers;
  if (motion_frame.actively_discarded_tracked_ids != nullptr) {
    num_prev_inliers_not_actively_discarded = std::count_if(
        curr_pos.inlier_ids().begin(), curr_pos.inlier_ids().end(),
        [&motion_frame](int id) {
          return !motion_frame.actively_discarded_tracked_ids->contains(id);
        });
    motion_frame.actively_discarded_tracked_ids->clear();
  }
  const int num_inliers = next_pos->inlier_ids_size();
  // Must be in [0, 1].
  const float continued_inlier_fraction =
      num_prev_inliers_not_actively_discarded == 0
          ? 1.0f
          : continued_inliers * 1.0f / num_prev_inliers_not_actively_discarded;

  // Within [0, M], where M is maximum number of features. Values of > 1
  // indicate significant number of inlieres were outliers in previous frame.
  const float swapped_inlier_fraction =
      num_prev_inliers == 0 ? 0.0f : swapped_inliers * 1.0f / num_prev_inliers;

  if (curr_pos.has_pnp_homography()) {
    MEASURE_TIME << "Estimate pnp homography.";

    // Use IRLS homography `inlier_weights` to determin inliers / outliers.
    // The rationale is: solving homography transform is 20X faster than solving
    // perspective transform (0.05ms vs 1ms). So, we use 5 iterations of
    // reweighted homography to filter out outliers first. And only use inliers
    // to solve for perspective.
    if (!EstimatePnpHomography(curr_pos, vectors, inlier_weights, domain_x,
                               domain_y, &pnp_homography)) {
      // Here, we can either cancel tracking or apply homography or even
      // translation as our best guess. But since some specific use cases of
      // pnp tracking (for example Augmented Images) prefer high precision
      // over high recall, we choose to cancel tracking once and for all.
      VLOG(1) << "Not enough motion vectors to solve pnp. Cancel tracking.";
      next_pos->set_track_status(MotionBoxState::BOX_UNTRACKED);
      return;
    }
  }

  // Compute disparity.
  if (num_good_inits > 0) {
    next_pos->set_motion_disparity(ComputeMotionDisparity(
        curr_pos, irls_scale, continued_inliers, num_inliers,
        valid_background_model ? object_translation : prev_object_motion));
  } else {
    // No good features, signal error.
    next_pos->set_motion_disparity(1.0);
  }

  VLOG(1) << "Motion inliers: " << motion_inliers
          << ", continued inliers: " << continued_inliers
          << ", continued ratio: " << continued_inlier_fraction
          << ", swapped fraction: " << swapped_inlier_fraction
          << ", motion disparity: " << next_pos->motion_disparity();

  if (options_.cancel_tracking_with_occlusion_options().activated() &&
      curr_pos.track_status() != MotionBoxState::BOX_DUPLICATED &&
      continued_inlier_fraction <
          options_.cancel_tracking_with_occlusion_options()
              .min_motion_continuity()) {
    next_pos->set_track_status(MotionBoxState::BOX_UNTRACKED);
    LOG(INFO) << "Occlusion detected. continued_inlier_fraction: "
              << continued_inlier_fraction << " too low. Stop tracking";
    return;
  }

  // Force reset of state when inlier continuity is severly violated,
  // disparity maxes out or significant of number of inliers were outliers in
  // the previous frame.
  if (std::max(continued_inlier_fraction, motion_inliers) < 0.15f ||
      std::abs(next_pos->motion_disparity()) >= 1.0f ||
      swapped_inlier_fraction >= 2.5) {
    VLOG(1) << "Track error, state reset.";
    // Bad tracking error occurred.
    // Current set of inliers is not reliable.
    ClearInlierState(next_pos);
    next_pos->set_motion_disparity(1.0f);
    inlier_weights.assign(inlier_weights.size(), 0);

    // Reuse previous motion and discrimination.
    object_translation = prev_object_motion;
    background_discrimination = curr_pos.background_discrimination();
  }

  next_pos->set_inlier_sum(
      std::accumulate(inlier_weights.begin(), inlier_weights.end(), 0.0f));
  if (history.empty()) {
    // Assign full confidence on first frame, otherwise all other stats
    // are zero and there is no way to compute.
    next_pos->set_tracking_confidence(1.0f);
    LOG(INFO) << "no history. confidence : 1.0";
  } else {
    next_pos->set_tracking_confidence(ComputeTrackingConfidence(*next_pos));
    VLOG(1) << "confidence: " << next_pos->tracking_confidence();
  }
  next_pos->set_background_discrimination(background_discrimination);

  // Slowly decay current kinetic energy. Blend with current measurement based
  // on disparity (high = use previous value, low = use current one).
  next_pos->set_kinetic_energy(std::max(
      options_.kinetic_energy_decay() * curr_pos.kinetic_energy(),
      kinetic_average * (1.0f - std::abs(next_pos->motion_disparity()))));

  float inlier_max = curr_pos.inlier_sum();
  int num_tracked_frames_in_history = 0;
  for (auto entry : history) {
    inlier_max = std::max(inlier_max, entry->inlier_sum());
    if (entry->track_status() == MotionBoxState::BOX_TRACKED) {
      num_tracked_frames_in_history++;
    }
  }

  const float inlier_ratio =
      inlier_max > 0 ? (next_pos->inlier_sum() / (inlier_max + 1e-3f)) : 0.0f;

  next_pos->set_inlier_ratio(inlier_ratio);

  const bool is_perfect_fit = inlier_ratio > 0.85f && inlier_ratio < 1.15f;

  // num_tracked_frames_in_history has to be greater than 1, since the first
  // frame is marked as BOX_TRACKED in ResetAtFrame.
  if (options_.cancel_tracking_with_occlusion_options().activated() &&
      curr_pos.track_status() != MotionBoxState::BOX_DUPLICATED &&
      num_tracked_frames_in_history > 1 &&
      inlier_ratio < options_.cancel_tracking_with_occlusion_options()
                         .min_inlier_ratio()) {
    next_pos->set_track_status(MotionBoxState::BOX_UNTRACKED);
    LOG(INFO) << "inlier_ratio: " << inlier_ratio
              << " too small. Stop tracking. inlier_max: " << inlier_max
              << ". length in history: " << history.size();
    return;
  }

  // Blend measured object motion based on motion disparity, i.e. the more the
  // measured and previous motion agree, the less the smoothing. This is to
  // propagate the box in the direction of the previous object motion
  // in case tracking has been lost.
  // Allow new measurements to propagate slowly.
  if (valid_background_model && !is_perfect_fit) {
    // Always move some fraction in the direction of the measured object
    // even if deemed in disagreement with previous motion.
    const float kMinimumBlend = 0.2f;
    object_translation = Lerp(
        object_translation, prev_object_motion,
        std::min(1.0f - kMinimumBlend, std::abs(next_pos->motion_disparity())));
  }

  if (curr_pos.has_pnp_homography()) {
    ApplyObjectMotionPerspectively(curr_pos, pnp_homography, domain_x, domain_y,
                                   next_pos);
  } else {
    ApplyObjectMotion(curr_pos, object_translation, object_similarity,
                      object_homography, options_, next_pos);

    ApplyCameraTrackingDegrees(curr_pos, motion_frame.background_model,
                               options_, Vector2_f(domain_x, domain_y),
                               next_pos);
  }

  if (next_pos->has_quad() &&
      !ObjectMotionValidator::IsValidQuad(next_pos->quad())) {
    LOG(ERROR) << "Quad is not convex. Cancel tracking.";
    next_pos->set_track_status(MotionBoxState::BOX_UNTRACKED);
    return;
  }

  // Storing pre-comp weights.
  const std::vector<float>& internal_weights =
      options_.use_post_estimation_weights_for_state() ? inlier_weights
                                                       : prior_weights;

  StoreInternalState(vectors, internal_weights, aspect_ratio,
                     next_pos->mutable_internal());

  // Compute center of inliers in next frame and change in scale for inliers.
  Vector2_f inlier_center;
  Vector2_f inlier_extent;
  float min_inlier_weight = 2.0f;  // Only update inlier_center if more
                                   // inliers than specified are found.
  ComputeInlierCenterAndExtent(vectors, inlier_weights, inlier_density,
                               *next_pos, &min_inlier_weight, &inlier_center,
                               &inlier_extent);

  // Determine difference to previous estimate.
  Vector2_f prev_inlier_center(InlierCenter(curr_pos));
  const float rel_inlier_center_diff =
      (inlier_center - prev_inlier_center).Norm() /
      MotionBoxSize(curr_pos).Norm();

  // Smooth with previous location, based on relative inlier difference.
  // A difference of 1.0 is mapped to a weight of 1.0 (total outlier).
  // Blend weight is maxed out at .6f to always allow measurements to propagate
  // over time (assuming high motion discrimination).
  const float center_blend =
      std::min(Lerp(0.95f, 0.6f, background_discrimination),
               rel_inlier_center_diff) *
      curr_pos.prior_weight();
  inlier_center = Lerp(inlier_center, prev_inlier_center, center_blend);

  next_pos->set_inlier_center_x(inlier_center.x());
  next_pos->set_inlier_center_y(inlier_center.y());

  // Update extent only when sufficient inliers are present.
  // TODO: This is too hacky, evaluate.
  if (min_inlier_weight > 30) {
    Vector2_f prev_inlier_extent(curr_pos.inlier_width(),
                                 curr_pos.inlier_height());
    // Blend with previous extent based on prior and discrimination.
    inlier_extent = Lerp(
        inlier_extent, prev_inlier_extent,
        curr_pos.prior_weight() * Lerp(1.0, 0.85, background_discrimination));
    next_pos->set_inlier_width(inlier_extent.x());
    next_pos->set_inlier_height(inlier_extent.y());
  }

  VLOG(1) << "Inlier extent " << next_pos->inlier_width() << " , "
          << next_pos->inlier_height();

  // Spring force applied to the inlier_center is modulated by the background
  // discrimination. Motivation: Low background discrimination leads to inlier
  // center more biased towards previous result due to update weight being
  // tampered down. Always apply a minimum force.
  // TODO: During challenging (low inlier) situations this can
  // save the lock onto objects. Cook up a condition to set min spring force to
  // 0.25 or so.
  constexpr float kMinSpringForceFraction = 0.0;
  ApplySpringForce(inlier_center, options_.inlier_center_relative_distance(),
                   std::min(1.0f, options_.inlier_spring_force() *
                                      std::max(kMinSpringForceFraction,
                                               background_discrimination)),
                   next_pos);

  if (options_.compute_spatial_prior()) {
    // Blend based on object multiplier using high prior weight for low
    // multipliers.
    // Magic update numbers, prior is not important for tracking, only for
    // visualization purposes.
    const float prior_weight = Lerp(0.98, 0.85, background_discrimination);
    ComputeSpatialPrior(true, true, prior_weight, next_pos);
  }

  // Update velocity.
  float velocity_update_weight =
      is_perfect_fit
          ? 0.0f
          : (options_.velocity_update_weight() * curr_pos.prior_weight());
  // Computed object motion is completely random when background model is
  // invalid. Use previous motion in this case.
  if (!valid_background_model) {
    velocity_update_weight = 1.0f;
  }

  next_pos->set_dx(
      Lerp(object_translation.x(), curr_pos.dx(), velocity_update_weight));
  next_pos->set_dy(
      Lerp(object_translation.y(), curr_pos.dy(), velocity_update_weight));

  // Update prior.
  if (valid_background_model) {
    next_pos->set_prior_weight(std::min(
        1.0f, curr_pos.prior_weight() + options_.prior_weight_increase()));
  } else {
    next_pos->set_prior_weight(std::max(
        0.0f, curr_pos.prior_weight() - options_.prior_weight_increase()));
  }

  next_pos->set_track_status(MotionBoxState::BOX_TRACKED);
}

void MotionVectorFrameFromTrackingData(const TrackingData& tracking_data,
                                       MotionVectorFrame* motion_vector_frame) {
  CHECK(motion_vector_frame != nullptr);

  const auto& motion_data = tracking_data.motion_data();
  float aspect_ratio = tracking_data.frame_aspect();
  if (aspect_ratio < 0.1 || aspect_ratio > 10.0f) {
    LOG(ERROR) << "Aspect ratio : " << aspect_ratio << " is out of bounds. "
               << "Resetting to 1.0.";
    aspect_ratio = 1.0f;
  }

  float scale_x, scale_y;
  // Normalize longest dimension to 1 under aspect ratio preserving scaling.
  ScaleFromAspect(aspect_ratio, false, &scale_x, &scale_y);

  scale_x /= tracking_data.domain_width();
  scale_y /= tracking_data.domain_height();

  const bool use_background_model =
      !(tracking_data.frame_flags() & TrackingData::FLAG_BACKGROUND_UNSTABLE);

  Homography homog_scale = HomographyAdapter::Embed(
      AffineAdapter::FromArgs(0, 0, scale_x, 0, 0, scale_y));

  Homography inv_homog_scale = HomographyAdapter::Embed(
      AffineAdapter::FromArgs(0, 0, 1.0f / scale_x, 0, 0, 1.0f / scale_y));

  // Might be just the identity if not set.
  const Homography background_model = tracking_data.background_model();
  const Homography background_model_scaled =
      ModelCompose3(homog_scale, background_model, inv_homog_scale);

  motion_vector_frame->background_model.CopyFrom(background_model_scaled);
  motion_vector_frame->valid_background_model = use_background_model;
  motion_vector_frame->is_duplicated =
      tracking_data.frame_flags() & TrackingData::FLAG_DUPLICATED;
  motion_vector_frame->is_chunk_boundary =
      tracking_data.frame_flags() & TrackingData::FLAG_CHUNK_BOUNDARY;
  motion_vector_frame->aspect_ratio = tracking_data.frame_aspect();
  motion_vector_frame->motion_vectors.reserve(motion_data.row_indices_size());

  motion_vector_frame->motion_vectors.clear();
  const bool long_tracks = motion_data.track_id_size() > 0;

  for (int c = 0; c < motion_data.col_starts_size() - 1; ++c) {
    const float x = c;
    const float scaled_x = x * scale_x;

    for (int r = motion_data.col_starts(c),
             r_end = motion_data.col_starts(c + 1);
         r < r_end; ++r) {
      MotionVector motion_vector;

      const float y = motion_data.row_indices(r);
      const float scaled_y = y * scale_y;

      const float dx = motion_data.vector_data(2 * r);
      const float dy = motion_data.vector_data(2 * r + 1);

      if (use_background_model) {
        Vector2_f loc(x, y);
        Vector2_f background_motion =
            HomographyAdapter::TransformPoint(background_model, loc) - loc;
        motion_vector.background = Vector2_f(background_motion.x() * scale_x,
                                             background_motion.y() * scale_y);
      }

      motion_vector.pos = Vector2_f(scaled_x, scaled_y);
      motion_vector.object = Vector2_f(dx * scale_x, dy * scale_y);

      if (long_tracks) {
        motion_vector.track_id = motion_data.track_id(r);
      }
      motion_vector_frame->motion_vectors.push_back(motion_vector);
    }
  }
}

void FeatureAndDescriptorFromTrackingData(
    const TrackingData& tracking_data, std::vector<Vector2_f>* features,
    std::vector<std::string>* descriptors) {
  const auto& motion_data = tracking_data.motion_data();
  float aspect_ratio = tracking_data.frame_aspect();
  if (aspect_ratio < 0.1 || aspect_ratio > 10.0f) {
    LOG(ERROR) << "Aspect ratio : " << aspect_ratio << " is out of bounds. "
               << "Resetting to 1.0.";
    aspect_ratio = 1.0f;
  }

  if (motion_data.feature_descriptors_size() == 0) {
    LOG(WARNING) << "Feature descriptors not exist";
    return;
  }

  float scale_x, scale_y;
  // Normalize longest dimension to 1 under aspect ratio preserving scaling.
  ScaleFromAspect(aspect_ratio, false, &scale_x, &scale_y);
  scale_x /= tracking_data.domain_width();
  scale_y /= tracking_data.domain_height();

  features->clear();
  descriptors->clear();

  for (int c = 0; c < motion_data.col_starts_size() - 1; ++c) {
    const float x = c;
    const float scaled_x = x * scale_x;

    for (int r = motion_data.col_starts(c),
             r_end = motion_data.col_starts(c + 1);
         r < r_end; ++r) {
      const std::string& descriptor = motion_data.feature_descriptors(r).data();

      if (absl::c_all_of(descriptor, [](char c) { return c == 0; })) {
        continue;
      }

      const float y = motion_data.row_indices(r);
      const float scaled_y = y * scale_y;

      features->emplace_back(scaled_x, scaled_y);
      descriptors->emplace_back(descriptor);
    }
  }
}

void InvertMotionVectorFrame(const MotionVectorFrame& input,
                             MotionVectorFrame* output) {
  CHECK(output != nullptr);

  output->background_model.CopyFrom(ModelInvert(input.background_model));
  output->valid_background_model = input.valid_background_model;
  output->is_duplicated = input.is_duplicated;
  output->is_chunk_boundary = input.is_chunk_boundary;
  output->duration_ms = input.duration_ms;
  output->aspect_ratio = input.aspect_ratio;
  output->motion_vectors.clear();
  output->motion_vectors.reserve(input.motion_vectors.size());
  output->actively_discarded_tracked_ids = input.actively_discarded_tracked_ids;

  const float aspect_ratio = input.aspect_ratio;
  float domain_x = 1.0f;
  float domain_y = 1.0f;
  ScaleFromAspect(aspect_ratio, false, &domain_x, &domain_y);

  // Explicit copy.
  for (auto motion_vec : input.motion_vectors) {
    motion_vec.background *= -1.0f;
    motion_vec.object *= -1.0f;

    motion_vec.pos -= motion_vec.background + motion_vec.object;

    // Inverted vector might be out of bound.
    if (motion_vec.pos.x() < 0.0f || motion_vec.pos.x() > domain_x ||
        motion_vec.pos.y() < 0.0f || motion_vec.pos.y() > domain_y) {
      continue;
    }

    // Approximately 40 - 60% of all inserts happen to be at the end.
    if (output->motion_vectors.empty() ||
        MotionVectorComparator()(output->motion_vectors.back(), motion_vec)) {
      output->motion_vectors.push_back(motion_vec);
    } else {
      output->motion_vectors.insert(
          std::lower_bound(output->motion_vectors.begin(),
                           output->motion_vectors.end(), motion_vec,
                           MotionVectorComparator()),
          motion_vec);
    }
  }
}

float TrackingDataDurationMs(const TrackingDataChunk::Item& item) {
  return (item.timestamp_usec() - item.prev_timestamp_usec()) * 1e-3f;
}

void GetFeatureIndicesWithinBox(const std::vector<Vector2_f>& features,
                                const MotionBoxState& box_state,
                                const Vector2_f& box_scaling,
                                float max_enlarge_size, int min_num_features,
                                std::vector<int>* inlier_indices) {
  CHECK(inlier_indices);
  inlier_indices->clear();

  if (features.empty()) return;
  std::array<Vector3_f, 4> box_lines;
  if (!MotionBoxLines(box_state, box_scaling, &box_lines)) {
    LOG(ERROR) << "Error in computing MotionBoxLines.";
    return;
  }

  // If the box size isn't big enough to cover sufficient features to
  // reacquire the box, the following code will try to iteratively enlarge the
  // box size by half of 'max_enlarge_size' to include more features, but
  // maximimum twice.
  float distance_threshold = 0.0f;
  int inliers_count = 0;
  std::vector<bool> chosen(features.size(), false);
  std::vector<float> signed_distance(features.size());

  for (int j = 0; j < features.size(); ++j) {
    float max_dist = std::numeric_limits<float>::lowest();
    for (const Vector3_f& line : box_lines) {
      float dist =
          line.DotProd(Vector3_f(features[j].x(), features[j].y(), 1.0f));
      if (dist > max_enlarge_size) {
        max_dist = dist;
        break;
      }
      max_dist = std::max(dist, max_dist);
    }

    signed_distance[j] = max_dist;
    if (signed_distance[j] < distance_threshold) {
      ++inliers_count;
      chosen[j] = true;
      inlier_indices->push_back(j);
    }
  }

  const float box_enlarge_step = max_enlarge_size * 0.5f;
  while (inliers_count < min_num_features) {
    distance_threshold += box_enlarge_step;
    if (distance_threshold > max_enlarge_size) break;
    for (int j = 0; j < features.size(); ++j) {
      if (chosen[j]) continue;
      if (signed_distance[j] < distance_threshold) {
        ++inliers_count;
        chosen[j] = true;
        inlier_indices->push_back(j);
      }
    }
  }
}

}  // namespace mediapipe.
