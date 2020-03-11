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

#include "mediapipe/graphs/object_detection_3d/calculators/decoder.h"

#include <limits>

#include "Eigen/Dense"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/graphs/object_detection_3d/calculators/annotation_data.pb.h"

namespace mediapipe {
constexpr int Decoder::kNumOffsetmaps = 16;

namespace {
void SetPoint3d(float x, float y, float z, Point3D* point_3d) {
  point_3d->set_x(x);
  point_3d->set_y(y);
  point_3d->set_z(z);
}
}  // namespace

FrameAnnotation Decoder::DecodeBoundingBoxKeypoints(
    const cv::Mat& heatmap, const cv::Mat& offsetmap) const {
  CHECK_EQ(1, heatmap.channels());
  CHECK_EQ(kNumOffsetmaps, offsetmap.channels());
  CHECK_EQ(heatmap.cols, offsetmap.cols);
  CHECK_EQ(heatmap.rows, offsetmap.rows);

  const float offset_scale = std::min(offsetmap.cols, offsetmap.rows);
  const std::vector<cv::Point> center_points = ExtractCenterKeypoints(heatmap);
  std::vector<BeliefBox> boxes;
  for (const auto& center_point : center_points) {
    BeliefBox box;
    box.box_2d.emplace_back(center_point.x, center_point.y);
    const int center_x = static_cast<int>(std::round(center_point.x));
    const int center_y = static_cast<int>(std::round(center_point.y));
    box.belief = heatmap.at<float>(center_y, center_x);
    if (config_.voting_radius() > 1) {
      DecodeByVoting(heatmap, offsetmap, center_x, center_y, offset_scale,
                     offset_scale, &box);
    } else {
      DecodeByPeak(offsetmap, center_x, center_y, offset_scale, offset_scale,
                   &box);
    }
    if (IsNewBox(&boxes, &box)) {
      boxes.push_back(std::move(box));
    }
  }

  const float x_scale = 1.0f / offsetmap.cols;
  const float y_scale = 1.0f / offsetmap.rows;
  FrameAnnotation frame_annotations;
  for (const auto& box : boxes) {
    auto* object = frame_annotations.add_annotations();
    for (const auto& point : box.box_2d) {
      auto* point2d = object->add_keypoints()->mutable_point_2d();
      point2d->set_x(point.first * x_scale);
      point2d->set_y(point.second * y_scale);
    }
  }
  return frame_annotations;
}

void Decoder::DecodeByPeak(const cv::Mat& offsetmap, int center_x, int center_y,
                           float offset_scale_x, float offset_scale_y,
                           BeliefBox* box) const {
  const auto& offset = offsetmap.at<cv::Vec<float, kNumOffsetmaps>>(
      /*row*/ center_y, /*col*/ center_x);
  for (int i = 0; i < kNumOffsetmaps / 2; ++i) {
    const float x_offset = offset[2 * i] * offset_scale_x;
    const float y_offset = offset[2 * i + 1] * offset_scale_y;
    box->box_2d.emplace_back(center_x + x_offset, center_y + y_offset);
  }
}

void Decoder::DecodeByVoting(const cv::Mat& heatmap, const cv::Mat& offsetmap,
                             int center_x, int center_y, float offset_scale_x,
                             float offset_scale_y, BeliefBox* box) const {
  // Votes at the center.
  const auto& center_offset = offsetmap.at<cv::Vec<float, kNumOffsetmaps>>(
      /*row*/ center_y, /*col*/ center_x);
  std::vector<float> center_votes(kNumOffsetmaps, 0.f);
  for (int i = 0; i < kNumOffsetmaps / 2; ++i) {
    center_votes[2 * i] = center_x + center_offset[2 * i] * offset_scale_x;
    center_votes[2 * i + 1] =
        center_y + center_offset[2 * i + 1] * offset_scale_y;
  }

  // Find voting window.
  int x_min = std::max(0, center_x - config_.voting_radius());
  int y_min = std::max(0, center_y - config_.voting_radius());
  int width = std::min(heatmap.cols - x_min, config_.voting_radius() * 2 + 1);
  int height = std::min(heatmap.rows - y_min, config_.voting_radius() * 2 + 1);
  cv::Rect rect(x_min, y_min, width, height);
  cv::Mat heat = heatmap(rect);
  cv::Mat offset = offsetmap(rect);

  for (int i = 0; i < kNumOffsetmaps / 2; ++i) {
    float x_sum = 0.f;
    float y_sum = 0.f;
    float votes = 0.f;
    for (int r = 0; r < heat.rows; ++r) {
      for (int c = 0; c < heat.cols; ++c) {
        const float belief = heat.at<float>(r, c);
        if (belief < config_.voting_threshold()) {
          continue;
        }
        float offset_x =
            offset.at<cv::Vec<float, kNumOffsetmaps>>(r, c)[2 * i] *
            offset_scale_x;
        float offset_y =
            offset.at<cv::Vec<float, kNumOffsetmaps>>(r, c)[2 * i + 1] *
            offset_scale_y;
        float vote_x = c + rect.x + offset_x;
        float vote_y = r + rect.y + offset_y;
        float x_diff = std::abs(vote_x - center_votes[2 * i]);
        float y_diff = std::abs(vote_y - center_votes[2 * i + 1]);
        if (x_diff > config_.voting_allowance() ||
            y_diff > config_.voting_allowance()) {
          continue;
        }
        x_sum += vote_x * belief;
        y_sum += vote_y * belief;
        votes += belief;
      }
    }
    box->box_2d.emplace_back(x_sum / votes, y_sum / votes);
  }
}

bool Decoder::IsNewBox(std::vector<BeliefBox>* boxes, BeliefBox* box) const {
  for (auto& b : *boxes) {
    if (IsIdentical(b, *box)) {
      if (b.belief < box->belief) {
        std::swap(b, *box);
      }
      return false;
    }
  }
  return true;
}

bool Decoder::IsIdentical(const BeliefBox& box_1,
                          const BeliefBox& box_2) const {
  // Skip the center point.
  for (int i = 1; i < box_1.box_2d.size(); ++i) {
    const float x_diff =
        std::abs(box_1.box_2d[i].first - box_2.box_2d[i].first);
    const float y_diff =
        std::abs(box_1.box_2d[i].second - box_2.box_2d[i].second);
    if (x_diff > config_.voting_allowance() ||
        y_diff > config_.voting_allowance()) {
      return false;
    }
  }
  return true;
}

std::vector<cv::Point> Decoder::ExtractCenterKeypoints(
    const cv::Mat& center_heatmap) const {
  cv::Mat max_filtered_heatmap(center_heatmap.rows, center_heatmap.cols,
                               center_heatmap.type());
  const int kernel_size =
      static_cast<int>(config_.local_max_distance() * 2 + 1 + 0.5f);
  const cv::Size morph_size(kernel_size, kernel_size);
  cv::dilate(center_heatmap, max_filtered_heatmap,
             cv::getStructuringElement(cv::MORPH_RECT, morph_size));
  cv::Mat peak_map;
  cv::bitwise_and((center_heatmap >= max_filtered_heatmap),
                  (center_heatmap >= config_.heatmap_threshold()), peak_map);
  std::vector<cv::Point> locations;  // output, locations of non-zero pixels
  cv::findNonZero(peak_map, locations);
  return locations;
}

absl::Status Decoder::Lift2DTo3D(
    const Eigen::Matrix<float, 4, 4, Eigen::RowMajor>& projection_matrix,
    bool portrait, FrameAnnotation* estimated_box) const {
  CHECK(estimated_box != nullptr);
  const float fx = projection_matrix(0, 0);
  const float fy = projection_matrix(1, 1);
  const float cx = projection_matrix(0, 2);
  const float cy = projection_matrix(1, 2);
  for (auto& annotation : *estimated_box->mutable_annotations()) {
    Eigen::Matrix<float, 16, 12, Eigen::RowMajor> m =
        Eigen::Matrix<float, 16, 12, Eigen::RowMajor>::Zero(16, 12);
    CHECK_EQ(9, annotation.keypoints_size());
    float u, v;
    for (int i = 0; i < 8; ++i) {
      const auto& keypoint2d = annotation.keypoints(i + 1).point_2d();
      if (portrait) {
        // swap x and y given that our image is in portrait orientation
        u = keypoint2d.y() * 2 - 1;
        v = keypoint2d.x() * 2 - 1;
      } else {
        u = keypoint2d.x() * 2 - 1;
        v = 1 - keypoint2d.y() * 2;  // (1 - keypoint2d.y()) * 2 - 1
      }
      for (int j = 0; j < 4; ++j) {
        // For each of the 4 control points, formulate two rows of the
        // m matrix (two equations).
        const float control_alpha = epnp_alpha_(i, j);
        m(i * 2, j * 3) = fx * control_alpha;
        m(i * 2, j * 3 + 2) = (cx + u) * control_alpha;
        m(i * 2 + 1, j * 3 + 1) = fy * control_alpha;
        m(i * 2 + 1, j * 3 + 2) = (cy + v) * control_alpha;
      }
    }
    // This is a self adjoint matrix. Use SelfAdjointEigenSolver for a fast
    // and stable solution.
    Eigen::Matrix<float, 12, 12, Eigen::RowMajor> mt_m = m.transpose() * m;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 12, 12, Eigen::RowMajor>>
        eigen_solver(mt_m);
    if (eigen_solver.info() != Eigen::Success) {
      return absl::AbortedError("Eigen decomposition failed.");
    }
    CHECK_EQ(12, eigen_solver.eigenvalues().size());
    // Eigenvalues are sorted in increasing order for SelfAdjointEigenSolver
    // only! If you use other Eigen Solvers, it's not guaranteed to be in
    // increasing order. Here, we just take the eigen vector corresponding
    // to first/smallest eigen value, since we used SelfAdjointEigenSolver.
    Eigen::VectorXf eigen_vec = eigen_solver.eigenvectors().col(0);
    Eigen::Map<Eigen::Matrix<float, 4, 3, Eigen::RowMajor>> control_matrix(
        eigen_vec.data());
    if (control_matrix(0, 2) > 0) {
      control_matrix = -control_matrix;
    }
    // First set the center keypoint.
    SetPoint3d(control_matrix(0, 0), control_matrix(0, 1), control_matrix(0, 2),
               annotation.mutable_keypoints(0)->mutable_point_3d());
    // Then set the 8 vertices.
    Eigen::Matrix<float, 8, 3, Eigen::RowMajor> vertices =
        epnp_alpha_ * control_matrix;
    for (int i = 0; i < 8; ++i) {
      SetPoint3d(vertices(i, 0), vertices(i, 1), vertices(i, 2),
                 annotation.mutable_keypoints(i + 1)->mutable_point_3d());
    }
  }
  return absl::OkStatus();
}

}  // namespace mediapipe
