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

#include "mediapipe/modules/objectron/calculators/decoder.h"

#include <limits>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/modules/objectron/calculators/annotation_data.pb.h"
#include "mediapipe/modules/objectron/calculators/box.h"
#include "mediapipe/modules/objectron/calculators/epnp.h"
#include "mediapipe/modules/objectron/calculators/types.h"

namespace mediapipe {

constexpr int Decoder::kNumOffsetmaps = 16;
constexpr int kNumKeypoints = 9;

namespace {

inline void SetPoint3d(const Eigen::Vector3f& point_vec, Point3D* point_3d) {
  point_3d->set_x(point_vec.x());
  point_3d->set_y(point_vec.y());
  point_3d->set_z(point_vec.z());
}

}  // namespace

FrameAnnotation Decoder::DecodeBoundingBoxKeypoints(
    const cv::Mat& heatmap, const cv::Mat& offsetmap) const {
  ABSL_CHECK_EQ(1, heatmap.channels());
  ABSL_CHECK_EQ(kNumOffsetmaps, offsetmap.channels());
  ABSL_CHECK_EQ(heatmap.cols, offsetmap.cols);
  ABSL_CHECK_EQ(heatmap.rows, offsetmap.rows);

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
  ABSL_CHECK(estimated_box != nullptr);

  for (auto& annotation : *estimated_box->mutable_annotations()) {
    ABSL_CHECK_EQ(kNumKeypoints, annotation.keypoints_size());

    // Fill input 2D Points;
    std::vector<Vector2f> input_points_2d;
    input_points_2d.reserve(kNumKeypoints);
    for (const auto& keypoint : annotation.keypoints()) {
      input_points_2d.emplace_back(keypoint.point_2d().x(),
                                   keypoint.point_2d().y());
    }

    // Run EPnP.
    std::vector<Vector3f> output_points_3d;
    output_points_3d.reserve(kNumKeypoints);
    auto status = SolveEpnp(projection_matrix, portrait, input_points_2d,
                            &output_points_3d);
    if (!status.ok()) {
      ABSL_LOG(ERROR) << status;
      return status;
    }

    // Fill 3D keypoints;
    for (int i = 0; i < kNumKeypoints; ++i) {
      SetPoint3d(output_points_3d[i],
                 annotation.mutable_keypoints(i)->mutable_point_3d());
    }

    // Fit a box to the 3D points to get box scale, rotation, translation.
    Box box("category");
    box.Fit(output_points_3d);
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotation =
        box.GetRotation();
    const Eigen::Vector3f translation = box.GetTranslation();
    const Eigen::Vector3f scale = box.GetScale();
    // Fill box rotation.
    *annotation.mutable_rotation() = {rotation.data(),
                                      rotation.data() + rotation.size()};
    // Fill box translation.
    *annotation.mutable_translation() = {
        translation.data(), translation.data() + translation.size()};
    // Fill box scale.
    *annotation.mutable_scale() = {scale.data(), scale.data() + scale.size()};
  }
  return absl::OkStatus();
}

}  // namespace mediapipe
