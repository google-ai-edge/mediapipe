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

#include "mediapipe/util/tracking/tracking_visualization_utilities.h"

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_format.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/util/tracking/box_tracker.h"
#include "mediapipe/util/tracking/tracking.h"

namespace mediapipe {

void RenderState(const MotionBoxState& box_state, bool print_stats,
                 cv::Mat* frame) {
#ifndef NO_RENDERING
  ABSL_CHECK(frame != nullptr);

  const int frame_width = frame->cols;
  const int frame_height = frame->rows;

  const Vector2_f top_left(box_state.pos_x() * frame_width,
                           box_state.pos_y() * frame_height);

  cv::rectangle(*frame,
                cv::Point(box_state.inlier_center_x() * frame_width - 2,
                          box_state.inlier_center_y() * frame_height - 2),
                cv::Point(box_state.inlier_center_x() * frame_width + 2,
                          box_state.inlier_center_y() * frame_height + 2),
                cv::Scalar(255, 255, 0, 255), 2);

  cv::rectangle(
      *frame,
      cv::Point(
          (box_state.inlier_center_x() - box_state.inlier_width() * 0.5) *
              frame_width,
          (box_state.inlier_center_y() - box_state.inlier_height() * 0.5) *
              frame_height),
      cv::Point(
          (box_state.inlier_center_x() + box_state.inlier_width() * 0.5) *
              frame_width,
          (box_state.inlier_center_y() + box_state.inlier_height() * 0.5) *
              frame_height),
      cv::Scalar(0, 0, 255, 255), 1);

  std::vector<Vector2_f> inlier_locs;
  std::vector<Vector2_f> outlier_locs;
  MotionBoxInlierLocations(box_state, &inlier_locs);
  MotionBoxOutlierLocations(box_state, &outlier_locs);
  float scale_x = 1.0f;
  float scale_y = 1.0f;
  ScaleFromAspect(frame_width * 1.0f / frame_height, true, &scale_x, &scale_y);

  for (const Vector2_f& loc : inlier_locs) {
    cv::circle(*frame,
               cv::Point(loc.x() * scale_x * frame_width,
                         loc.y() * scale_y * frame_height),
               4.0, cv::Scalar(0, 255, 0, 128), 1);
  }
  for (const Vector2_f& loc : outlier_locs) {
    cv::circle(*frame,
               cv::Point(loc.x() * scale_x * frame_width,
                         loc.y() * scale_y * frame_height),
               4.0, cv::Scalar(255, 0, 0, 128), 1);
  }

  if (!print_stats) {
    return;
  }

  std::vector<std::string> stats;
  stats.push_back(
      absl::StrFormat("Motion: %.4f, %.4f", box_state.dx(), box_state.dy()));
  stats.push_back(
      absl::StrFormat("KinEnergy: %.4f", box_state.kinetic_energy()));
  stats.push_back(
      absl::StrFormat("Disparity: %.2f", box_state.motion_disparity()));
  stats.push_back(absl::StrFormat("Discrimination: %.2f",
                                  box_state.background_discrimination()));
  stats.push_back(
      absl::StrFormat("InlierRatio: %2.2f", box_state.inlier_ratio()));
  stats.push_back(
      absl::StrFormat("InlierNum: %3d", box_state.inlier_ids_size()));

  stats.push_back(absl::StrFormat("Prior: %.2f", box_state.prior_weight()));
  stats.push_back(absl::StrFormat("TrackingConfidence: %.2f",
                                  box_state.tracking_confidence()));

  for (int k = 0; k < stats.size(); ++k) {
    // Display some stats below the box.
    cv::putText(*frame, stats[k],
                cv::Point(top_left.x(), top_left.y() + (k + 1) * 12),
                cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 0, 255));

    cv::putText(*frame, stats[k],
                cv::Point(top_left.x() + 1, top_left.y() + (k + 1) * 12 + 1),
                cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 255, 255, 255));
  }

  // Visualize locking state via heuristically chosen thresholds. Locking
  // state is purely for visualization/illustration and has no further
  // meaning.
  std::string lock_text;
  cv::Scalar lock_color;
  if (box_state.motion_disparity() > 0.8) {
    lock_text = "Lock lost";
    lock_color = cv::Scalar(255, 0, 0, 255);
  } else if (box_state.motion_disparity() > 0.4) {
    lock_text = "Aquiring lock";
    lock_color = cv::Scalar(255, 255, 0, 255);
  } else if (box_state.motion_disparity() > 0.1) {
    lock_text = "Locked";
    lock_color = cv::Scalar(0, 255, 0, 255);
  }

  cv::putText(*frame, lock_text, cv::Point(top_left.x() + 1, top_left.y() - 4),
              cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(255, 255, 255, 255));

  cv::putText(*frame, lock_text, cv::Point(top_left.x(), top_left.y() - 5),
              cv::FONT_HERSHEY_PLAIN, 0.8, lock_color);
#else
  ABSL_LOG(FATAL) << "Code stripped out because of NO_RENDERING";
#endif
}

void RenderInternalState(const MotionBoxInternalState& internal,
                         cv::Mat* frame) {
#ifndef NO_RENDERING
  ABSL_CHECK(frame != nullptr);

  const int num_vectors = internal.pos_x_size();

  // Determine max inlier score for visualization.
  float max_score = 0;
  for (int k = 0; k < num_vectors; ++k) {
    max_score = std::max(max_score, internal.inlier_score(k));
  }

  float alpha_scale = 1.0f;
  if (max_score > 0) {
    alpha_scale = 1.0f / max_score;
  }

  const int frame_width = frame->cols;
  const int frame_height = frame->rows;
  for (int k = 0; k < num_vectors; ++k) {
    const MotionVector v = MotionVector::FromInternalState(internal, k);
    cv::Point p1(v.pos.x() * frame_width, v.pos.y() * frame_height);
    Vector2_f match = v.pos + v.object;

    cv::Point p2(match.x() * frame_width, match.y() * frame_height);
    const float alpha = internal.inlier_score(k) * alpha_scale;

    cv::Scalar color_scaled(0 * alpha + (1.0f - alpha) * 255,
                            255 * alpha + (1.0f - alpha) * 0,
                            0 * alpha + (1.0f - alpha) * 0);

    cv::line(*frame, p1, p2, color_scaled, 1, cv::LINE_AA);
    cv::circle(*frame, p1, 2.0, color_scaled, 1);
  }
#else
  ABSL_LOG(FATAL) << "Code stripped out because of NO_RENDERING";
#endif
}

void RenderTrackingData(const TrackingData& data, cv::Mat* mat,
                        bool antialiasing) {
#ifndef NO_RENDERING
  ABSL_CHECK(mat != nullptr);

  MotionVectorFrame mvf;
  MotionVectorFrameFromTrackingData(data, &mvf);
  const float aspect_ratio = mvf.aspect_ratio;
  float scale_x, scale_y;
  ScaleFromAspect(aspect_ratio, true, &scale_x, &scale_y);
  scale_x *= mat->cols;
  scale_y *= mat->rows;

  for (const auto& motion_vector : mvf.motion_vectors) {
    Vector2_f pos = motion_vector.Location();
    Vector2_f match = motion_vector.MatchLocation();

    cv::Point p1(pos.x() * scale_x, pos.y() * scale_y);
    cv::Point p2(match.x() * scale_x, match.y() * scale_y);

    // iOS cannot display a width 1 antialiased line, so we provide
    // an option for 8-connected drawing instead.
    cv::line(*mat, p1, p2, cv::Scalar(0, 255, 0, 255), 1,
             antialiasing ? cv::LINE_AA : 8);
  }
#else
  ABSL_LOG(FATAL) << "Code stripped out because of NO_RENDERING";
#endif
}

void RenderBox(const TimedBoxProto& box_proto, cv::Mat* mat) {
#ifndef NO_RENDERING
  ABSL_CHECK(mat != nullptr);

  TimedBox box = TimedBox::FromProto(box_proto);
  std::array<Vector2_f, 4> corners = box.Corners(mat->cols, mat->rows);

  for (int k = 0; k < 4; ++k) {
    cv::line(*mat, cv::Point2f(corners[k].x(), corners[k].y()),
             cv::Point2f(corners[(k + 1) % 4].x(), corners[(k + 1) % 4].y()),
             cv::Scalar(255, 0, 0, 255),  // Red.
             4);
  }
#else
  ABSL_LOG(FATAL) << "Code stripped out because of NO_RENDERING";
#endif
}

}  // namespace mediapipe
