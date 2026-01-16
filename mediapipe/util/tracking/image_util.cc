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

#include "mediapipe/util/tracking/image_util.h"

#include <algorithm>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "mediapipe/util/tracking/motion_models.h"
#include "mediapipe/util/tracking/region_flow.h"

namespace mediapipe {

// Returns median of the L1 color distance between img_1 and img_2
float FrameDifferenceMedian(const cv::Mat& img_1, const cv::Mat& img_2) {
  ABSL_CHECK(img_1.size() == img_2.size());
  ABSL_CHECK_EQ(img_1.channels(), img_2.channels());

  std::vector<float> color_diffs;
  color_diffs.reserve(img_1.cols * img_1.rows);
  const int channels = img_1.channels();
  for (int j = 0; j < img_1.rows; ++j) {
    const uint8_t* src_1 = img_1.ptr<uint8_t>(j);
    const uint8_t* src_2 = img_2.ptr<uint8_t>(j);
    const int end_i = channels * img_1.cols;
    const float inverse = 1.0f / channels;
    for (int i = 0; i < end_i;) {
      float color_diff = 0.0f;
      for (int k = 0; k < channels; ++k, ++i) {
        color_diff +=
            std::abs(static_cast<int>(src_1[i]) - static_cast<int>(src_2[i]));
      }
      color_diffs.push_back(color_diff * inverse);
    }
  }

  auto median = color_diffs.begin() + color_diffs.size() / 2;
  std::nth_element(color_diffs.begin(), median, color_diffs.end());
  return *median;
}

void JetColoring(int steps, std::vector<Vector3_f>* color_map) {
  ABSL_CHECK(color_map != nullptr);
  color_map->resize(steps);
  for (int i = 0; i < steps; ++i) {
    const float frac = 2.0f * (i * (1.0f / steps) - 0.5f);
    if (frac < -0.8f) {
      (*color_map)[i] = Vector3_f(0, 0, 0.6f + (frac + 1.0) * 2.0f) * 255.0f;
    } else if (frac >= -0.8f && frac < -0.25f) {
      (*color_map)[i] = Vector3_f(0, (frac + 0.8f) * 1.82f, 1.0f) * 255.0f;
    } else if (frac >= -0.25f && frac < 0.25f) {
      (*color_map)[i] =
          Vector3_f((frac + 0.25f) * 2.0f, 1.0f, 1.0 + (frac + 0.25f) * -2.0f) *
          255.0f;
    } else if (frac >= 0.25f && frac < 0.8f) {
      (*color_map)[i] =
          Vector3_f(1.0f, 1.0f + (frac - 0.25f) * -1.81f, 0.0f) * 255.0f;
    } else if (frac >= 0.8f) {
      (*color_map)[i] =
          Vector3_f(1.0f + (frac - 0.8f) * -2.0f, 0.0f, 0.0f) * 255.0f;
    } else {
      ABSL_LOG(ERROR) << "Out of bound value. Should not occur.";
    }
  }
}

void RenderSaliency(const SalientPointFrame& salient_points,
                    const cv::Scalar& line_color, int line_thickness,
                    bool render_bounding_box, cv::Mat* image) {
  // Visualize salient points.
  for (const auto& point : salient_points.point()) {
    if (point.weight() > 0) {
      SalientPoint copy = point;
      ScaleSalientPoint(image->cols, image->rows, &copy);
      Vector2_f pt(copy.norm_point_x(), copy.norm_point_y());

      cv::ellipse(*image, cv::Point(pt.x(), pt.y()),
                  cv::Size(copy.norm_major(), copy.norm_minor()),
                  copy.angle() / M_PI * 180.0,
                  0,    // start angle
                  360,  // end angle
                  line_color, line_thickness);

      if (render_bounding_box) {
        std::vector<Vector2_f> ellipse_bounding_box;
        BoundingBoxFromEllipse(pt, copy.norm_major(), copy.norm_minor(),
                               copy.angle(), &ellipse_bounding_box);

        std::vector<cv::Point> corners;
        corners.reserve(4);
        for (const Vector2_f& corner : ellipse_bounding_box) {
          corners.emplace_back(cv::Point(corner.x(), corner.y()));
        }

        for (int k = 0; k < 4; ++k) {
          cv::line(*image, corners[k], corners[(k + 1) % 4], line_color,
                   line_thickness, cv::LINE_AA);
        }
      }
    }
  }
}

}  // namespace mediapipe
