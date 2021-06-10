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

#include "mediapipe/util/tracking/region_flow_visualization.h"

#include <stddef.h>

#include <memory>
#include <numeric>

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/util/tracking/measure_time.h"
#include "mediapipe/util/tracking/parallel_invoker.h"
#include "mediapipe/util/tracking/region_flow.h"

namespace mediapipe {

void VisualizeRegionFlowImpl(const RegionFlowFrame& region_flow_frame,
                             cv::Mat* output) {
  // Just draw a green line for each feature.
  for (const auto& region_flow : region_flow_frame.region_flow()) {
    cv::Scalar color(0, 255, 0);

    for (const auto& feature : region_flow.feature()) {
      cv::Point p1(static_cast<int>(feature.x() + 0.5f),
                   static_cast<int>(feature.y() + 0.5f));

      cv::Point p2(static_cast<int>(feature.x() + feature.dx() + 0.5f),
                   static_cast<int>(feature.y() + feature.dy() + 0.5f));

      cv::line(*output, p1, p2, color);
    }
  }
}

void VisualizeRegionFlow(const RegionFlowFrame& region_flow_frame,
                         cv::Mat* output) {
  CHECK(output);
  VisualizeRegionFlowImpl(region_flow_frame, output);
}

void VisualizeRegionFlowFeaturesImpl(const RegionFlowFeatureList& feature_list,
                                     const cv::Scalar& color,
                                     const cv::Scalar& outlier,
                                     bool irls_visualization, float scale_x,
                                     float scale_y, cv::Mat* output) {
  const float kLineScaleOneRows = 1080.0;
  const int line_size =
      std::max(std::min<int>(output->rows / kLineScaleOneRows, 4), 1);

  const float text_scale = std::max(output->rows, output->cols) * 3e-4;
  for (const auto& feature : feature_list.feature()) {
    Vector2_i v1 = FeatureIntLocation(feature);
    Vector2_i v2 = FeatureMatchIntLocation(feature);
    cv::Point p1(v1.x() * scale_x, v1.y() * scale_y);
    cv::Point p2(v2.x() * scale_x, v2.y() * scale_y);

    float alpha = 0;
    if (irls_visualization) {
      if (feature.irls_weight() < 0) {
        continue;
      }

      if (feature.irls_weight() == 0.0f) {
        // Ignored feature, draw as circle.
        cv::circle(*output, p1, 4.0 * line_size, outlier, 2 * line_size);
        continue;
      }

      // Scale irls_weight, such that error of one pixel equals alpha of 0.5
      // (evenly mix inlier and outlier color).
      alpha = std::min(1.0f, feature.irls_weight() * 0.5f);

    } else {
      const float feature_stdev_l1 =
          PatchDescriptorColorStdevL1(feature.feature_descriptor());
      if (feature_stdev_l1 >= 0.0f) {
        // Normalize w.r.t. maximum per channel standard deviation (128.0)
        // and scale (scale could be user-defined and was chosen based on some
        // test videos such that only very low textured features are colored
        // with outlier color).
        alpha = std::min(1.0f, feature_stdev_l1 / 128.f * 6.f);
      } else {
        alpha = 0.5;  // Dummy value indicating, descriptor is not present.
      }
    }

#ifdef __APPLE__
    cv::Scalar color_scaled(color.mul(alpha) + outlier.mul(1.0f - alpha));
#else
    cv::Scalar color_scaled(color * alpha + outlier * (1.0f - alpha));
#endif
    cv::line(*output, p1, p2, color_scaled, line_size, cv::LINE_AA);
    cv::circle(*output, p1, 2.0 * line_size, color_scaled, line_size);

    if (feature.has_label()) {
      cv::putText(*output, absl::StrCat(" ", feature.label()), p1,
                  cv::FONT_HERSHEY_SIMPLEX, text_scale, color_scaled,
                  3.0 * text_scale, cv::LINE_AA);
    }
  }
}

void VisualizeRegionFlowFeatures(const RegionFlowFeatureList& feature_list,
                                 const cv::Scalar& color,
                                 const cv::Scalar& outlier,
                                 bool irls_visualization, float scale_x,
                                 float scale_y, cv::Mat* output) {
  CHECK(output);
  VisualizeRegionFlowFeaturesImpl(feature_list, color, outlier,
                                  irls_visualization, scale_x, scale_y, output);
}

void VisualizeLongFeatureStreamImpl(const LongFeatureStream& stream,
                                    const cv::Scalar& color,
                                    const cv::Scalar& outlier,
                                    int min_track_length,
                                    int max_points_per_track, float scale_x,
                                    float scale_y, cv::Mat* output) {
  const float text_scale = std::max(output->cols, output->rows) * 3e-4;
  for (const auto& track : stream) {
    std::vector<Vector2_f> pts;
    std::vector<float> irls_weights;
    stream.FlattenTrack(track.second, &pts, &irls_weights, nullptr);

    if (min_track_length > 0 && pts.size() < min_track_length) {
      continue;
    }
    CHECK_GT(pts.size(), 1);  // Should have at least two points per track.

    // Tracks are ordered with oldest point first, most recent one last.
    const int start_k =
        max_points_per_track > 0
            ? std::max<int>(0, pts.size() - max_points_per_track)
            : 0;
    // Draw points in last to first order.
    for (int k = start_k; k < pts.size(); ++k) {
      cv::Point p1(pts[k].x() * scale_x, pts[k].y() * scale_y);

      if (irls_weights[k] == 0.0f) {
        // Ignored feature, draw as circle.
        cv::circle(*output, p1, 4.0, outlier, 2);
        continue;
      }

      const float alpha = std::min(1.0f, irls_weights[k] * 0.5f);
#ifdef __APPLE__
      cv::Scalar color_scaled(color.mul(alpha) + outlier.mul(1.0f - alpha));
#else
      cv::Scalar color_scaled(color * alpha + outlier * (1.0f - alpha));
#endif

      if (k < pts.size() - 1) {
        // Draw line connecting points.
        cv::Point p2(pts[k + 1].x() * scale_x, pts[k + 1].y() * scale_y);
        cv::line(*output, p1, p2, color_scaled, 1.0, cv::LINE_AA);
      }
      if (k + 1 == pts.size()) {  // Last iteration.
        cv::circle(*output, p1, 2.0, color_scaled, 1.0);

        const RegionFlowFeature& latest_feature = track.second.back();
        if (latest_feature.has_label()) {
          cv::putText(*output, absl::StrCat(" ", latest_feature.label()), p1,
                      cv::FONT_HERSHEY_SIMPLEX, text_scale, color_scaled,
                      3 * text_scale, cv::LINE_AA);
        }
      }
    }
  }
}

void VisualizeLongFeatureStream(const LongFeatureStream& stream,
                                const cv::Scalar& color,
                                const cv::Scalar& outlier, int min_track_length,
                                int max_points_per_track, float scale_x,
                                float scale_y, cv::Mat* output) {
  CHECK(output);
  VisualizeLongFeatureStreamImpl(stream, color, outlier, min_track_length,

                                 max_points_per_track, scale_x, scale_y,
                                 output);
}

}  // namespace mediapipe
