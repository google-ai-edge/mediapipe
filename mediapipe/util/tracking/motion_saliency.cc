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

#include "mediapipe/util/tracking/motion_saliency.h"

#include <math.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <iterator>
#include <list>
#include <memory>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "mediapipe/util/tracking/camera_motion.h"
#include "mediapipe/util/tracking/measure_time.h"
#include "mediapipe/util/tracking/region_flow.h"
#include "mediapipe/util/tracking/region_flow.pb.h"

namespace mediapipe {

MotionSaliency::MotionSaliency(const MotionSaliencyOptions& options,
                               int frame_width, int frame_height)
    : options_(options),
      frame_width_(frame_width),
      frame_height_(frame_height) {}

MotionSaliency::~MotionSaliency() {}

void MotionSaliency::SaliencyFromFeatures(
    const RegionFlowFeatureList& feature_list,
    std::vector<float>* irls_weights,  // optional.
    SalientPointFrame* salient_frame) {
  ABSL_CHECK(salient_frame);
  ABSL_CHECK_EQ(frame_width_, feature_list.frame_width());
  ABSL_CHECK_EQ(frame_height_, feature_list.frame_height());

  if (irls_weights) {
    ABSL_CHECK_EQ(feature_list.feature_size(), irls_weights->size());
  }

  if (feature_list.feature_size() < 1) {
    return;
  }

  float max_irls_weight = 0;
  if (irls_weights) {
    max_irls_weight =
        *std::max_element(irls_weights->begin(), irls_weights->end());
  } else {
    struct FeatureIRLSComparator {
      bool operator()(const RegionFlowFeature& lhs,
                      const RegionFlowFeature& rhs) const {
        return lhs.irls_weight() < rhs.irls_weight();
      }
    };
    max_irls_weight =
        std::max_element(feature_list.feature().begin(),
                         feature_list.feature().end(), FeatureIRLSComparator())
            ->irls_weight();
  }

  // Max weight is too small for meaningful mode finding, terminate.
  if (max_irls_weight < 1e-2f) {
    return;
  }

  // Discard small weights that just slow clustering down.
  const float irls_cutoff = max_irls_weight * 1e-2f;

  int feat_idx = 0;

  // Create SalientLocation's from input feature_list.
  std::vector<SalientLocation> features;
  for (const auto& src_feature : feature_list.feature()) {
    const float weight =
        irls_weights ? (*irls_weights)[feat_idx] : src_feature.irls_weight();
    ++feat_idx;

    // Discard all features with small measure or zero weight from mode finding.
    if (weight < irls_cutoff) {
      continue;
    }

    features.push_back(SalientLocation(FeatureLocation(src_feature), weight));
  }

  DetermineSalientFrame(features, salient_frame);
}

void MotionSaliency::SaliencyFromPoints(const std::vector<Vector2_f>* points,
                                        const std::vector<float>* weights,
                                        SalientPointFrame* salient_frame) {
  // TODO: Handle vectors of size zero.
  ABSL_CHECK(salient_frame);
  ABSL_CHECK_EQ(points->size(), weights->size());

  float max_weight = *std::max_element(weights->begin(), weights->end());

  // Max weight is too small for meaningful mode finding, terminate.
  if (max_weight < 1e-2f) {
    return;
  }

  // Discard small weights that just slow clustering down.
  const float weight_cutoff = max_weight * 1e-2f;

  // Create SalientLocation's from input points.
  std::vector<SalientLocation> features;
  for (int point_idx = 0; point_idx < points->size(); ++point_idx) {
    const float weight = (*weights)[point_idx];
    // Discard all features with small measure or zero weight from mode finding.
    if (weight < weight_cutoff) {
      continue;
    }

    features.push_back(SalientLocation((*points)[point_idx], weight));
  }

  DetermineSalientFrame(features, salient_frame);
}

// We only keep those salient points that have neighbors along the temporal
// dimension.
void MotionSaliency::SelectSaliencyInliers(
    std::vector<SalientPointFrame*>* motion_saliency,
    bool rescale_to_median_saliency_weight) {
  float scale = 1.0;

  if (rescale_to_median_saliency_weight) {
    // Compute median saliency weight across all frames, to rescale saliency.
    std::vector<float> saliency_weights;
    for (int i = 0; i < motion_saliency->size(); ++i) {
      for (const auto& salient_point : (*motion_saliency)[i]->point()) {
        saliency_weights.push_back(salient_point.weight());
      }
    }

    // Nothing to filter in the frame chunk.
    if (saliency_weights.empty()) {
      return;
    }

    auto median_iter = saliency_weights.begin() + saliency_weights.size() / 2;
    std::nth_element(saliency_weights.begin(), median_iter,
                     saliency_weights.end());

    const float median_weight = *median_iter;
    if (median_weight > 0) {
      scale = options_.saliency_weight() / median_weight;
    }
  }

  SaliencyPointList inlier_saliency(motion_saliency->size());
  const float sq_support_distance = options_.selection_support_distance() *
                                    options_.selection_support_distance();

  // Test each point salient point for inlierness.
  for (int i = 0; i < motion_saliency->size(); ++i) {
    for (const auto& salient_point : (*motion_saliency)[i]->point()) {
      int support = 0;
      Vector2_f salient_location(salient_point.norm_point_x(),
                                 salient_point.norm_point_y());

      // Find supporting points (saliency points close enough to current one)
      // in adjacent frames. Linear Complexity.
      for (int j = std::max<int>(0, i - options_.selection_frame_radius()),
               end_j = std::min<int>(i + options_.selection_frame_radius(),
                                     motion_saliency->size() - 1);

           j <= end_j; ++j) {
        if (i == j) {
          continue;
        }

        for (const auto& compare_point : (*motion_saliency)[j]->point()) {
          Vector2_f compare_location(compare_point.norm_point_x(),
                                     compare_point.norm_point_y());

          if ((salient_location - compare_location).Norm2() <=
              sq_support_distance) {
            ++support;
          }
        }
      }  // end neighbor frames iteration.

      if (support >= options_.selection_minimum_support()) {
        SalientPoint* scaled_point = inlier_saliency[i].add_point();
        scaled_point->CopyFrom(salient_point);
        scaled_point->set_weight(scaled_point->weight() * scale);
      }
    }  // end point traversal.
  }  // end frame traversal.

  for (int k = 0; k < motion_saliency->size(); ++k) {
    (*motion_saliency)[k]->Swap(&inlier_saliency[k]);
  }
}

void MotionSaliency::FilterMotionSaliency(
    std::vector<SalientPointFrame*>* saliency_point_list) {
  ABSL_CHECK(saliency_point_list != nullptr);
  const float sigma_time = options_.filtering_sigma_time();
  const float sigma_space = options_.filtering_sigma_space();

  const int time_radius = ceil(sigma_time * 1.5);
  const int time_diameter = 2 * time_radius + 1;

  // Create lookup table for weights.
  std::vector<float> time_weights(time_diameter);
  const float time_coeff = -0.5f / (sigma_time * sigma_time);

  for (int i = -time_radius, time_idx = 0; i <= time_radius; ++i, ++time_idx) {
    time_weights[time_idx] = std::exp(time_coeff * i * i);
  }

  // Ignore points further than 1.65 sigmas away (includes 90% of distribution).
  const float space_cutoff = 1.65 * sigma_space;
  const float space_exp_scale = -0.5f / (sigma_space * sigma_space);

  // Copy saliency points.
  const int num_frames = saliency_point_list->size();
  std::vector<SalientPointFrame> points(num_frames + 2 * time_radius);

  for (int k = 0; k < saliency_point_list->size(); ++k) {
    points[time_radius + k].CopyFrom(*(*saliency_point_list)[k]);
  }

  // Copy border.
  std::copy(points.rbegin() + time_radius, points.rbegin() + 2 * time_radius,
            points.end() - time_radius);
  std::copy(points.begin() + time_radius, points.begin() + 2 * time_radius,
            points.rend() - time_radius);

  // Apply filter.
  for (int i = time_radius; i < num_frames + time_radius; ++i) {
    const int frame_idx = i - time_radius;
    for (auto& sample_point :
         *(*saliency_point_list)[frame_idx]->mutable_point()) {
      Vector2_f point_sum(0, 0);
      // Sum for left, bottom, right, top tuple.
      Vector4_f bound_sum;
      Vector3_f ellipse_sum(0, 0, 0);  // Captures major, minor and angle.
      float weight_sum = 0;
      float filter_sum = 0;

      const float sample_angle = sample_point.angle();
      for (int k = i - time_radius, time_idx = 0; k <= i + time_radius;
           ++k, ++time_idx) {
        for (const auto& test_point : points[k].point()) {
          const float diff = std::hypot(
              test_point.norm_point_y() - sample_point.norm_point_y(),
              test_point.norm_point_x() - sample_point.norm_point_x());
          if (diff > space_cutoff) {
            continue;
          }

          const float weight = time_weights[time_idx] * test_point.weight() *
                               std::exp(diff * diff * space_exp_scale);

          filter_sum += weight;
          point_sum +=
              Vector2_f(test_point.norm_point_x(), test_point.norm_point_y()) *
              weight;
          bound_sum += Vector4_f(test_point.left(), test_point.bottom(),
                                 test_point.right(), test_point.top()) *
                       weight;
          weight_sum += test_point.weight() * weight;

          // Ensure test_point and sample are less than pi / 2 apart.
          float test_angle = test_point.angle();
          if (fabs(test_angle - sample_angle) > M_PI / 2) {
            if (sample_angle > M_PI / 2) {
              test_angle += M_PI;
            } else {
              test_angle -= M_PI;
            }
          }

          ellipse_sum += Vector3_f(test_point.norm_major(),
                                   test_point.norm_minor(), test_angle) *
                         weight;
        }
      }

      if (filter_sum > 0) {
        const float inv_filter_sum = 1.0f / filter_sum;
        point_sum *= inv_filter_sum;
        bound_sum *= inv_filter_sum;
        weight_sum *= inv_filter_sum;
        ellipse_sum *= inv_filter_sum;
      }

      sample_point.set_norm_point_x(point_sum.x());
      sample_point.set_norm_point_y(point_sum.y());
      sample_point.set_left(bound_sum.x());
      sample_point.set_bottom(bound_sum.y());
      sample_point.set_right(bound_sum.z());
      sample_point.set_top(bound_sum.w());

      sample_point.set_weight(weight_sum);
      sample_point.set_norm_major(ellipse_sum.x());
      sample_point.set_norm_minor(ellipse_sum.y());
      sample_point.set_angle(ellipse_sum.z());

      if (sample_point.angle() > M_PI) {
        sample_point.set_angle(sample_point.angle() - M_PI);
      }
      if (sample_point.angle() < 0) {
        sample_point.set_angle(sample_point.angle() + M_PI);
      }
    }
  }
}

void MotionSaliency::CollapseMotionSaliency(
    const SaliencyPointList& input_saliency, const Vector4_f& bounds,
    SaliencyPointList* output_saliency) {
  ABSL_CHECK(output_saliency);
  output_saliency->clear();
  output_saliency->resize(input_saliency.size());

  for (int f = 0; f < input_saliency.size(); ++f) {  // traverse frames.
    Vector2_f mean_saliency(0, 0);
    float weight_sum = 0;
    for (const auto& salient_point : input_saliency[f].point()) {
      mean_saliency +=
          Vector2_f(salient_point.norm_point_x(), salient_point.norm_point_y());
      weight_sum += 1;
    }

    if (weight_sum > 0) {
      SalientPoint* collapsed = (*output_saliency)[f].add_point();
      collapsed->set_norm_point_x(mean_saliency.x() / weight_sum);
      collapsed->set_norm_point_y(mean_saliency.y() / weight_sum);
      collapsed->set_left(bounds.x());
      collapsed->set_bottom(bounds.y());
      collapsed->set_right(bounds.z());
      collapsed->set_top(bounds.w());
      collapsed->set_weight(1.0f);
    }
  }
}

namespace {

// Describes feature mode for a feature in a RegionFlowFeatureList stored
// at index feature_idx.
struct FeatureMode {
  Vector2_f location;
  float irls_weight;
  int feature_idx;
  int mode_bin;
};

// Determines mode for each feature in feature_view.
// Returns modes as list of pointers in mode_ptrs. Actual modes are stored
// binned into a grid of equal size as the passed FeatureGrid
// (bins of size grid_resolution x grid_resolution).
void DetermineFeatureModes(
    const FeatureFrame<MotionSaliency::SalientLocation>& features,
    float grid_resolution, const Vector2_i& grid_dims, float band_width,
    const FeatureGrid<MotionSaliency::SalientLocation>& feature_grid,
    const std::vector<std::vector<int>>& feature_taps,
    const std::vector<float>& space_lut, float space_scale,
    std::vector<std::list<FeatureMode>>* mode_grid,
    std::vector<FeatureMode*>* mode_ptrs) {
  ABSL_CHECK(mode_grid);
  ABSL_CHECK(mode_ptrs);
  const int num_features = features.size();
  mode_ptrs->reserve(num_features);

  const float grid_scale = 1.0f / grid_resolution;
  int feature_idx = 0;
  const int kMaxIter = 100;
  // Set convergence radius to 0.1% of bandwidth.
  const float sq_conv_radius = band_width * band_width * 1e-6f;
  for (const auto& feature_ptr : features) {
    Vector2_f center = feature_ptr->pt;
    int iter = 0;
    for (; iter < kMaxIter; ++iter) {
      const int bin_x = center.x() * grid_scale;
      const int bin_y = center.y() * grid_scale;
      const int grid_loc = bin_y * grid_dims.x() + bin_x;

      float sum_weight = 0;
      Vector2_f new_center;
      for (const auto& bin : feature_taps[grid_loc]) {
        for (const auto& test_feat_ptr : feature_grid[bin]) {
          const float dist = (test_feat_ptr->pt - center).Norm();
          const float weight = space_lut[static_cast<int>(dist * space_scale)] *
                               test_feat_ptr->weight;
          sum_weight += weight;
          new_center += weight * test_feat_ptr->pt;
        }
      }

      if (sum_weight > 0) {
        new_center *= (1.0f / sum_weight);
        if ((center - new_center).Norm2() < sq_conv_radius) {
          center = new_center;
          break;
        } else {
          center = new_center;
        }
      } else {
        ABSL_LOG(WARNING) << "No features found in band_width radius, "
                          << "should not happen. ";
        break;
      }
    }

    const int mode_bin_x = center.x() * grid_scale;
    const int mode_bin_y = center.y() * grid_scale;
    const int mode_grid_loc = mode_bin_y * grid_dims.x() + mode_bin_x;
    FeatureMode mode{center, feature_ptr->weight, feature_idx, mode_grid_loc};

    (*mode_grid)[mode_grid_loc].push_back(mode);
    FeatureMode* added_mode = &(*mode_grid)[mode_grid_loc].back();
    (*mode_ptrs).push_back(added_mode);
    ++feature_idx;
  }
}

}  // namespace.

void MotionSaliency::SalientModeFinding(std::vector<SalientLocation>* locations,
                                        std::vector<SalientMode>* modes) {
  ABSL_CHECK(modes);
  ABSL_CHECK(locations);
  if (locations->empty()) {
    return;
  }

  // Scale band_width to image domain.
  const float band_width =
      hypot(frame_width_, frame_height_) * options_.mode_band_width();

  // Select all salient locations with non-zero weight.
  FeatureFrame<SalientLocation> salient_features;
  salient_features.reserve(locations->size());
  for (auto& loc : *locations) {
    if (loc.weight > 1e-6) {
      salient_features.push_back(&loc);
    }
  }

  const int num_features = salient_features.size();
  if (num_features == 0) {
    return;
  }

  // Build feature grid according to bandwith.
  std::vector<FeatureGrid<SalientLocation>> feature_grids;
  std::vector<std::vector<int>> feature_taps;

  // Guarantee at least 1.5 sigmas in each direction are captured with
  // tap 3 filtering (86 % of the data).
  const float grid_resolution = 1.5f * band_width;
  Vector2_i grid_dims;
  BuildFeatureGrid(
      frame_width_, frame_height_, grid_resolution, {salient_features},
      [](const SalientLocation& l) -> Vector2_f { return l.pt; }, &feature_taps,
      nullptr, &grid_dims, &feature_grids);

  // Just one frame input, expect one grid as output.
  ABSL_CHECK_EQ(1, feature_grids.size());
  const auto& feature_grid = feature_grids[0];

  // Setup Gaussian LUT for smoothing in space, using 2^10 discretization bins.
  const int lut_bins = 1 << 10;
  std::vector<float> space_lut(lut_bins);

  // Using 3 tap smoothing, max distance is 2 bin diagonals.
  // We use maximum of 2 * sqrt(2) * bin_radius plus 1% room in case maximum
  // value is attained.
  const float max_space_diff = sqrt(2.0) * 2.f * grid_resolution * 1.01f;

  const float space_bin_size = max_space_diff / lut_bins;
  const float space_scale = 1.0f / space_bin_size;
  const float space_coeff = -0.5f / (band_width * band_width);
  for (int i = 0; i < lut_bins; ++i) {
    const float value = i * space_bin_size;
    space_lut[i] = std::exp(value * value * space_coeff);
  }

  // Store modes for each grid bin (to be averaged later).
  std::vector<std::list<FeatureMode>> mode_grid(grid_dims.x() * grid_dims.y());
  std::vector<FeatureMode*> mode_ptrs;

  DetermineFeatureModes(salient_features, grid_resolution, grid_dims,
                        band_width, feature_grid, feature_taps, space_lut,
                        space_scale, &mode_grid, &mode_ptrs);

  // Read out modes, ordered by decreasing weight.
  struct FeatureModeComparator {
    bool operator()(const FeatureMode* mode_lhs,
                    const FeatureMode* mode_rhs) const {
      return mode_lhs->irls_weight > mode_rhs->irls_weight;
    }
  };

  // Sort pointers, to keep order immutable during flagging operations.
  std::sort(mode_ptrs.begin(), mode_ptrs.end(), FeatureModeComparator());

  for (int m = 0; m < mode_ptrs.size(); ++m) {
    // We mark a mode as processed by assigning -1 to its index.
    if (mode_ptrs[m]->feature_idx < 0) {
      continue;
    }

    FeatureMode* mode = mode_ptrs[m];

    // Average modes within band_width based on irls_weight * spatial weight.
    double sum_weight = mode->irls_weight;
    double mode_x = sum_weight * mode->location.x();
    double mode_y = sum_weight * mode->location.y();

    const Vector2_f& feat_loc = salient_features[mode->feature_idx]->pt;
    double feat_x = sum_weight * feat_loc.x();
    double feat_y = sum_weight * feat_loc.y();
    double feat_xx = sum_weight * feat_loc.x() * feat_loc.x();
    double feat_xy = sum_weight * feat_loc.x() * feat_loc.y();
    double feat_yy = sum_weight * feat_loc.y() * feat_loc.y();

    mode->feature_idx = -1;  // Flag as processed, does not change order
                             // of traversal.

    for (const auto& bin : feature_taps[mode->mode_bin]) {
      for (auto& test_mode : mode_grid[bin]) {
        if (test_mode.feature_idx >= 0) {
          const float dist = (test_mode.location - mode->location).Norm();
          if (dist <= band_width) {
            const Vector2_f test_loc =
                salient_features[test_mode.feature_idx]->pt;
            const float weight =
                space_lut[static_cast<int>(dist * space_scale)] *
                test_mode.irls_weight;

            sum_weight += weight;
            mode_x += weight * test_mode.location.x();
            mode_y += weight * test_mode.location.y();

            const float test_loc_x_w = weight * test_loc.x();
            const float test_loc_y_w = weight * test_loc.y();
            feat_x += test_loc_x_w;
            feat_y += test_loc_y_w;

            feat_xx += test_loc_x_w * test_loc.x();
            feat_xy += test_loc_x_w * test_loc.y();
            feat_yy += test_loc_y_w * test_loc.y();

            // Flag as processed, does not change order of traversal.
            test_mode.feature_idx = -1;
          }
        }
      }
    }

    if (sum_weight >= options_.min_irls_mode_weight()) {
      double inv_sum_weight = 1.0f / sum_weight;
      mode_x *= inv_sum_weight;
      mode_y *= inv_sum_weight;
      feat_x *= inv_sum_weight;
      feat_y *= inv_sum_weight;
      feat_xx *= inv_sum_weight;
      feat_xy *= inv_sum_weight;
      feat_yy *= inv_sum_weight;

      // Covariance matrix.
      const float a = feat_xx - 2.0 * feat_x * mode_x + mode_x * mode_x;
      const float bc =
          feat_xy - feat_x * mode_y - feat_y * mode_x + mode_x * mode_y;
      const float d = feat_yy - 2.0 * feat_y * mode_y + mode_y * mode_y;

      Vector2_f axis_magnitude;
      float angle;
      if (!EllipseFromCovariance(a, bc, d, &axis_magnitude, &angle)) {
        angle = 0;
        axis_magnitude = Vector2_f(1, 1);
      } else {
        if (angle < 0) {
          angle += M_PI;
        }
        ABSL_CHECK_GE(angle, 0);
        ABSL_CHECK_LE(angle, M_PI + 1e-3);
      }

      SalientMode irls_mode;
      irls_mode.location = Vector2_f(mode_x, mode_y);
      irls_mode.assignment_weight = sum_weight;
      irls_mode.axis_magnitude = axis_magnitude;
      irls_mode.angle = angle;
      modes->push_back(irls_mode);
    }
  }

  // Sort modes by descreasing weight.
  struct ModeWeightCompare {
    bool operator()(const SalientMode& lhs, const SalientMode& rhs) const {
      return lhs.assignment_weight > rhs.assignment_weight;
    }
  };

  std::sort(modes->begin(), modes->end(), ModeWeightCompare());
}

// Determines the salient frame for a list of SalientLocations by performing
// mode finding and scales each point based on frame size.
void MotionSaliency::DetermineSalientFrame(
    std::vector<SalientLocation> locations, SalientPointFrame* salient_frame) {
  ABSL_CHECK(salient_frame);

  std::vector<SalientMode> modes;
  {
    MEASURE_TIME << "Mode finding";
    SalientModeFinding(&locations, &modes);
  }

  const float denom_x = 1.0f / frame_width_;
  const float denom_y = 1.0f / frame_height_;

  // Convert to salient points.
  for (int mode_idx = 0,
           mode_sz = std::min<int>(modes.size(), options_.num_top_irls_modes());
       mode_idx < mode_sz; ++mode_idx) {
    SalientPoint* pt = salient_frame->add_point();
    pt->set_norm_point_x(modes[mode_idx].location.x());
    pt->set_norm_point_y(modes[mode_idx].location.y());
    pt->set_left(options_.bound_left());
    pt->set_bottom(options_.bound_bottom());
    pt->set_right(options_.bound_right());
    pt->set_top(options_.bound_top());

    pt->set_norm_major(modes[mode_idx].axis_magnitude.x());
    pt->set_norm_minor(modes[mode_idx].axis_magnitude.y());
    pt->set_angle(modes[mode_idx].angle);
    pt->set_weight(modes[mode_idx].assignment_weight *
                   options_.saliency_weight());

    ScaleSalientPoint(denom_x, denom_y, pt);
  }
}

void ForegroundWeightsFromFeatures(const RegionFlowFeatureList& feature_list,
                                   float foreground_threshold,
                                   float foreground_gamma,
                                   const CameraMotion* camera_motion,
                                   std::vector<float>* weights) {
  ABSL_CHECK(weights != nullptr);
  weights->clear();

  constexpr float kEpsilon = 1e-4f;

  ABSL_CHECK_GT(foreground_threshold, 0.0f);
  if (camera_motion) {
    foreground_threshold *=
        std::max(kEpsilon, InlierCoverage(*camera_motion, false));
  }

  const float weight_denom = 1.0f / foreground_threshold;

  // Map weights to foreground measure and determine minimum irls weight.
  for (const auto& feature : feature_list.feature()) {
    // Skip marked outliers.
    if (feature.irls_weight() == 0) {
      weights->push_back(0.0f);
      continue;
    }

    // Maps an irls_weight of magnitude weight_denom (from above) to zero,
    // with values below weight_denom assigned linearly mapped (zero is mapped
    // ot 1). Avoid mapping to zero as it used to mark outliers.
    const float foreground_measure =
        std::max(0.0f, 1.0f - feature.irls_weight() * weight_denom);

    if (std::abs(foreground_gamma - 1.0f) < 1e-3f) {
      weights->push_back(std::max(kEpsilon, foreground_measure));
    } else {
      weights->push_back(
          std::max(kEpsilon, std::pow(foreground_measure, foreground_gamma)));
    }
  }
  ABSL_CHECK_EQ(feature_list.feature_size(), weights->size());
}

}  // namespace mediapipe
