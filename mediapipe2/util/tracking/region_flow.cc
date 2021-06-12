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

#include "mediapipe/util/tracking/region_flow.h"

#include <stddef.h>

#include <cmath>
#include <memory>
#include <numeric>

#include "absl/container/node_hash_map.h"
#include "absl/container/node_hash_set.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/util/tracking/measure_time.h"
#include "mediapipe/util/tracking/parallel_invoker.h"

namespace mediapipe {

namespace {

bool IsPointWithinBounds(const Vector2_f& pt, float bounds, int frame_width,
                         int frame_height) {
  // Ensure stability under float -> rounding operations.
  if (pt.x() - 0.5f >= bounds && pt.x() + 0.5f <= frame_width - 1 - bounds &&
      pt.y() - 0.5f >= bounds && pt.y() + 0.5f <= frame_height - 1 - bounds) {
    return true;
  } else {
    return false;
  }
}

}  // namespace.

void GetRegionFlowFeatureList(const RegionFlowFrame& region_flow_frame,
                              int distance_from_border,
                              RegionFlowFeatureList* flow_feature_list) {
  CHECK(flow_feature_list);
  flow_feature_list->clear_feature();
  const int frame_width = region_flow_frame.frame_width();
  const int frame_height = region_flow_frame.frame_height();
  flow_feature_list->set_frame_width(frame_width);
  flow_feature_list->set_frame_height(frame_height);
  flow_feature_list->set_unstable(region_flow_frame.unstable_frame());
  flow_feature_list->set_distance_from_border(distance_from_border);
  flow_feature_list->set_blur_score(region_flow_frame.blur_score());

  for (const auto& region_flow : region_flow_frame.region_flow()) {
    for (const auto& feature : region_flow.feature()) {
      if (distance_from_border > 0) {
        if (!IsPointWithinBounds(FeatureLocation(feature), distance_from_border,
                                 frame_width, frame_height) ||
            !IsPointWithinBounds(FeatureMatchLocation(feature),
                                 distance_from_border, frame_width,
                                 frame_height)) {
          continue;
        }
      }

      flow_feature_list->add_feature()->CopyFrom(feature);
    }
  }
}

float RegionFlowFeatureDistance(const PatchDescriptor& patch_desc_1,
                                const PatchDescriptor& patch_desc_2) {
  DCHECK_EQ(patch_desc_1.data_size(), patch_desc_2.data_size());
  DCHECK_GE(patch_desc_1.data_size(), 3);

  constexpr int kNumMeans = 3;
  float sq_distance_sum = 0;
  for (int n = 0; n < kNumMeans; ++n) {
    const float distance = patch_desc_1.data(n) - patch_desc_2.data(n);
    sq_distance_sum += distance * distance;
  }

  return std::sqrt(sq_distance_sum);
}

void ResetRegionFlowFeatureIRLSWeights(
    float value, RegionFlowFeatureList* flow_feature_list) {
  for (auto& feature : *flow_feature_list->mutable_feature()) {
    feature.set_irls_weight(value);
  }
}

double RegionFlowFeatureIRLSSum(const RegionFlowFeatureList& feature_list) {
  double sum = 0.0;
  for (const auto& feature : feature_list.feature()) {
    sum += feature.irls_weight();
  }
  return sum;
}

void ClampRegionFlowFeatureIRLSWeights(float lower, float upper,
                                       RegionFlowFeatureView* feature_view) {
  for (auto& feature_ptr : *feature_view) {
    if (feature_ptr->irls_weight() < lower) {
      feature_ptr->set_irls_weight(lower);
    } else if (feature_ptr->irls_weight() > upper) {
      feature_ptr->set_irls_weight(upper);
    }
  }
}

void ComputeRegionFlowFeatureTexturedness(
    const RegionFlowFeatureList& flow_feature_list, bool use_15percent_as_max,
    std::vector<float>* texturedness) {
  CHECK(texturedness != nullptr);
  *texturedness = std::vector<float>(flow_feature_list.feature_size(), 1.0f);

  int texture_idx = 0;
  for (auto feature = flow_feature_list.feature().begin();
       feature != flow_feature_list.feature().end(); ++feature, ++texture_idx) {
    const float feature_stdev_l1 =
        PatchDescriptorColorStdevL1(feature->feature_descriptor());

    if (feature_stdev_l1 < 0.0f) {
      LOG_IF(WARNING,
             []() {
               static int k = 0;
               return k++ < 2;
             }())
          << "Feature descriptor does not contain variance information. Was "
          << "ComputeRegionFlowFeatureDescriptors called?";
      continue;
    }

    // feature_stdev_l1  is within [0, 3 * 128 = 384]
    float alpha = feature_stdev_l1 / 384.f;

    // In [0, 1], 0 = low texture, 1 = high texture. Scale such that around
    // 15% of per channel maximum stdev is considered totally textured
    // (1 / 0.15 * 3 ~ 18);
    if (use_15percent_as_max) {
      alpha = std::min(1.0f, alpha * 18.f);
    }

    (*texturedness)[texture_idx] = alpha;
  }
}

void TextureFilteredRegionFlowFeatureIRLSWeights(
    float low_texture_threshold, float low_texture_outlier_clamp,
    RegionFlowFeatureList* flow_feature_list) {
  std::vector<float> texturedness;
  ComputeRegionFlowFeatureTexturedness(*flow_feature_list, true, &texturedness);

  int texture_idx = 0;
  for (auto feature = flow_feature_list->mutable_feature()->begin();
       feature != flow_feature_list->mutable_feature()->end();
       ++feature, ++texture_idx) {
    if (feature->irls_weight() == 0.0f) {
      continue;
    }

    if (texturedness[texture_idx] < low_texture_threshold &&
        feature->irls_weight() < low_texture_outlier_clamp) {
      feature->set_irls_weight(low_texture_outlier_clamp);
    } else {
      feature->set_irls_weight(feature->irls_weight() /
                               (texturedness[texture_idx] + 1.e-6f));
    }
  }
}

void CornerFilteredRegionFlowFeatureIRLSWeights(
    float low_corner_threshold, float low_corner_outlier_clamp,
    RegionFlowFeatureList* flow_feature_list) {
  for (auto feature = flow_feature_list->mutable_feature()->begin();
       feature != flow_feature_list->mutable_feature()->end(); ++feature) {
    if (feature->irls_weight() == 0.0f) {
      continue;
    }

    const float corner_response = feature->corner_response();

    if (corner_response < low_corner_threshold &&
        feature->irls_weight() < low_corner_threshold) {
      feature->set_irls_weight(low_corner_outlier_clamp);
    } else {
      feature->set_irls_weight(feature->irls_weight() /
                               (corner_response + 1.e-6f));
    }
  }
}

void GetRegionFlowFeatureIRLSWeights(
    const RegionFlowFeatureList& flow_feature_list,
    std::vector<float>* irls_weights) {
  CHECK(irls_weights != nullptr);
  irls_weights->clear();
  irls_weights->reserve(flow_feature_list.feature_size());
  for (auto feature = flow_feature_list.feature().begin();
       feature != flow_feature_list.feature().end(); ++feature) {
    irls_weights->push_back(feature->irls_weight());
  }
}

void SetRegionFlowFeatureIRLSWeights(const std::vector<float>& irls_weights,
                                     RegionFlowFeatureList* flow_feature_list) {
  CHECK(flow_feature_list != nullptr);
  CHECK_EQ(irls_weights.size(), flow_feature_list->feature_size());
  int idx = 0;
  for (auto feature = flow_feature_list->mutable_feature()->begin();
       feature != flow_feature_list->mutable_feature()->end();
       ++feature, ++idx) {
    feature->set_irls_weight(irls_weights[idx]);
  }
}

int CountIgnoredRegionFlowFeatures(
    const RegionFlowFeatureList& flow_feature_list, float threshold) {
  int count = 0;
  for (auto feature = flow_feature_list.feature().begin();
       feature != flow_feature_list.feature().end(); ++feature) {
    if (feature->irls_weight() <= threshold) {
      ++count;
    }
  }
  return count;
}

namespace {

struct RegionFlowLocator {
  bool operator()(const RegionFlow& lhs, const RegionFlow& rhs) const {
    return lhs.region_id() < rhs.region_id();
  }
};

}  // namespace.

const RegionFlow* GetRegionFlowById(int region_id,
                                    const RegionFlowFrame& flow_frame) {
  RegionFlow region_flow;
  region_flow.set_region_id(region_id);

  const auto& region_pos = std::lower_bound(flow_frame.region_flow().begin(),
                                            flow_frame.region_flow().end(),
                                            region_flow, RegionFlowLocator());

  if (region_pos == flow_frame.region_flow().end() ||
      region_pos->region_id() != region_id) {
    return NULL;
  } else {
    return &(*region_pos);
  }
}

RegionFlow* GetMutableRegionFlowById(int region_id,
                                     RegionFlowFrame* flow_frame) {
  RegionFlow region_flow;
  region_flow.set_region_id(region_id);

  auto region_pos = std::lower_bound(flow_frame->mutable_region_flow()->begin(),
                                     flow_frame->mutable_region_flow()->end(),
                                     region_flow, RegionFlowLocator());

  if (region_pos == flow_frame->mutable_region_flow()->end() ||
      region_pos->region_id() != region_id) {
    return NULL;
  } else {
    return &(*region_pos);
  }
}

void SortRegionFlowById(RegionFlowFrame* flow_frame) {
  std::sort(flow_frame->mutable_region_flow()->begin(),
            flow_frame->mutable_region_flow()->end(), RegionFlowLocator());
}

void InvertRegionFlow(const RegionFlowFrame& region_flow_frame,
                      RegionFlowFrame* inverted_flow_frame) {
  CHECK(inverted_flow_frame);
  inverted_flow_frame->CopyFrom(region_flow_frame);
  for (auto& region_flow : *inverted_flow_frame->mutable_region_flow()) {
    region_flow.set_centroid_x(region_flow.centroid_x() + region_flow.flow_x());
    region_flow.set_centroid_y(region_flow.centroid_y() + region_flow.flow_y());
    region_flow.set_flow_x(-region_flow.flow_x());
    region_flow.set_flow_y(-region_flow.flow_y());

    for (auto& feature : *region_flow.mutable_feature()) {
      feature.set_x(feature.x() + feature.dx());
      feature.set_y(feature.y() + feature.dy());
      feature.set_dx(-feature.dx());
      feature.set_dy(-feature.dy());
    }
  }
}

void InvertRegionFlowFeatureList(const RegionFlowFeatureList& feature_list,
                                 RegionFlowFeatureList* inverted_feature_list) {
  CHECK(inverted_feature_list);
  *inverted_feature_list = feature_list;
  for (auto& feature : *inverted_feature_list->mutable_feature()) {
    InvertRegionFlowFeature(&feature);
  }
}

void InvertRegionFlowFeature(RegionFlowFeature* feature) {
  Vector2_f pt_match = FeatureMatchLocation(*feature);
  feature->set_x(pt_match.x());
  feature->set_y(pt_match.y());
  Vector2_f flow = FeatureFlow(*feature);
  feature->set_dx(-flow.x());
  feature->set_dy(-flow.y());
}

void LimitFeaturesToBounds(int frame_width, int frame_height, float bounds,
                           RegionFlowFeatureList* feature_list) {
  RegionFlowFeatureList limited;
  for (const auto& feature : feature_list->feature()) {
    if (!IsPointWithinBounds(FeatureLocation(feature), bounds, frame_width,
                             frame_height)) {
      continue;
    } else {
      limited.add_feature()->CopyFrom(feature);
    }
  }
  feature_list->Swap(&limited);
}

void NormalizeRegionFlowFeatureList(RegionFlowFeatureList* feature_list) {
  const LinearSimilarityModel norm_model =
      LinearSimilarityAdapter::NormalizationTransform(
          feature_list->frame_width(), feature_list->frame_height());
  TransformRegionFlowFeatureList(norm_model, feature_list);
}

void DeNormalizeRegionFlowFeatureList(RegionFlowFeatureList* feature_list) {
  const LinearSimilarityModel norm_model =
      LinearSimilarityAdapter::NormalizationTransform(
          feature_list->frame_width(), feature_list->frame_height());
  TransformRegionFlowFeatureList(ModelInvert(norm_model), feature_list);
}

void ScaleSalientPoint(float scale_x, float scale_y, SalientPoint* sp) {
  sp->set_norm_point_x(sp->norm_point_x() * scale_x);
  sp->set_norm_point_y(sp->norm_point_y() * scale_y);

  const float cos_angle = std::cos(sp->angle());
  const float sin_angle = std::sin(sp->angle());
  Vector2_f major_axis = Vector2_f(cos_angle, sin_angle) * sp->norm_major();
  Vector2_f minor_axis = Vector2_f(-sin_angle, cos_angle) * sp->norm_minor();

  major_axis[0] *= scale_x;
  major_axis[1] *= scale_y;
  minor_axis[0] *= scale_x;
  minor_axis[1] *= scale_y;

  sp->set_norm_major(major_axis.Norm());
  sp->set_norm_minor(minor_axis.Norm());
  sp->set_angle(std::atan2(major_axis.y(), major_axis.x()));
  if (sp->angle() < 0) {
    sp->set_angle(sp->angle() + M_PI);
  }
}

void ScaleSaliencyList(float scale, bool normalize_to_scale,
                       SaliencyPointList* saliency_list) {
  CHECK(saliency_list != nullptr);
  for (auto& point_frame : *saliency_list) {
    ScaleSalientPointFrame(scale, normalize_to_scale, &point_frame);
  }
}

void ScaleSalientPointFrame(float scale, bool normalize_to_scale,
                            SalientPointFrame* saliency) {
  CHECK(saliency != nullptr);
  float saliency_scale = scale;
  if (normalize_to_scale) {
    float weight_sum = 0.0f;
    for (const auto& salient_point : saliency->point()) {
      weight_sum += salient_point.weight();
    }

    if (weight_sum > 1e-6f) {
      saliency_scale /= weight_sum;
    }
  }

  for (auto& salient_point : *saliency->mutable_point()) {
    salient_point.set_weight(salient_point.weight() * saliency_scale);
  }
}

void ResetSaliencyBounds(float left, float bottom, float right, float top,
                         SaliencyPointList* saliency_list) {
  CHECK(saliency_list != nullptr);
  for (auto& point_frame : *saliency_list) {
    for (auto& salient_point : *point_frame.mutable_point()) {
      salient_point.set_left(left);
      salient_point.set_bottom(bottom);
      salient_point.set_right(right);
      salient_point.set_top(top);
    }
  }
}

bool EllipseFromCovariance(float a, float bc, float d,
                           Vector2_f* axis_magnitude, float* angle) {
  CHECK(axis_magnitude != nullptr);
  CHECK(angle != nullptr);

  // Get trace and determinant
  const float trace = a + d;
  const float det = a * d - bc * bc;

  // If area is very small (small axis in at least one direction)
  // axis are unreliable -> return false.
  if (det < 4) {  // Measured in sq. pixels.
    *axis_magnitude = Vector2_f(1, 1);
    *angle = 0;
    return false;
  }

  const float discriminant = std::max(0.f, trace * trace * 0.25f - det);
  const float sqrt_discrm = std::sqrt(discriminant);

  // Get eigenvalues.
  float eig_1 = trace * 0.5f + sqrt_discrm;
  float eig_2 = trace * 0.5f - sqrt_discrm;

  // Compute eigenvectors.
  Vector2_f vec_1, vec_2;

  if (fabs(bc) < 1e-6) {
    // Right-most case, we already have diagonal matrix.
    vec_1.Set(1, 0);
    vec_2.Set(0, 1);
  } else {
    vec_1.Set(eig_1 - d, bc);
    vec_2.Set(eig_2 - d, bc);

    // Normalize. Norm is always > 0, as bc is > 0 via above if.
    vec_1 /= vec_1.Norm();
    vec_2 /= vec_2.Norm();
  }

  // Select positive eigenvector.
  if (eig_1 < 0) {
    eig_1 *= -1.f;
  }

  if (eig_2 < 0) {
    eig_2 *= -1.f;
  }

  // Major first.
  if (eig_1 < eig_2) {
    using std::swap;
    swap(vec_1, vec_2);
    swap(eig_1, eig_2);
  }

  *axis_magnitude = Vector2_f(std::sqrt(eig_1), std::sqrt(eig_2));
  *angle = std::atan2(vec_1.y(), vec_1.x());

  return eig_2 >= 1.5f;  // Measurement in pixels.
}

void BoundingBoxFromEllipse(const Vector2_f& center, float norm_major_axis,
                            float norm_minor_axis, float angle,
                            std::vector<Vector2_f>* bounding_box) {
  CHECK(bounding_box != nullptr);
  float dim_x;
  float dim_y;
  if (angle < M_PI * 0.25 || angle > M_PI * 0.75) {
    dim_x = norm_major_axis;
    dim_y = norm_minor_axis;
  } else {
    dim_y = norm_major_axis;
    dim_x = norm_minor_axis;
  }

  // Construct bounding for for axes aligned ellipse.
  *bounding_box = {
      Vector2_f(-dim_x, -dim_y),
      Vector2_f(-dim_x, dim_y),
      Vector2_f(dim_x, dim_y),
      Vector2_f(dim_x, -dim_y),
  };

  for (Vector2_f& corner : *bounding_box) {
    corner += center;
  }
}

void CopyToEmptyFeatureList(RegionFlowFeatureList* src,
                            RegionFlowFeatureList* dst) {
  CHECK(src != nullptr);
  CHECK(dst != nullptr);

  // Swap out features for empty list.
  RegionFlowFeatureList empty_list;
  empty_list.mutable_feature()->Swap(src->mutable_feature());

  // Copy.
  dst->CopyFrom(*src);

  // Swap back.
  src->mutable_feature()->Swap(empty_list.mutable_feature());

  // src_features should be empty as in the beginning.
  CHECK_EQ(0, empty_list.feature_size());
}

void IntersectRegionFlowFeatureList(
    const RegionFlowFeatureList& to,
    std::function<Vector2_f(const RegionFlowFeature&)> to_location_eval,
    RegionFlowFeatureList* from, RegionFlowFeatureList* result,
    std::vector<int>* source_indices) {
  CHECK(from != nullptr);
  CHECK(result != nullptr);
  CHECK(from->long_tracks()) << "Intersection only works for long features";
  CHECK(to.long_tracks()) << "Intersection only works for long features";

  // Hash features in to, based on track_id.
  absl::node_hash_map<int, const RegionFlowFeature*> track_map;
  for (const auto& feature : to.feature()) {
    track_map[feature.track_id()] = &feature;
  }

  // Initialize result.
  CopyToEmptyFeatureList(from, result);
  const int num_from_features = from->feature_size();
  result->mutable_feature()->Reserve(num_from_features);

  int feature_idx = 0;
  for (const auto& feature : from->feature()) {
    auto find_location = track_map.find(feature.track_id());
    if (find_location != track_map.end()) {
      const Vector2_f diff =
          to_location_eval(*find_location->second) - FeatureLocation(feature);

      RegionFlowFeature* new_feature = result->add_feature();
      new_feature->CopyFrom(feature);
      new_feature->set_dx(diff.x());
      new_feature->set_dy(diff.y());
      if (source_indices != nullptr) {
        source_indices->push_back(feature_idx);
      }
    }

    ++feature_idx;
  }
}

void LongFeatureStream::AddFeatures(const RegionFlowFeatureList& feature_list,
                                    bool check_connectivity,
                                    bool purge_non_present_features) {
  if (!feature_list.long_tracks()) {
    LOG(ERROR) << "Feature stream should be used only used with long feature "
               << "tracks. Ensure POLICY_LONG_FEATURE was used for "
               << "RegionFlowComputation.";
    return;
  }

  if (feature_list.match_frame() == 0) {
    // Skip first frame.
    return;
  }

  if (std::abs(feature_list.match_frame()) != 1) {
    LOG(ERROR) << "Only matching frames one frame from current one are "
               << "supported";
    return;
  }

  // Record id of each track that is present in the current feature_list.
  absl::node_hash_set<int> present_tracks;
  for (auto feature : feature_list.feature()) {  // Copy feature.
    if (feature.track_id() < 0) {
      LOG_IF(WARNING, []() {
        static int k = 0;
        return k++ < 2;
      }()) << "Feature does not have a valid track id assigned. Ignoring.";
      continue;
    }
    present_tracks.insert(feature.track_id());
    if (check_connectivity) {
      // A new feature should never have been erased before.
      CHECK(old_ids_.find(feature.track_id()) == old_ids_.end())
          << "Feature : " << feature.track_id() << "was already removed.";
    }

    // Invert the features to be foward or backward according to forward_ flag.
    if ((!forward_ && feature_list.match_frame() > 0) ||
        (forward_ && feature_list.match_frame() < 0)) {
      InvertRegionFlowFeature(&feature);
    }

    auto find_pos = tracks_.find(feature.track_id());
    if (find_pos != tracks_.end()) {
      // Track is present, add to it.
      if (check_connectivity) {
        CHECK_LT((FeatureLocation(find_pos->second.back()) -
                  FeatureMatchLocation(feature))
                     .Norm2(),
                 1e-4);
      }
      find_pos->second.push_back(feature);
    } else {
      tracks_[feature.track_id()] = std::vector<RegionFlowFeature>(1, feature);
    }
  }

  if (purge_non_present_features) {
    std::vector<int> to_be_removed;
    for (const auto& track : tracks_) {
      if (present_tracks.find(track.first) == present_tracks.end()) {
        to_be_removed.push_back(track.first);
        if (check_connectivity) {
          old_ids_.insert(track.first);
        }
      }
    }
    for (int id : to_be_removed) {
      tracks_.erase(id);
    }
  }
}

void LongFeatureStream::FlattenTrack(
    const std::vector<RegionFlowFeature>& features,
    std::vector<Vector2_f>* result, std::vector<float>* irls_weight,
    std::vector<Vector2_f>* flow) const {
  CHECK(result != nullptr);
  if (features.empty()) {
    return;
  }

  if (irls_weight) {
    irls_weight->clear();
  }

  if (flow) {
    flow->clear();
  }

  if (!forward_) {
    // Backward tracking, add first match.
    result->push_back(FeatureMatchLocation(features[0]));
  }

  for (const auto& feature : features) {
    result->push_back(FeatureLocation(feature));
    if (flow) {
      flow->push_back(FeatureFlow(feature));
    }
    if (irls_weight) {
      irls_weight->push_back(feature.irls_weight());
    }
  }

  if (forward_) {
    // Forward tracking, add last match.
    result->push_back(FeatureMatchLocation(features.back()));
  }

  // Replicate last irls weight.
  if (irls_weight) {
    irls_weight->push_back(irls_weight->back());
  }
}

const std::vector<RegionFlowFeature>* LongFeatureStream::TrackById(
    int id) const {
  auto track_pos = tracks_.find(id);
  if (track_pos == tracks_.end()) {
    return nullptr;
  } else {
    return &track_pos->second;
  }
}

std::vector<Vector2_f> LongFeatureStream::FlattenedTrackById(int id) const {
  const auto* track = TrackById(id);
  if (track != nullptr) {
    std::vector<Vector2_f> points;
    FlattenTrack(*track, &points, nullptr, nullptr);
    return points;
  } else {
    return std::vector<Vector2_f>();
  }
}

void LongFeatureInfo::AddFeatures(const RegionFlowFeatureList& feature_list) {
  if (!feature_list.long_tracks()) {
    LOG(ERROR) << "Passed feature list was not computed with long tracks. ";
    return;
  }

  for (const auto& feature : feature_list.feature()) {
    AddFeature(feature);
  }

  IncrementFrame();
}

void LongFeatureInfo::AddFeature(const RegionFlowFeature& feature) {
  if (feature.irls_weight() == 0) {  // Skip outliers.
    return;
  }
  const int track_id = feature.track_id();
  if (track_id < 0) {  // Skip unassigned ids.
    return;
  }
  auto insert_pos = track_info_.find(track_id);
  if (insert_pos == track_info_.end()) {
    track_info_[track_id].start = num_frames_;
    track_info_[track_id].length = 1;
  } else {
    ++insert_pos->second.length;
  }
}

void LongFeatureInfo::TrackLengths(const RegionFlowFeatureList& feature_list,
                                   std::vector<int>* track_lengths) const {
  CHECK(track_lengths);
  const int feature_size = feature_list.feature_size();
  track_lengths->resize(feature_size);
  for (int k = 0; k < feature_size; ++k) {
    (*track_lengths)[k] = TrackLength(feature_list.feature(k));
  }
}

int LongFeatureInfo::TrackLength(const RegionFlowFeature& feature) const {
  const auto insert_pos = track_info_.find(feature.track_id());
  return insert_pos != track_info_.end() ? insert_pos->second.length : 0;
}

int LongFeatureInfo::TrackStart(const RegionFlowFeature& feature) const {
  const auto insert_pos = track_info_.find(feature.track_id());
  return insert_pos != track_info_.end() ? insert_pos->second.start : -1;
}

void LongFeatureInfo::Reset() {
  track_info_.clear();
  num_frames_ = 0;
}

int LongFeatureInfo::GlobalTrackLength(float percentile) const {
  std::vector<int> track_lengths;
  track_lengths.reserve(track_info_.size());

  for (const auto& pair : track_info_) {
    track_lengths.push_back(pair.second.length);
  }

  if (track_lengths.empty()) {
    return 0;
  }

  auto percentile_item =
      track_lengths.begin() + percentile * track_lengths.size();

  std::nth_element(track_lengths.begin(), percentile_item, track_lengths.end());

  return *percentile_item;
}

void GridTaps(int dim_x, int dim_y, int tap_radius,
              std::vector<std::vector<int>>* taps) {
  CHECK(taps);
  const int grid_size = dim_x * dim_y;
  const int diam = 2 * tap_radius + 1;
  taps->resize(grid_size);

  for (int i = 0; i < dim_y; ++i) {
    for (int j = 0; j < dim_x; ++j) {
      std::vector<int>& grid_bin = (*taps)[i * dim_x + j];
      grid_bin.clear();
      grid_bin.reserve(diam * diam);
      for (int k = std::max(0, i - tap_radius),
               end_k = std::min(dim_y - 1, i + tap_radius);
           k <= end_k; ++k) {
        for (int l = std::max(0, j - tap_radius),
                 end_l = std::min(dim_x - 1, j + tap_radius);
             l <= end_l; ++l) {
          grid_bin.push_back(k * dim_x + l);
        }
      }
    }
  }
}

}  // namespace mediapipe
