// Copyright 2023 The MediaPipe Authors.
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

#include "mediapipe/calculators/util/landmarks_smoothing_calculator_utils.h"

#include <iostream>

#include "mediapipe/calculators/util/landmarks_smoothing_calculator.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/filtering/one_euro_filter.h"
#include "mediapipe/util/filtering/relative_velocity_filter.h"

namespace mediapipe {
namespace landmarks_smoothing {

namespace {

using ::mediapipe::NormalizedRect;
using ::mediapipe::OneEuroFilter;
using ::mediapipe::Rect;
using ::mediapipe::RelativeVelocityFilter;

// Estimate object scale to use its inverse value as velocity scale for
// RelativeVelocityFilter. If value will be too small (less than
// `options_.min_allowed_object_scale`) smoothing will be disabled and
// landmarks will be returned as is.
// Object scale is calculated as average between bounding box width and height
// with sides parallel to axis.
float GetObjectScale(const LandmarkList& landmarks) {
  const auto& lm_minmax_x = absl::c_minmax_element(
      landmarks.landmark(),
      [](const auto& a, const auto& b) { return a.x() < b.x(); });
  const float x_min = lm_minmax_x.first->x();
  const float x_max = lm_minmax_x.second->x();

  const auto& lm_minmax_y = absl::c_minmax_element(
      landmarks.landmark(),
      [](const auto& a, const auto& b) { return a.y() < b.y(); });
  const float y_min = lm_minmax_y.first->y();
  const float y_max = lm_minmax_y.second->y();

  const float object_width = x_max - x_min;
  const float object_height = y_max - y_min;

  return (object_width + object_height) / 2.0f;
}

// Returns landmarks as is without smoothing.
class NoFilter : public LandmarksFilter {
 public:
  absl::Status Apply(const LandmarkList& in_landmarks,
                     const absl::Duration& timestamp,
                     const absl::optional<float> object_scale_opt,
                     LandmarkList& out_landmarks) override {
    out_landmarks = in_landmarks;
    return absl::OkStatus();
  }
};

// Please check RelativeVelocityFilter documentation for details.
class VelocityFilter : public LandmarksFilter {
 public:
  VelocityFilter(int window_size, float velocity_scale,
                 float min_allowed_object_scale, bool disable_value_scaling)
      : window_size_(window_size),
        velocity_scale_(velocity_scale),
        min_allowed_object_scale_(min_allowed_object_scale),
        disable_value_scaling_(disable_value_scaling) {}

  absl::Status Reset() override {
    x_filters_.clear();
    y_filters_.clear();
    z_filters_.clear();
    return absl::OkStatus();
  }

  absl::Status Apply(const LandmarkList& in_landmarks,
                     const absl::Duration& timestamp,
                     const absl::optional<float> object_scale_opt,
                     LandmarkList& out_landmarks) override {
    // Get value scale as inverse value of the object scale.
    // If value is too small smoothing will be disabled and landmarks will be
    // returned as is.
    float value_scale = 1.0f;
    if (!disable_value_scaling_) {
      const float object_scale =
          object_scale_opt ? *object_scale_opt : GetObjectScale(in_landmarks);
      if (object_scale < min_allowed_object_scale_) {
        out_landmarks = in_landmarks;
        return absl::OkStatus();
      }
      value_scale = 1.0f / object_scale;
    }

    // Initialize filters once.
    MP_RETURN_IF_ERROR(InitializeFiltersIfEmpty(in_landmarks.landmark_size()));

    // Filter landmarks. Every axis of every landmark is filtered separately.
    for (int i = 0; i < in_landmarks.landmark_size(); ++i) {
      const auto& in_landmark = in_landmarks.landmark(i);

      auto* out_landmark = out_landmarks.add_landmark();
      *out_landmark = in_landmark;
      out_landmark->set_x(
          x_filters_[i].Apply(timestamp, value_scale, in_landmark.x()));
      out_landmark->set_y(
          y_filters_[i].Apply(timestamp, value_scale, in_landmark.y()));
      out_landmark->set_z(
          z_filters_[i].Apply(timestamp, value_scale, in_landmark.z()));
    }

    return absl::OkStatus();
  }

 private:
  // Initializes filters for the first time or after Reset. If initialized then
  // check the size.
  absl::Status InitializeFiltersIfEmpty(const int n_landmarks) {
    if (!x_filters_.empty()) {
      RET_CHECK_EQ(x_filters_.size(), n_landmarks);
      RET_CHECK_EQ(y_filters_.size(), n_landmarks);
      RET_CHECK_EQ(z_filters_.size(), n_landmarks);
      return absl::OkStatus();
    }

    x_filters_.resize(n_landmarks,
                      RelativeVelocityFilter(window_size_, velocity_scale_));
    y_filters_.resize(n_landmarks,
                      RelativeVelocityFilter(window_size_, velocity_scale_));
    z_filters_.resize(n_landmarks,
                      RelativeVelocityFilter(window_size_, velocity_scale_));

    return absl::OkStatus();
  }

  int window_size_;
  float velocity_scale_;
  float min_allowed_object_scale_;
  bool disable_value_scaling_;

  std::vector<RelativeVelocityFilter> x_filters_;
  std::vector<RelativeVelocityFilter> y_filters_;
  std::vector<RelativeVelocityFilter> z_filters_;
};

// Please check OneEuroFilter documentation for details.
class OneEuroFilterImpl : public LandmarksFilter {
 public:
  OneEuroFilterImpl(double frequency, double min_cutoff, double beta,
                    double derivate_cutoff, float min_allowed_object_scale,
                    bool disable_value_scaling)
      : frequency_(frequency),
        min_cutoff_(min_cutoff),
        beta_(beta),
        derivate_cutoff_(derivate_cutoff),
        min_allowed_object_scale_(min_allowed_object_scale),
        disable_value_scaling_(disable_value_scaling) {}

  absl::Status Reset() override {
    x_filters_.clear();
    y_filters_.clear();
    z_filters_.clear();
    return absl::OkStatus();
  }

  absl::Status Apply(const LandmarkList& in_landmarks,
                     const absl::Duration& timestamp,
                     const absl::optional<float> object_scale_opt,
                     LandmarkList& out_landmarks) override {
    // Initialize filters once.
    MP_RETURN_IF_ERROR(InitializeFiltersIfEmpty(in_landmarks.landmark_size()));

    // Get value scale as inverse value of the object scale.
    // If value is too small smoothing will be disabled and landmarks will be
    // returned as is.
    float value_scale = 1.0f;
    if (!disable_value_scaling_) {
      const float object_scale =
          object_scale_opt ? *object_scale_opt : GetObjectScale(in_landmarks);
      if (object_scale < min_allowed_object_scale_) {
        out_landmarks = in_landmarks;
        return absl::OkStatus();
      }
      value_scale = 1.0f / object_scale;
    }

    // Filter landmarks. Every axis of every landmark is filtered separately.
    for (int i = 0; i < in_landmarks.landmark_size(); ++i) {
      const auto& in_landmark = in_landmarks.landmark(i);

      auto* out_landmark = out_landmarks.add_landmark();
      *out_landmark = in_landmark;
      out_landmark->set_x(
          x_filters_[i].Apply(timestamp, value_scale, in_landmark.x()));
      out_landmark->set_y(
          y_filters_[i].Apply(timestamp, value_scale, in_landmark.y()));
      out_landmark->set_z(
          z_filters_[i].Apply(timestamp, value_scale, in_landmark.z()));
    }

    return absl::OkStatus();
  }

 private:
  // Initializes filters for the first time or after Reset. If initialized then
  // check the size.
  absl::Status InitializeFiltersIfEmpty(const int n_landmarks) {
    if (!x_filters_.empty()) {
      RET_CHECK_EQ(x_filters_.size(), n_landmarks);
      RET_CHECK_EQ(y_filters_.size(), n_landmarks);
      RET_CHECK_EQ(z_filters_.size(), n_landmarks);
      return absl::OkStatus();
    }

    for (int i = 0; i < n_landmarks; ++i) {
      x_filters_.push_back(
          OneEuroFilter(frequency_, min_cutoff_, beta_, derivate_cutoff_));
      y_filters_.push_back(
          OneEuroFilter(frequency_, min_cutoff_, beta_, derivate_cutoff_));
      z_filters_.push_back(
          OneEuroFilter(frequency_, min_cutoff_, beta_, derivate_cutoff_));
    }

    return absl::OkStatus();
  }

  double frequency_;
  double min_cutoff_;
  double beta_;
  double derivate_cutoff_;
  double min_allowed_object_scale_;
  bool disable_value_scaling_;

  std::vector<OneEuroFilter> x_filters_;
  std::vector<OneEuroFilter> y_filters_;
  std::vector<OneEuroFilter> z_filters_;
};

}  // namespace

void NormalizedLandmarksToLandmarks(
    const NormalizedLandmarkList& norm_landmarks, const int image_width,
    const int image_height, LandmarkList& landmarks) {
  for (int i = 0; i < norm_landmarks.landmark_size(); ++i) {
    const auto& norm_landmark = norm_landmarks.landmark(i);

    auto* landmark = landmarks.add_landmark();
    landmark->set_x(norm_landmark.x() * image_width);
    landmark->set_y(norm_landmark.y() * image_height);
    // Scale Z the same way as X (using image width).
    landmark->set_z(norm_landmark.z() * image_width);

    if (norm_landmark.has_visibility()) {
      landmark->set_visibility(norm_landmark.visibility());
    } else {
      landmark->clear_visibility();
    }

    if (norm_landmark.has_presence()) {
      landmark->set_presence(norm_landmark.presence());
    } else {
      landmark->clear_presence();
    }
  }
}

void LandmarksToNormalizedLandmarks(const LandmarkList& landmarks,
                                    const int image_width,
                                    const int image_height,
                                    NormalizedLandmarkList& norm_landmarks) {
  for (int i = 0; i < landmarks.landmark_size(); ++i) {
    const auto& landmark = landmarks.landmark(i);

    auto* norm_landmark = norm_landmarks.add_landmark();
    norm_landmark->set_x(landmark.x() / image_width);
    norm_landmark->set_y(landmark.y() / image_height);
    // Scale Z the same way as X (using image width).
    norm_landmark->set_z(landmark.z() / image_width);

    if (landmark.has_visibility()) {
      norm_landmark->set_visibility(landmark.visibility());
    } else {
      norm_landmark->clear_visibility();
    }

    if (landmark.has_presence()) {
      norm_landmark->set_presence(landmark.presence());
    } else {
      norm_landmark->clear_presence();
    }
  }
}

float GetObjectScale(const NormalizedRect& roi, const int image_width,
                     const int image_height) {
  const float object_width = roi.width() * image_width;
  const float object_height = roi.height() * image_height;

  return (object_width + object_height) / 2.0f;
}

float GetObjectScale(const Rect& roi) {
  return (roi.width() + roi.height()) / 2.0f;
}

absl::StatusOr<std::unique_ptr<LandmarksFilter>> InitializeLandmarksFilter(
    const LandmarksSmoothingCalculatorOptions& options) {
  if (options.has_no_filter()) {
    return absl::make_unique<NoFilter>();
  } else if (options.has_velocity_filter()) {
    return absl::make_unique<VelocityFilter>(
        options.velocity_filter().window_size(),
        options.velocity_filter().velocity_scale(),
        options.velocity_filter().min_allowed_object_scale(),
        options.velocity_filter().disable_value_scaling());
  } else if (options.has_one_euro_filter()) {
    return absl::make_unique<OneEuroFilterImpl>(
        options.one_euro_filter().frequency(),
        options.one_euro_filter().min_cutoff(),
        options.one_euro_filter().beta(),
        options.one_euro_filter().derivate_cutoff(),
        options.one_euro_filter().min_allowed_object_scale(),
        options.one_euro_filter().disable_value_scaling());
  } else {
    RET_CHECK_FAIL()
        << "Landmarks filter is either not specified or not supported";
  }
}

absl::StatusOr<LandmarksFilter*> MultiLandmarkFilters::GetOrCreate(
    const int64_t tracking_id,
    const mediapipe::LandmarksSmoothingCalculatorOptions& options) {
  const auto it = filters_.find(tracking_id);
  if (it != filters_.end()) {
    return it->second.get();
  }

  MP_ASSIGN_OR_RETURN(auto landmarks_filter,
                      InitializeLandmarksFilter(options));
  filters_[tracking_id] = std::move(landmarks_filter);
  return filters_[tracking_id].get();
}

void MultiLandmarkFilters::ClearUnused(
    const std::vector<int64_t>& tracking_ids) {
  std::vector<int64_t> unused_tracking_ids;
  for (const auto& it : filters_) {
    bool unused = true;
    for (int64_t tracking_id : tracking_ids) {
      if (tracking_id == it.first) unused = false;
    }
    if (unused) unused_tracking_ids.push_back(it.first);
  }

  for (int64_t tracking_id : unused_tracking_ids) {
    filters_.erase(tracking_id);
  }
}

void MultiLandmarkFilters::Clear() { filters_.clear(); }

}  // namespace landmarks_smoothing
}  // namespace mediapipe
