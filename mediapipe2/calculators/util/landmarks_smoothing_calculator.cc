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

#include <memory>

#include "absl/algorithm/container.h"
#include "mediapipe/calculators/util/landmarks_smoothing_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/util/filtering/one_euro_filter.h"
#include "mediapipe/util/filtering/relative_velocity_filter.h"

namespace mediapipe {

namespace {

constexpr char kNormalizedLandmarksTag[] = "NORM_LANDMARKS";
constexpr char kLandmarksTag[] = "LANDMARKS";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kObjectScaleRoiTag[] = "OBJECT_SCALE_ROI";
constexpr char kNormalizedFilteredLandmarksTag[] = "NORM_FILTERED_LANDMARKS";
constexpr char kFilteredLandmarksTag[] = "FILTERED_LANDMARKS";

using mediapipe::OneEuroFilter;
using mediapipe::RelativeVelocityFilter;

void NormalizedLandmarksToLandmarks(
    const NormalizedLandmarkList& norm_landmarks, const int image_width,
    const int image_height, LandmarkList* landmarks) {
  for (int i = 0; i < norm_landmarks.landmark_size(); ++i) {
    const auto& norm_landmark = norm_landmarks.landmark(i);

    auto* landmark = landmarks->add_landmark();
    landmark->set_x(norm_landmark.x() * image_width);
    landmark->set_y(norm_landmark.y() * image_height);
    // Scale Z the same way as X (using image width).
    landmark->set_z(norm_landmark.z() * image_width);
    landmark->set_visibility(norm_landmark.visibility());
    landmark->set_presence(norm_landmark.presence());
  }
}

void LandmarksToNormalizedLandmarks(const LandmarkList& landmarks,
                                    const int image_width,
                                    const int image_height,
                                    NormalizedLandmarkList* norm_landmarks) {
  for (int i = 0; i < landmarks.landmark_size(); ++i) {
    const auto& landmark = landmarks.landmark(i);

    auto* norm_landmark = norm_landmarks->add_landmark();
    norm_landmark->set_x(landmark.x() / image_width);
    norm_landmark->set_y(landmark.y() / image_height);
    // Scale Z the same way as X (using image width).
    norm_landmark->set_z(landmark.z() / image_width);
    norm_landmark->set_visibility(landmark.visibility());
    norm_landmark->set_presence(landmark.presence());
  }
}

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

float GetObjectScale(const NormalizedRect& roi, const int image_width,
                     const int image_height) {
  const float object_width = roi.width() * image_width;
  const float object_height = roi.height() * image_height;

  return (object_width + object_height) / 2.0f;
}

float GetObjectScale(const Rect& roi) {
  return (roi.width() + roi.height()) / 2.0f;
}

// Abstract class for various landmarks filters.
class LandmarksFilter {
 public:
  virtual ~LandmarksFilter() = default;

  virtual absl::Status Reset() { return absl::OkStatus(); }

  virtual absl::Status Apply(const LandmarkList& in_landmarks,
                             const absl::Duration& timestamp,
                             const absl::optional<float> object_scale_opt,
                             LandmarkList* out_landmarks) = 0;
};

// Returns landmarks as is without smoothing.
class NoFilter : public LandmarksFilter {
 public:
  absl::Status Apply(const LandmarkList& in_landmarks,
                     const absl::Duration& timestamp,
                     const absl::optional<float> object_scale_opt,
                     LandmarkList* out_landmarks) override {
    *out_landmarks = in_landmarks;
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
                     LandmarkList* out_landmarks) override {
    // Get value scale as inverse value of the object scale.
    // If value is too small smoothing will be disabled and landmarks will be
    // returned as is.
    float value_scale = 1.0f;
    if (!disable_value_scaling_) {
      const float object_scale =
          object_scale_opt ? *object_scale_opt : GetObjectScale(in_landmarks);
      if (object_scale < min_allowed_object_scale_) {
        *out_landmarks = in_landmarks;
        return absl::OkStatus();
      }
      value_scale = 1.0f / object_scale;
    }

    // Initialize filters once.
    MP_RETURN_IF_ERROR(InitializeFiltersIfEmpty(in_landmarks.landmark_size()));

    // Filter landmarks. Every axis of every landmark is filtered separately.
    for (int i = 0; i < in_landmarks.landmark_size(); ++i) {
      const auto& in_landmark = in_landmarks.landmark(i);

      auto* out_landmark = out_landmarks->add_landmark();
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
                     LandmarkList* out_landmarks) override {
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
        *out_landmarks = in_landmarks;
        return absl::OkStatus();
      }
      value_scale = 1.0f / object_scale;
    }

    // Filter landmarks. Every axis of every landmark is filtered separately.
    for (int i = 0; i < in_landmarks.landmark_size(); ++i) {
      const auto& in_landmark = in_landmarks.landmark(i);

      auto* out_landmark = out_landmarks->add_landmark();
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

// A calculator to smooth landmarks over time.
//
// Inputs:
//   NORM_LANDMARKS: A NormalizedLandmarkList of landmarks you want to smooth.
//   IMAGE_SIZE: A std::pair<int, int> represention of image width and height.
//     Required to perform all computations in absolute coordinates to avoid any
//     influence of normalized values.
//   OBJECT_SCALE_ROI (optional): A NormRect or Rect (depending on the format of
//     input landmarks) used to determine the object scale for some of the
//     filters. If not provided - object scale will be calculated from
//     landmarks.
//
// Outputs:
//   NORM_FILTERED_LANDMARKS: A NormalizedLandmarkList of smoothed landmarks.
//
// Example config:
//   node {
//     calculator: "LandmarksSmoothingCalculator"
//     input_stream: "NORM_LANDMARKS:pose_landmarks"
//     input_stream: "IMAGE_SIZE:image_size"
//     input_stream: "OBJECT_SCALE_ROI:roi"
//     output_stream: "NORM_FILTERED_LANDMARKS:pose_landmarks_filtered"
//     options: {
//       [mediapipe.LandmarksSmoothingCalculatorOptions.ext] {
//         velocity_filter: {
//           window_size: 5
//           velocity_scale: 10.0
//         }
//       }
//     }
//   }
//
class LandmarksSmoothingCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  std::unique_ptr<LandmarksFilter> landmarks_filter_;
};
REGISTER_CALCULATOR(LandmarksSmoothingCalculator);

absl::Status LandmarksSmoothingCalculator::GetContract(CalculatorContract* cc) {
  if (cc->Inputs().HasTag(kNormalizedLandmarksTag)) {
    cc->Inputs().Tag(kNormalizedLandmarksTag).Set<NormalizedLandmarkList>();
    cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>();
    cc->Outputs()
        .Tag(kNormalizedFilteredLandmarksTag)
        .Set<NormalizedLandmarkList>();

    if (cc->Inputs().HasTag(kObjectScaleRoiTag)) {
      cc->Inputs().Tag(kObjectScaleRoiTag).Set<NormalizedRect>();
    }
  } else {
    cc->Inputs().Tag(kLandmarksTag).Set<LandmarkList>();
    cc->Outputs().Tag(kFilteredLandmarksTag).Set<LandmarkList>();

    if (cc->Inputs().HasTag(kObjectScaleRoiTag)) {
      cc->Inputs().Tag(kObjectScaleRoiTag).Set<Rect>();
    }
  }

  return absl::OkStatus();
}

absl::Status LandmarksSmoothingCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  // Pick landmarks filter.
  const auto& options = cc->Options<LandmarksSmoothingCalculatorOptions>();
  if (options.has_no_filter()) {
    landmarks_filter_ = absl::make_unique<NoFilter>();
  } else if (options.has_velocity_filter()) {
    landmarks_filter_ = absl::make_unique<VelocityFilter>(
        options.velocity_filter().window_size(),
        options.velocity_filter().velocity_scale(),
        options.velocity_filter().min_allowed_object_scale(),
        options.velocity_filter().disable_value_scaling());
  } else if (options.has_one_euro_filter()) {
    landmarks_filter_ = absl::make_unique<OneEuroFilterImpl>(
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

  return absl::OkStatus();
}

absl::Status LandmarksSmoothingCalculator::Process(CalculatorContext* cc) {
  // Check that landmarks are not empty and reset the filter if so.
  // Don't emit an empty packet for this timestamp.
  if ((cc->Inputs().HasTag(kNormalizedLandmarksTag) &&
       cc->Inputs().Tag(kNormalizedLandmarksTag).IsEmpty()) ||
      (cc->Inputs().HasTag(kLandmarksTag) &&
       cc->Inputs().Tag(kLandmarksTag).IsEmpty())) {
    MP_RETURN_IF_ERROR(landmarks_filter_->Reset());
    return absl::OkStatus();
  }

  const auto& timestamp =
      absl::Microseconds(cc->InputTimestamp().Microseconds());

  if (cc->Inputs().HasTag(kNormalizedLandmarksTag)) {
    const auto& in_norm_landmarks =
        cc->Inputs().Tag(kNormalizedLandmarksTag).Get<NormalizedLandmarkList>();

    int image_width;
    int image_height;
    std::tie(image_width, image_height) =
        cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();

    absl::optional<float> object_scale;
    if (cc->Inputs().HasTag(kObjectScaleRoiTag) &&
        !cc->Inputs().Tag(kObjectScaleRoiTag).IsEmpty()) {
      auto& roi = cc->Inputs().Tag(kObjectScaleRoiTag).Get<NormalizedRect>();
      object_scale = GetObjectScale(roi, image_width, image_height);
    }

    auto in_landmarks = absl::make_unique<LandmarkList>();
    NormalizedLandmarksToLandmarks(in_norm_landmarks, image_width, image_height,
                                   in_landmarks.get());

    auto out_landmarks = absl::make_unique<LandmarkList>();
    MP_RETURN_IF_ERROR(landmarks_filter_->Apply(
        *in_landmarks, timestamp, object_scale, out_landmarks.get()));

    auto out_norm_landmarks = absl::make_unique<NormalizedLandmarkList>();
    LandmarksToNormalizedLandmarks(*out_landmarks, image_width, image_height,
                                   out_norm_landmarks.get());

    cc->Outputs()
        .Tag(kNormalizedFilteredLandmarksTag)
        .Add(out_norm_landmarks.release(), cc->InputTimestamp());
  } else {
    const auto& in_landmarks =
        cc->Inputs().Tag(kLandmarksTag).Get<LandmarkList>();

    absl::optional<float> object_scale;
    if (cc->Inputs().HasTag(kObjectScaleRoiTag) &&
        !cc->Inputs().Tag(kObjectScaleRoiTag).IsEmpty()) {
      auto& roi = cc->Inputs().Tag(kObjectScaleRoiTag).Get<Rect>();
      object_scale = GetObjectScale(roi);
    }

    auto out_landmarks = absl::make_unique<LandmarkList>();
    MP_RETURN_IF_ERROR(landmarks_filter_->Apply(
        in_landmarks, timestamp, object_scale, out_landmarks.get()));

    cc->Outputs()
        .Tag(kFilteredLandmarksTag)
        .Add(out_landmarks.release(), cc->InputTimestamp());
  }

  return absl::OkStatus();
}

}  // namespace mediapipe
