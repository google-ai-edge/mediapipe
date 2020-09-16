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

#include "absl/algorithm/container.h"
#include "mediapipe/calculators/util/landmarks_smoothing_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/util/filtering/relative_velocity_filter.h"

namespace mediapipe {

namespace {

constexpr char kNormalizedLandmarksTag[] = "NORM_LANDMARKS";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kNormalizedFilteredLandmarksTag[] = "NORM_FILTERED_LANDMARKS";

using ::mediapipe::RelativeVelocityFilter;

// Estimate object scale to use its inverse value as velocity scale for
// RelativeVelocityFilter. If value will be too small (less than
// `options_.min_allowed_object_scale`) smoothing will be disabled and
// landmarks will be returned as is.
// Object scale is calculated as average between bounding box width and height
// with sides parallel to axis.
float GetObjectScale(const NormalizedLandmarkList& landmarks, int image_width,
                     int image_height) {
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

  const float object_width = (x_max - x_min) * image_width;
  const float object_height = (y_max - y_min) * image_height;

  return (object_width + object_height) / 2.0f;
}

// Abstract class for various landmarks filters.
class LandmarksFilter {
 public:
  virtual ~LandmarksFilter() = default;

  virtual ::mediapipe::Status Reset() { return ::mediapipe::OkStatus(); }

  virtual ::mediapipe::Status Apply(const NormalizedLandmarkList& in_landmarks,
                                    const std::pair<int, int>& image_size,
                                    const absl::Duration& timestamp,
                                    NormalizedLandmarkList* out_landmarks) = 0;
};

// Returns landmarks as is without smoothing.
class NoFilter : public LandmarksFilter {
 public:
  ::mediapipe::Status Apply(const NormalizedLandmarkList& in_landmarks,
                            const std::pair<int, int>& image_size,
                            const absl::Duration& timestamp,
                            NormalizedLandmarkList* out_landmarks) override {
    *out_landmarks = in_landmarks;
    return ::mediapipe::OkStatus();
  }
};

// Please check RelativeVelocityFilter documentation for details.
class VelocityFilter : public LandmarksFilter {
 public:
  VelocityFilter(int window_size, float velocity_scale,
                 float min_allowed_object_scale)
      : window_size_(window_size),
        velocity_scale_(velocity_scale),
        min_allowed_object_scale_(min_allowed_object_scale) {}

  ::mediapipe::Status Reset() override {
    x_filters_.clear();
    y_filters_.clear();
    z_filters_.clear();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Apply(const NormalizedLandmarkList& in_landmarks,
                            const std::pair<int, int>& image_size,
                            const absl::Duration& timestamp,
                            NormalizedLandmarkList* out_landmarks) override {
    // Get image size.
    int image_width;
    int image_height;
    std::tie(image_width, image_height) = image_size;

    // Get value scale as inverse value of the object scale.
    // If value is too small smoothing will be disabled and landmarks will be
    // returned as is.
    const float object_scale =
        GetObjectScale(in_landmarks, image_width, image_height);
    if (object_scale < min_allowed_object_scale_) {
      *out_landmarks = in_landmarks;
      return ::mediapipe::OkStatus();
    }
    const float value_scale = 1.0f / object_scale;

    // Initialize filters once.
    MP_RETURN_IF_ERROR(InitializeFiltersIfEmpty(in_landmarks.landmark_size()));

    // Filter landmarks. Every axis of every landmark is filtered separately.
    for (int i = 0; i < in_landmarks.landmark_size(); ++i) {
      const NormalizedLandmark& in_landmark = in_landmarks.landmark(i);

      NormalizedLandmark* out_landmark = out_landmarks->add_landmark();
      out_landmark->set_x(x_filters_[i].Apply(timestamp, value_scale,
                                              in_landmark.x() * image_width) /
                          image_width);
      out_landmark->set_y(y_filters_[i].Apply(timestamp, value_scale,
                                              in_landmark.y() * image_height) /
                          image_height);
      // Scale Z the save was as X (using image width).
      out_landmark->set_z(z_filters_[i].Apply(timestamp, value_scale,
                                              in_landmark.z() * image_width) /
                          image_width);
      // Keep visibility as is.
      out_landmark->set_visibility(in_landmark.visibility());
    }

    return ::mediapipe::OkStatus();
  }

 private:
  // Initializes filters for the first time or after Reset. If initialized then
  // check the size.
  ::mediapipe::Status InitializeFiltersIfEmpty(const int n_landmarks) {
    if (!x_filters_.empty()) {
      RET_CHECK_EQ(x_filters_.size(), n_landmarks);
      RET_CHECK_EQ(y_filters_.size(), n_landmarks);
      RET_CHECK_EQ(z_filters_.size(), n_landmarks);
      return ::mediapipe::OkStatus();
    }

    x_filters_.resize(n_landmarks,
                      RelativeVelocityFilter(window_size_, velocity_scale_));
    y_filters_.resize(n_landmarks,
                      RelativeVelocityFilter(window_size_, velocity_scale_));
    z_filters_.resize(n_landmarks,
                      RelativeVelocityFilter(window_size_, velocity_scale_));

    return ::mediapipe::OkStatus();
  }

  int window_size_;
  float velocity_scale_;
  float min_allowed_object_scale_;

  std::vector<RelativeVelocityFilter> x_filters_;
  std::vector<RelativeVelocityFilter> y_filters_;
  std::vector<RelativeVelocityFilter> z_filters_;
};

}  // namespace

// A calculator to smooth landmarks over time.
//
// Inputs:
//   NORM_LANDMARKS: A NormalizedLandmarkList of landmarks you want to smooth.
//   IMAGE_SIZE: A std::pair<int, int> represention of image width and height.
//     Required to perform all computations in absolute coordinates to avoid any
//     influence of normalized values.
//
// Outputs:
//   NORM_FILTERED_LANDMARKS: A NormalizedLandmarkList of smoothed landmarks.
//
// Example config:
//   node {
//     calculator: "LandmarksSmoothingCalculator"
//     input_stream: "NORM_LANDMARKS:pose_landmarks"
//     input_stream: "IMAGE_SIZE:image_size"
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
  static ::mediapipe::Status GetContract(CalculatorContract* cc);
  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  LandmarksFilter* landmarks_filter_;
};
REGISTER_CALCULATOR(LandmarksSmoothingCalculator);

::mediapipe::Status LandmarksSmoothingCalculator::GetContract(
    CalculatorContract* cc) {
  cc->Inputs().Tag(kNormalizedLandmarksTag).Set<NormalizedLandmarkList>();
  cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>();
  cc->Outputs()
      .Tag(kNormalizedFilteredLandmarksTag)
      .Set<NormalizedLandmarkList>();

  return ::mediapipe::OkStatus();
}

::mediapipe::Status LandmarksSmoothingCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  // Pick landmarks filter.
  const auto& options = cc->Options<LandmarksSmoothingCalculatorOptions>();
  if (options.has_no_filter()) {
    landmarks_filter_ = new NoFilter();
  } else if (options.has_velocity_filter()) {
    landmarks_filter_ = new VelocityFilter(
        options.velocity_filter().window_size(),
        options.velocity_filter().velocity_scale(),
        options.velocity_filter().min_allowed_object_scale());
  } else {
    RET_CHECK_FAIL()
        << "Landmarks filter is either not specified or not supported";
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status LandmarksSmoothingCalculator::Process(
    CalculatorContext* cc) {
  // Check that landmarks are not empty and reset the filter if so.
  // Don't emit an empty packet for this timestamp.
  if (cc->Inputs().Tag(kNormalizedLandmarksTag).IsEmpty()) {
    MP_RETURN_IF_ERROR(landmarks_filter_->Reset());
    return ::mediapipe::OkStatus();
  }

  const auto& in_landmarks =
      cc->Inputs().Tag(kNormalizedLandmarksTag).Get<NormalizedLandmarkList>();
  const auto& image_size =
      cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();
  const auto& timestamp =
      absl::Microseconds(cc->InputTimestamp().Microseconds());

  auto out_landmarks = absl::make_unique<NormalizedLandmarkList>();
  MP_RETURN_IF_ERROR(landmarks_filter_->Apply(in_landmarks, image_size,
                                              timestamp, out_landmarks.get()));

  cc->Outputs()
      .Tag(kNormalizedFilteredLandmarksTag)
      .Add(out_landmarks.release(), cc->InputTimestamp());

  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
