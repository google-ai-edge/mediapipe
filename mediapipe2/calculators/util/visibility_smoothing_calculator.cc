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
#include "mediapipe/calculators/util/visibility_smoothing_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/util/filtering/low_pass_filter.h"

namespace mediapipe {

namespace {

constexpr char kNormalizedLandmarksTag[] = "NORM_LANDMARKS";
constexpr char kLandmarksTag[] = "LANDMARKS";
constexpr char kNormalizedFilteredLandmarksTag[] = "NORM_FILTERED_LANDMARKS";
constexpr char kFilteredLandmarksTag[] = "FILTERED_LANDMARKS";

using mediapipe::LowPassFilter;

// Abstract class for various visibility filters.
class VisibilityFilter {
 public:
  virtual ~VisibilityFilter() = default;

  virtual absl::Status Reset() { return absl::OkStatus(); }

  virtual absl::Status Apply(const LandmarkList& in_landmarks,
                             const absl::Duration& timestamp,
                             LandmarkList* out_landmarks) = 0;

  virtual absl::Status Apply(const NormalizedLandmarkList& in_landmarks,
                             const absl::Duration& timestamp,
                             NormalizedLandmarkList* out_landmarks) = 0;
};

// Returns visibility as is without smoothing.
class NoFilter : public VisibilityFilter {
 public:
  absl::Status Apply(const NormalizedLandmarkList& in_landmarks,
                     const absl::Duration& timestamp,
                     NormalizedLandmarkList* out_landmarks) override {
    *out_landmarks = in_landmarks;
    return absl::OkStatus();
  }

  absl::Status Apply(const LandmarkList& in_landmarks,
                     const absl::Duration& timestamp,
                     LandmarkList* out_landmarks) override {
    *out_landmarks = in_landmarks;
    return absl::OkStatus();
  }
};

// Please check LowPassFilter documentation for details.
class LowPassVisibilityFilter : public VisibilityFilter {
 public:
  LowPassVisibilityFilter(float alpha) : alpha_(alpha) {}

  absl::Status Reset() override {
    visibility_filters_.clear();
    return absl::OkStatus();
  }

  absl::Status Apply(const LandmarkList& in_landmarks,
                     const absl::Duration& timestamp,
                     LandmarkList* out_landmarks) override {
    return ApplyImpl<LandmarkList>(in_landmarks, timestamp, out_landmarks);
  }

  absl::Status Apply(const NormalizedLandmarkList& in_landmarks,
                     const absl::Duration& timestamp,
                     NormalizedLandmarkList* out_landmarks) override {
    return ApplyImpl<NormalizedLandmarkList>(in_landmarks, timestamp,
                                             out_landmarks);
  }

 private:
  template <class LandmarksType>
  absl::Status ApplyImpl(const LandmarksType& in_landmarks,
                         const absl::Duration& timestamp,
                         LandmarksType* out_landmarks) {
    // Initializes filters for the first time or after Reset. If initialized
    // then check the size.
    int n_landmarks = in_landmarks.landmark_size();
    if (!visibility_filters_.empty()) {
      RET_CHECK_EQ(visibility_filters_.size(), n_landmarks);
    } else {
      visibility_filters_.resize(n_landmarks, LowPassFilter(alpha_));
    }

    // Filter visibilities.
    for (int i = 0; i < in_landmarks.landmark_size(); ++i) {
      const auto& in_landmark = in_landmarks.landmark(i);

      auto* out_landmark = out_landmarks->add_landmark();
      *out_landmark = in_landmark;
      out_landmark->set_visibility(
          visibility_filters_[i].Apply(in_landmark.visibility()));
    }

    return absl::OkStatus();
  }

  float alpha_;
  std::vector<LowPassFilter> visibility_filters_;
};

}  // namespace

// A calculator to smooth landmark visibilities over time.
//
// Exactly one landmarks input stream is expected. Output stream type should be
// the same as the input one.
//
// Inputs:
//   LANDMARKS (optional): A LandmarkList of landmarks you want to smooth.
//   NORM_LANDMARKS (optional): A NormalizedLandmarkList of landmarks you want
//     to smooth.
//
// Outputs:
//   FILTERED_LANDMARKS (optional): A LandmarkList of smoothed landmarks.
//   NORM_FILTERED_LANDMARKS (optional): A NormalizedLandmarkList of smoothed
//     landmarks.
//
// Example config:
//   node {
//     calculator: "VisibilitySmoothingCalculator"
//     input_stream: "NORM_LANDMARKS:pose_landmarks"
//     output_stream: "NORM_FILTERED_LANDMARKS:pose_landmarks_filtered"
//     options: {
//       [mediapipe.VisibilitySmoothingCalculatorOptions.ext] {
//         low_pass_filter: {
//           alpha: 0.1
//         }
//       }
//     }
//   }
//
class VisibilitySmoothingCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  std::unique_ptr<VisibilityFilter> visibility_filter_;
};
REGISTER_CALCULATOR(VisibilitySmoothingCalculator);

absl::Status VisibilitySmoothingCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag(kNormalizedLandmarksTag) ^
            cc->Inputs().HasTag(kLandmarksTag))
      << "Exactly one landmarks input stream is expected";
  if (cc->Inputs().HasTag(kNormalizedLandmarksTag)) {
    cc->Inputs().Tag(kNormalizedLandmarksTag).Set<NormalizedLandmarkList>();
    RET_CHECK(cc->Outputs().HasTag(kNormalizedFilteredLandmarksTag))
        << "Landmarks output stream should of the same type as input one";
    cc->Outputs()
        .Tag(kNormalizedFilteredLandmarksTag)
        .Set<NormalizedLandmarkList>();
  } else {
    cc->Inputs().Tag(kLandmarksTag).Set<LandmarkList>();
    RET_CHECK(cc->Outputs().HasTag(kFilteredLandmarksTag))
        << "Landmarks output stream should of the same type as input one";
    cc->Outputs().Tag(kFilteredLandmarksTag).Set<LandmarkList>();
  }

  return absl::OkStatus();
}

absl::Status VisibilitySmoothingCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  // Pick visibility filter.
  const auto& options = cc->Options<VisibilitySmoothingCalculatorOptions>();
  if (options.has_no_filter()) {
    visibility_filter_ = absl::make_unique<NoFilter>();
  } else if (options.has_low_pass_filter()) {
    visibility_filter_ = absl::make_unique<LowPassVisibilityFilter>(
        options.low_pass_filter().alpha());
  } else {
    RET_CHECK_FAIL()
        << "Visibility filter is either not specified or not supported";
  }

  return absl::OkStatus();
}

absl::Status VisibilitySmoothingCalculator::Process(CalculatorContext* cc) {
  // Check that landmarks are not empty and reset the filter if so.
  // Don't emit an empty packet for this timestamp.
  if ((cc->Inputs().HasTag(kNormalizedLandmarksTag) &&
       cc->Inputs().Tag(kNormalizedLandmarksTag).IsEmpty()) ||
      (cc->Inputs().HasTag(kLandmarksTag) &&
       cc->Inputs().Tag(kLandmarksTag).IsEmpty())) {
    MP_RETURN_IF_ERROR(visibility_filter_->Reset());
    return absl::OkStatus();
  }

  const auto& timestamp =
      absl::Microseconds(cc->InputTimestamp().Microseconds());

  if (cc->Inputs().HasTag(kNormalizedLandmarksTag)) {
    const auto& in_landmarks =
        cc->Inputs().Tag(kNormalizedLandmarksTag).Get<NormalizedLandmarkList>();
    auto out_landmarks = absl::make_unique<NormalizedLandmarkList>();
    MP_RETURN_IF_ERROR(visibility_filter_->Apply(in_landmarks, timestamp,
                                                 out_landmarks.get()));
    cc->Outputs()
        .Tag(kNormalizedFilteredLandmarksTag)
        .Add(out_landmarks.release(), cc->InputTimestamp());
  } else {
    const auto& in_landmarks =
        cc->Inputs().Tag(kLandmarksTag).Get<LandmarkList>();
    auto out_landmarks = absl::make_unique<LandmarkList>();
    MP_RETURN_IF_ERROR(visibility_filter_->Apply(in_landmarks, timestamp,
                                                 out_landmarks.get()));
    cc->Outputs()
        .Tag(kFilteredLandmarksTag)
        .Add(out_landmarks.release(), cc->InputTimestamp());
  }

  return absl::OkStatus();
}

}  // namespace mediapipe
