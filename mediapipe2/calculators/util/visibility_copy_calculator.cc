// Copyright 2021 The MediaPipe Authors.
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
#include "mediapipe/calculators/util/visibility_copy_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {

namespace {

constexpr char kLandmarksFromTag[] = "LANDMARKS_FROM";
constexpr char kNormalizedLandmarksFromTag[] = "NORM_LANDMARKS_FROM";
constexpr char kLandmarksToTag[] = "LANDMARKS_TO";
constexpr char kNormalizedLandmarksToTag[] = "NORM_LANDMARKS_TO";

}  // namespace

// A calculator to copy visibility and presence between landmarks.
//
// Landmarks to copy from and to copy to can be of different type (normalized or
// non-normalized), but ladnmarks to copy to and output landmarks should be of
// the same type. Exactly one stream to copy landmarks from, to copy to and to
// output should be provided.
//
// Inputs:
//   LANDMARKS_FROM (optional): A LandmarkList of landmarks to copy from.
//   NORM_LANDMARKS_FROM (optional): A NormalizedLandmarkList of landmarks to
//     copy from.
//   LANDMARKS_TO (optional): A LandmarkList of landmarks to copy to.
//   NORM_LANDMARKS_TO (optional): A NormalizedLandmarkList of landmarks to copy
//     to.
//
// Outputs:
//   LANDMARKS_TO (optional): A LandmarkList of landmarks from LANDMARKS_TO and
//     visibility/presence from LANDMARKS_FROM or NORM_LANDMARKS_FROM.
//   NORM_LANDMARKS_TO (optional): A NormalizedLandmarkList of landmarks to copy
//     to.
//
// Example config:
//   node {
//     calculator: "VisibilityCopyCalculator"
//     input_stream: "NORM_LANDMARKS_FROM:pose_landmarks"
//     input_stream: "LANDMARKS_TO:pose_world_landmarks"
//     output_stream: "LANDMARKS_TO:pose_world_landmarks_with_visibility"
//     options: {
//       [mediapipe.VisibilityCopyCalculatorOptions.ext] {
//         copy_visibility: true
//         copy_presence: true
//       }
//     }
//   }
//
class VisibilityCopyCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  template <class LandmarkFromType, class LandmarkToType>
  absl::Status CopyVisibility(CalculatorContext* cc,
                              const std::string& landmarks_from_tag,
                              const std::string& landmarks_to_tag);

  bool copy_visibility_;
  bool copy_presence_;
};
REGISTER_CALCULATOR(VisibilityCopyCalculator);

absl::Status VisibilityCopyCalculator::GetContract(CalculatorContract* cc) {
  // Landmarks to copy from.
  RET_CHECK(cc->Inputs().HasTag(kLandmarksFromTag) ^
            cc->Inputs().HasTag(kNormalizedLandmarksFromTag))
      << "Exatly one landmarks stream to copy from should be provided";
  if (cc->Inputs().HasTag(kLandmarksFromTag)) {
    cc->Inputs().Tag(kLandmarksFromTag).Set<LandmarkList>();
  } else {
    cc->Inputs().Tag(kNormalizedLandmarksFromTag).Set<NormalizedLandmarkList>();
  }

  // Landmarks to copy to and corresponding output landmarks.
  RET_CHECK(cc->Inputs().HasTag(kLandmarksToTag) ^
            cc->Inputs().HasTag(kNormalizedLandmarksToTag))
      << "Exatly one landmarks stream to copy to should be provided";
  if (cc->Inputs().HasTag(kLandmarksToTag)) {
    cc->Inputs().Tag(kLandmarksToTag).Set<LandmarkList>();

    RET_CHECK(cc->Outputs().HasTag(kLandmarksToTag))
        << "Landmarks to copy to and output stream types should be the same";
    cc->Outputs().Tag(kLandmarksToTag).Set<LandmarkList>();
  } else {
    cc->Inputs().Tag(kNormalizedLandmarksToTag).Set<NormalizedLandmarkList>();

    RET_CHECK(cc->Outputs().HasTag(kNormalizedLandmarksToTag))
        << "Landmarks to copy to and output stream types should be the same";
    cc->Outputs().Tag(kNormalizedLandmarksToTag).Set<NormalizedLandmarkList>();
  }

  return absl::OkStatus();
}

absl::Status VisibilityCopyCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  const auto& options = cc->Options<VisibilityCopyCalculatorOptions>();
  copy_visibility_ = options.copy_visibility();
  copy_presence_ = options.copy_presence();

  return absl::OkStatus();
}

absl::Status VisibilityCopyCalculator::Process(CalculatorContext* cc) {
  // Switch between all four possible combinations of landmarks from and
  // landmarks to types (normalized and non-normalized).
  auto status = absl::OkStatus();
  if (cc->Inputs().HasTag(kLandmarksFromTag)) {
    if (cc->Inputs().HasTag(kLandmarksToTag)) {
      status = CopyVisibility<LandmarkList, LandmarkList>(cc, kLandmarksFromTag,
                                                          kLandmarksToTag);
    } else {
      status = CopyVisibility<LandmarkList, NormalizedLandmarkList>(
          cc, kLandmarksFromTag, kNormalizedLandmarksToTag);
    }
  } else {
    if (cc->Inputs().HasTag(kLandmarksToTag)) {
      status = CopyVisibility<NormalizedLandmarkList, LandmarkList>(
          cc, kNormalizedLandmarksFromTag, kLandmarksToTag);
    } else {
      status = CopyVisibility<NormalizedLandmarkList, NormalizedLandmarkList>(
          cc, kNormalizedLandmarksFromTag, kNormalizedLandmarksToTag);
    }
  }

  return status;
}

template <class LandmarkFromType, class LandmarkToType>
absl::Status VisibilityCopyCalculator::CopyVisibility(
    CalculatorContext* cc, const std::string& landmarks_from_tag,
    const std::string& landmarks_to_tag) {
  // Check that both landmarks to copy from and to copy to are non empty.
  if (cc->Inputs().Tag(landmarks_from_tag).IsEmpty() ||
      cc->Inputs().Tag(landmarks_to_tag).IsEmpty()) {
    return absl::OkStatus();
  }

  const auto landmarks_from =
      cc->Inputs().Tag(landmarks_from_tag).Get<LandmarkFromType>();
  const auto landmarks_to =
      cc->Inputs().Tag(landmarks_to_tag).Get<LandmarkToType>();
  auto landmarks_out = absl::make_unique<LandmarkToType>();

  for (int i = 0; i < landmarks_from.landmark_size(); ++i) {
    const auto& landmark_from = landmarks_from.landmark(i);
    const auto& landmark_to = landmarks_to.landmark(i);

    // Create output landmark and copy all fields from the `to` landmark.
    const auto& landmark_out = landmarks_out->add_landmark();
    *landmark_out = landmark_to;

    // Copy visibility and presence from the `from` landmark.
    if (copy_visibility_) {
      landmark_out->set_visibility(landmark_from.visibility());
    }
    if (copy_presence_) {
      landmark_out->set_presence(landmark_from.presence());
    }
  }

  cc->Outputs()
      .Tag(landmarks_to_tag)
      .Add(landmarks_out.release(), cc->InputTimestamp());

  return absl::OkStatus();
}

}  // namespace mediapipe
