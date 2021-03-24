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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

namespace {

constexpr char kNormalizedLandmarksTag[] = "NORM_LANDMARKS";
constexpr char kVisibilityTag[] = "VISIBILITY";

}  // namespace

// A calculator to set landmark visibility.
//
// Inputs:
//   NORM_LANDMARKS: A NormalizedLandmarkList with only a single landmark to set
//     visibility to. It's a list and not single landmark as split/concatenate
//     calculators work with lists.
//
//   VISIBILITY: Float visibility of the given landmark.
//
// Outputs:
//   NORM_LANDMARKS: A NormalizedLandmarkList with only single landmark with
//     updated visibility.
//
// Example config:
//   node {
//     calculator: "SetLandmarkVisibility"
//     input_stream: "NORM_LANDMARKS:landmarks"
//     input_stream: "VISIBILITY:visibility"
//     output_stream: "NORM_LANDMARKS:landmarks_with_visibility"
//   }
//
class SetLandmarkVisibilityCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
};
REGISTER_CALCULATOR(SetLandmarkVisibilityCalculator);

absl::Status SetLandmarkVisibilityCalculator::GetContract(
    CalculatorContract* cc) {
  cc->Inputs().Tag(kNormalizedLandmarksTag).Set<NormalizedLandmarkList>();
  cc->Inputs().Tag(kVisibilityTag).Set<float>();
  cc->Outputs().Tag(kNormalizedLandmarksTag).Set<NormalizedLandmarkList>();

  return absl::OkStatus();
}

absl::Status SetLandmarkVisibilityCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  return absl::OkStatus();
}

absl::Status SetLandmarkVisibilityCalculator::Process(CalculatorContext* cc) {
  // Check that landmark and visibility are not empty.
  // Don't emit an empty packet for this timestamp.
  if (cc->Inputs().Tag(kNormalizedLandmarksTag).IsEmpty() ||
      cc->Inputs().Tag(kVisibilityTag).IsEmpty()) {
    return absl::OkStatus();
  }

  const auto& in_landmarks =
      cc->Inputs().Tag(kNormalizedLandmarksTag).Get<NormalizedLandmarkList>();
  RET_CHECK_EQ(in_landmarks.landmark_size(), 1);
  const NormalizedLandmark& in_landmark = in_landmarks.landmark(0);

  const auto& visibility = cc->Inputs().Tag(kVisibilityTag).Get<float>();

  auto out_landmarks = absl::make_unique<NormalizedLandmarkList>();
  NormalizedLandmark* out_landmark = out_landmarks->add_landmark();
  *out_landmark = in_landmark;
  // Update visibility.
  out_landmark->set_visibility(visibility);

  cc->Outputs()
      .Tag(kNormalizedLandmarksTag)
      .Add(out_landmarks.release(), cc->InputTimestamp());

  return absl::OkStatus();
}

}  // namespace mediapipe
