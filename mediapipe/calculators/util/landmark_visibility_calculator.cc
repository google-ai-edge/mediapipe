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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

namespace {

constexpr char kNormalizedLandmarksTag[] = "NORM_LANDMARKS";
constexpr char kVisibilityTag[] = "VISIBILITY";

}  // namespace

// A calculator to extract visibility from the landmark.
//
// Inputs:
//   NORM_LANDMARKS: A NormalizedLandmarkList with only a single landmark to
//     take visibility from. It's a list and not single landmark as
//     split/concatenate calculators work with lists.
//
// Outputs:
//   VISIBILITY: Float visibility of the given landmark.
//
// Example config:
//   node {
//     calculator: "LandmarkVisibilityCalculator"
//     input_stream: "NORM_LANDMARKS:landmarks"
//     output_stream: "VISIBILITY:visibility"
//   }
//
class LandmarkVisibilityCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
};
REGISTER_CALCULATOR(LandmarkVisibilityCalculator);

absl::Status LandmarkVisibilityCalculator::GetContract(CalculatorContract* cc) {
  cc->Inputs().Tag(kNormalizedLandmarksTag).Set<NormalizedLandmarkList>();
  cc->Outputs().Tag(kVisibilityTag).Set<float>();

  return absl::OkStatus();
}

absl::Status LandmarkVisibilityCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  return absl::OkStatus();
}

absl::Status LandmarkVisibilityCalculator::Process(CalculatorContext* cc) {
  // Check that landmark is not empty.
  // Don't emit an empty packet for this timestamp.
  if (cc->Inputs().Tag(kNormalizedLandmarksTag).IsEmpty()) {
    return absl::OkStatus();
  }

  const auto& landmarks =
      cc->Inputs().Tag(kNormalizedLandmarksTag).Get<NormalizedLandmarkList>();
  RET_CHECK_EQ(landmarks.landmark_size(), 1);
  float visibility = landmarks.landmark(0).visibility();

  cc->Outputs()
      .Tag(kVisibilityTag)
      .AddPacket(MakePacket<float>(visibility).At(cc->InputTimestamp()));

  return absl::OkStatus();
}

}  // namespace mediapipe
