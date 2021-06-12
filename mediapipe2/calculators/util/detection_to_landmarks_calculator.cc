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
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"

namespace mediapipe {

namespace {

constexpr char kDetectionTag[] = "DETECTION";
constexpr char kLandmarksTag[] = "LANDMARKS";

absl::Status ConvertDetectionToLandmarks(const Detection& detection,
                                         NormalizedLandmarkList* landmarks) {
  const auto& location_data = detection.location_data();
  for (int i = 0; i < location_data.relative_keypoints_size(); ++i) {
    const auto& keypoint = location_data.relative_keypoints(i);

    auto* landmark = landmarks->add_landmark();
    landmark->set_x(keypoint.x());
    landmark->set_y(keypoint.y());
  }

  return absl::OkStatus();
}

}  // namespace

// Converts a detection into a normalized landmark list by extracting the
// location data relative keypoints as landmarks.
//
// Input:
//   DETECTION - `Detection`
//     A detection to be converted.
//
// Output:
//   LANDMARKS - `NormalizedLandmarkList`
//     A converted normalized landmark list.
//
// Example:
//
//   node {
//     calculator: "DetectionToLandmarksCalculator"
//     input_stream: "DETECTION:detection"
//     output_stream: "LANDMARKS:landmarks"
//   }
//
class DetectionToLandmarksCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(cc->Inputs().HasTag(kDetectionTag));
    RET_CHECK(cc->Outputs().HasTag(kLandmarksTag));

    cc->Inputs().Tag(kDetectionTag).Set<Detection>();
    cc->Outputs().Tag(kLandmarksTag).Set<NormalizedLandmarkList>();

    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const auto& detection = cc->Inputs().Tag(kDetectionTag).Get<Detection>();

    auto landmarks = absl::make_unique<NormalizedLandmarkList>();
    MP_RETURN_IF_ERROR(ConvertDetectionToLandmarks(detection, landmarks.get()));

    cc->Outputs()
        .Tag(kLandmarksTag)
        .Add(landmarks.release(), cc->InputTimestamp());

    return absl::OkStatus();
  }
};

REGISTER_CALCULATOR(DetectionToLandmarksCalculator);

}  // namespace mediapipe
