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

#include <cmath>
#include <vector>

#include "mediapipe/calculators/util/landmark_projection_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

namespace {

constexpr char kLandmarksTag[] = "NORM_LANDMARKS";
constexpr char kRectTag[] = "NORM_RECT";

}  // namespace

// Projects normalized landmarks in a rectangle to its original coordinates. The
// rectangle must also be in normalized coordinates.
// Input:
//   NORM_LANDMARKS: An std::vector<NormalizedLandmark> representing landmarks
//                   in a normalized rectangle.
//   NORM_RECT: An NormalizedRect representing a normalized rectangle in image
//              coordinates.
//
// Output:
//   NORM_LANDMARKS: An std::vector<NormalizedLandmark> representing landmarks
//                   with their locations adjusted to the image.
//
// Usage example:
// node {
//   calculator: "LandmarkProjectionCalculator"
//   input_stream: "NORM_LANDMARKS:landmarks"
//   input_stream: "NORM_RECT:rect"
//   output_stream: "NORM_LANDMARKS:projected_landmarks"
// }
class LandmarkProjectionCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(cc->Inputs().HasTag(kLandmarksTag) &&
              cc->Inputs().HasTag(kRectTag))
        << "Missing one or more input streams.";

    cc->Inputs().Tag(kLandmarksTag).Set<std::vector<NormalizedLandmark>>();
    cc->Inputs().Tag(kRectTag).Set<NormalizedRect>();

    cc->Outputs().Tag(kLandmarksTag).Set<std::vector<NormalizedLandmark>>();

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    const auto& options =
        cc->Options<::mediapipe::LandmarkProjectionCalculatorOptions>();
    // Only process if there's input landmarks.
    if (cc->Inputs().Tag(kLandmarksTag).IsEmpty()) {
      return ::mediapipe::OkStatus();
    }

    const auto& input_landmarks =
        cc->Inputs().Tag(kLandmarksTag).Get<std::vector<NormalizedLandmark>>();
    const auto& input_rect = cc->Inputs().Tag(kRectTag).Get<NormalizedRect>();

    auto output_landmarks =
        absl::make_unique<std::vector<NormalizedLandmark>>();
    for (const auto& landmark : input_landmarks) {
      NormalizedLandmark new_landmark;

      const float x = landmark.x() - 0.5f;
      const float y = landmark.y() - 0.5f;
      const float angle = options.ignore_rotation() ? 0 : input_rect.rotation();
      float new_x = std::cos(angle) * x - std::sin(angle) * y;
      float new_y = std::sin(angle) * x + std::cos(angle) * y;

      new_x = new_x * input_rect.width() + input_rect.x_center();
      new_y = new_y * input_rect.height() + input_rect.y_center();

      new_landmark.set_x(new_x);
      new_landmark.set_y(new_y);
      // Keep z-coord as is.
      new_landmark.set_z(landmark.z());

      output_landmarks->emplace_back(new_landmark);
    }

    cc->Outputs()
        .Tag(kLandmarksTag)
        .Add(output_landmarks.release(), cc->InputTimestamp());
    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(LandmarkProjectionCalculator);

}  // namespace mediapipe
