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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

namespace {

constexpr char kLandmarksTag[] = "LANDMARKS";
constexpr char kRectTag[] = "NORM_RECT";

}  // namespace

// Projects world landmarks from the rectangle to original coordinates.
//
// World landmarks are predicted in meters rather than in pixels of the image
// and have origin in the middle of the hips rather than in the corner of the
// pose image (cropped with given rectangle). Thus only rotation (but not scale
// and translation) is applied to the landmarks to transform them back to
// original coordinates.
//
// Input:
//   LANDMARKS: A LandmarkList representing world landmarks in the rectangle.
//   NORM_RECT: An NormalizedRect representing a normalized rectangle in image
//              coordinates. (Optional)
//
// Output:
//   LANDMARKS: A LandmarkList representing world landmarks projected (rotated
//              but not scaled or translated) from the rectangle to original
//              coordinates.
//
// Usage example:
// node {
//   calculator: "WorldLandmarkProjectionCalculator"
//   input_stream: "LANDMARKS:landmarks"
//   input_stream: "NORM_RECT:rect"
//   output_stream: "LANDMARKS:projected_landmarks"
// }
//
class WorldLandmarkProjectionCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag(kLandmarksTag).Set<LandmarkList>();
    if (cc->Inputs().HasTag(kRectTag)) {
      cc->Inputs().Tag(kRectTag).Set<NormalizedRect>();
    }
    cc->Outputs().Tag(kLandmarksTag).Set<LandmarkList>();

    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));

    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    // Check that landmarks and rect are not empty.
    if (cc->Inputs().Tag(kLandmarksTag).IsEmpty() ||
        (cc->Inputs().HasTag(kRectTag) &&
         cc->Inputs().Tag(kRectTag).IsEmpty())) {
      return absl::OkStatus();
    }

    const auto& in_landmarks =
        cc->Inputs().Tag(kLandmarksTag).Get<LandmarkList>();
    std::function<void(const Landmark&, Landmark*)> rotate_fn;
    if (cc->Inputs().HasTag(kRectTag)) {
      const auto& in_rect = cc->Inputs().Tag(kRectTag).Get<NormalizedRect>();
      const float cosa = std::cos(in_rect.rotation());
      const float sina = std::sin(in_rect.rotation());
      rotate_fn = [cosa, sina](const Landmark& in_landmark,
                               Landmark* out_landmark) {
        out_landmark->set_x(cosa * in_landmark.x() - sina * in_landmark.y());
        out_landmark->set_y(sina * in_landmark.x() + cosa * in_landmark.y());
      };
    }

    auto out_landmarks = absl::make_unique<LandmarkList>();
    for (int i = 0; i < in_landmarks.landmark_size(); ++i) {
      const auto& in_landmark = in_landmarks.landmark(i);

      Landmark* out_landmark = out_landmarks->add_landmark();
      *out_landmark = in_landmark;

      if (rotate_fn) {
        rotate_fn(in_landmark, out_landmark);
      }
    }

    cc->Outputs()
        .Tag(kLandmarksTag)
        .Add(out_landmarks.release(), cc->InputTimestamp());

    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(WorldLandmarkProjectionCalculator);

}  // namespace mediapipe
