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
//   NORM_LANDMARKS: A NormalizedLandmarkList representing landmarks
//                   in a normalized rectangle.
//   NORM_RECT: An NormalizedRect representing a normalized rectangle in image
//              coordinates.
//
// Output:
//   NORM_LANDMARKS: A NormalizedLandmarkList representing landmarks
//                   with their locations adjusted to the image.
//
// Usage example:
// node {
//   calculator: "LandmarkProjectionCalculator"
//   input_stream: "NORM_LANDMARKS:landmarks"
//   input_stream: "NORM_RECT:rect"
//   output_stream: "NORM_LANDMARKS:projected_landmarks"
// }
//
// node {
//   calculator: "LandmarkProjectionCalculator"
//   input_stream: "NORM_LANDMARKS:0:landmarks_0"
//   input_stream: "NORM_LANDMARKS:1:landmarks_1"
//   input_stream: "NORM_RECT:rect"
//   output_stream: "NORM_LANDMARKS:0:projected_landmarks_0"
//   output_stream: "NORM_LANDMARKS:1:projected_landmarks_1"
// }
class LandmarkProjectionCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(cc->Inputs().HasTag(kLandmarksTag) &&
              cc->Inputs().HasTag(kRectTag))
        << "Missing one or more input streams.";

    RET_CHECK_EQ(cc->Inputs().NumEntries(kLandmarksTag),
                 cc->Outputs().NumEntries(kLandmarksTag))
        << "Same number of input and output landmarks is required.";

    for (CollectionItemId id = cc->Inputs().BeginId(kLandmarksTag);
         id != cc->Inputs().EndId(kLandmarksTag); ++id) {
      cc->Inputs().Get(id).Set<NormalizedLandmarkList>();
    }
    cc->Inputs().Tag(kRectTag).Set<NormalizedRect>();

    for (CollectionItemId id = cc->Outputs().BeginId(kLandmarksTag);
         id != cc->Outputs().EndId(kLandmarksTag); ++id) {
      cc->Outputs().Get(id).Set<NormalizedLandmarkList>();
    }

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    if (cc->Inputs().Tag(kRectTag).IsEmpty()) {
      return ::mediapipe::OkStatus();
    }
    const auto& input_rect = cc->Inputs().Tag(kRectTag).Get<NormalizedRect>();

    const auto& options =
        cc->Options<::mediapipe::LandmarkProjectionCalculatorOptions>();

    CollectionItemId input_id = cc->Inputs().BeginId(kLandmarksTag);
    CollectionItemId output_id = cc->Outputs().BeginId(kLandmarksTag);
    // Number of inputs and outpus is the same according to the contract.
    for (; input_id != cc->Inputs().EndId(kLandmarksTag);
         ++input_id, ++output_id) {
      const auto& input_packet = cc->Inputs().Get(input_id);
      if (input_packet.IsEmpty()) {
        continue;
      }

      const auto& input_landmarks = input_packet.Get<NormalizedLandmarkList>();
      NormalizedLandmarkList output_landmarks;
      for (int i = 0; i < input_landmarks.landmark_size(); ++i) {
        const NormalizedLandmark& landmark = input_landmarks.landmark(i);
        NormalizedLandmark* new_landmark = output_landmarks.add_landmark();

        const float x = landmark.x() - 0.5f;
        const float y = landmark.y() - 0.5f;
        const float angle =
            options.ignore_rotation() ? 0 : input_rect.rotation();
        float new_x = std::cos(angle) * x - std::sin(angle) * y;
        float new_y = std::sin(angle) * x + std::cos(angle) * y;

        new_x = new_x * input_rect.width() + input_rect.x_center();
        new_y = new_y * input_rect.height() + input_rect.y_center();

        new_landmark->set_x(new_x);
        new_landmark->set_y(new_y);
        // Keep z-coord as is.
        new_landmark->set_z(landmark.z());
      }

      cc->Outputs().Get(output_id).AddPacket(
          MakePacket<NormalizedLandmarkList>(output_landmarks)
              .At(cc->InputTimestamp()));
    }
    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(LandmarkProjectionCalculator);

}  // namespace mediapipe
