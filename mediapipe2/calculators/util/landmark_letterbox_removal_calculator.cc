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
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

namespace {

constexpr char kLandmarksTag[] = "LANDMARKS";
constexpr char kLetterboxPaddingTag[] = "LETTERBOX_PADDING";

}  // namespace

// Adjusts landmark locations on a letterboxed image to the corresponding
// locations on the same image with the letterbox removed. This is useful to map
// the landmarks inferred from a letterboxed image, for example, output of
// the ImageTransformationCalculator when the scale mode is FIT, back to the
// corresponding input image before letterboxing.
//
// Input:
//   LANDMARKS: A NormalizedLandmarkList representing landmarks on an
//   letterboxed image.
//
//   LETTERBOX_PADDING: An std::array<float, 4> representing the letterbox
//   padding from the 4 sides ([left, top, right, bottom]) of the letterboxed
//   image, normalized to [0.f, 1.f] by the letterboxed image dimensions.
//
// Output:
//   LANDMARKS: An NormalizedLandmarkList proto representing landmarks with
//   their locations adjusted to the letterbox-removed (non-padded) image.
//
// Usage example:
// node {
//   calculator: "LandmarkLetterboxRemovalCalculator"
//   input_stream: "LANDMARKS:landmarks"
//   input_stream: "LETTERBOX_PADDING:letterbox_padding"
//   output_stream: "LANDMARKS:adjusted_landmarks"
// }
//
// node {
//   calculator: "LandmarkLetterboxRemovalCalculator"
//   input_stream: "LANDMARKS:0:landmarks_0"
//   input_stream: "LANDMARKS:1:landmarks_1"
//   input_stream: "LETTERBOX_PADDING:letterbox_padding"
//   output_stream: "LANDMARKS:0:adjusted_landmarks_0"
//   output_stream: "LANDMARKS:1:adjusted_landmarks_1"
// }
class LandmarkLetterboxRemovalCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(cc->Inputs().HasTag(kLandmarksTag) &&
              cc->Inputs().HasTag(kLetterboxPaddingTag))
        << "Missing one or more input streams.";

    RET_CHECK_EQ(cc->Inputs().NumEntries(kLandmarksTag),
                 cc->Outputs().NumEntries(kLandmarksTag))
        << "Same number of input and output landmarks is required.";

    for (CollectionItemId id = cc->Inputs().BeginId(kLandmarksTag);
         id != cc->Inputs().EndId(kLandmarksTag); ++id) {
      cc->Inputs().Get(id).Set<NormalizedLandmarkList>();
    }
    cc->Inputs().Tag(kLetterboxPaddingTag).Set<std::array<float, 4>>();

    for (CollectionItemId id = cc->Outputs().BeginId(kLandmarksTag);
         id != cc->Outputs().EndId(kLandmarksTag); ++id) {
      cc->Outputs().Get(id).Set<NormalizedLandmarkList>();
    }

    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));

    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (cc->Inputs().Tag(kLetterboxPaddingTag).IsEmpty()) {
      return absl::OkStatus();
    }
    const auto& letterbox_padding =
        cc->Inputs().Tag(kLetterboxPaddingTag).Get<std::array<float, 4>>();
    const float left = letterbox_padding[0];
    const float top = letterbox_padding[1];
    const float left_and_right = letterbox_padding[0] + letterbox_padding[2];
    const float top_and_bottom = letterbox_padding[1] + letterbox_padding[3];

    CollectionItemId input_id = cc->Inputs().BeginId(kLandmarksTag);
    CollectionItemId output_id = cc->Outputs().BeginId(kLandmarksTag);
    // Number of inputs and outpus is the same according to the contract.
    for (; input_id != cc->Inputs().EndId(kLandmarksTag);
         ++input_id, ++output_id) {
      const auto& input_packet = cc->Inputs().Get(input_id);
      if (input_packet.IsEmpty()) {
        continue;
      }

      const NormalizedLandmarkList& input_landmarks =
          input_packet.Get<NormalizedLandmarkList>();
      NormalizedLandmarkList output_landmarks;
      for (int i = 0; i < input_landmarks.landmark_size(); ++i) {
        const NormalizedLandmark& landmark = input_landmarks.landmark(i);
        NormalizedLandmark* new_landmark = output_landmarks.add_landmark();
        const float new_x = (landmark.x() - left) / (1.0f - left_and_right);
        const float new_y = (landmark.y() - top) / (1.0f - top_and_bottom);
        const float new_z =
            landmark.z() / (1.0f - left_and_right);  // Scale Z coordinate as X.
        *new_landmark = landmark;
        new_landmark->set_x(new_x);
        new_landmark->set_y(new_y);
        new_landmark->set_z(new_z);
      }

      cc->Outputs().Get(output_id).AddPacket(
          MakePacket<NormalizedLandmarkList>(output_landmarks)
              .At(cc->InputTimestamp()));
    }
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(LandmarkLetterboxRemovalCalculator);

}  // namespace mediapipe
