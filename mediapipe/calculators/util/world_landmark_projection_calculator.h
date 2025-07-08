// Copyright 2025 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_CALCULATORS_UTIL_WORLD_LANDMARK_PROJECTION_CALCULATOR_H_
#define MEDIAPIPE_CALCULATORS_UTIL_WORLD_LANDMARK_PROJECTION_CALCULATOR_H_

#include "absl/strings/string_view.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe::api3 {

inline constexpr absl::string_view kWorldLandmarkProjectionNodeName =
    "WorldLandmarkProjectionCalculator";

// Projects world landmarks from the rectangle to original coordinates.
//
// World landmarks are predicted in meters rather than in pixels of the image
// and have origin in the middle of the hips rather than in the corner of the
// pose image (cropped with given rectangle). Thus only rotation (but not scale
// and translation) is applied to the landmarks to transform them back to
// original coordinates.
//
// `CalculatorGraphConfig` usage example:
// node {
//   calculator: "WorldLandmarkProjectionCalculator"
//   input_stream: "LANDMARKS:landmarks"
//   input_stream: "NORM_RECT:rect"
//   output_stream: "LANDMARKS:projected_landmarks"
// }
//
struct WorldLandmarkProjectionNode : Node<kWorldLandmarkProjectionNodeName> {
  template <typename S>
  struct Contract {
    // A LandmarkList representing world landmarks in the rectangle.
    Input<S, mediapipe::LandmarkList> input_landmarks{"LANDMARKS"};

    // An NormalizedRect representing a normalized rectangle in image
    // coordinates.
    Optional<Input<S, mediapipe::NormalizedRect>> input_rect{"NORM_RECT"};

    // A LandmarkList representing world landmarks projected (rotated but not
    // scaled or translated) from the rectangle to original coordinates.
    Output<S, mediapipe::LandmarkList> output_landmarks{"LANDMARKS"};
  };
};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_CALCULATORS_UTIL_WORLD_LANDMARK_PROJECTION_CALCULATOR_H_
