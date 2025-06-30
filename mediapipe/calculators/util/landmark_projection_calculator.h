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

#ifndef MEDIAPIPE_CALCULATORS_UTIL_LANDMARK_PROJECTION_CALCULATOR_H_
#define MEDIAPIPE_CALCULATORS_UTIL_LANDMARK_PROJECTION_CALCULATOR_H_

#include <array>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/util/landmark_projection_calculator.pb.h"
#include "mediapipe/framework/api3/calculator_contract.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe::api3 {

inline constexpr absl::string_view kLandmarkProjectionNodeName =
    "LandmarkProjectionCalculator";

// Projects normalized landmarks to its original coordinates.
//
// NOTE: landmark's Z is projected in a custom way - it's scaled by width of
// the normalized region of interest used during landmarks detection.
//
// Usage examples (`CalculatorGraphConfig` proto):
//
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
//
// node {
//   calculator: "LandmarkProjectionCalculator"
//   input_stream: "NORM_LANDMARKS:landmarks"
//   input_stream: "PROECTION_MATRIX:matrix"
//   output_stream: "NORM_LANDMARKS:projected_landmarks"
// }
//
// node {
//   calculator: "LandmarkProjectionCalculator"
//   input_stream: "NORM_LANDMARKS:0:landmarks_0"
//   input_stream: "NORM_LANDMARKS:1:landmarks_1"
//   input_stream: "PROECTION_MATRIX:matrix"
//   output_stream: "NORM_LANDMARKS:0:projected_landmarks_0"
//   output_stream: "NORM_LANDMARKS:1:projected_landmarks_1"
// }
struct LandmarkProjectionNode : Node<kLandmarkProjectionNodeName> {
  template <typename S>
  struct Contract {
    // Represents landmarks in a normalized rectangle if NORM_RECT is specified
    // or landmarks that should be projected using PROJECTION_MATRIX if
    // specified. (Prefer using PROJECTION_MATRIX as it eliminates need of
    // letterbox removal step.)
    Repeated<Input<S, mediapipe::NormalizedLandmarkList>> input_landmarks{
        "NORM_LANDMARKS"};

    // Represents a normalized rectangle in image coordinates and results in
    // landmarks with their locations adjusted to the image.
    //
    // NOTE: either NORM_RECT or PROJECTION_MATRIX has to be specified.
    Optional<Input<S, NormalizedRect>> norm_rect{"NORM_RECT"};

    // The dimensions of the original image. Original image dimensions are
    // needed to properly scale the landmarks in the general, non-square
    // NORM_RECT case. It can be unset if NORM_RECT is a square, and is allowed
    // for backwards compatibility.
    //
    // NOTE: only works when NORM_RECT is used.
    Optional<Input<S, std::pair<int, int>>> image_dimensions{
        "IMAGE_DIMENSIONS"};

    // A 4x4 row-major-order matrix that maps landmarks' locations from one
    // coordinate system to another. In this case from the coordinate system of
    // the normalized region of interest to the coordinate system of the image.
    //
    // NOTE: either NORM_RECT or PROJECTION_MATRIX has to be specified.
    Optional<Input<S, std::array<float, 16>>> projection_matrix{
        "PROJECTION_MATRIX"};

    // Landmarks with their locations adjusted according to the inputs.
    Repeated<Output<S, mediapipe::NormalizedLandmarkList>> output_landmarks{
        "NORM_LANDMARKS"};

    // Node options.
    Options<S, mediapipe::LandmarkProjectionCalculatorOptions> options;

    // Extra validation for optionals and multi inputs.
    static absl::Status UpdateContract(
        CalculatorContract<LandmarkProjectionNode>& cc) {
      RET_CHECK_GT(cc.input_landmarks.Count(), 0)
          << "Missing input landmarks input.";

      RET_CHECK_EQ(cc.input_landmarks.Count(), cc.output_landmarks.Count())
          << "Same number of input and output landmarks is required.";

      RET_CHECK(cc.norm_rect.IsConnected() ^ cc.projection_matrix.IsConnected())
          << "Either NORM_RECT or PROJECTION_MATRIX must be specified.";
      if (cc.image_dimensions.IsConnected()) {
        RET_CHECK(cc.norm_rect.IsConnected())
            << "IMAGE_DIMENSIONS can only be specified with NORM_RECT";
      }
      return absl::OkStatus();
    }
  };
};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_CALCULATORS_UTIL_LANDMARK_PROJECTION_CALCULATOR_H_
