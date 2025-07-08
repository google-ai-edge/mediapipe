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

#ifndef MEDIAPIPE_CALCULATORS_UTIL_LANDMARKS_REFINEMENT_CALCULATOR_H_
#define MEDIAPIPE_CALCULATORS_UTIL_LANDMARKS_REFINEMENT_CALCULATOR_H_

#include "absl/strings/string_view.h"
#include "mediapipe/calculators/util/landmarks_refinement_calculator.pb.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/formats/landmark.pb.h"

namespace mediapipe::api3 {

inline constexpr absl::string_view kLandmarksRefinementNodeName =
    "LandmarksRefinementCalculator";

// A calculator to refine one set of landmarks with another.
//
// Example config:
//   node {
//     calculator: "LandmarksRefinementCalculator"
//     input_stream: "LANDMARKS:0:mesh_landmarks"
//     input_stream: "LANDMARKS:1:lips_landmarks"
//     input_stream: "LANDMARKS:2:left_eye_landmarks"
//     input_stream: "LANDMARKS:3:right_eye_landmarks"
//     output_stream: "REFINED_LANDMARKS:landmarks"
//     options: {
//       [mediapipe.LandmarksRefinementCalculatorOptions.ext] {
//         refinement: {
//           indexes_mapping: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
//           z_refinement: { copy {} }
//         }
//         refinement: {
//           indexes_mapping: [0, 1, 2, 3]
//           z_refinement: { none {} }
//         }
//         refinement: {
//           indexes_mapping: [4, 5]
//           z_refinement: { none {} }
//         }
//         refinement: {
//           indexes_mapping: [6, 7]
//           z_refinement: { none {} }
//         }
//       }
//     }
//   }
//
struct LandmarksRefinementNode : Node<kLandmarksRefinementNodeName> {
  template <typename S>
  struct Contract {
    // Multiple NormalizedLandmarkList to use for
    // refinement. They will be applied to the resulting REFINED_LANDMARKS
    // in the provided order. Each list should be non empty and contain the
    // same amount of landmarks as indexes in mapping. Number of lists
    // should be the same as number of refinements in options.
    Repeated<Input<S, mediapipe::NormalizedLandmarkList>> landmarks{
        "LANDMARKS"};

    // A NormalizedLandmarkList with refined landmarks. Number
    // of produced landmarks is equal to to the maximum index mapping number in
    // calculator options (calculator verifies that there are no gaps in the
    // mapping).
    Output<S, mediapipe::NormalizedLandmarkList> refined_landmarks{
        "REFINED_LANDMARKS"};

    Options<S, mediapipe::LandmarksRefinementCalculatorOptions> options;
  };
};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_CALCULATORS_UTIL_LANDMARKS_REFINEMENT_CALCULATOR_H_
