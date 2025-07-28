#ifndef MEDIAPIPE_CALCULATORS_UTIL_VISIBILITY_COPY_CALCULATOR_H_
#define MEDIAPIPE_CALCULATORS_UTIL_VISIBILITY_COPY_CALCULATOR_H_

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

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/util/visibility_copy_calculator.pb.h"
#include "mediapipe/framework/api3/calculator_contract.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe::api3 {

inline constexpr absl::string_view kVisibilityCopyNodeName =
    "VisibilityCopyCalculator";

// A calculator to copy visibility and presence between landmarks.
//
// Landmarks to copy from and to copy to can be of different type (normalized or
// non-normalized), but landmarks to copy to and output landmarks should be of
// the same type. Exactly one stream to copy landmarks from, to copy to and to
// output should be provided.
//
// Example config:
//   node {
//     calculator: "VisibilityCopyCalculator"
//     input_stream: "NORM_LANDMARKS_FROM:pose_landmarks"
//     input_stream: "LANDMARKS_TO:pose_world_landmarks"
//     output_stream: "LANDMARKS_TO:pose_world_landmarks_with_visibility"
//     options: {
//       [mediapipe.VisibilityCopyCalculatorOptions.ext] {
//         copy_visibility: true
//         copy_presence: true
//       }
//     }
//   }
//
// WARNING: the fact that every input/output is optional is an unfortunate
// design choice at the time of calculator creation - this should habe been
// distinct calculators.
struct VisibilityCopyNode : Node<kVisibilityCopyNodeName> {
  template <typename S>
  struct Contract {
    // A LandmarkList of landmarks to copy from.
    Optional<Input<S, mediapipe::LandmarkList>> in_landmarks_from{
        "LANDMARKS_FROM"};
    // A NormalizedLandmarkList of landmarks to copy from.
    Optional<Input<S, mediapipe::NormalizedLandmarkList>>
        in_norm_landmarks_from{"NORM_LANDMARKS_FROM"};

    // A LandmarkList of landmarks to copy to.
    Optional<Input<S, mediapipe::LandmarkList>> in_landmarks_to{"LANDMARKS_TO"};
    // An output landmarks.
    Optional<Output<S, mediapipe::LandmarkList>> out_landmarks_to{
        "LANDMARKS_TO"};

    // A NormalizedLandmarkList of landmarks to copy to.
    Optional<Input<S, mediapipe::NormalizedLandmarkList>> in_norm_landmarks_to{
        "NORM_LANDMARKS_TO"};
    // An output NormalizedLandmarkList.
    Optional<Output<S, mediapipe::NormalizedLandmarkList>>
        out_norm_landmarks_to{"NORM_LANDMARKS_TO"};

    Options<S, mediapipe::VisibilityCopyCalculatorOptions> options;
  };

  // Validates node is configured properly.
  static absl::Status UpdateContract(
      CalculatorContract<VisibilityCopyNode>& cc) {
    RET_CHECK(cc.in_landmarks_from.IsConnected() ^
              cc.in_norm_landmarks_from.IsConnected())
        << "Exatly one landmarks stream to copy from should be provided";

    RET_CHECK(cc.in_landmarks_to.IsConnected() ^
              cc.in_norm_landmarks_to.IsConnected())
        << "Exatly one landmarks stream to copy to should be provided";

    RET_CHECK(cc.out_landmarks_to.IsConnected() ^
              cc.out_norm_landmarks_to.IsConnected())
        << "Exatly one output landmarks stream is required.";

    RET_CHECK(cc.in_landmarks_to.IsConnected() ==
              cc.out_landmarks_to.IsConnected())
        << "Landmarks to copy to and output landmarks stream types should be "
           "the same";
    return absl::OkStatus();
  }
};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_CALCULATORS_UTIL_VISIBILITY_COPY_CALCULATOR_H_
