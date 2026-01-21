/* Copyright 2025 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MEDIAPIPE_TASKS_CC_VISION_GESTURE_RECOGNIZER_CALCULATORS_HANDEDNESS_TO_MATRIX_CALCULATOR_H_
#define MEDIAPIPE_TASKS_CC_VISION_GESTURE_RECOGNIZER_CALCULATORS_HANDEDNESS_TO_MATRIX_CALCULATOR_H_

#include "absl/strings/string_view.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/matrix.h"

namespace mediapipe {
namespace tasks {

inline constexpr absl::string_view kHandednessToMatrixNodeName =
    "HandednessToMatrixCalculator";

// Convert single hand handedness into a matrix.
//
// Input:
//   HANDEDNESS - Single hand handedness.
// Output:
//   HANDEDNESS_MATRIX - Matrix for handedness.
//
// Usage example:
// node {
//   calculator: "HandednessToMatrixCalculator"
//   input_stream: "HANDEDNESS:handedness"
//   output_stream: "HANDEDNESS_MATRIX:handedness_matrix"
// }
struct HandednessToMatrixNode : api3::Node<kHandednessToMatrixNodeName> {
  template <typename S>
  struct Contract {
    // Input stream containing the handedness classification for a single hand.
    api3::Input<S, mediapipe::ClassificationList> in_handedness{"HANDEDNESS"};

    // Output stream containing the handedness classification for a single hand
    // in matrix format.
    api3::Output<S, Matrix> out_handedness_matrix{"HANDEDNESS_MATRIX"};
  };
};

}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_GESTURE_RECOGNIZER_CALCULATORS_HANDEDNESS_TO_MATRIX_CALCULATOR_H_
