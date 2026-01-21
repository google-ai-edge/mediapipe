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

#ifndef MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_CALCULATORS_HAND_ASSOCIATION_CALCULATOR_H_
#define MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_CALCULATORS_HAND_ASSOCIATION_CALCULATOR_H_

#include <vector>

#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/calculators/hand_association_calculator.pb.h"

namespace mediapipe {
namespace tasks {

// This calculator checks for overlap among rects from input streams tagged
// with "RECTS". Rects are prioritized based on their index in the vector and
// input streams to the calculator. When two rects overlap, the rect that
// comes from an input stream with lower tag-index is kept in the output.
//
// Input:
//  BASE_RECTS - Vector of NormalizedRect.
//  RECTS - Vector of NormalizedRect.
//
// Output:
//  No tag - Vector of NormalizedRect.
//
// Example use:
// node {
//   calculator: "HandAssociationCalculator"
//   input_stream: "BASE_RECTS:base_rects"
//   input_stream: "RECTS:0:rects0"
//   input_stream: "RECTS:1:rects1"
//   input_stream: "RECTS:2:rects2"
//   output_stream: "output_rects"
//   options {
//     [mediapipe.HandAssociationCalculatorOptions.ext] {
//       min_similarity_threshold: 0.1
//   }
// }
//
// IMPORTANT Notes:
//  - Rects from input streams tagged with "BASE_RECTS" are always preserved.
//  - Example of inputs for the node above:
//      "base_rects": rect 0, rect 1
//      "rects0": rect 2, rect 3
//      "rects1": rect 4, rect 5
//      "rects2": rect 6, rect 7
//    (Conceptually) flattened list: 0, 1, 2, 3, 4, 5, 6, 7.
//    Rects 0, 1 will be preserved. Rects 2, 3, 4, 5, 6, 7 will be checked for
//    overlap. If a rect with a higher index overlaps with a rect with lower
//    index, beyond a specified IOU threshold, the rect with the lower index
//    will be in the output, and the rect with higher index will be discarded.
struct HandAssociationNode : public api3::Node<"HandAssociationCalculator"> {
  template <typename S>
  struct Contract {
    // Repeated input streams of NormalizedRect.
    // Rects from input streams tagged with "BASE_RECTS" are always preserved.
    api3::Repeated<api3::Input<S, std::vector<NormalizedRect>>> base_rects{
        "BASE_RECTS"};

    // Repeated input streams of NormalizedRect.
    // Rects from input streams tagged with "RECTS" are checked for overlap.
    api3::Repeated<api3::Input<S, std::vector<NormalizedRect>>> rects{"RECTS"};

    // Output stream of vector of NormalizedRect.
    api3::Output<S, std::vector<NormalizedRect>> output_rects{""};

    // HandAssociationCalculator options.
    api3::Options<S, mediapipe::HandAssociationCalculatorOptions> options;
  };
};

}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_CALCULATORS_HAND_ASSOCIATION_CALCULATOR_H_
