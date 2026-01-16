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
#ifndef MEDIAPIPE_TASKS_CC_VISION_GESTURE_RECOGNIZER_CALCULATORS_LANDMARKS_TO_MATRIX_CALCULATOR_H_
#define MEDIAPIPE_TASKS_CC_VISION_GESTURE_RECOGNIZER_CALCULATORS_LANDMARKS_TO_MATRIX_CALCULATOR_H_

#include <utility>

#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/vision/gesture_recognizer/calculators/landmarks_to_matrix_calculator.pb.h"

namespace mediapipe {
namespace tasks {

// Convert landmarks into a matrix. The landmarks are normalized
// w.r.t. the image's aspect ratio (if they are NormalizedLandmarksList)
// and optionally w.r.t and "origin" landmark. This pre-processing step
// is required for the some models.
//
// Usage example:
// node {
//   calculator: "LandmarksToMatrixCalculator"
//   input_stream: "LANDMARKS:hand_landmarks"
//   input_stream: "IMAGE_SIZE:image_size"
//   output_stream: "LANDMARKS_MATRIX:landmarks_matrix"
//   options {
//     [mediapipe.LandmarksToMatrixCalculatorOptions.ext] {
//       object_normalization: true
//       object_normalization_origin_offset: 0
//     }
//   }
// }
struct LandmarksToMatrixNode : api3::Node<"LandmarksToMatrixCalculator"> {
  template <typename S>
  struct Contract {
    // Landmarks of one object.
    // Use either LANDMARKS or WORLD_LANDMARKS.
    api3::Optional<api3::Input<S, mediapipe::NormalizedLandmarkList>> landmarks{
        "LANDMARKS"};

    // World 3d landmarks of one object.
    // Use either LANDMARKS or WORLD_LANDMARKS.
    api3::Optional<api3::Input<S, mediapipe::LandmarkList>> world_landmarks{
        "WORLD_LANDMARKS"};

    // Width and Height of the image
    api3::Optional<api3::Input<S, std::pair<int, int>>> image_size{
        "IMAGE_SIZE"};

    // Optional NormalizedRect object whose 'rotation' field is used
    // to rotate the landmarks.
    api3::Optional<api3::Input<S, NormalizedRect>> norm_rect{"NORM_RECT"};

    // LANDMARKS_MATRIX: Matrix for the landmarks.
    api3::Output<S, Matrix> landmarks_matrix{"LANDMARKS_MATRIX"};

    // options in LandmarksToMatrixCalculatorOptions.
    api3::Options<S, mediapipe::LandmarksToMatrixCalculatorOptions> options;
  };
};

}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_GESTURE_RECOGNIZER_CALCULATORS_LANDMARKS_TO_MATRIX_CALCULATOR_H_
