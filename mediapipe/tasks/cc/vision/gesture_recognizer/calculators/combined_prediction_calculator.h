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

#ifndef MEDIAPIPE_TASKS_CC_VISION_GESTURE_RECOGNIZER_CALCULATORS_COMBINED_PREDICTION_CALCULATOR_H_
#define MEDIAPIPE_TASKS_CC_VISION_GESTURE_RECOGNIZER_CALCULATORS_COMBINED_PREDICTION_CALCULATOR_H_

#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/tasks/cc/vision/gesture_recognizer/calculators/combined_prediction_calculator.pb.h"

namespace mediapipe {
namespace tasks {

// This calculator accepts multiple ClassificationList input streams. Each
// ClassificationList should contain classifications with labels and
// corresponding softmax scores. The calculator computes the best prediction for
// each ClassificationList input stream via argmax and thresholding.
// Please note that for this calculator to work as designed, the class names
// other than the background class in the ClassificationList objects must be
// different, but the background class name has to be the same. This background
// label name can be set via `background_label` in
// `CombinedPredictionCalculatorOptions`.
//
// Usage example:
// node {
//   calculator: "CombinedPredictionCalculator"
//   input_stream: "classification_list_0"
//   input_stream: "classification_list_1"
//   output_stream: "PREDICTION:prediction"
//   options {
//     [mediapipe.CombinedPredictionCalculatorOptions.ext] {
//       class {
//         label: "A"
//         score_threshold: 0.7
//       }
//       default_global_threshold: 0.1
//       background_label: "B"
//     }
//   }
// }
struct CombinedPredictionNode : api3::Node<"CombinedPredictionCalculator"> {
  template <typename S>
  struct Contract {
    // Repeated ClassificationList input streams. Each ClassificationList should
    // contain classifications with labels and corresponding softmax scores.
    // NOTE: At least one input is expected.
    api3::Repeated<api3::Input<S, mediapipe::ClassificationList>>
        classification_list_in{""};

    // The ClassificationList in the PREDICTION output stream contains the label
    // of the winning class and corresponding softmax score. If none of the
    // ClassificationList objects has a non-background winning class, the output
    // contains the background class and score of the background class in the
    // first ClassificationList. If multiple ClassificationList objects have a
    // non-background winning class, the output contains the winning prediction
    // from the ClassificationList with the highest priority. Priority is in
    // decreasing order of input streams to the graph node using this
    // calculator.
    api3::Output<S, mediapipe::ClassificationList> prediction_out{"PREDICTION"};

    // Thresholds for all classes can be specified in the
    // `CombinedPredictionCalculatorOptions`, along with a default global
    // threshold.
    api3::Options<S, mediapipe::CombinedPredictionCalculatorOptions> options;
  };
};

}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_GESTURE_RECOGNIZER_CALCULATORS_COMBINED_PREDICTION_CALCULATOR_H_
