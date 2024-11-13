//
//  INTEL CONFIDENTIAL
//
//  Copyright (C) 2023 Intel Corporation
//
//  This software and the related documents are Intel copyrighted materials, and
// your use of them is governed by the express license under which they were
// provided to you ("License"). Unless the License provides otherwise, you may
// not use, modify, copy, publish, distribute, disclose or transmit this
// software or the related documents without Intel's prior written permission.
//
//  This software and the related documents are provided as is, with no express
// or implied warranties, other than those that are expressly stated in the
// License.
//

#include "detection_classification_combiner_calculator.h"

#include <memory>
#include <string>

#include "../inference/utils.h"
#include "../utils/data_structures.h"

namespace mediapipe {
absl::Status DetectionClassificationCombinerCalculator::GetContract(
    CalculatorContract *cc) {
  cc->Inputs().Tag("DETECTION").Set<geti::RectanglePrediction>();
  cc->Inputs().Tag("INFERENCE_RESULT").Set<geti::InferenceResult>().Optional();
  cc->Inputs().Tag("CLASSIFICATION").Set<geti::InferenceResult>().Optional();
  cc->Outputs()
      .Tag("DETECTION_CLASSIFICATIONS")
      .Set<geti::RectanglePrediction>();

  return absl::OkStatus();
}
absl::Status DetectionClassificationCombinerCalculator::Open(
    CalculatorContext *cc) {
  return absl::OkStatus();
}
absl::Status DetectionClassificationCombinerCalculator::GetiProcess(
    CalculatorContext *cc) {
  if (cc->Inputs().Tag("DETECTION").IsEmpty()) {
    return absl::OkStatus();
  }

  std::string input_tag =
      geti::get_input_tag("INFERENCE_RESULT", {"CLASSIFICATION"}, cc);

  const auto &detection_input =
      cc->Inputs().Tag("DETECTION").Get<geti::RectanglePrediction>();
  const auto &classifications =
      cc->Inputs().Tag(input_tag).Get<geti::InferenceResult>();

  std::unique_ptr<geti::RectanglePrediction> result(
      new geti::RectanglePrediction(detection_input));
  if (classifications.rectangles.size() > 0) {
    result->labels.insert(result->labels.end(),
                          classifications.rectangles[0].labels.begin(),
                          classifications.rectangles[0].labels.end());
  }

  cc->Outputs()
      .Tag("DETECTION_CLASSIFICATIONS")
      .Add(result.release(), cc->InputTimestamp());

  return absl::OkStatus();
}
absl::Status DetectionClassificationCombinerCalculator::Close(
    CalculatorContext *cc) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(DetectionClassificationCombinerCalculator);

}  // namespace mediapipe
