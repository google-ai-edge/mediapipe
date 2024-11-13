/**
 *  INTEL CONFIDENTIAL
 *
 *  Copyright (C) 2023-2024 Intel Corporation
 *
 *  This software and the related documents are Intel copyrighted materials, and
 * your use of them is governed by the express license under which they were
 * provided to you ("License"). Unless the License provides otherwise, you may
 * not use, modify, copy, publish, distribute, disclose or transmit this
 * software or the related documents without Intel's prior written permission.
 *
 *  This software and the related documents are provided as is, with no express
 * or implied warranties, other than those that are expressly stated in the
 * License.
 */
#include "detection_classification_result_calculator.h"

#include <memory>

#include "../utils/data_structures.h"
namespace mediapipe {
absl::Status DetectionClassificationResultCalculator::GetContract(
    CalculatorContract *cc) {
  cc->Inputs().Tag("DETECTION").Set<geti::InferenceResult>();
  cc->Inputs()
      .Tag("DETECTION_CLASSIFICATIONS")
      .Set<std::vector<geti::RectanglePrediction>>();
  cc->Outputs()
      .Tag("DETECTION_CLASSIFICATION_RESULT")
      .Set<geti::InferenceResult>();

  return absl::OkStatus();
}
absl::Status DetectionClassificationResultCalculator::Open(
    CalculatorContext *cc) {
  return absl::OkStatus();
}
absl::Status DetectionClassificationResultCalculator::GetiProcess(
    CalculatorContext *cc) {
  const auto &detection =
      cc->Inputs().Tag("DETECTION").Get<geti::InferenceResult>();

  if (!cc->Inputs().Tag("DETECTION_CLASSIFICATIONS").IsEmpty()) {
    const auto &classifications =
        cc->Inputs()
            .Tag("DETECTION_CLASSIFICATIONS")
            .Get<std::vector<geti::RectanglePrediction>>();

    if (classifications.size() > 0) {
      std::unique_ptr<geti::InferenceResult> result =
          std::make_unique<geti::InferenceResult>();

      result->rectangles = classifications;
      result->saliency_maps = detection.saliency_maps;
      result->roi = detection.roi;

      cc->Outputs()
          .Tag("DETECTION_CLASSIFICATION_RESULT")
          .Add(result.release(), cc->InputTimestamp());

      return absl::OkStatus();
    }
  }

  std::unique_ptr<geti::InferenceResult> result(
      new geti::InferenceResult(detection));

  cc->Outputs()
      .Tag("DETECTION_CLASSIFICATION_RESULT")
      .Add(result.release(), cc->InputTimestamp());

  return absl::OkStatus();
}
absl::Status DetectionClassificationResultCalculator::Close(
    CalculatorContext *cc) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(DetectionClassificationResultCalculator);

}  // namespace mediapipe
