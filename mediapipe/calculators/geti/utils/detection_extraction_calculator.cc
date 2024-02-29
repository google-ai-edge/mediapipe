/**
 *  INTEL CONFIDENTIAL
 *
 *  Copyright (C) 2023 Intel Corporation
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

#include "detection_extraction_calculator.h"

#include <memory>
#include <string>
#include <vector>

#include "../inference/utils.h"

namespace mediapipe {

absl::Status DetectionExtractionCalculator::GetContract(
    CalculatorContract *cc) {
  LOG(INFO) << "DetectionExtractionCalculator::GetContract()";
  cc->Inputs().Tag("DETECTIONS").Set<geti::InferenceResult>().Optional();
  cc->Inputs().Tag("INFERENCE_RESULT").Set<geti::InferenceResult>().Optional();
  cc->Outputs()
      .Tag("RECTANGLE_PREDICTION")
      .Set<std::vector<geti::RectanglePrediction>>()
      .Optional();

  cc->Outputs()
      .Tag("DETECTED_OBJECTS")
      .Set<std::vector<geti::RectanglePrediction>>()
      .Optional();
  return absl::OkStatus();
}

absl::Status DetectionExtractionCalculator::Open(CalculatorContext *cc) {
  LOG(INFO) << "DetectionExtractionCalculator::Open()";
  return absl::OkStatus();
}

absl::Status DetectionExtractionCalculator::GetiProcess(CalculatorContext *cc) {
  LOG(INFO) << "DetectionExtractionCalculator::GetiProcess()";

  std::string input_tag =
      geti::get_input_tag("INFERENCE_RESULT", {"DETECTIONS"}, cc);

  const auto &result = cc->Inputs().Tag(input_tag).Get<geti::InferenceResult>();
  auto detections = std::make_unique<std::vector<geti::RectanglePrediction>>(
      result.rectangles);

  std::string tag =
      geti::get_output_tag("RECTANGLE_PREDICTION", {"DETECTED_OBJECTS"}, cc);
  cc->Outputs().Tag(tag).Add(detections.release(), cc->InputTimestamp());
  return absl::OkStatus();
}

absl::Status DetectionExtractionCalculator::Close(CalculatorContext *cc) {
  LOG(INFO) << "DetectionExtractionCalculator::Close()";
  return absl::OkStatus();
}

REGISTER_CALCULATOR(DetectionExtractionCalculator);

}  // namespace mediapipe
