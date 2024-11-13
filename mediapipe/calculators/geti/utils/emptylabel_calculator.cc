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

#include "emptylabel_calculator.h"

#include <memory>
#include <string>

#include "../utils/data_structures.h"

namespace mediapipe {

absl::Status EmptyLabelCalculator::GetContract(CalculatorContract *cc) {
  LOG(INFO) << "EmptyLabelCalculator::GetContract()";
  cc->Inputs().Tag("PREDICTION").Set<geti::InferenceResult>();
  cc->Outputs().Tag("PREDICTION").Set<geti::InferenceResult>();

  return absl::OkStatus();
}

absl::Status EmptyLabelCalculator::Open(CalculatorContext *cc) {
  LOG(INFO) << "EmptyLabelCalculator::Open()";
  return absl::OkStatus();
}

absl::Status EmptyLabelCalculator::GetiProcess(CalculatorContext *cc) {
  LOG(INFO) << "EmptyLabelCalculator::GetiProcess()";
  auto prediction = cc->Inputs().Tag("PREDICTION").Get<geti::InferenceResult>();
  size_t n_predictions =
      prediction.polygons.size() + prediction.rectangles.size() +
      prediction.circles.size() + prediction.rotated_rectangles.size();
  if (n_predictions == 0) {
    const auto &options = cc->Options<EmptyLabelOptions>();
    auto label = get_label_from_options(options);
    prediction.rectangles.push_back(
        {{geti::LabelResult{0.0f, label}}, prediction.roi});
  }

  cc->Outputs()
      .Tag("PREDICTION")
      .AddPacket(MakePacket<geti::InferenceResult>(prediction)
                     .At(cc->InputTimestamp()));

  return absl::OkStatus();
}

absl::Status EmptyLabelCalculator::Close(CalculatorContext *cc) {
  LOG(INFO) << "EmptyLabelCalculator::Close()";
  return absl::OkStatus();
}

geti::Label EmptyLabelCalculator::get_label_from_options(
    const mediapipe::EmptyLabelOptions &options) {
  std::string label_name = options.label().empty() ? "empty" : options.label();
  return {options.id(), label_name};
}

REGISTER_CALCULATOR(EmptyLabelCalculator);
REGISTER_CALCULATOR(EmptyLabelDetectionCalculator);
REGISTER_CALCULATOR(EmptyLabelClassificationCalculator);
REGISTER_CALCULATOR(EmptyLabelRotatedDetectionCalculator);
REGISTER_CALCULATOR(EmptyLabelSegmentationCalculator);

}  // namespace mediapipe
