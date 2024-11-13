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
#include "detection_segmentation_result_calculator.h"

#include "../utils/data_structures.h"

namespace mediapipe {
absl::Status DetectionSegmentationResultCalculator::GetContract(
    CalculatorContract *cc) {
  cc->Inputs().Tag("DETECTION").Set<geti::InferenceResult>();
  cc->Inputs()
      .Tag("DETECTION_SEGMENTATIONS")
      .Set<std::vector<std::vector<geti::PolygonPrediction>>>();
  cc->Outputs()
      .Tag("DETECTION_SEGMENTATION_RESULT")
      .Set<geti::InferenceResult>();

  return absl::OkStatus();
}
absl::Status DetectionSegmentationResultCalculator::Open(
    CalculatorContext *cc) {
  return absl::OkStatus();
}
absl::Status DetectionSegmentationResultCalculator::GetiProcess(
    CalculatorContext *cc) {
  const auto &detection =
      cc->Inputs().Tag("DETECTION").Get<geti::InferenceResult>();

  auto result = detection;  // copy

  if (!cc->Inputs().Tag("DETECTION_SEGMENTATIONS").IsEmpty()) {
    const auto &segmentations =
        cc->Inputs()
            .Tag("DETECTION_SEGMENTATIONS")
            .Get<std::vector<std::vector<geti::PolygonPrediction>>>();

    for (auto &polygons : segmentations) {
      result.polygons.insert(result.polygons.end(), polygons.begin(),
                             polygons.end());
    }
  }

  cc->Outputs()
      .Tag("DETECTION_SEGMENTATION_RESULT")
      .AddPacket(
          MakePacket<geti::InferenceResult>(result).At(cc->InputTimestamp()));

  return absl::OkStatus();
}
absl::Status DetectionSegmentationResultCalculator::Close(
    CalculatorContext *cc) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(DetectionSegmentationResultCalculator);

}  // namespace mediapipe
