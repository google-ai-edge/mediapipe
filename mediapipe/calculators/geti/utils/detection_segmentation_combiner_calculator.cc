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
#include "detection_segmentation_combiner_calculator.h"

#include "../utils/data_structures.h"

namespace mediapipe {
absl::Status DetectionSegmentationCombinerCalculator::GetContract(
    CalculatorContract *cc) {
  cc->Inputs().Tag("DETECTION").Set<geti::RectanglePrediction>();
  cc->Inputs().Tag("SEGMENTATION").Set<geti::InferenceResult>();
  cc->Outputs()
      .Tag("DETECTION_SEGMENTATIONS")
      .Set<std::vector<geti::PolygonPrediction>>();

  return absl::OkStatus();
}
absl::Status DetectionSegmentationCombinerCalculator::Open(
    CalculatorContext *cc) {
  return absl::OkStatus();
}
absl::Status DetectionSegmentationCombinerCalculator::GetiProcess(
    CalculatorContext *cc) {
  const auto &detection =
      cc->Inputs().Tag("DETECTION").Get<geti::RectanglePrediction>();
  const auto &segmentation =
      cc->Inputs().Tag("SEGMENTATION").Get<geti::InferenceResult>();

  auto polygons = segmentation.polygons;

  for (auto &contour : polygons) {
    for (auto &point : contour.shape) {
      point.x += detection.shape.x;
      point.y += detection.shape.y;
    }
  }

  cc->Outputs()
      .Tag("DETECTION_SEGMENTATIONS")
      .AddPacket(MakePacket<std::vector<geti::PolygonPrediction>>(polygons).At(
          cc->InputTimestamp()));

  return absl::OkStatus();
}
absl::Status DetectionSegmentationCombinerCalculator::Close(
    CalculatorContext *cc) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(DetectionSegmentationCombinerCalculator);

}  // namespace mediapipe
