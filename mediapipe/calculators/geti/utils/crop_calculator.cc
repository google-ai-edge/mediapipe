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
#include "crop_calculator.h"

#include <memory>

namespace mediapipe {

absl::Status CropCalculator::GetContract(CalculatorContract *cc) {
  LOG(INFO) << "CropCalculator::GetContract()";
  cc->Inputs().Tag("IMAGE").Set<cv::Mat>();
  cc->Inputs().Tag("DETECTION").Set<geti::RectanglePrediction>();
  cc->Outputs().Tag("IMAGE").Set<cv::Mat>();

  return absl::OkStatus();
}

absl::Status CropCalculator::Open(CalculatorContext *cc) {
  LOG(INFO) << "CropCalculator::Open()";
  return absl::OkStatus();
}

absl::Status CropCalculator::GetiProcess(CalculatorContext *cc) {
  LOG(INFO) << "CropCalculator::GetiProcess()";
  const cv::Mat &image = cc->Inputs().Tag("IMAGE").Get<cv::Mat>();
  const auto &detection =
      cc->Inputs().Tag("DETECTION").Get<geti::RectanglePrediction>();
  cv::Mat croppedImage = image(detection.shape).clone();
  cc->Outputs().Tag("IMAGE").AddPacket(
      MakePacket<cv::Mat>(croppedImage).At(cc->InputTimestamp()));
  return absl::OkStatus();
}

absl::Status CropCalculator::Close(CalculatorContext *cc) {
  LOG(INFO) << "CropCalculator::Close()";
  return absl::OkStatus();
}

REGISTER_CALCULATOR(CropCalculator);

}  // namespace mediapipe
