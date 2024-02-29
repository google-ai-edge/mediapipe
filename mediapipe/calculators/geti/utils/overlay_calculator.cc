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
#include "overlay_calculator.h"

namespace mediapipe {

absl::Status OverlayCalculator::GetContract(CalculatorContract *cc) {
  LOG(INFO) << "DetectionOverlayCalculator::GetContract()";
  cc->Inputs().Tag("IMAGE").Set<cv::Mat>();
  cc->Inputs().Tag("INFERENCE_RESULT").Set<geti::InferenceResult>();
  cc->Outputs().Tag("IMAGE").Set<cv::Mat>();

  return absl::OkStatus();
}

absl::Status OverlayCalculator::Open(CalculatorContext *cc) {
  LOG(INFO) << "DetectionOverlayCalculator::Open()";
  cc->SetOffset(TimestampDiff(0));
  return absl::OkStatus();
}

absl::Status OverlayCalculator::GetiProcess(CalculatorContext *cc) {
  LOG(INFO) << "DetectionOverlayCalculator::GetiProcess()";
  if (cc->Inputs().Tag("IMAGE").IsEmpty()) {
    return absl::OkStatus();
  }

  // Get inputs
  const cv::Mat &cvimage = cc->Inputs().Tag("IMAGE").Get<cv::Mat>();
  const auto result =
      cc->Inputs().Tag("INFERENCE_RESULT").Get<geti::InferenceResult>();

  // Apply results
  cv::Mat output_img = cvimage.clone();
  auto color = cv::Scalar(255, 0, 0);
  for (auto &detection : result.rectangles) {
    auto position = cv::Point2f(detection.shape.x, detection.shape.y + 20);
    std::ostringstream predictions;
    for (auto &label : detection.labels) {
      predictions << ":" << label.label.label << "("
                  << std::to_string(label.probability) << ")";
    }
    cv::rectangle(output_img, detection.shape, color, 2);
    cv::putText(output_img, predictions.str(), position,
                cv::FONT_HERSHEY_COMPLEX_SMALL, 1., cv::Scalar(255, 255, 255),
                3);
    cv::putText(output_img, predictions.str(), position,
                cv::FONT_HERSHEY_COMPLEX_SMALL, 1., color, 2);
  }

  for (auto &obj : result.polygons) {
    auto br = cv::boundingRect(obj.shape);
    auto position =
        cv::Point2f(br.x + br.width / 2.0f, br.y + br.height / 2.0f);
    std::ostringstream conf;
    conf << ":" << std::fixed << std::setprecision(1)
         << obj.labels[0].probability * 100 << '%';
    std::vector<std::vector<cv::Point>> contours = {obj.shape};
    cv::drawContours(output_img, contours, 0, 255);
    cv::putText(output_img, obj.labels[0].label.label + conf.str(), position,
                cv::FONT_HERSHEY_COMPLEX_SMALL, 1., cv::Scalar(255, 255, 255),
                3);
    cv::putText(output_img, obj.labels[0].label.label + conf.str(), position,
                cv::FONT_HERSHEY_COMPLEX_SMALL, 1., color, 2);
  }

  for (auto &obj : result.rotated_rectangles) {
    auto position = obj.shape.center;
    std::ostringstream conf;
    cv::Point2f vertices[4];
    obj.shape.points(vertices);

    for (int i = 0; i < 4; i++) {
      cv::line(output_img, vertices[i], vertices[(i + 1) % 4], color);
    }

    conf << ":" << std::fixed << std::setprecision(1)
         << obj.labels[0].probability * 100 << '%';

    cv::putText(output_img, obj.labels[0].label.label + conf.str(), position,
                cv::FONT_HERSHEY_COMPLEX_SMALL, 1., cv::Scalar(255, 255, 255),
                3);
    cv::putText(output_img, obj.labels[0].label.label + conf.str(), position,
                cv::FONT_HERSHEY_COMPLEX_SMALL, 1., color, 2);
  }

  cc->Outputs().Tag("IMAGE").AddPacket(
      MakePacket<cv::Mat>(output_img).At(cc->InputTimestamp()));

  return absl::OkStatus();
}

absl::Status OverlayCalculator::Close(CalculatorContext *cc) {
  LOG(INFO) << "DetectionOverlayCalculator::Close()";
  return absl::OkStatus();
}

REGISTER_CALCULATOR(OverlayCalculator);

}  // namespace mediapipe
