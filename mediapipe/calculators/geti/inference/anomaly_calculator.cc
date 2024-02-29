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

#include "anomaly_calculator.h"

#include <memory>
#include <string>

#include "../inference/utils.h"
#include "models/image_model.h"
#include "../utils/data_structures.h"

namespace mediapipe {

absl::Status AnomalyCalculator::GetContract(CalculatorContract *cc) {
  LOG(INFO) << "AnomalyCalculator::GetContract()";
  cc->Inputs().Tag("IMAGE").Set<cv::Mat>();
#ifdef USE_MODELADAPTER
  cc->InputSidePackets()
      .Tag("INFERENCE_ADAPTER")
      .Set<std::shared_ptr<InferenceAdapter>>();
#else
  cc->InputSidePackets().Tag("MODEL_PATH").Set<std::string>();
#endif
  cc->Outputs().Tag("INFERENCE_RESULT").Set<geti::InferenceResult>().Optional();
  cc->Outputs().Tag("RESULT").Set<geti::InferenceResult>().Optional();
  return absl::OkStatus();
}

absl::Status AnomalyCalculator::Open(CalculatorContext *cc) {
  LOG(INFO) << "AnomalyCalculator::Open()";
  cc->SetOffset(TimestampDiff(0));
#ifdef USE_MODELADAPTER
  ia = cc->InputSidePackets()
           .Tag("INFERENCE_ADAPTER")
           .Get<std::shared_ptr<InferenceAdapter>>();

  auto configuration = ia->getModelConfig();
  auto task_iter = configuration.find("task");
  if (task_iter != configuration.end()) {
    task = task_iter->second.as<std::string>();
  }
  auto labels = geti::get_labels_from_configuration(configuration);
  normal_label = labels[0];
  anomalous_label = labels[1];

  model = AnomalyModel::create_model(ia);
#else
  auto model_path = cc->InputSidePackets().Tag("MODEL_PATH").Get<std::string>();
  model = AnomalyModel::create_model(model_path);
#endif

  return absl::OkStatus();
}

absl::Status AnomalyCalculator::Process(CalculatorContext *cc) {
  LOG(INFO) << "AnomalyCalculator::GetiProcess()";
  if (cc->Inputs().Tag("IMAGE").IsEmpty()) {
    return absl::OkStatus();
  }

  const cv::Mat &cvimage = cc->Inputs().Tag("IMAGE").Get<cv::Mat>();

  auto infer_result = model->infer(cvimage);

  auto result = std::make_unique<geti::InferenceResult>();

  cv::Rect image_roi(0, 0, cvimage.cols, cvimage.rows);
  result->roi = image_roi;

  {  // global classification is added as full image detection
    auto label = infer_result->pred_label == normal_label.label
                     ? normal_label
                     : anomalous_label;
    result->rectangles.push_back(
        {{geti::LabelResult{(float)infer_result->pred_score, label}},
         image_roi});
  }

  if (infer_result->pred_label != normal_label.label) {
    if (task == "detection") {
      for (auto &box : infer_result->pred_boxes) {
        double box_score;
        cv::minMaxLoc(infer_result->anomaly_map(box), NULL, &box_score);

        result->rectangles.push_back(
            {{geti::LabelResult{(float)box_score / 255, anomalous_label}},
             box});
      }
    }
    if (task == "segmentation") {
      cv::Mat mask;
      cv::threshold(infer_result->pred_mask, mask, 0, 255, cv::THRESH_BINARY);
      double box_score;
      std::vector<std::vector<cv::Point>> contours, approxCurve;
      cv::findContours(mask, contours, cv::RETR_EXTERNAL,
                       cv::CHAIN_APPROX_SIMPLE);

      for (size_t i = 0; i < contours.size(); i++) {
        std::vector<cv::Point> approx;
        if (contours[i].size() > 0) {
          cv::approxPolyDP(contours[i], approx, 1.0f, true);
          if (approx.size() > 2) approxCurve.push_back(approx);
        }
      }
      for (size_t i = 0; i < approxCurve.size(); i++) {
        cv::Mat contour_mask =
            cv::Mat::zeros(infer_result->anomaly_map.size(), CV_8UC1);
        cv::drawContours(contour_mask, approxCurve, i, 255, -1);
        cv::minMaxLoc(infer_result->anomaly_map, &box_score, 0, 0, 0,
                      contour_mask);

        result->polygons.push_back(
            {{geti::LabelResult{(float)box_score / 255, anomalous_label}},
             approxCurve[i]});
      }
    }
  }

  result->saliency_maps.push_back(
      {infer_result->anomaly_map, image_roi, anomalous_label});

  std::string tag = geti::get_output_tag("INFERENCE_RESULT", {"RESULT"}, cc);
  cc->Outputs().Tag(tag).Add(result.release(), cc->InputTimestamp());

  return absl::OkStatus();
}

absl::Status AnomalyCalculator::Close(CalculatorContext *cc) {
  LOG(INFO) << "AnomalyCalculator::Close()";
  return absl::OkStatus();
}

REGISTER_CALCULATOR(AnomalyCalculator);

}  // namespace mediapipe
