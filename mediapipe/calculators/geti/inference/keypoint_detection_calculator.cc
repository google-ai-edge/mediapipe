/**
 *  INTEL CONFIDENTIAL
 *
 *  Copyright (C) 2024 Intel Corporation
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

#include "keypoint_detection_calculator.h"

#include <memory>
#include <string>
#include <vector>

#include "../inference/utils.h"
#include "../utils/data_structures.h"

namespace mediapipe {

absl::Status KeypointDetectionCalculator::GetContract(CalculatorContract *cc) {
  LOG(INFO) << "KeypointDetectionCalculator::GetContract()";
  cc->Inputs().Tag("IMAGE").Set<cv::Mat>();
#ifdef USE_MODELADAPTER
  cc->InputSidePackets()
      .Tag("INFERENCE_ADAPTER")
      .Set<std::shared_ptr<InferenceAdapter>>();
#else
  cc->InputSidePackets().Tag("MODEL_PATH").Set<std::string>();
#endif

  cc->Outputs().Tag("INFERENCE_RESULT").Set<geti::InferenceResult>().Optional();
  return absl::OkStatus();
}

absl::Status KeypointDetectionCalculator::Open(CalculatorContext *cc) {
  LOG(INFO) << "KeypointDetectionCalculator::Open()";
  cc->SetOffset(TimestampDiff(0));
#ifdef USE_MODELADAPTER
  ia = cc->InputSidePackets()
           .Tag("INFERENCE_ADAPTER")
           .Get<std::shared_ptr<InferenceAdapter>>();
  auto configuration = ia->getModelConfig();
  labels = geti::get_labels_from_configuration(configuration);
  model = KeypointDetectionModel::create_model(ia);
#else
  auto model_path = cc->InputSidePackets().Tag("MODEL_PATH").Get<std::string>();
  model = DetectionModel::create_model(model_path);
#endif

  return absl::OkStatus();
}

absl::Status KeypointDetectionCalculator::GetiProcess(CalculatorContext *cc) {
  LOG(INFO) << "KeypointDetectionCalculator::GetiProcess()";
  if (cc->Inputs().Tag("IMAGE").IsEmpty()) {
    return absl::OkStatus();
  }

  LOG(INFO) << "Starting Keypoint Detection inference";

  // Get image
  const cv::Mat &cvimage = cc->Inputs().Tag("IMAGE").Get<cv::Mat>();

  // Run Inference model
  std::unique_ptr<geti::InferenceResult> result =
      std::make_unique<geti::InferenceResult>();
  std::unique_ptr<KeypointDetectionResult> inference_result;
  inference_result = model->infer(cvimage);
  result->roi = cv::Rect(0, 0, cvimage.cols, cvimage.rows);
  result->poses.clear();

  if (inference_result->poses.size() > 0) {
    geti::DetectedKeypointsWithLabels keypoints;
    for (size_t i = 0; i < inference_result->poses[0].keypoints.size(); ++i) {
      geti::KeypointWithLabel keypoint;
      keypoint.x = inference_result->poses[0].keypoints[i].x;
      keypoint.y = inference_result->poses[0].keypoints[i].y;
      keypoint.score = inference_result->poses[0].scores[i];
      if (i < labels.size()) {
        keypoint.label_id = labels[i].label_id;
        keypoint.label = labels[i].label;
      }
      keypoints.keypoints.push_back(keypoint);
    }
    result->poses.push_back(keypoints);
  }

  LOG(INFO) << "Completed keypoint detection inference";
  cc->Outputs()
      .Tag("INFERENCE_RESULT")
      .Add(result.release(), cc->InputTimestamp());

  return absl::OkStatus();
}

absl::Status KeypointDetectionCalculator::Close(CalculatorContext *cc) {
  LOG(INFO) << "KeypointDetectionCalculator::Close()";
  return absl::OkStatus();
}

REGISTER_CALCULATOR(KeypointDetectionCalculator);

}  // namespace mediapipe
