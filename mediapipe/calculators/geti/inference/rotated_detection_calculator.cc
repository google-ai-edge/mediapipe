/*
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

#include "rotated_detection_calculator.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../inference/utils.h"
#include "../utils/data_structures.h"

namespace mediapipe {

absl::Status RotatedDetectionCalculator::GetContract(CalculatorContract *cc) {
  LOG(INFO) << "RotatedDetectionCalculator::GetContract()";
  cc->Inputs().Tag("IMAGE").Set<cv::Mat>();
#ifdef USE_MODELADAPTER
  cc->InputSidePackets()
      .Tag("INFERENCE_ADAPTER")
      .Set<std::shared_ptr<InferenceAdapter>>();
#else
  cc->InputSidePackets().Tag("MODEL_PATH").Set<std::string>();
#endif
  cc->Outputs().Tag("INFERENCE_RESULT").Set<geti::InferenceResult>().Optional();
  cc->Outputs().Tag("DETECTIONS").Set<geti::InferenceResult>().Optional();
  return absl::OkStatus();
}

absl::Status RotatedDetectionCalculator::Open(CalculatorContext *cc) {
  LOG(INFO) << "RotatedDetectionCalculator::Open()";
  cc->SetOffset(TimestampDiff(0));

#ifdef USE_MODELADAPTER
  ia = cc->InputSidePackets()
           .Tag("INFERENCE_ADAPTER")
           .Get<std::shared_ptr<InferenceAdapter>>();

  auto configuration = ia->getModelConfig();
  labels = geti::get_labels_from_configuration(configuration);

  auto property = configuration.find("tile_size");
  if (property == configuration.end()) {
    model = MaskRCNNModel::create_model(ia);
  } else {
    tiler = std::unique_ptr<InstanceSegmentationTiler>(
        new InstanceSegmentationTiler(
            std::move(MaskRCNNModel::create_model(ia)), {}));
  }
#else
  auto model_path = cc->InputSidePackets().Tag("MODEL_PATH").Get<std::string>();
  model = MaskRCNNModel::create_model(model_path);
#endif

  return absl::OkStatus();
}

absl::Status RotatedDetectionCalculator::GetiProcess(CalculatorContext *cc) {
  LOG(INFO) << "RotatedDetectionCalculator::GetiProcess()";
  if (cc->Inputs().Tag("IMAGE").IsEmpty()) {
    return absl::OkStatus();
  }

  // Get image
  const cv::Mat &cvimage = cc->Inputs().Tag("IMAGE").Get<cv::Mat>();

  std::unique_ptr<InstanceSegmentationResult> inference_result;
  // Run Inference model

  if (tiler) {
    auto tiler_result = tiler->run(cvimage);
    inference_result = std::unique_ptr<InstanceSegmentationResult>(
        static_cast<InstanceSegmentationResult *>(tiler_result.release()));
  } else {
    inference_result = model->infer(cvimage);
  }

  auto rotated_rects = add_rotated_rects(inference_result->segmentedObjects);

  std::unique_ptr<geti::InferenceResult> result =
      std::make_unique<geti::InferenceResult>();

  for (auto &obj : rotated_rects) {
    if (labels.size() > obj.labelID)
      result->rotated_rectangles.push_back(
          {{geti::LabelResult{obj.confidence, labels[obj.labelID]}},
           obj.rotated_rect});
  }

  cv::Rect roi(0, 0, cvimage.cols, cvimage.rows);
  result->roi = roi;
  for (size_t i = 0; i < inference_result->saliency_map.size(); i++) {
    if (labels.size() > i + 1)
      result->saliency_maps.push_back(
          {inference_result->saliency_map[i], roi, labels[i + 1]});
  }

  std::string tag =
      geti::get_output_tag("INFERENCE_RESULT", {"DETECTIONS"}, cc);
  cc->Outputs().Tag(tag).Add(result.release(), cc->InputTimestamp());

  return absl::OkStatus();
}

absl::Status RotatedDetectionCalculator::Close(CalculatorContext *cc) {
  LOG(INFO) << "RotatedDetectionCalculator::Close()";
  return absl::OkStatus();
}

REGISTER_CALCULATOR(RotatedDetectionCalculator);

}  // namespace mediapipe
