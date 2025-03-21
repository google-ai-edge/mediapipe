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

#include "detection_calculator.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../inference/utils.h"
#include "../utils/data_structures.h"
#include "mediapipe/calculators/geti/utils/emptylabel.pb.h"

namespace mediapipe {

absl::Status DetectionCalculator::GetContract(CalculatorContract *cc) {
  LOG(INFO) << "DetectionCalculator::GetContract()";
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

absl::Status DetectionCalculator::Open(CalculatorContext *cc) {
  LOG(INFO) << "DetectionCalculator::Open()";
  cc->SetOffset(TimestampDiff(0));
#ifdef USE_MODELADAPTER
  ia = cc->InputSidePackets()
           .Tag("INFERENCE_ADAPTER")
           .Get<std::shared_ptr<InferenceAdapter>>();
  auto configuration = ia->getModelConfig();
  labels = geti::get_labels_from_configuration(configuration);

  auto tile_size_iter = configuration.find("tile_size");
  if (tile_size_iter == configuration.end()) {
    model = DetectionModel::create_model(ia);
  } else {
    tiler = std::unique_ptr<DetectionTiler>(
        new DetectionTiler(std::move(DetectionModel::create_model(ia)), {}));
  }
#else
  auto model_path = cc->InputSidePackets().Tag("MODEL_PATH").Get<std::string>();
  model = DetectionModel::create_model(model_path);
#endif

  return absl::OkStatus();
}

absl::Status DetectionCalculator::GetiProcess(CalculatorContext *cc) {
  LOG(INFO) << "DetectionCalculator::GetiProcess()";
  if (cc->Inputs().Tag("IMAGE").IsEmpty()) {
    return absl::OkStatus();
  }

  LOG(INFO) << "starting detection inference";

  // Get image
  const cv::Mat &cvimage = cc->Inputs().Tag("IMAGE").Get<cv::Mat>();

  // Run Inference model
  std::unique_ptr<geti::InferenceResult> result =
      std::make_unique<geti::InferenceResult>();
  std::unique_ptr<DetectionResult> inference_result;
  if (tiler) {
    inference_result = std::unique_ptr<DetectionResult>(
        static_cast<DetectionResult *>(tiler->run(cvimage).release()));
  } else {
    inference_result = model->infer(cvimage);
  }

  const auto &options = cc->Options<EmptyLabelOptions>();
  std::string no_object_name =
      options.label().empty() ? geti::GETI_NOOBJECT_LABEL : options.label();

  for (auto &obj : inference_result->objects) {
    if (labels.size() > obj.labelID) {
      if (labels[obj.labelID].label != no_object_name) {
        result->rectangles.push_back(
            {{geti::LabelResult{obj.confidence, labels[obj.labelID]}}, obj});
      }
    }
  }

  result->roi = cv::Rect(0, 0, cvimage.cols, cvimage.rows);

  if ((inference_result->saliency_map) &&
      (inference_result->saliency_map.get_shape().size() > 1)) {
    size_t shape_shift =
        (inference_result->saliency_map.get_shape().size() > 3) ? 1 : 0;

    for (size_t i = 0; i < labels.size(); i++) {
      result->saliency_maps.push_back(
          {geti::get_mat_from_ov_tensor(inference_result->saliency_map,
                                        shape_shift, i),
           result->roi, labels[i]});
    }
  }
  LOG(INFO) << "completed detection inference";

  std::string tag =
      geti::get_output_tag("INFERENCE_RESULT", {"DETECTIONS"}, cc);
  cc->Outputs().Tag(tag).Add(result.release(), cc->InputTimestamp());

  return absl::OkStatus();
}

absl::Status DetectionCalculator::Close(CalculatorContext *cc) {
  LOG(INFO) << "DetectionCalculator::Close()";
  return absl::OkStatus();
}

REGISTER_CALCULATOR(DetectionCalculator);

}  // namespace mediapipe
