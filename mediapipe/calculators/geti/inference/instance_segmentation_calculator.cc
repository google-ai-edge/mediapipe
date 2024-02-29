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

#include "instance_segmentation_calculator.h"

#include <memory>
#include <opencv2/imgproc.hpp>
#include <string>

#include "utils.h"
#include "models/image_model.h"
#include "mediapipe/calculators/geti/utils/emptylabel.pb.h"

namespace mediapipe {

absl::Status InstanceSegmentationCalculator::GetContract(
    CalculatorContract *cc) {
  LOG(INFO) << "InstanceSegmentationCalculator::GetContract()";
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

absl::Status InstanceSegmentationCalculator::Open(CalculatorContext *cc) {
  LOG(INFO) << "InstanceSegmentationCalculator::Open()";
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

absl::Status InstanceSegmentationCalculator::GetiProcess(
    CalculatorContext *cc) {
  LOG(INFO) << "InstanceSegmentationCalculator::GetiProcess()";
  if (cc->Inputs().Tag("IMAGE").IsEmpty()) {
    return absl::OkStatus();
  }

  const cv::Mat &cvimage = cc->Inputs().Tag("IMAGE").Get<cv::Mat>();

  std::unique_ptr<InstanceSegmentationResult> inference_result;

  if (tiler) {
    auto tiler_result = tiler->run(cvimage);
    inference_result = std::unique_ptr<InstanceSegmentationResult>(
        static_cast<InstanceSegmentationResult *>(tiler_result.release()));
  } else {
    inference_result = model->infer(cvimage);
  }

  // Build contours ourselves since model api does not handle multiple contours
  // from one segmented object. Model API resolves the issue by throwing an
  // exception Our solution returns the biggest area contour.
  const auto &options = cc->Options<EmptyLabelOptions>();
  std::string label_name =
      options.label().empty() ? geti::GETI_EMPTY_LABEL : options.label();
  std::unique_ptr<geti::InferenceResult> result =
      std::make_unique<geti::InferenceResult>();
  bool isEmpty = false;
  cv::Rect roi(0, 0, cvimage.cols, cvimage.rows);
  result->roi = roi;

  for (auto &obj : inference_result->segmentedObjects) {
    if (labels.size() > obj.labelID) {
      if (labels[obj.labelID].label == label_name) {
        if (!isEmpty) {
          result->rectangles.push_back(
              {{geti::LabelResult{obj.confidence, labels[obj.labelID]}}, roi});
          isEmpty = true;
        }
      } else {
        auto mask = obj.mask.clone();
        std::vector<std::vector<cv::Point>> contours;
        cv::threshold(mask, mask, 1, 999, cv::THRESH_OTSU);
        cv::findContours(mask, contours, cv::RETR_EXTERNAL,
                         cv::CHAIN_APPROX_NONE);

        if (contours.size() == 0) {
          LOG(INFO) << "findContours() returned no contours";
        } else {
          double biggest_area = 0.0;
          std::vector<cv::Point> biggest_contour, approxCurve;
          for (auto contour : contours) {
            double area = cv::contourArea(contour);
            if (biggest_area < area) {
              biggest_area = area;
              biggest_contour = contour;
            }
          }

          if (biggest_contour.size() > 0) {
            cv::approxPolyDP(biggest_contour, approxCurve, 1.0f, true);
            if (approxCurve.size() > 2)
              result->polygons.push_back(
                  {{geti::LabelResult{obj.confidence, labels[obj.labelID]}},
                   approxCurve});
          }
        }
      }
    }
  }

  for (size_t i = 0; i < inference_result->saliency_map.size(); i++) {
    if (labels.size() > i + 1) {
      result->saliency_maps.push_back(
          {inference_result->saliency_map[i], roi, labels[i + 1]});
    }
  }

  std::string tag = geti::get_output_tag("INFERENCE_RESULT", {"RESULT"}, cc);
  cc->Outputs().Tag(tag).Add(result.release(), cc->InputTimestamp());
  return absl::OkStatus();
}

absl::Status InstanceSegmentationCalculator::Close(CalculatorContext *cc) {
  LOG(INFO) << "InstanceSegmentationCalculator::Close()";
  return absl::OkStatus();
}

REGISTER_CALCULATOR(InstanceSegmentationCalculator);

}  // namespace mediapipe
