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
#ifndef DETECTION_CALCULATOR_H
#define DETECTION_CALCULATOR_H

#include <models/detection_model.h>
#include <models/input_data.h>
#include <models/results.h>
#include <tilers/detection.h>

#include <memory>
#include <vector>

#include "../inference/geti_calculator_base.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/status.h"
#include "../utils/data_structures.h"

namespace mediapipe {

// Runs detection inference on the provided image and OpenVINO model.
//
// Input:
//  IMAGE - cv::Mat
//
// Output:
//  DETECTIONS - DetectionResult
//
// Input side packet:
//  INFERENCE_ADAPTER - std::shared_ptr<InferenceAdapter>
//

class DetectionCalculator : public GetiCalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract *cc);
  absl::Status Open(CalculatorContext *cc) override;
  absl::Status GetiProcess(CalculatorContext *cc) override;
  absl::Status Close(CalculatorContext *cc) override;

 private:
  std::shared_ptr<InferenceAdapter> ia;
  std::unique_ptr<DetectionModel> model;
  std::unique_ptr<DetectionTiler> tiler;
  std::vector<geti::Label> labels;
};

}  // namespace mediapipe

#endif  // DETECTION_CALCULATOR_H
