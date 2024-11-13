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
#ifndef ANOMALY_CALCULATOR_H
#define ANOMALY_CALCULATOR_H

#include <models/anomaly_model.h>
#include <models/input_data.h>
#include <models/results.h>

#include <memory>
#include <string>

#include "../inference/geti_calculator_base.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/status.h"
#include "../utils/data_structures.h"

inline std::string getEnvVar(std::string const &key) {
  char *val = getenv(key.c_str());
  return val == nullptr ? std::string("") : std::string(val);
}

namespace mediapipe {

// Runs anomaly inference on the provided image and OpenVINO model.
//
// Input:
//  IMAGE
//
// Output:
//  RESULT
//
// Input side packet:
//  INFERENCE_ADAPTER
//

class AnomalyCalculator : public GetiCalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract *cc);
  absl::Status Open(CalculatorContext *cc) override;
  absl::Status GetiProcess(CalculatorContext *cc) override;
  absl::Status Close(CalculatorContext *cc) override;

 private:
  std::shared_ptr<InferenceAdapter> ia;
  std::unique_ptr<AnomalyModel> model;
  std::string task;
  geti::Label normal_label;
  geti::Label anomalous_label;
};

}  // namespace mediapipe

#endif  // ANOMALY_CALCULATOR_H
