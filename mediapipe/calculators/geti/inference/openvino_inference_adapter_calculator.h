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
#ifndef OPENVINO_INFERENCE_ADAPTER_CALCULATOR_H
#define OPENVINO_INFERENCE_ADAPTER_CALCULATOR_H
#include <adapters/inference_adapter.h>

#include <memory>

#include "../inference/geti_calculator_base.h"
#include "mediapipe/calculators/geti/inference/openvino_inference_adapter_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

// Create inference adapter on the provided model and device
//
// Input side packet:
//  MODEL_PATH
//  DEVICE
//
// Output side packet:
//  INFERENCE_ADAPTER
//

class OpenVINOInferenceAdapterCalculator : public GetiCalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract *cc);
  absl::Status Open(CalculatorContext *cc) override;
  absl::Status GetiProcess(CalculatorContext *cc) override;
  absl::Status Close(CalculatorContext *cc) override;

 private:
  std::shared_ptr<InferenceAdapter> ia;
};

}  // namespace mediapipe

#endif  // OPENVINO_INFERENCE_ADAPTER_CALCULATOR_H
