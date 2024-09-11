// Copyright 2024 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_RUNNER_ML_DRIFT_OPENCL_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_RUNNER_ML_DRIFT_OPENCL_H_

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/inference_io_mapper.h"
#include "mediapipe/calculators/tensor/inference_runner.h"
#include "mediapipe/calculators/tensor/tensor_span.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/util/tflite/tflite_model_loader.h"
#include "tensorflow/lite/interpreter.h"
#include "third_party/ml_drift/contrib/tflite_op_resolver.h"

namespace mediapipe::api2 {

// Inference runner implementation that uses the ML Drift OpenCL Delegate.
class InferenceRunnerMlDriftOpenClDelegate : public InferenceRunner {
 public:
  ~InferenceRunnerMlDriftOpenClDelegate() override = default;

  absl::Status Init(
      const mediapipe::InferenceCalculatorOptions& options,
      Packet<TfLiteModelPtr> model_packet,
      Packet<ml_drift::contrib::TfLiteOpResolver> op_resolver_packet);

  absl::StatusOr<std::vector<Tensor>> Run(
      CalculatorContext* cc, const TensorSpan& input_tensors) override;

  const InputOutputTensorNames& GetInputOutputTensorNames() const override;

 private:
  static absl::StatusOr<std::vector<Tensor>> AllocateOutputTensors(
      const tflite::Interpreter& interpreter);

  // TfLite requires us to keep the model alive as long as the interpreter is.
  Packet<TfLiteModelPtr> model_packet_;
  InputOutputTensorNames input_output_tensor_names_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
};

}  // namespace mediapipe::api2

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_RUNNER_ML_DRIFT_OPENCL_H_
