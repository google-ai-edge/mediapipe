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

#ifndef MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_RUNNER_ML_DRIFT_OPENCL_RUNTIME_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_RUNNER_ML_DRIFT_OPENCL_RUNTIME_H_

#include <cstdint>
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
#include "mediapipe/gpu/gl_base.h"
#include "mediapipe/util/tflite/tflite_model_loader.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/model_builder.h"
#include "third_party/ml_drift/cl/api.h"
#include "third_party/ml_drift/common/model.h"
#include "third_party/ml_drift/common/shape.h"
#include "third_party/ml_drift/contrib/tflite_op_resolver.h"
#include "third_party/ml_drift/delegate/api_cl_gl_vk.h"

namespace mediapipe::api2 {

// Inference runner implementation that uses the ML Drift OpenCL runtime with
// GPU bindings.
class InferenceRunnerMlDriftOpenClRuntime : public InferenceRunner {
 public:
  ~InferenceRunnerMlDriftOpenClRuntime() override = default;

  absl::Status Init(
      const mediapipe::InferenceCalculatorOptions& options,
      Packet<TfLiteModelPtr> model_packet,
      Packet<ml_drift::contrib::TfLiteOpResolver> op_resolver_packet);

  // This method must be executed on current OpenGL context / thread.
  absl::StatusOr<std::vector<Tensor>> Run(
      CalculatorContext* cc, const TensorSpan& input_tensors) override;

  const InputOutputTensorNames& GetInputOutputTensorNames() const override;

 private:
  absl::Status InitTFLiteGpuRunner(
      CalculatorContext* cc,
      const mediapipe::InferenceCalculatorOptions::Delegate& delegate);

  absl::StatusOr<ml_drift::GraphFloat32> InitModelFromFlatBuffer(
      const tflite::FlatBufferModel& flatbuffer,
      const tflite::OpResolver& op_resolver, bool allow_quant_ops = false);

  absl::Status BindSsboToInputTensor(GLuint ssbo_id, int input_id);
  absl::Status BindSsboToOutputTensor(GLuint ssbo_id, int output_id);

  absl::Status InitializeMlDriftRuntime(
      ml_drift::GraphFloat32&& graph_cl,
      const ml_drift::cl::InferenceOptions& options);

  InputOutputTensorNames input_output_tensor_names_;

  std::unique_ptr<ml_drift::cl::InferenceEnvironment> cl_environment_;

  std::unique_ptr<ml_drift::InferenceRunner> runner_;

  std::vector<ml_drift::BHWC> input_shapes_;
  std::vector<ml_drift::BHWC> output_shapes_;
  std::vector<Tensor::Shape> tensor_output_shapes_;
};

};  // namespace mediapipe::api2

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_RUNNER_ML_DRIFT_OPENCL_RUNTIME_H_
