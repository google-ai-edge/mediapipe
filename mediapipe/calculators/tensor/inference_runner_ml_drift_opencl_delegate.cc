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

#include "mediapipe/calculators/tensor/inference_runner_ml_drift_opencl_delegate.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/inference_calculator_utils.h"
#include "mediapipe/calculators/tensor/inference_io_mapper.h"
#include "mediapipe/calculators/tensor/tensor_span.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/util/tflite/tflite_model_loader.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/util.h"
#include "third_party/ml_drift/contrib/tflite_op_resolver.h"
#include "third_party/odml/infra/ml_drift_delegate/ml_drift_cl.h"
#include "third_party/odml/infra/ml_drift_delegate/ml_drift_delegate.h"

namespace mediapipe::api2 {

using ::tflite::ml_drift::MlDriftClDelegateDefaultOptionsPtr;
using ::tflite::ml_drift::TfLiteCreateMlDriftClDelegate;

absl::Status InferenceRunnerMlDriftOpenClDelegate::Init(
    const mediapipe::InferenceCalculatorOptions& options,
    Packet<TfLiteModelPtr> model_packet,
    Packet<ml_drift::contrib::TfLiteOpResolver> op_resolver_packet) {
  RET_CHECK_EQ(options.delegate().gpu().api(),
               InferenceCalculatorOptions::Delegate::Gpu::ML_DRIFT_OPENCL);
  model_packet_ = model_packet;
  tflite::InterpreterBuilder builder(*model_packet_.Get(),
                                     op_resolver_packet.Get());
  builder(&interpreter_);
  ABSL_CHECK(interpreter_ != nullptr);
  MP_ASSIGN_OR_RETURN(
      input_output_tensor_names_,
      InferenceIoMapper::GetInputOutputTensorNamesFromInterpreter(
          *interpreter_));
  // Initialize ML Drift CL.
  auto delegate_options = MlDriftClDelegateDefaultOptionsPtr();
  delegate_options->enable_fast_tuning = true;
  if (options.delegate().gpu().allow_precision_loss()) {
    delegate_options->precision = MlDriftDelegatePrecision::kDefault;
  } else {
    delegate_options->precision = MlDriftDelegatePrecision::kFp32;
  }
  tflite::Interpreter::TfLiteDelegatePtr delegate =
      TfLiteCreateMlDriftClDelegate(std::move(delegate_options));
  ABSL_CHECK_EQ(interpreter_->ModifyGraphWithDelegate(std::move(delegate)),
                kTfLiteOk);
  interpreter_->AllocateTensors();
  return absl::OkStatus();
}

absl::StatusOr<std::vector<Tensor>> InferenceRunnerMlDriftOpenClDelegate::Run(
    CalculatorContext* cc, const TensorSpan& input_tensors) {
  // If the input tensors have dynamic shape, then the tensors need to be
  // resized and reallocated before we can copy the tensor values.
  bool resized_tensor_shapes = false;
  for (int i = 0; i < input_tensors.size(); ++i) {
    const Tensor& input_tensor = input_tensors[i];
    if (input_tensor.shape().is_dynamic) {
      const TfLiteTensor* interpreter_tensor =
          interpreter_->tensor(interpreter_->inputs()[i]);
      // TODO: Can avoid copying even these <= 4 values in the future.
      std::vector<int> interpreter_dims{
          interpreter_tensor->dims->data,
          interpreter_tensor->dims->data + interpreter_tensor->dims->size};
      if (interpreter_dims != input_tensor.shape().dims) {
        interpreter_->ResizeInputTensorStrict(i, input_tensor.shape().dims);
        resized_tensor_shapes = true;
      }
    }
  }
  // Reallocation is needed for memory sanity.
  if (resized_tensor_shapes) {
    interpreter_->AllocateTensors();
  }

  for (int i = 0; i < input_tensors.size(); ++i) {
    const Tensor& input_tensor = input_tensors[i];
    MP_RETURN_IF_ERROR(
        CopyCpuInputIntoInterpreterTensor(input_tensor, *interpreter_, i));
  }
  ABSL_CHECK_EQ(interpreter_->Invoke(), kTfLiteOk);
  MP_ASSIGN_OR_RETURN(auto output_tensors,
                      AllocateOutputTensors(*interpreter_));
  for (int i = 0; i < output_tensors.size(); ++i) {
    const int output_tensor_index = interpreter_->outputs()[i];
    MP_RETURN_IF_ERROR(CopyInterpreterTensorIntoCpuOutput(
        *interpreter_, output_tensor_index, output_tensors[i]));
  }
  return output_tensors;
}

absl::StatusOr<std::vector<Tensor>>
InferenceRunnerMlDriftOpenClDelegate::AllocateOutputTensors(
    const tflite::Interpreter& interpreter) {
  const int num_outputs = interpreter.outputs().size();
  std::vector<Tensor> output_tensors;
  output_tensors.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    const TfLiteTensor* reference_tensor =
        interpreter.tensor(interpreter.outputs()[i]);
    MP_ASSIGN_OR_RETURN(Tensor output_tensor,
                        CreateTensorWithTfLiteTensorSpecs(
                            *reference_tensor, /*memory_manager=*/nullptr,
                            tflite::kDefaultTensorAlignment));
    output_tensors.push_back(std::move(output_tensor));
  }
  return output_tensors;
}

const InputOutputTensorNames&
InferenceRunnerMlDriftOpenClDelegate::GetInputOutputTensorNames() const {
  return input_output_tensor_names_;
}

}  // namespace mediapipe::api2
