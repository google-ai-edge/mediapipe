// Copyright 2022 The MediaPipe Authors.
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

#include "mediapipe/calculators/tensor/inference_interpreter_delegate_runner.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/inference_feedback_manager.h"
#include "mediapipe/calculators/tensor/inference_io_mapper.h"
#include "mediapipe/calculators/tensor/tensor_span.h"
#include "mediapipe/calculators/tensor/tflite_delegate_ptr.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/mediapipe_profiling.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/util/tflite/tflite_model_loader.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/string_util.h"

namespace mediapipe {

namespace {

using Interpreter = ::tflite::Interpreter;
using InterpreterBuilder = ::tflite::InterpreterBuilder;

template <typename T>
void CopyTensorBufferToInterpreter(const Tensor& input_tensor,
                                   Interpreter* interpreter,
                                   int input_tensor_index) {
  auto input_tensor_view = input_tensor.GetCpuReadView();
  auto input_tensor_buffer = input_tensor_view.buffer<T>();
  T* local_tensor_buffer =
      interpreter->typed_input_tensor<T>(input_tensor_index);
  std::memcpy(local_tensor_buffer, input_tensor_buffer, input_tensor.bytes());
}

template <>
void CopyTensorBufferToInterpreter<char>(const Tensor& input_tensor,
                                         Interpreter* interpreter,
                                         int input_tensor_index) {
  const char* input_tensor_buffer =
      input_tensor.GetCpuReadView().buffer<char>();
  tflite::DynamicBuffer dynamic_buffer;
  dynamic_buffer.AddString(input_tensor_buffer,
                           input_tensor.shape().num_elements());
  dynamic_buffer.WriteToTensorAsVector(
      interpreter->tensor(interpreter->inputs()[input_tensor_index]));
}

template <typename T>
void CopyTensorBufferFromInterpreter(Interpreter* interpreter,
                                     int output_tensor_index,
                                     Tensor* output_tensor) {
  auto output_tensor_view = output_tensor->GetCpuWriteView();
  auto output_tensor_buffer = output_tensor_view.buffer<T>();
  T* local_tensor_buffer =
      interpreter->typed_output_tensor<T>(output_tensor_index);
  std::memcpy(output_tensor_buffer, local_tensor_buffer,
              output_tensor->bytes());
}

}  // namespace

class InferenceInterpreterDelegateRunner : public InferenceRunner {
 public:
  InferenceInterpreterDelegateRunner(
      api2::Packet<TfLiteModelPtr> model,
      std::unique_ptr<Interpreter> interpreter, TfLiteDelegatePtr delegate,
      InputOutputTensorNames&& input_output_tensor_names,
      std::unique_ptr<InferenceFeedbackManager> feedback_manager)
      : model_(std::move(model)),
        interpreter_(std::move(interpreter)),
        delegate_(std::move(delegate)),
        input_output_tensor_names_(std::move(input_output_tensor_names)),
        feedback_manager_(std::move(feedback_manager)) {}

  absl::StatusOr<std::vector<Tensor>> Run(
      CalculatorContext* cc, const TensorSpan& tensor_span) override;

  const InputOutputTensorNames& GetInputOutputTensorNames() const override {
    return input_output_tensor_names_;
  }

 private:
  api2::Packet<TfLiteModelPtr> model_;
  std::unique_ptr<Interpreter> interpreter_;
  TfLiteDelegatePtr delegate_;
  InputOutputTensorNames input_output_tensor_names_;
  std::unique_ptr<InferenceFeedbackManager> feedback_manager_;
};

absl::StatusOr<std::vector<Tensor>> InferenceInterpreterDelegateRunner::Run(
    CalculatorContext* cc, const TensorSpan& tensor_span) {
  const int num_feedback_tensors =
      feedback_manager_ ? feedback_manager_->GetNumberOfFeedbackTensors() : 0;

  RET_CHECK_EQ(tensor_span.size() + num_feedback_tensors,
               interpreter_->inputs().size());

  // If the input tensors have dynamic shape, then the tensors need to be
  // resized and reallocated before we can copy the tensor values.
  bool resized_tensor_shapes = false;
  for (int i = 0; i < tensor_span.size(); ++i) {
    int input_model_index;
    if (feedback_manager_) {
      // Feedback tensors are stripped from the InferenceRunner input. Calling
      // MapInputTensorToModelIndex assigns the input tensors to the correct
      // model index.
      MP_ASSIGN_OR_RETURN(input_model_index,
                          feedback_manager_->MapInputTensorToModelIndex(i));
    } else {
      input_model_index = i;
    }
    const Tensor& input_tensor = tensor_span[i];
    if (input_tensor.shape().is_dynamic) {
      const TfLiteTensor* interpreter_tensor =
          interpreter_->tensor(interpreter_->inputs()[input_model_index]);
      // TODO: Can avoid copying even these <= 4 values in the future.
      std::vector<int> interpreter_dims{
          interpreter_tensor->dims->data,
          interpreter_tensor->dims->data + interpreter_tensor->dims->size};
      if (interpreter_dims != input_tensor.shape().dims) {
        interpreter_->ResizeInputTensorStrict(input_model_index,
                                              input_tensor.shape().dims);
        resized_tensor_shapes = true;
      }
    }
  }
  // Reallocation is needed for memory sanity.
  if (resized_tensor_shapes) interpreter_->AllocateTensors();

  // TODO: Replace this using the util function in
  // inference_calculator_utils.
  for (int i = 0; i < tensor_span.size(); ++i) {
    int input_model_index;
    if (feedback_manager_) {
      // Feedback tensors are stripped from the InferenceRunner input. Calling
      // MapInputTensorToModelIndex assigns the input tensors to the correct
      // model index.
      MP_ASSIGN_OR_RETURN(input_model_index,
                          feedback_manager_->MapInputTensorToModelIndex(i));
    } else {
      input_model_index = i;
    }
    const TfLiteType input_tensor_type =
        interpreter_->tensor(interpreter_->inputs()[input_model_index])->type;
    const Tensor& input_tensor = tensor_span[i];
    switch (input_tensor_type) {
      case TfLiteType::kTfLiteFloat16:
      case TfLiteType::kTfLiteFloat32: {
        CopyTensorBufferToInterpreter<float>(input_tensor, interpreter_.get(),
                                             input_model_index);
        break;
      }
      case TfLiteType::kTfLiteUInt8: {
        CopyTensorBufferToInterpreter<uint8_t>(input_tensor, interpreter_.get(),
                                               input_model_index);
        break;
      }
      case TfLiteType::kTfLiteInt8: {
        CopyTensorBufferToInterpreter<int8_t>(input_tensor, interpreter_.get(),
                                              input_model_index);
        break;
      }
      case TfLiteType::kTfLiteInt32: {
        CopyTensorBufferToInterpreter<int32_t>(input_tensor, interpreter_.get(),
                                               input_model_index);
        break;
      }
      case TfLiteType::kTfLiteString: {
        CopyTensorBufferToInterpreter<char>(input_tensor, interpreter_.get(),
                                            input_model_index);
        break;
      }
      case TfLiteType::kTfLiteBool: {
        CopyTensorBufferToInterpreter<bool>(input_tensor, interpreter_.get(),
                                            input_model_index);
        break;
      }
      default:
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported input tensor type:", input_tensor_type));
    }
  }

  // Run inference.
  {
    MEDIAPIPE_PROFILING(CPU_TASK_INVOKE, cc);
    RET_CHECK_EQ(interpreter_->Invoke(), kTfLiteOk);
  }
  // Output result tensors (CPU).
  const auto& tensor_indexes = interpreter_->outputs();
  std::vector<Tensor> output_tensors;
  output_tensors.reserve(tensor_indexes.size() - num_feedback_tensors);
  for (int i = 0; i < tensor_indexes.size(); ++i) {
    if (feedback_manager_ &&
        feedback_manager_->IsFeedbackOutputTensorAtIndex(i)) {
      // Exclude feedback tensors from InferenceRunner output.
      continue;
    }
    TfLiteTensor* tensor = interpreter_->tensor(tensor_indexes[i]);
    Tensor::Shape shape{std::vector<int>{
        tensor->dims->data, tensor->dims->data + tensor->dims->size}};
    switch (tensor->type) {
      case TfLiteType::kTfLiteFloat16:
      case TfLiteType::kTfLiteFloat32:
        output_tensors.emplace_back(Tensor::ElementType::kFloat32, shape);
        CopyTensorBufferFromInterpreter<float>(interpreter_.get(), i,
                                               &output_tensors.back());
        break;
      case TfLiteType::kTfLiteUInt8:
        output_tensors.emplace_back(
            Tensor::ElementType::kUInt8, shape,
            Tensor::QuantizationParameters{tensor->params.scale,
                                           tensor->params.zero_point});
        CopyTensorBufferFromInterpreter<uint8_t>(interpreter_.get(), i,
                                                 &output_tensors.back());
        break;
      case TfLiteType::kTfLiteInt8:
        output_tensors.emplace_back(
            Tensor::ElementType::kInt8, shape,
            Tensor::QuantizationParameters{tensor->params.scale,
                                           tensor->params.zero_point});
        CopyTensorBufferFromInterpreter<int8_t>(interpreter_.get(), i,
                                                &output_tensors.back());
        break;
      case TfLiteType::kTfLiteInt32:
        output_tensors.emplace_back(Tensor::ElementType::kInt32, shape);
        CopyTensorBufferFromInterpreter<int32_t>(interpreter_.get(), i,
                                                 &output_tensors.back());
        break;
      case TfLiteType::kTfLiteBool:
        output_tensors.emplace_back(Tensor::ElementType::kBool, shape,
                                    Tensor::QuantizationParameters{1.0f, 0});
        CopyTensorBufferFromInterpreter<bool>(interpreter_.get(), i,
                                              &output_tensors.back());
        break;
      case TfLiteType::kTfLiteString:
        // No current use-case for copying TfLiteTensors with string type to
        // MediaPipe Tensors.
      default:
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported output tensor type:",
                         TfLiteTypeGetName(tensor->type)));
    }
  }
  if (feedback_manager_) {
    feedback_manager_->SwapFeedbackTensors();
  }
  return output_tensors;
}

absl::StatusOr<std::unique_ptr<InferenceRunner>>
CreateInferenceInterpreterDelegateRunner(
    api2::Packet<TfLiteModelPtr> model,
    api2::Packet<tflite::OpResolver> op_resolver, TfLiteDelegatePtr delegate,
    int interpreter_num_threads,
    const mediapipe::InferenceCalculatorOptions::InputOutputConfig*
        input_output_config) {
  InterpreterBuilder interpreter_builder(*model.Get(), op_resolver.Get());
  if (delegate) {
    interpreter_builder.AddDelegate(delegate.get());
  }
#if defined(__EMSCRIPTEN__)
  interpreter_builder.SetNumThreads(1);
#else
  interpreter_builder.SetNumThreads(interpreter_num_threads);
#endif  // __EMSCRIPTEN__
  std::unique_ptr<Interpreter> interpreter;
  RET_CHECK_EQ(interpreter_builder(&interpreter), kTfLiteOk);
  RET_CHECK(interpreter);
  RET_CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  MP_ASSIGN_OR_RETURN(
      auto input_output_tensor_names,
      InferenceIoMapper::GetInputOutputTensorNamesFromInterpreter(
          *interpreter));
  std::unique_ptr<InferenceFeedbackManager> inference_feedback_manager;
  if (input_output_config) {
    // Create inference_feedback_manager if input_output_config is available.
    inference_feedback_manager = std::make_unique<InferenceFeedbackManager>();
    MP_RETURN_IF_ERROR(inference_feedback_manager->Init(
        *input_output_config, input_output_tensor_names, interpreter.get()));
  }
  return std::make_unique<InferenceInterpreterDelegateRunner>(
      std::move(model), std::move(interpreter), std::move(delegate),
      std::move(input_output_tensor_names),
      std::move(inference_feedback_manager));
}

}  // namespace mediapipe
