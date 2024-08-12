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

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/calculators/tensor/inference_calculator_utils.h"
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
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/util.h"

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

absl::Status VerifyModelTensorsForCustomAllocation(
    const Interpreter& interpreter) {
  absl::flat_hash_set<int> input_tensor_indices_set(
      interpreter.inputs().begin(), interpreter.inputs().end());
  RET_CHECK(input_tensor_indices_set.size() == interpreter.inputs().size())
      << "Custom allocation is not supported for models with duplicate input "
         "tensor indices.";
  absl::flat_hash_set<int> output_tensor_indices_set;
  for (const int output_tensor_index : interpreter.outputs()) {
    const auto [_, inserted] =
        output_tensor_indices_set.insert(output_tensor_index);
    RET_CHECK(inserted) << "Custom allocation is not supported for models with "
                           "duplicate output tensor indices: "
                        << output_tensor_index;
    RET_CHECK(!input_tensor_indices_set.contains(output_tensor_index))
        << "Custom allocation is not supported for models with input->output "
           "passthrough tensors, i.e. the same tensor index appears in the "
           "model input and output tensors: "
        << output_tensor_index;
  }
  return absl::OkStatus();
}

absl::StatusOr<Tensor> AllocateOutputTensor(const int tensor_index,
                                            const Interpreter& interpreter) {
  const TfLiteTensor* tensor = interpreter.tensor(tensor_index);
  Tensor::Shape shape{std::vector<int>{
      tensor->dims->data, tensor->dims->data + tensor->dims->size}};
  switch (tensor->type) {
    case TfLiteType::kTfLiteFloat16:
    case TfLiteType::kTfLiteFloat32:
      return Tensor(Tensor::ElementType::kFloat32, shape,
                    Tensor::QuantizationParameters{tensor->params.scale,
                                                   tensor->params.zero_point},
                    /*memory_manager=*/nullptr,
                    tflite::kDefaultTensorAlignment);
      break;
    case TfLiteType::kTfLiteUInt8:
      return Tensor(Tensor::ElementType::kUInt8, shape,
                    Tensor::QuantizationParameters{tensor->params.scale,
                                                   tensor->params.zero_point},
                    /*memory_manager=*/nullptr,
                    tflite::kDefaultTensorAlignment);
      break;
    case TfLiteType::kTfLiteInt8:
      return Tensor(Tensor::ElementType::kInt8, shape,
                    Tensor::QuantizationParameters{tensor->params.scale,
                                                   tensor->params.zero_point},
                    /*memory_manager=*/nullptr,
                    tflite::kDefaultTensorAlignment);
      break;
    case TfLiteType::kTfLiteInt32:
      return Tensor(Tensor::ElementType::kInt32, shape,
                    Tensor::QuantizationParameters{tensor->params.scale,
                                                   tensor->params.zero_point},
                    /*memory_manager=*/nullptr,
                    tflite::kDefaultTensorAlignment);
      break;
    case TfLiteType::kTfLiteBool:
      return Tensor(Tensor::ElementType::kBool, shape,
                    Tensor::QuantizationParameters{1.0f, 0},
                    /*memory_manager=*/nullptr,
                    tflite::kDefaultTensorAlignment);
      break;
    case TfLiteType::kTfLiteString:
      // No current use-case for copying TfLiteTensors with string type to
      // MediaPipe Tensors.
    default:
      break;
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Unsupported output tensor type:", TfLiteTypeGetName(tensor->type)));
}

absl::Status CopyTensorToInterpreter(const Tensor& input_tensor,
                                     Interpreter* interpreter,
                                     int model_input_index) {
  const TfLiteType input_tensor_type =
      interpreter->tensor(interpreter->inputs()[model_input_index])->type;
  switch (input_tensor_type) {
    case TfLiteType::kTfLiteFloat16:
    case TfLiteType::kTfLiteFloat32: {
      CopyTensorBufferToInterpreter<float>(input_tensor, interpreter,
                                           model_input_index);
      break;
    }
    case TfLiteType::kTfLiteUInt8: {
      CopyTensorBufferToInterpreter<uint8_t>(input_tensor, interpreter,
                                             model_input_index);
      break;
    }
    case TfLiteType::kTfLiteInt8: {
      CopyTensorBufferToInterpreter<int8_t>(input_tensor, interpreter,
                                            model_input_index);
      break;
    }
    case TfLiteType::kTfLiteInt32: {
      CopyTensorBufferToInterpreter<int32_t>(input_tensor, interpreter,
                                             model_input_index);
      break;
    }
    case TfLiteType::kTfLiteString: {
      CopyTensorBufferToInterpreter<char>(input_tensor, interpreter,
                                          model_input_index);
      break;
    }
    case TfLiteType::kTfLiteBool: {
      CopyTensorBufferToInterpreter<bool>(input_tensor, interpreter,
                                          model_input_index);
      break;
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported input tensor type:", input_tensor_type));
  }
  return absl::OkStatus();
}

absl::Status CopyTensorFromInterpreter(Interpreter* interpreter,
                                       int output_model_index,
                                       Tensor* output_tensor) {
  TfLiteTensor* tensor =
      interpreter->tensor(interpreter->outputs()[output_model_index]);
  switch (tensor->type) {
    case TfLiteType::kTfLiteFloat16:
    case TfLiteType::kTfLiteFloat32:
      CopyTensorBufferFromInterpreter<float>(interpreter, output_model_index,
                                             output_tensor);
      break;
    case TfLiteType::kTfLiteUInt8:
      CopyTensorBufferFromInterpreter<uint8_t>(interpreter, output_model_index,
                                               output_tensor);
      break;
    case TfLiteType::kTfLiteInt8:
      CopyTensorBufferFromInterpreter<int8_t>(interpreter, output_model_index,
                                              output_tensor);
      break;
    case TfLiteType::kTfLiteInt32:
      CopyTensorBufferFromInterpreter<int32_t>(interpreter, output_model_index,
                                               output_tensor);
      break;
    case TfLiteType::kTfLiteBool:
      CopyTensorBufferFromInterpreter<bool>(interpreter, output_model_index,
                                            output_tensor);
      break;
    case TfLiteType::kTfLiteString:
      // No current use-case for copying TfLiteTensors with string type to
      // MediaPipe Tensors.
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported output tensor type:", TfLiteTypeGetName(tensor->type)));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<Tensor>> AllocateOutputTensors(
    const std::vector<int>& model_output_indexes,
    const Interpreter& interpreter) {
  std::vector<Tensor> output_tensors;
  output_tensors.reserve(model_output_indexes.size());
  for (int i = 0; i < model_output_indexes.size(); ++i) {
    MP_ASSIGN_OR_RETURN(
        Tensor output_tensor,
        AllocateOutputTensor(interpreter.outputs()[model_output_indexes[i]],
                             interpreter));
    output_tensors.push_back(std::move(output_tensor));
  }
  return output_tensors;
}

}  // namespace

class InferenceInterpreterDelegateRunner : public InferenceRunner {
 public:
  InferenceInterpreterDelegateRunner(
      api2::Packet<TfLiteModelPtr> model,
      std::unique_ptr<Interpreter> interpreter, TfLiteDelegatePtr delegate,
      InputOutputTensorNames&& input_output_tensor_names,
      std::unique_ptr<InferenceFeedbackManager> feedback_manager,
      bool enable_zero_copy_tensor_io)
      : model_(std::move(model)),
        interpreter_(std::move(interpreter)),
        delegate_(std::move(delegate)),
        input_output_tensor_names_(std::move(input_output_tensor_names)),
        feedback_manager_(std::move(feedback_manager)),
        enable_zero_copy_tensor_io_(enable_zero_copy_tensor_io) {}

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
  bool enable_zero_copy_tensor_io_ = false;
};

absl::StatusOr<std::vector<Tensor>> InferenceInterpreterDelegateRunner::Run(
    CalculatorContext* cc, const TensorSpan& tensor_span) {
  const int num_feedback_tensors =
      feedback_manager_ ? feedback_manager_->GetNumberOfFeedbackTensors() : 0;

  RET_CHECK_EQ(tensor_span.size() + num_feedback_tensors,
               interpreter_->inputs().size());

  std::vector<int> input_indices_excluding_feedback_tensors;
  input_indices_excluding_feedback_tensors.reserve(tensor_span.size());
  for (int i = 0; i < interpreter_->inputs().size(); ++i) {
    if (feedback_manager_ &&
        feedback_manager_->IsFeedbackInputTensorAtIndex(i)) {
      // Feedback tensors are stripped from the InferenceRunner input.
      continue;
    }
    input_indices_excluding_feedback_tensors.push_back(i);
  }
  std::vector<int> output_indices_excluding_feedback_tensors;
  output_indices_excluding_feedback_tensors.reserve(
      interpreter_->outputs().size() - num_feedback_tensors);
  for (int i = 0; i < interpreter_->outputs().size(); ++i) {
    if (feedback_manager_ &&
        feedback_manager_->IsFeedbackOutputTensorAtIndex(i)) {
      // Exclude feedback tensors from InferenceRunner output.
      continue;
    }
    output_indices_excluding_feedback_tensors.push_back(i);
  }

  // Input tensor views for TfLite custom allocation. They must outlive the
  // inference call to provide Tensor read access to the interpreter.
  std::vector<Tensor::CpuReadView> input_tensor_views;
  input_tensor_views.reserve(tensor_span.size());

  // If the input tensors have dynamic shape, then the tensors need to be
  // resized and reallocated before we can copy the tensor values.
  bool resized_tensor_shapes = false;
  for (int i = 0; i < input_indices_excluding_feedback_tensors.size(); ++i) {
    const int input_tensor_index = input_indices_excluding_feedback_tensors[i];
    const Tensor& input_tensor = tensor_span[i];
    if (input_tensor.shape().is_dynamic) {
      const TfLiteTensor* interpreter_tensor =
          interpreter_->tensor(interpreter_->inputs()[input_tensor_index]);
      // TODO: Can avoid copying even these <= 4 values in the future.
      std::vector<int> interpreter_dims{
          interpreter_tensor->dims->data,
          interpreter_tensor->dims->data + interpreter_tensor->dims->size};
      if (interpreter_dims != input_tensor.shape().dims) {
        interpreter_->ResizeInputTensorStrict(input_tensor_index,
                                              input_tensor.shape().dims);
        resized_tensor_shapes = true;
      }
    }
  }
  // Reallocation is needed for memory sanity.
  if (resized_tensor_shapes) interpreter_->AllocateTensors();

  // TODO: Replace this using the util function in
  // inference_calculator_utils.
  for (int i = 0; i < input_indices_excluding_feedback_tensors.size(); ++i) {
    const int input_tensor_index = input_indices_excluding_feedback_tensors[i];
    const Tensor& input_tensor = tensor_span[i];
    // TODO b/329100795 - can TfLite custom allocation work with dynamic
    // tensors?
    if (enable_zero_copy_tensor_io_) {
      auto input_tensor_view = input_tensor.GetCpuReadView();
      RET_CHECK(IsAlignedWithTFLiteDefaultAlignment(
          input_tensor_view.buffer<const void>()))
          << "TfLite custom tensor allocation of input tensors is enabled but "
             "tensor memory is not aligned to tflite::kDefaultTensorAlignment.";
      MP_RETURN_IF_ERROR(SetTfLiteCustomAllocation(
          *interpreter_, input_tensor_view.buffer<const void>(),
          input_tensor.bytes(), interpreter_->inputs()[input_tensor_index]));
      input_tensor_views.emplace_back(std::move(input_tensor_view));
      continue;
    }

    MP_RETURN_IF_ERROR(CopyTensorToInterpreter(input_tensor, interpreter_.get(),
                                               input_tensor_index));
  }

  MP_ASSIGN_OR_RETURN(
      std::vector<Tensor> output_tensors,
      AllocateOutputTensors(output_indices_excluding_feedback_tensors,
                            *interpreter_));

  std::vector<Tensor::CpuWriteView> output_tensor_views;
  if (enable_zero_copy_tensor_io_) {
    for (int i = 0; i < output_indices_excluding_feedback_tensors.size(); ++i) {
      const int output_tensor_index =
          output_indices_excluding_feedback_tensors[i];
      Tensor& tensor = output_tensors[i];
      auto write_view = output_tensors[i].GetCpuWriteView();
      MP_RETURN_IF_ERROR(SetTfLiteCustomAllocation(
          *interpreter_, write_view.buffer<void>(), tensor.bytes(),
          interpreter_->outputs()[output_tensor_index]));
      output_tensor_views.push_back(std::move(write_view));
    }
  }

  // Reallocation is needed for memory sanity.
  if (resized_tensor_shapes || !input_tensor_views.empty() ||
      !output_tensor_views.empty()) {
    interpreter_->AllocateTensors();
  }

  // Run inference.
  {
    MEDIAPIPE_PROFILING(CPU_TASK_INVOKE, cc);
    RET_CHECK_EQ(interpreter_->Invoke(), kTfLiteOk);
  }
  input_tensor_views.clear();
  output_tensor_views.clear();

  if (enable_zero_copy_tensor_io_) {
    // TODO b/340643988 -To avoid dangling pointers to Tensors that are not
    // owned anymore by the InferenceRunner (once output tensors are passed to
    // downstream calculators), we should invalidate TfLiteCustomAllocation
    // assignments here.
  } else {
    // Copy output tensors from the interpreter.
    for (int i = 0; i < output_indices_excluding_feedback_tensors.size(); ++i) {
      MP_RETURN_IF_ERROR(CopyTensorFromInterpreter(
          interpreter_.get(), output_indices_excluding_feedback_tensors[i],
          &output_tensors[i]));
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
        input_output_config,
    bool enable_zero_copy_tensor_io) {
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
  if (enable_zero_copy_tensor_io) {
    MP_RETURN_IF_ERROR(VerifyModelTensorsForCustomAllocation(*interpreter));
  }
  return std::make_unique<InferenceInterpreterDelegateRunner>(
      std::move(model), std::move(interpreter), std::move(delegate),
      std::move(input_output_tensor_names),
      std::move(inference_feedback_manager), enable_zero_copy_tensor_io);
}

}  // namespace mediapipe
