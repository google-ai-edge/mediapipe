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

#include "mediapipe/calculators/tensor/inference_calculator_utils.h"

#include <cstring>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port.h"  // NOLINT: provides MEDIAPIPE_ANDROID/IOS
#include "mediapipe/framework/port/status_macros.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"

#if !defined(__EMSCRIPTEN__) || defined(__EMSCRIPTEN_PTHREADS__)
#include "mediapipe/util/cpu_util.h"
#endif  // !__EMSCRIPTEN__ || __EMSCRIPTEN_PTHREADS__

namespace mediapipe {

namespace {

int GetXnnpackDefaultNumThreads() {
#if defined(MEDIAPIPE_ANDROID) || defined(MEDIAPIPE_IOS) || \
    defined(__EMSCRIPTEN_PTHREADS__)
  constexpr int kMinNumThreadsByDefault = 1;
  constexpr int kMaxNumThreadsByDefault = 4;
  return std::clamp(NumCPUCores() / 2, kMinNumThreadsByDefault,
                    kMaxNumThreadsByDefault);
#else
  return 1;
#endif  // MEDIAPIPE_ANDROID || MEDIAPIPE_IOS || __EMSCRIPTEN_PTHREADS__
}

// Checks if a MediaPipe Tensor's type matches a TfLite's data type.
bool DoTypesMatch(Tensor::ElementType tensor_type, TfLiteType tflite_type) {
  switch (tensor_type) {
    // Do these two match?
    case Tensor::ElementType::kNone:
      return tflite_type == TfLiteType::kTfLiteNoType;
    case Tensor::ElementType::kFloat16:
      return tflite_type == TfLiteType::kTfLiteFloat16;
    case Tensor::ElementType::kFloat32:
      return tflite_type == TfLiteType::kTfLiteFloat32;
    case Tensor::ElementType::kUInt8:
      return tflite_type == TfLiteType::kTfLiteUInt8;
    case Tensor::ElementType::kInt8:
      return tflite_type == TfLiteType::kTfLiteInt8;
    case Tensor::ElementType::kInt32:
      return tflite_type == TfLiteType::kTfLiteInt32;
    case Tensor::ElementType::kBool:
      return tflite_type == TfLiteType::kTfLiteBool;
    // Seems like TfLite does not have a char type support?
    default:
      return false;
  }
}

template <typename T>
absl::Status CopyTensorBufferToInterpreter(const Tensor& input_tensor,
                                           tflite::Interpreter& interpreter,
                                           int input_tensor_index) {
  auto input_tensor_view = input_tensor.GetCpuReadView();
  const T* input_tensor_buffer = input_tensor_view.buffer<T>();
  if (input_tensor_buffer == nullptr) {
    return absl::InternalError("Input tensor buffer is null.");
  }
  T* local_tensor_buffer =
      interpreter.typed_input_tensor<T>(input_tensor_index);
  if (local_tensor_buffer == nullptr) {
    return absl::InvalidArgumentError(
        "Interpreter's input tensor buffer is null, may because it does not "
        "support the input type specified.");
  }
  const TfLiteTensor* local_tensor =
      interpreter.input_tensor(input_tensor_index);
  if (local_tensor->bytes != input_tensor.bytes()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Interpreter's input size do not match the input tensor's "
                     "size for index ",
                     input_tensor_index, "."));
  }
  std::memcpy(local_tensor_buffer, input_tensor_buffer, input_tensor.bytes());
  return absl::OkStatus();
}

}  // namespace

int GetXnnpackNumThreads(
    const bool opts_has_delegate,
    const mediapipe::InferenceCalculatorOptions::Delegate& opts_delegate) {
  static constexpr int kDefaultNumThreads = -1;
  if (opts_has_delegate && opts_delegate.has_xnnpack() &&
      opts_delegate.xnnpack().num_threads() != kDefaultNumThreads) {
    return opts_delegate.xnnpack().num_threads();
  }
  return GetXnnpackDefaultNumThreads();
}

absl::Status CopyCpuInputIntoInterpreterTensor(const Tensor& input_tensor,
                                               tflite::Interpreter& interpreter,
                                               int input_tensor_index) {
  const TfLiteType interpreter_tensor_type =
      interpreter.input_tensor(input_tensor_index)->type;
  const Tensor::ElementType input_tensor_type = input_tensor.element_type();
  if (!DoTypesMatch(input_tensor_type, interpreter_tensor_type)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Input and interpreter tensor type mismatch: ", input_tensor_type,
        " vs. ", interpreter_tensor_type));
  }
  switch (interpreter_tensor_type) {
    case TfLiteType::kTfLiteFloat16:
    case TfLiteType::kTfLiteFloat32: {
      MP_RETURN_IF_ERROR(CopyTensorBufferToInterpreter<float>(
          input_tensor, interpreter, input_tensor_index));
      break;
    }
    case TfLiteType::kTfLiteInt32: {
      MP_RETURN_IF_ERROR(CopyTensorBufferToInterpreter<int>(
          input_tensor, interpreter, input_tensor_index));
      break;
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported input data type: ", input_tensor_type));
  }
  return absl::OkStatus();
}

}  // namespace mediapipe
