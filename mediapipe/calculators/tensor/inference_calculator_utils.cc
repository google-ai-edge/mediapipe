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

#include <cstdint>
#include <cstring>
#include <ostream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/port.h"  // NOLINT: provides MEDIAPIPE_ANDROID/IOS
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/portable_type_to_tflitetype.h"
#include "tensorflow/lite/string_util.h"

ABSL_FLAG(int, xnnpack_default_num_threads, 0,
          "Default number of xnnpack threads to use. If unset, determines a "
          "good default number based on the platform.");

#if !defined(__EMSCRIPTEN__) || defined(__EMSCRIPTEN_PTHREADS__)
#include "mediapipe/util/cpu_util.h"
#endif  // !__EMSCRIPTEN__ || __EMSCRIPTEN_PTHREADS__

namespace mediapipe {
namespace {

int GetXnnpackDefaultNumThreads() {
  int default_from_flag = absl::GetFlag(FLAGS_xnnpack_default_num_threads);
  if (default_from_flag > 0) {
    return default_from_flag;
  }
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
bool operator==(Tensor::ElementType tensor_type, TfLiteType tflite_type) {
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
    case Tensor::ElementType::kInt64:
      return tflite_type == TfLiteType::kTfLiteInt64;
    case Tensor::ElementType::kBool:
      return tflite_type == TfLiteType::kTfLiteBool;
    case Tensor::ElementType::kChar:
      return tflite_type == TfLiteType::kTfLiteString;
    default:
      return false;
  }
}

std::string GetTensorTypeString(const Tensor::ElementType& tensor_type) {
  switch (tensor_type) {
    case Tensor::ElementType::kNone:
      return "kNone";
    case Tensor::ElementType::kFloat16:
      return "kFloat16";
    case Tensor::ElementType::kFloat32:
      return "kFloat32";
    case Tensor::ElementType::kUInt8:
      return "kUInt8";
    case Tensor::ElementType::kInt8:
      return "kInt8";
    case Tensor::ElementType::kInt32:
      return "kInt32";
    case Tensor::ElementType::kBool:
      return "kBool";
    case Tensor::ElementType::kChar:
      return "kChar";
    default:
      return "Unknown";
  }
}

std::string GetTfLiteTensorDebugInfo(const TfLiteTensor& tflite_tensor) {
  absl::Span<int> dims(tflite_tensor.dims->data, tflite_tensor.dims->size);
  return absl::StrFormat(
      "TfLiteTensor dims: [%s], type: %s, bytes: %d", absl::StrJoin(dims, ", "),
      TfLiteTypeGetName(tflite_tensor.type), tflite_tensor.bytes);
}

std::string GetMpTensorDebugInfo(const Tensor& tensor) {
  return absl::StrFormat("MP Tensor dims: [%s], type: %s, bytes: %d",
                         absl::StrJoin(tensor.shape().dims, ", "),
                         GetTensorTypeString(tensor.element_type()),
                         tensor.bytes());
}

template <typename T>
absl::Status CopyTensorToTfLiteTensor(const Tensor& input_tensor,
                                      TfLiteTensor& tflite_tensor) {
  auto input_tensor_view = input_tensor.GetCpuReadView();
  const T* input_tensor_buffer = input_tensor_view.buffer<T>();
  RET_CHECK(input_tensor_buffer) << "Input tensor buffer is null.";
  RET_CHECK_EQ(tflite_tensor.type, tflite::typeToTfLiteType<T>())
          .SetCode(absl::StatusCode::kInvalidArgument)
      << "Tensor and TfLiteTensor types do not match.";
  void* local_tensor_buffer = tflite_tensor.data.raw;
  RET_CHECK(local_tensor_buffer) << "TfLiteTensor data is null.";
  RET_CHECK_EQ(tflite_tensor.bytes, input_tensor.bytes())
          .SetCode(absl::StatusCode::kInvalidArgument)
      << "TfLiteTensor and Tensor sizes do not match. "
      << GetTfLiteTensorDebugInfo(tflite_tensor) << " vs. "
      << GetMpTensorDebugInfo(input_tensor);
  std::memcpy(local_tensor_buffer, input_tensor_buffer, input_tensor.bytes());
  return absl::OkStatus();
}

template <>
absl::Status CopyTensorToTfLiteTensor<char>(const Tensor& input_tensor,
                                            TfLiteTensor& tflite_tensor) {
  const char* input_tensor_buffer =
      input_tensor.GetCpuReadView().buffer<char>();
  RET_CHECK(input_tensor_buffer) << "Char-typed input tensor buffer is null.";
  RET_CHECK_EQ(tflite_tensor.type, TfLiteType::kTfLiteString)
          .SetCode(absl::StatusCode::kInvalidArgument)
      << "TfLiteTensor type is not kTfLiteString while Tensor type is kChar.";
  tflite::DynamicBuffer dynamic_buffer;
  dynamic_buffer.AddString(input_tensor_buffer,
                           input_tensor.shape().num_elements());
  dynamic_buffer.WriteToTensorAsVector(&tflite_tensor);
  return absl::OkStatus();
}

template <typename T>
absl::Status CopyTfLiteTensorToTensor(const TfLiteTensor& tflite_tensor,
                                      Tensor& output_tensor) {
  auto output_tensor_view = output_tensor.GetCpuWriteView();
  T* output_tensor_buffer = output_tensor_view.buffer<T>();
  RET_CHECK(output_tensor_buffer) << "Output tensor buffer is null.";
  RET_CHECK_EQ(tflite_tensor.type, tflite::typeToTfLiteType<T>())
          .SetCode(absl::StatusCode::kInvalidArgument)
      << "TfLite tensor type and requested output type do not match.";
  const Tensor::ElementType output_tensor_type = output_tensor.element_type();
  RET_CHECK(output_tensor_type == tflite_tensor.type)
          .SetCode(absl::StatusCode::kInvalidArgument)
      << "Output and TfLiteTensor types do not match";
  const void* local_tensor_buffer = tflite_tensor.data.raw;
  RET_CHECK(local_tensor_buffer) << "TfLiteTensor tensor buffer is null.";
  if (!TfLiteIntArrayEqualsArray(tflite_tensor.dims,
                                 output_tensor.shape().dims.size(),
                                 output_tensor.shape().dims.data())) {
    return absl::InvalidArgumentError(
        absl::StrCat("TfLiteTensor and Tensor shape do not match: ",
                     GetTfLiteTensorDebugInfo(tflite_tensor), " vs. ",
                     GetMpTensorDebugInfo(output_tensor)));
  }

  std::memcpy(output_tensor_buffer, local_tensor_buffer, output_tensor.bytes());
  return absl::OkStatus();
}

template <>
absl::Status CopyTfLiteTensorToTensor<char>(const TfLiteTensor& tflite_tensor,
                                            Tensor& output_tensor) {
  auto output_tensor_view = output_tensor.GetCpuWriteView();
  char* output_tensor_buffer = output_tensor_view.buffer<char>();
  RET_CHECK(output_tensor_buffer) << "Output tensor buffer is null.";
  RET_CHECK_EQ(tflite_tensor.type, kTfLiteString)
          .SetCode(absl::StatusCode::kInvalidArgument)
      << "TfLiteTensor type and requested output type do not match.";
  const Tensor::ElementType output_tensor_type = output_tensor.element_type();
  RET_CHECK(output_tensor_type == Tensor::ElementType::kChar)
          .SetCode(absl::StatusCode::kInvalidArgument)
      << "Output and TfLiteTensor types do not match";

  // Only one string expected.
  RET_CHECK_EQ(tflite::GetStringCount(&tflite_tensor), 1);
  const tflite::StringRef string_ref = tflite::GetString(&tflite_tensor, 0);
  std::string str(string_ref.str, string_ref.len);
  RET_CHECK(str.size() == output_tensor.shape().num_elements())
          .SetCode(absl::StatusCode::kInvalidArgument)
      << absl::StrFormat(
             "TfLiteTensor and Tensor shape do not match: %d vs [%s]",
             str.size(), absl::StrJoin(output_tensor.shape().dims, ","));
  std::memcpy(output_tensor_buffer, str.data(), str.size());
  return absl::OkStatus();
}

}  // namespace

int GetXnnpackNumThreads(
    bool opts_has_delegate,
    const mediapipe::InferenceCalculatorOptions::Delegate& opts_delegate) {
  static constexpr int kDefaultNumThreads = -1;
  if (opts_has_delegate && opts_delegate.has_xnnpack() &&
      opts_delegate.xnnpack().num_threads() != kDefaultNumThreads) {
    return opts_delegate.xnnpack().num_threads();
  }
  return GetXnnpackDefaultNumThreads();
}

absl::Status CopyCpuInputIntoTfLiteTensor(const Tensor& input_tensor,
                                          TfLiteTensor& tflite_tensor) {
  const TfLiteType interpreter_tensor_type = tflite_tensor.type;
  const Tensor::ElementType input_tensor_type = input_tensor.element_type();
  RET_CHECK(input_tensor_type == interpreter_tensor_type)
          .SetCode(absl::StatusCode::kInvalidArgument)
      << "Input and interpreter tensor type do not match.";
  switch (interpreter_tensor_type) {
    case TfLiteType::kTfLiteFloat16:
    case TfLiteType::kTfLiteFloat32: {
      MP_RETURN_IF_ERROR(
          CopyTensorToTfLiteTensor<float>(input_tensor, tflite_tensor));
      break;
    }
    case TfLiteType::kTfLiteUInt8: {
      MP_RETURN_IF_ERROR(
          CopyTensorToTfLiteTensor<uint8_t>(input_tensor, tflite_tensor));
      break;
    }
    case TfLiteType::kTfLiteInt8: {
      MP_RETURN_IF_ERROR(
          CopyTensorToTfLiteTensor<int8_t>(input_tensor, tflite_tensor));
      break;
    }
    case TfLiteType::kTfLiteInt32: {
      MP_RETURN_IF_ERROR(
          CopyTensorToTfLiteTensor<int32_t>(input_tensor, tflite_tensor));
      break;
    }
    case TfLiteType::kTfLiteInt64: {
      MP_RETURN_IF_ERROR(
          CopyTensorToTfLiteTensor<int64_t>(input_tensor, tflite_tensor));
      break;
    }
    case TfLiteType::kTfLiteString: {
      MP_RETURN_IF_ERROR(
          CopyTensorToTfLiteTensor<char>(input_tensor, tflite_tensor));
      break;
    }
    case TfLiteType::kTfLiteBool: {
      MP_RETURN_IF_ERROR(
          CopyTensorToTfLiteTensor<bool>(input_tensor, tflite_tensor));
      break;
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported input data type: ", input_tensor_type));
  }
  return absl::OkStatus();
}

absl::Status CopyTfLiteTensorIntoCpuOutput(const TfLiteTensor& tflite_tensor,
                                           Tensor& output_tensor) {
  const TfLiteType tflite_tensor_type = tflite_tensor.type;
  switch (tflite_tensor_type) {
    case TfLiteType::kTfLiteFloat16:
    case TfLiteType::kTfLiteFloat32: {
      MP_RETURN_IF_ERROR(
          CopyTfLiteTensorToTensor<float>(tflite_tensor, output_tensor));
      break;
    }
    case TfLiteType::kTfLiteUInt8: {
      MP_RETURN_IF_ERROR(
          CopyTfLiteTensorToTensor<uint8_t>(tflite_tensor, output_tensor));
      break;
    }
    case TfLiteType::kTfLiteInt8: {
      MP_RETURN_IF_ERROR(
          CopyTfLiteTensorToTensor<int8_t>(tflite_tensor, output_tensor));
      break;
    }
    case TfLiteType::kTfLiteInt32: {
      MP_RETURN_IF_ERROR(
          CopyTfLiteTensorToTensor<int>(tflite_tensor, output_tensor));
      break;
    }
    case TfLiteType::kTfLiteInt64: {
      MP_RETURN_IF_ERROR(
          CopyTfLiteTensorToTensor<int64_t>(tflite_tensor, output_tensor));
      break;
    }
    case TfLiteType::kTfLiteString: {
      MP_RETURN_IF_ERROR(
          CopyTfLiteTensorToTensor<char>(tflite_tensor, output_tensor));
      break;
    }
    case TfLiteType::kTfLiteBool: {
      MP_RETURN_IF_ERROR(
          CopyTfLiteTensorToTensor<bool>(tflite_tensor, output_tensor));
      break;
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported output data type: ", tflite_tensor_type));
  }
  return absl::OkStatus();
}

absl::StatusOr<Tensor> ConvertTfLiteTensorToTensor(
    const TfLiteTensor& tflite_tensor) {
  Tensor::Shape shape(
      std::vector<int>(tflite_tensor.dims->data,
                       tflite_tensor.dims->data + tflite_tensor.dims->size));
  switch (tflite_tensor.type) {
    case TfLiteType::kTfLiteFloat16:
    case TfLiteType::kTfLiteFloat32: {
      Tensor output_tensor(Tensor::ElementType::kFloat32, shape);
      MP_RETURN_IF_ERROR(
          CopyTfLiteTensorToTensor<float>(tflite_tensor, output_tensor));
      return output_tensor;
    }
    case TfLiteType::kTfLiteInt32: {
      Tensor output_tensor(Tensor::ElementType::kInt32, shape);
      MP_RETURN_IF_ERROR(
          CopyTfLiteTensorToTensor<int32_t>(tflite_tensor, output_tensor));
      return output_tensor;
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported output data type: ", tflite_tensor.type));
  }
}

absl::StatusOr<Tensor> CreateTensorWithTfLiteTensorSpecs(
    const TfLiteTensor& reference_tflite_tensor, MemoryManager* memory_manager,
    int alignment) {
  Tensor::Shape shape;
  if (reference_tflite_tensor.dims->size > 0) {
    shape = std::vector<int>(reference_tflite_tensor.dims->data,
                             reference_tflite_tensor.dims->data +
                                 reference_tflite_tensor.dims->size);
  } else {
    ABSL_LOG(ERROR) << "TfLite tensor with empty dimensions: "
                    << GetTfLiteTensorDebugInfo(reference_tflite_tensor)
                    << ", likely due to malformed model signature.";
    // TODO b/362911393 - remove hack once hades tests are fixed.
    if (reference_tflite_tensor.type == TfLiteType::kTfLiteUInt8) {
      shape = std::vector<int>(
          {1, static_cast<int>(reference_tflite_tensor.bytes)});
    }
  }

  switch (reference_tflite_tensor.type) {
    case TfLiteType::kTfLiteFloat16:
    case TfLiteType::kTfLiteFloat32:
      return Tensor(Tensor::ElementType::kFloat32, shape,
                    Tensor::QuantizationParameters{
                        reference_tflite_tensor.params.scale,
                        reference_tflite_tensor.params.zero_point},
                    memory_manager, alignment);
    case TfLiteType::kTfLiteUInt8:
      return Tensor(Tensor::ElementType::kUInt8, shape,
                    Tensor::QuantizationParameters{
                        reference_tflite_tensor.params.scale,
                        reference_tflite_tensor.params.zero_point},
                    memory_manager, alignment);
    case TfLiteType::kTfLiteInt8:
      return Tensor(Tensor::ElementType::kInt8, shape,
                    Tensor::QuantizationParameters{
                        reference_tflite_tensor.params.scale,
                        reference_tflite_tensor.params.zero_point},
                    memory_manager, alignment);
    case TfLiteType::kTfLiteInt32:
      return Tensor(Tensor::ElementType::kInt32, shape,
                    Tensor::QuantizationParameters{
                        reference_tflite_tensor.params.scale,
                        reference_tflite_tensor.params.zero_point},
                    memory_manager, alignment);
    case TfLiteType::kTfLiteBool:
      return Tensor(Tensor::ElementType::kBool, shape,
                    Tensor::QuantizationParameters{1.0f, 0}, memory_manager,
                    alignment);
    case TfLiteType::kTfLiteString:
      // No current use-case for allocating TfLiteTensors with string type.
    default:
      break;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported output tensor type:",
                   TfLiteTypeGetName(reference_tflite_tensor.type)));
}

}  // namespace mediapipe
