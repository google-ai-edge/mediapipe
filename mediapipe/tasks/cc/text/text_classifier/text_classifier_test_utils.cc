/* Copyright 2022 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/tasks/cc/text/text_classifier/text_classifier_test_utils.h"

#include <cstring>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/common.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/portable_type_to_tflitetype.h"
#include "tensorflow/lite/string_util.h"

namespace mediapipe {
namespace tasks {
namespace text {
namespace {

using ::mediapipe::tasks::CreateStatusWithPayload;
using ::tflite::GetInput;
using ::tflite::GetOutput;
using ::tflite::GetString;
using ::tflite::StringRef;

constexpr absl::string_view kInputStr = "hello";
constexpr bool kBooleanData[] = {true, true, false};
constexpr size_t kBooleanDataSize = std::size(kBooleanData);

// Checks and returns type of a tensor, fails if tensor type is not T.
template <typename T>
absl::StatusOr<T*> AssertAndReturnTypedTensor(const TfLiteTensor* tensor) {
  if (!tensor->data.raw) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInternal,
        absl::StrFormat("Tensor (%s) has no raw data.", tensor->name));
  }

  // Checks if data type of tensor is T and returns the pointer casted to T if
  // applicable, returns nullptr if tensor type is not T.
  // See type_to_tflitetype.h for a mapping from plain C++ type to TfLiteType.
  if (tensor->type == tflite::typeToTfLiteType<T>()) {
    return reinterpret_cast<T*>(tensor->data.raw);
  }
  return CreateStatusWithPayload(
      absl::StatusCode::kInternal,
      absl::StrFormat("Type mismatch for tensor %s. Required %d, got %d.",
                      tensor->name, tflite::typeToTfLiteType<T>(),
                      tensor->bytes));
}

// Populates tensor with array of data, fails if data type doesn't match tensor
// type or they don't have the same number of elements.
template <typename T, typename = std::enable_if_t<
                          std::negation_v<std::is_same<T, std::string>>>>
absl::Status PopulateTensor(const T* data, int num_elements,
                            TfLiteTensor* tensor) {
  MP_ASSIGN_OR_RETURN(T * v, AssertAndReturnTypedTensor<T>(tensor));
  size_t bytes = num_elements * sizeof(T);
  if (tensor->bytes != bytes) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInternal,
        absl::StrFormat("tensor->bytes (%d) != bytes (%d)", tensor->bytes,
                        bytes));
  }
  std::memcpy(v, data, bytes);
  return absl::OkStatus();
}

TfLiteStatus PrepareStringToBool(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* output = GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, output != nullptr);
  TfLiteIntArray* dims = TfLiteIntArrayCreate(1);
  dims->data[0] = kBooleanDataSize;
  return context->ResizeTensor(context, output, dims);
}

TfLiteStatus InvokeStringToBool(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input_tensor = GetInput(context, node, 0);
  TF_LITE_ENSURE(context, input_tensor != nullptr);
  StringRef input_str_ref = GetString(input_tensor, 0);
  std::string input_str(input_str_ref.str, input_str_ref.len);
  if (input_str != kInputStr) {
    return kTfLiteError;
  }
  TfLiteTensor* output = GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, output != nullptr);
  TF_LITE_ENSURE(context, PopulateTensor(kBooleanData, 3, output).ok());
  return kTfLiteOk;
}

// This custom op takes a string tensor in and outputs a bool tensor with
// value{true, true, false}, it's used to mimic a real text classification model
// which classifies a string into scores of different categories.
TfLiteRegistration* RegisterStringToBool() {
  // Dummy implementation of custom OP
  // This op takes string as input and outputs bool[]
  static TfLiteRegistration r = {/* init= */ nullptr, /* free= */ nullptr,
                                 /* prepare= */ PrepareStringToBool,
                                 /* invoke= */ InvokeStringToBool};
  return &r;
}
}  // namespace

std::unique_ptr<tflite::MutableOpResolver> CreateCustomResolver() {
  tflite::MutableOpResolver resolver;
  resolver.AddCustom("CUSTOM_OP_STRING_TO_BOOLS", RegisterStringToBool());
  return std::make_unique<tflite::MutableOpResolver>(resolver);
}

}  // namespace text
}  // namespace tasks
}  // namespace mediapipe
