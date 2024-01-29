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

#include "mediapipe/calculators/tensor/inference_calculator_utils.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"

namespace mediapipe {
namespace {

using ElementType = ::mediapipe::Tensor::ElementType;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::tflite::Interpreter;

// Adds a tensor of certain type and size inside the interpreter, and update
// the tensor index.
void AddInterpreterInput(TfLiteType type, int size, int& tensor_index,
                         bool allocate_tensor, Interpreter& interpreter) {
  ABSL_CHECK_EQ(interpreter.AddTensors(1, &tensor_index), kTfLiteOk);
  TfLiteQuantizationParams quant;
  interpreter.SetTensorParametersReadWrite(tensor_index, type, "", {size},
                                           quant);
  interpreter.SetInputs({tensor_index});
  ABSL_CHECK_EQ(interpreter.tensor(interpreter.inputs()[tensor_index])->type,
                type);
  if (allocate_tensor) {
    ABSL_CHECK_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  }
}

template <typename T>
std::vector<T> TfLiteTensorData(const Interpreter& interpreter,
                                int tensor_index) {
  const TfLiteTensor* tensor =
      interpreter.tensor(interpreter.inputs()[tensor_index]);
  const T* tensor_ptr = reinterpret_cast<T*>(tensor->data.data);
  ABSL_CHECK_NE(tensor_ptr, nullptr);
  size_t tensor_size = tensor->bytes / sizeof(T);
  return std::vector<T>(tensor_ptr, tensor_ptr + tensor_size);
}

TEST(InferenceCalculatorUtilsTest,
     CopyCpuInputIntoInterpreterTensorWorksCorrectlyForInt32) {
  tflite::Interpreter interpreter;
  int tensor_index, tensor_len = 4;
  AddInterpreterInput(kTfLiteInt32, tensor_len, tensor_index,
                      /*allocate_tensor=*/true, interpreter);
  std::vector<int32_t> values{1, 2, 3, 4};
  int values_len = values.size();
  Tensor tensor(ElementType::kInt32, Tensor::Shape({values_len}));
  std::memcpy(tensor.GetCpuWriteView().buffer<int32_t>(), values.data(),
              values_len * sizeof(int32_t));
  MP_EXPECT_OK(
      CopyCpuInputIntoInterpreterTensor(tensor, interpreter, tensor_index));
  EXPECT_THAT(TfLiteTensorData<int32_t>(interpreter, tensor_index),
              ElementsAreArray(values));
}

TEST(InferenceCalculatorUtilsTest,
     CopyCpuInputIntoInterpreterTensorWorksCorrectlyForFloat32) {
  tflite::Interpreter interpreter;
  int tensor_index, tensor_len = 4;
  AddInterpreterInput(kTfLiteFloat32, tensor_len, tensor_index,
                      /*allocate_tensor=*/true, interpreter);
  std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f};
  int values_len = values.size();
  Tensor tensor(ElementType::kFloat32, Tensor::Shape({values_len}));
  std::memcpy(tensor.GetCpuWriteView().buffer<float>(), values.data(),
              values_len * sizeof(float));
  MP_EXPECT_OK(
      CopyCpuInputIntoInterpreterTensor(tensor, interpreter, tensor_index));
  EXPECT_THAT(TfLiteTensorData<float>(interpreter, tensor_index),
              ElementsAreArray(values));
}

TEST(InferenceCalculatorUtilsTest,
     CopyCpuInputIntoInterpreterTensorTypeMismatch) {
  tflite::Interpreter interpreter;
  int tensor_index, tensor_len = 4;
  AddInterpreterInput(kTfLiteInt32, tensor_len, tensor_index,
                      /*allocate_tensor=*/true, interpreter);
  std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f};
  int values_len = values.size();
  Tensor tensor(ElementType::kFloat32, Tensor::Shape({values_len}));
  std::memcpy(tensor.GetCpuWriteView().buffer<float>(), values.data(),
              values_len * sizeof(float));
  absl::Status status =
      CopyCpuInputIntoInterpreterTensor(tensor, interpreter, tensor_index);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Input and interpreter tensor type mismatch:"));
}

TEST(InferenceCalculatorUtilsTest,
     CopyCpuInputIntoInterpreterTensorSizeMismatch) {
  tflite::Interpreter interpreter;
  int tensor_index, tensor_len = 5;
  AddInterpreterInput(kTfLiteFloat32, tensor_len, tensor_index,
                      /*allocate_tensor=*/true, interpreter);
  std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f};
  int values_len = values.size();
  Tensor tensor(ElementType::kFloat32, Tensor::Shape({values_len}));
  std::memcpy(tensor.GetCpuWriteView().buffer<float>(), values.data(),
              values_len * sizeof(float));
  absl::Status status =
      CopyCpuInputIntoInterpreterTensor(tensor, interpreter, tensor_index);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Interpreter's input size do not match the input "
                        "tensor's size for index"));
}

TEST(InferenceCalculatorUtilsTest,
     CopyCpuInputIntoInterpreterTensorNullBuffer) {
  tflite::Interpreter interpreter;
  int tensor_index, tensor_len = 4;
  // Make TFLite interpreter's buffer null.
  AddInterpreterInput(kTfLiteFloat32, tensor_len, tensor_index,
                      /*allocate_tensor=*/false, interpreter);
  std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f};
  int values_len = values.size();
  Tensor tensor(ElementType::kFloat32, Tensor::Shape({values_len}));
  std::memcpy(tensor.GetCpuWriteView().buffer<float>(), values.data(),
              values_len * sizeof(float));
  absl::Status status =
      CopyCpuInputIntoInterpreterTensor(tensor, interpreter, tensor_index);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Interpreter's input tensor buffer is null"));
}

TEST(InferenceCalculatorUtilsTest,
     CopyCpuInputIntoInterpreterTensorUnsupportedType) {
  tflite::Interpreter interpreter;
  int tensor_index, tensor_len = 4;
  AddInterpreterInput(kTfLiteUInt8, tensor_len, tensor_index,
                      /*allocate_tensor=*/true, interpreter);
  std::vector<uint8_t> values{1, 2, 3, 4};
  int values_len = values.size();
  Tensor tensor(ElementType::kUInt8, Tensor::Shape({values_len}));
  std::memcpy(tensor.GetCpuWriteView().buffer<uint8_t>(), values.data(),
              values_len * sizeof(uint8_t));
  absl::Status status =
      CopyCpuInputIntoInterpreterTensor(tensor, interpreter, tensor_index);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(), HasSubstr("Unsupported input data type:"));
}

}  // namespace
}  // namespace mediapipe
