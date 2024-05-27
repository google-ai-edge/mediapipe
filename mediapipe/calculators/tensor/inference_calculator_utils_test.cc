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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/cast_test_common.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/util.h"

namespace mediapipe {
namespace {

using ElementType = ::mediapipe::Tensor::ElementType;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::tflite::Interpreter;
using ::tflite::TensorType_FLOAT32;
using ::tflite::TensorType_INT32;

// Adds an input tensor of certain type and size inside the interpreter, and
// update the tensor index.
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

// Adds an output tensor of certain type and size inside the interpreter, and
// update the tensor index.
void AddInterpreterOutput(TfLiteType type, int size, int& tensor_index,
                          bool allocate_tensor, Interpreter& interpreter) {
  ABSL_CHECK_EQ(interpreter.AddTensors(1, &tensor_index), kTfLiteOk);
  TfLiteQuantizationParams quant;
  interpreter.SetTensorParametersReadWrite(tensor_index, type, "", {size},
                                           quant);
  interpreter.SetOutputs({tensor_index});
  ABSL_CHECK_EQ(interpreter.tensor(tensor_index)->type, type);
  if (allocate_tensor) {
    ABSL_CHECK_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  }
}

template <typename T>
std::vector<T> TfLiteInputTensorData(const Interpreter& interpreter,
                                     int tensor_index) {
  ABSL_CHECK_LT(tensor_index, interpreter.inputs().size());
  const TfLiteTensor* tensor =
      interpreter.tensor(interpreter.inputs()[tensor_index]);
  const T* tensor_ptr = reinterpret_cast<T*>(tensor->data.data);
  ABSL_CHECK_NE(tensor_ptr, nullptr);
  size_t tensor_size = tensor->bytes / sizeof(T);
  return std::vector<T>(tensor_ptr, tensor_ptr + tensor_size);
}

template <>
std::vector<char> TfLiteInputTensorData<char>(const Interpreter& interpreter,
                                              int tensor_index) {
  ABSL_CHECK_LT(tensor_index, interpreter.inputs().size());
  const TfLiteTensor* tensor =
      interpreter.tensor(interpreter.inputs()[tensor_index]);
  int num_strings = tflite::GetStringCount(tensor);
  ABSL_CHECK_EQ(num_strings, 1) << "Only one string expected inside tensor";
  const tflite::StringRef string_ref = tflite::GetString(tensor, 0);
  std::string str(string_ref.str, string_ref.len);
  return std::vector<char>(str.begin(), str.end());
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
  EXPECT_THAT(TfLiteInputTensorData<int32_t>(interpreter, tensor_index),
              ElementsAreArray(values));
}

TEST(InferenceCalculatorUtilsTest,
     CopyCpuInputIntoInterpreterTensorWorksCorrectlyForInt64) {
  tflite::Interpreter interpreter;
  int tensor_index, tensor_len = 4;
  AddInterpreterInput(kTfLiteInt64, tensor_len, tensor_index,
                      /*allocate_tensor=*/true, interpreter);
  std::vector<int64_t> values{1, 2, 3, 4};
  int values_len = values.size();
  Tensor tensor(ElementType::kInt64, Tensor::Shape({values_len}));
  std::memcpy(tensor.GetCpuWriteView().buffer<int64_t>(), values.data(),
              values_len * sizeof(int64_t));
  MP_EXPECT_OK(
      CopyCpuInputIntoInterpreterTensor(tensor, interpreter, tensor_index));
  EXPECT_THAT(TfLiteInputTensorData<int64_t>(interpreter, tensor_index),
              ElementsAreArray(values));
}

TEST(InferenceCalculatorUtilsTest,
     CopyCpuInputIntoInterpreterTensorWorksCorrectlyForUInt8) {
  tflite::Interpreter interpreter;
  int tensor_index, tensor_len = 4;
  AddInterpreterInput(kTfLiteUInt8, tensor_len, tensor_index,
                      /*allocate_tensor=*/true, interpreter);
  std::vector<uint8_t> values{1, 2, 3, 4};
  int values_len = values.size();
  Tensor tensor(ElementType::kUInt8, Tensor::Shape({values_len}));
  std::memcpy(tensor.GetCpuWriteView().buffer<uint8_t>(), values.data(),
              values_len * sizeof(uint8_t));
  MP_EXPECT_OK(
      CopyCpuInputIntoInterpreterTensor(tensor, interpreter, tensor_index));
  EXPECT_THAT(TfLiteInputTensorData<uint8_t>(interpreter, tensor_index),
              ElementsAreArray(values));
}

TEST(InferenceCalculatorUtilsTest,
     CopyCpuInputIntoInterpreterTensorWorksCorrectlyForInt8) {
  tflite::Interpreter interpreter;
  int tensor_index, tensor_len = 4;
  AddInterpreterInput(kTfLiteInt8, tensor_len, tensor_index,
                      /*allocate_tensor=*/true, interpreter);
  std::vector<int8_t> values{-1, -2, 3, 4};
  int values_len = values.size();
  Tensor tensor(ElementType::kInt8, Tensor::Shape({values_len}));
  std::memcpy(tensor.GetCpuWriteView().buffer<int8_t>(), values.data(),
              values_len * sizeof(int8_t));
  MP_EXPECT_OK(
      CopyCpuInputIntoInterpreterTensor(tensor, interpreter, tensor_index));
  EXPECT_THAT(TfLiteInputTensorData<int8_t>(interpreter, tensor_index),
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
  EXPECT_THAT(TfLiteInputTensorData<float>(interpreter, tensor_index),
              ElementsAreArray(values));
}

TEST(InferenceCalculatorUtilsTest,
     CopyCpuInputIntoInterpreterTensorWorksCorrectlyForString) {
  tflite::Interpreter interpreter;
  int tensor_index, tensor_len = 4;
  AddInterpreterInput(kTfLiteString, tensor_len, tensor_index,
                      /*allocate_tensor=*/true, interpreter);
  std::vector<char> values{'a', 'b', 'c', 'd'};
  int values_len = values.size();
  Tensor tensor(ElementType::kChar, Tensor::Shape({values_len}));
  std::memcpy(tensor.GetCpuWriteView().buffer<char>(), values.data(),
              values_len * sizeof(char));
  MP_EXPECT_OK(
      CopyCpuInputIntoInterpreterTensor(tensor, interpreter, tensor_index));
  EXPECT_THAT(TfLiteInputTensorData<char>(interpreter, tensor_index),
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
              HasSubstr("Input and interpreter tensor type do not match"));
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
              HasSubstr("TfLiteTensor and Tensor sizes do not match"));
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
  EXPECT_THAT(status.message(), HasSubstr("TfLiteTensor data is null"));
}

TEST(InferenceCalculatorUtilsTest,
     CopyCpuInputIntoInterpreterTensorUnsupportedType) {
  tflite::Interpreter interpreter;
  int tensor_index, tensor_len = 4;

  // Add TFLite interpreter's input tensor.
  ABSL_CHECK_EQ(interpreter.AddTensors(1, &tensor_index), kTfLiteOk);
  TfLiteQuantizationParams quant;
  interpreter.SetTensorParametersReadWrite(tensor_index, kTfLiteFloat32, "",
                                           {tensor_len}, quant);
  interpreter.SetInputs({tensor_index});
  // Manually set the input type as NoType, to not trigger type mismatch but
  // trigger unsupported types.
  interpreter.tensor(interpreter.inputs()[tensor_index])->type = kTfLiteNoType;

  std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f};
  int values_len = values.size();
  Tensor tensor(ElementType::kNone, Tensor::Shape({values_len}));
  absl::Status status =
      CopyCpuInputIntoInterpreterTensor(tensor, interpreter, tensor_index);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(), HasSubstr("Unsupported input data type:"));
}

TEST(InferenceCalculatorUtilsTest,
     CopyInterpreterTensorIntoCpuOutputWorksCorrectlyForFloat32) {
  std::vector<float> values{100.f, 200.f, 300.f, 400.f, 500.f, 600.f};

  tflite::CastOpModel m({TensorType_INT32, {2, 3}},
                        {TensorType_FLOAT32, {2, 3}});
  m.PopulateTensor<int32_t>(m.input(), {100, 200, 300, 400, 500, 600});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()), ElementsAreArray(values));

  Tensor tensor(ElementType::kFloat32, Tensor::Shape({2, 3}));
  MP_EXPECT_OK(CopyTfLiteTensorIntoCpuOutput(*m.GetOutputTensor(0), tensor));
  EXPECT_THAT(absl::MakeConstSpan(tensor.GetCpuReadView().buffer<float>(),
                                  tensor.shape().num_elements()),
              ElementsAreArray(values));
}

TEST(InferenceCalculatorUtilsTest,
     CopyInterpreterTensorIntoCpuOutputWorksCorrectlyForString) {
  std::vector<char> values{'a', 'b', 'c', 'd'};
  int values_len = values.size();
  Tensor tensor(ElementType::kChar, Tensor::Shape({values_len}));

  Interpreter interpreter;
  int tensor_index, tensor_len = 4;
  AddInterpreterOutput(kTfLiteString, tensor_len, tensor_index,
                       /*allocate_tensor=*/true, interpreter);
  // Copy the chars as a string into interpreter's output tensor.
  tflite::DynamicBuffer dynamic_buffer;
  dynamic_buffer.AddString(values.data(), values.size());
  dynamic_buffer.WriteToTensorAsVector(
      interpreter.tensor(interpreter.outputs()[tensor_index]));
  MP_EXPECT_OK(
      CopyInterpreterTensorIntoCpuOutput(interpreter, tensor_index, tensor));
  EXPECT_THAT(absl::MakeConstSpan(tensor.GetCpuReadView().buffer<char>(),
                                  tensor.shape().num_elements()),
              ElementsAreArray(values));
}

TEST(InferenceCalculatorUtilsTest,
     CopyInterpreterTensorIntoCpuOutputTypeMismatch) {
  std::vector<float> values{100.f, 200.f, 300.f, 400.f, 500.f, 600.f};

  tflite::CastOpModel m({TensorType_INT32, {2, 3}},
                        {TensorType_FLOAT32, {2, 3}});
  m.PopulateTensor<int32_t>(m.input(), {100, 200, 300, 400, 500, 600});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()), ElementsAreArray(values));

  Tensor tensor(ElementType::kInt32, Tensor::Shape({2, 3}));
  absl::Status status =
      CopyTfLiteTensorIntoCpuOutput(*m.GetOutputTensor(0), tensor);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Output and TfLiteTensor types do not match"));
}

TEST(InferenceCalculatorUtilsTest,
     CopyInterpreterTensorIntoCpuOutputSizeMismatch) {
  std::vector<float> values{100.f, 200.f, 300.f, 400.f, 500.f, 600.f};

  tflite::CastOpModel m({TensorType_INT32, {2, 3}},
                        {TensorType_FLOAT32, {2, 3}});
  m.PopulateTensor<int32_t>(m.input(), {100, 200, 300, 400, 500, 600});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()), ElementsAreArray(values));

  Tensor tensor(ElementType::kFloat32, Tensor::Shape({4}));
  absl::Status status =
      CopyTfLiteTensorIntoCpuOutput(*m.GetOutputTensor(0), tensor);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("TfLiteTensor and Tensor shape do not match"));
}

TEST(InferenceCalculatorUtilsTest,
     CopyInterpreterTensorIntoCpuOutputNullBuffer) {
  tflite::Interpreter interpreter;
  int tensor_index, tensor_len = 4;
  // Make TFLite interpreter's buffer null.
  AddInterpreterOutput(kTfLiteFloat32, tensor_len, tensor_index,
                       /*allocate_tensor=*/false, interpreter);
  std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f};
  int values_len = values.size();
  Tensor tensor(ElementType::kFloat32, Tensor::Shape({values_len}));
  std::memcpy(tensor.GetCpuWriteView().buffer<float>(), values.data(),
              values_len * sizeof(float));
  absl::Status status =
      CopyInterpreterTensorIntoCpuOutput(interpreter, tensor_index, tensor);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("TfLiteTensor tensor buffer is null"));
}

TEST(InferenceCalculatorUtilsTest,
     CopyInterpreterTensorIntoCpuOutputUnsupportedType) {
  tflite::Interpreter interpreter;
  int tensor_index, tensor_len = 4;

  // Add TFLite interpreter's output tensor.
  ABSL_CHECK_EQ(interpreter.AddTensors(1, &tensor_index), kTfLiteOk);
  TfLiteQuantizationParams quant;
  interpreter.SetTensorParametersReadWrite(tensor_index, kTfLiteUInt8, "",
                                           {tensor_len}, quant);
  interpreter.SetOutputs({tensor_index});
  // Manually set the output type as NoType, to not trigger type mismatch but
  // trigger unsupported types.
  interpreter.tensor(interpreter.outputs()[tensor_index])->type = kTfLiteNoType;

  std::vector<uint8_t> values{1, 2, 3, 4};
  int values_len = values.size();
  Tensor tensor(ElementType::kNone, Tensor::Shape({values_len}));
  absl::Status status =
      CopyInterpreterTensorIntoCpuOutput(interpreter, tensor_index, tensor);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(), HasSubstr("Unsupported output data type:"));
}

TEST(InferenceCalculatorUtilsTest, ConvertTfLiteTensorToFloat32) {
  const std::vector<float> expected_values{100.f, 200.f, 300.f,
                                           400.f, 500.f, 600.f};

  tflite::CastOpModel m({TensorType_INT32, {2, 3}},
                        {TensorType_FLOAT32, {2, 3}});
  m.PopulateTensor<int32_t>(m.input(), {100, 200, 300, 400, 500, 600});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray(expected_values));

  MP_ASSERT_OK_AND_ASSIGN(auto tensor,
                          ConvertTfLiteTensorToTensor(*m.GetOutputTensor(0)));
  EXPECT_THAT(absl::MakeConstSpan(tensor.GetCpuReadView().buffer<float>(),
                                  tensor.shape().num_elements()),
              ElementsAreArray(expected_values));
}

TEST(InferenceCalculatorUtilsTest, ShouldSetCustomAllocatorForCpuWriteView) {
  tflite::Interpreter interpreter;
  int tensor_index, tensor_size = 4;
  AddInterpreterInput(kTfLiteInt32, tensor_size, tensor_index,
                      /*allocate_tensor=*/true, interpreter);
  std::vector<int32_t> values{1, 2, 3, 4};
  int values_len = values.size();
  Tensor tensor(ElementType::kInt32, Tensor::Shape({values_len}),
                /*memory_manager=*/nullptr, tflite::kDefaultTensorAlignment);
  {
    auto write_view = tensor.GetCpuWriteView();
    MP_EXPECT_OK(SetTfLiteCustomAllocation<void>(
        interpreter, write_view.buffer<void>(), tensor.bytes(), tensor_index));
    interpreter.AllocateTensors();
    const TfLiteTensor* tf_lite_tensor =
        interpreter.tensor(interpreter.inputs()[tensor_index]);
    int32_t* tensor_ptr = reinterpret_cast<int32_t*>(tf_lite_tensor->data.data);
    std::copy(values.begin(), values.end(), tensor_ptr);

    EXPECT_THAT(TfLiteInputTensorData<int32_t>(interpreter, tensor_index),
                ElementsAreArray(values));
  }
  auto read_view = tensor.GetCpuReadView();
  const int32_t* tensor_ptr = read_view.buffer<int32_t>();
  for (int i = 0; i < values.size(); ++i) {
    EXPECT_EQ(tensor_ptr[i], values[i]);
  }
}

TEST(InferenceCalculatorUtilsTest, ShouldSetCustomAllocatorForCpuReadView) {
  tflite::Interpreter interpreter;
  int tensor_index, tensor_size = 4;
  AddInterpreterInput(kTfLiteInt32, tensor_size, tensor_index,
                      /*allocate_tensor=*/true, interpreter);
  std::vector<int32_t> values{1, 2, 3, 4};
  int values_len = values.size();
  Tensor tensor(ElementType::kInt32, Tensor::Shape({values_len}),
                /*memory_manager=*/nullptr, tflite::kDefaultTensorAlignment);
  std::memcpy(tensor.GetCpuWriteView().buffer<int32_t>(), values.data(),
              values_len * sizeof(int32_t));

  auto read_view = tensor.GetCpuReadView();
  MP_EXPECT_OK(SetTfLiteCustomAllocation<const void>(
      interpreter, read_view.buffer<void>(), tensor.bytes(), tensor_index));
  interpreter.AllocateTensors();

  EXPECT_THAT(TfLiteInputTensorData<int32_t>(interpreter, tensor_index),
              ElementsAreArray(values));
}

TEST(InferenceCalculatorUtilsTest, ShouldConfirmTfLiteMemoryAlignment) {
  std::vector<int32_t> values{1, 2, 3, 4};
  int values_len = values.size();
  Tensor tensor(ElementType::kInt32, Tensor::Shape({values_len}),
                /*memory_manager=*/nullptr, tflite::kDefaultTensorAlignment);
  std::memcpy(tensor.GetCpuWriteView().buffer<int32_t>(), values.data(),
              values_len * sizeof(int32_t));
  const auto read_view = tensor.GetCpuReadView();
  EXPECT_TRUE(IsAlignedWithTFLiteDefaultAlignment(read_view.buffer<int32_t>()));
}

TEST(InferenceCalculatorUtilsTest, ShouldNotConfirmTfLiteMemoryAlignment) {
  std::vector<int32_t> values{1, 2, 3, 4};
  int values_len = values.size();
  Tensor tensor(ElementType::kInt32, Tensor::Shape({values_len}),
                /*memory_manager=*/nullptr, tflite::kDefaultTensorAlignment);
  std::memcpy(tensor.GetCpuWriteView().buffer<int32_t>(), values.data(),
              values_len * sizeof(int32_t));
  const auto read_view = tensor.GetCpuReadView();
  EXPECT_FALSE(IsAlignedWithTFLiteDefaultAlignment(read_view.buffer<int32_t>() +
                                                   sizeof(int32_t)));
}

}  // namespace
}  // namespace mediapipe
