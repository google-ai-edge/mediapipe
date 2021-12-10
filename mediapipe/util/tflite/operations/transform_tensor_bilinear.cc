// Copyright 2021 The MediaPipe Authors.
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

#include "mediapipe/util/tflite/operations/transform_tensor_bilinear.h"

#include "tensorflow/lite/delegates/gpu/common/mediapipe/transform_tensor_bilinear.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mediapipe {
namespace tflite_operations {
namespace {

constexpr int kDataInput0Tensor = 0;
constexpr int kDataInput1Tensor = 1;
constexpr int kOutputTensor = 0;

float DotProduct(const tflite::gpu::float4& l, const tflite::gpu::float4& r) {
  return l.x * r.x + l.y * r.y + l.z * r.z + l.w * r.w;
}
namespace v1 {

inline void TransformTensor(
    const tflite::gpu::TransformTensorBilinearAttributes& params,
    const tflite::RuntimeShape& input0_shape,
    const float* input_data_0,  // data
    const tflite::RuntimeShape& input1_shape,
    const float* input_data_1,  // transformation matrix
    const tflite::RuntimeShape& output_shape, float* output_data) {
  TFLITE_CHECK_EQ(input0_shape.DimensionsCount(), 4);
  TFLITE_CHECK_EQ(output_shape.DimensionsCount(), 4);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_channels = output_shape.Dims(3);

  const int input_height = input0_shape.Dims(1);
  const int input_width = input0_shape.Dims(2);
  const int input_channels = input0_shape.Dims(3);

  tflite::RuntimeShape input_shape_with_batch{/*batch=*/1, input_height,
                                              input_width, input_channels};
  tflite::RuntimeShape output_shape_with_batch{/*batch=*/1, output_height,
                                               output_width, output_channels};

  // Read first two rows of transformation matrix
  tflite::gpu::float4 x_transform(input_data_1[0], input_data_1[1],
                                  input_data_1[2], input_data_1[3]);
  tflite::gpu::float4 y_transform(input_data_1[4], input_data_1[5],
                                  input_data_1[6], input_data_1[7]);

  for (int out_y = 0; out_y < output_height; ++out_y) {
    for (int out_x = 0; out_x < output_width; ++out_x) {
      tflite::gpu::float4 coord(
          static_cast<float>(out_x), static_cast<float>(out_y),
          static_cast<float>(0.0), static_cast<float>(1.0));

      // Transformed coordinates.
      tflite::gpu::float2 tc(DotProduct(x_transform, coord),
                             DotProduct(y_transform, coord));

      bool out_of_bound = tc.x < 0.0 || tc.x > input_width - 1 || tc.y < 0.0 ||
                          tc.y > input_height - 1;

      for (int out_z = 0; out_z < output_channels; ++out_z) {
        float result = 0;
        if (!out_of_bound) {
          // Corners position:
          // q_11 --- q_21
          // ----     ----
          // q_12 --- q_22

          auto ReadValue = [&](int h, int w) -> float {
            return h < 0 || w < 0 || h >= input_height || w >= input_width
                       ? 0
                       : input_data_0[Offset(input_shape_with_batch, 0, h, w,
                                             out_z)];
          };

          float q_11 = ReadValue(floor(tc.y), floor(tc.x));
          float q_21 = ReadValue(floor(tc.y), floor(tc.x) + 1);
          float q_12 = ReadValue(floor(tc.y) + 1, floor(tc.x));
          float q_22 = ReadValue(floor(tc.y) + 1, floor(tc.x) + 1);

          float right_contrib = tc.x - floor(tc.x);
          float lower_contrib = tc.y - floor(tc.y);

          float upper = (1.0 - right_contrib) * q_11 + right_contrib * q_21;
          float lower = (1.0 - right_contrib) * q_12 + right_contrib * q_22;

          result = lower_contrib * lower + (1.0 - lower_contrib) * upper;
        }

        const int out_offset =
            Offset(output_shape_with_batch, 0, out_y, out_x, out_z);

        output_data[out_offset] = result;
      }
    }
  }
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, tflite::NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, tflite::NumOutputs(node), 1);
  const TfLiteTensor* input =
      tflite::GetInput(context, node, kDataInput0Tensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output = tflite::GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_EQ(context, tflite::NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteFloat32);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  tflite::gpu::TransformTensorBilinearAttributes op_params;
  tflite::gpu::BHWC output_shape;
  auto status = tflite::gpu::ParseTransformTensorBilinearV1Attributes(
      node->custom_initial_data, node->custom_initial_data_size, &op_params,
      &output_shape);
  if (!status.ok()) {
    context->ReportError(context, status.message().data());
    return kTfLiteError;
  }

  const TfLiteTensor* input0 =
      tflite::GetInput(context, node, kDataInput0Tensor);
  TF_LITE_ENSURE(context, input0 != nullptr);
  const TfLiteTensor* input1 =
      tflite::GetInput(context, node, kDataInput1Tensor);
  TF_LITE_ENSURE(context, input1 != nullptr);
  TfLiteTensor* output = tflite::GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TransformTensor(
      op_params, tflite::GetTensorShape(input0),
      tflite::GetTensorData<float>(input0), tflite::GetTensorShape(input1),
      tflite::GetTensorData<float>(input1), tflite::GetTensorShape(output),
      tflite::GetTensorData<float>(output));
  return kTfLiteOk;
}
}  // namespace v1

namespace v2 {

inline void TransformTensorBilinearV2(
    const tflite::gpu::TransformTensorBilinearAttributes& params,
    const tflite::RuntimeShape& input0_shape,
    const float* input_data_0,  // data
    const tflite::RuntimeShape& input1_shape,
    const float* input_data_1,  // transformation matrix
    const tflite::RuntimeShape& output_shape, float* output_data) {
  TFLITE_CHECK_EQ(input0_shape.DimensionsCount(), 4);
  TFLITE_CHECK_EQ(output_shape.DimensionsCount(), 4);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_channels = output_shape.Dims(3);

  const int input_height = input0_shape.Dims(1);
  const int input_width = input0_shape.Dims(2);
  const int input_channels = input0_shape.Dims(3);

  tflite::RuntimeShape input_shape_with_batch{/*batch=*/1, input_height,
                                              input_width, input_channels};
  tflite::RuntimeShape output_shape_with_batch{/*batch=*/1, output_height,
                                               output_width, output_channels};

  // Read first two rows of transformation matrix
  tflite::gpu::float4 x_transform(input_data_1[0], input_data_1[1],
                                  input_data_1[2], input_data_1[3]);
  tflite::gpu::float4 y_transform(input_data_1[4], input_data_1[5],
                                  input_data_1[6], input_data_1[7]);

  // Align corners correction: T -> S * ( T * A ), where T is a
  // transformation matrix, and subtruction and addition matrices are:
  // S            A
  // 1 0 0 -0.5   1 0 0 0.5
  // 0 1 0 -0.5   0 1 0 0.5
  // 0 0 1 0      0 0 1 0
  // 0 0 0 1      0 0 0 1
  // Transformation matrix column 3 and rows 3, 4 are identity, which makes
  // the final formula pretty simple and easy to get if doing a manual
  // multiuplication.
  x_transform[3] += x_transform[0] * 0.5 + x_transform[1] * 0.5 - 0.5;
  y_transform[3] += y_transform[0] * 0.5 + y_transform[1] * 0.5 - 0.5;

  for (int out_y = 0; out_y < output_height; ++out_y) {
    for (int out_x = 0; out_x < output_width; ++out_x) {
      tflite::gpu::float4 coord(
          static_cast<float>(out_x), static_cast<float>(out_y),
          static_cast<float>(0.0), static_cast<float>(1.0));

      // Transformed coordinates.
      tflite::gpu::float2 tc(DotProduct(x_transform, coord),
                             DotProduct(y_transform, coord));

      bool out_of_bound = tc.x < 0.0 || tc.x > input_width - 1 || tc.y < 0.0 ||
                          tc.y > input_height - 1;

      for (int out_z = 0; out_z < output_channels; ++out_z) {
        float result = 0;
        if (!out_of_bound) {
          // Corners position:
          // q_11 --- q_21
          // ----     ----
          // q_12 --- q_22

          auto ReadValue = [&](int h, int w) -> float {
            return h < 0 || w < 0 || h >= input_height || w >= input_width
                       ? 0
                       : input_data_0[Offset(input_shape_with_batch, 0, h, w,
                                             out_z)];
          };

          float q_11 = ReadValue(floor(tc.y), floor(tc.x));
          float q_21 = ReadValue(floor(tc.y), floor(tc.x) + 1);
          float q_12 = ReadValue(floor(tc.y) + 1, floor(tc.x));
          float q_22 = ReadValue(floor(tc.y) + 1, floor(tc.x) + 1);

          float right_contrib = tc.x - floor(tc.x);
          float lower_contrib = tc.y - floor(tc.y);

          float upper = (1.0 - right_contrib) * q_11 + right_contrib * q_21;
          float lower = (1.0 - right_contrib) * q_12 + right_contrib * q_22;

          result = lower_contrib * lower + (1.0 - lower_contrib) * upper;
        }

        const int out_offset =
            Offset(output_shape_with_batch, 0, out_y, out_x, out_z);

        output_data[out_offset] = result;
      }
    }
  }
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, tflite::NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, tflite::NumOutputs(node), 1);
  const TfLiteTensor* input =
      tflite::GetInput(context, node, kDataInput0Tensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output = tflite::GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_EQ(context, tflite::NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteFloat32);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  tflite::gpu::TransformTensorBilinearAttributes op_params;
  tflite::gpu::BHWC output_shape;
  auto status = tflite::gpu::ParseTransformTensorBilinearV2Attributes(
      node->custom_initial_data, node->custom_initial_data_size, &op_params,
      &output_shape);
  if (!status.ok()) {
    context->ReportError(context, status.message().data());
    return kTfLiteError;
  }

  const TfLiteTensor* input0 =
      tflite::GetInput(context, node, kDataInput0Tensor);
  TF_LITE_ENSURE(context, input0 != nullptr);
  const TfLiteTensor* input1 =
      tflite::GetInput(context, node, kDataInput1Tensor);
  TF_LITE_ENSURE(context, input1 != nullptr);
  TfLiteTensor* output = tflite::GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TransformTensorBilinearV2(
      op_params, tflite::GetTensorShape(input0),
      tflite::GetTensorData<float>(input0), tflite::GetTensorShape(input1),
      tflite::GetTensorData<float>(input1), tflite::GetTensorShape(output),
      tflite::GetTensorData<float>(output));
  return kTfLiteOk;
}
}  // namespace v2

}  // namespace

TfLiteRegistration* RegisterTransformTensorBilinearV1() {
  static TfLiteRegistration reg = {
      /*.init=*/nullptr,
      /*.free=*/nullptr,
      /*.prepare=*/v1::Prepare,
      /*.invoke=*/v1::Eval,
      /*.profiling_string=*/nullptr,
      /*.builtin_code=*/tflite::BuiltinOperator_CUSTOM,
      /*.custom_name=*/"TransformTensor",
      /*.version=*/1,
  };
  return &reg;
}

TfLiteRegistration* RegisterTransformTensorBilinearV2() {
  static TfLiteRegistration reg = {
      /*.init=*/nullptr,
      /*.free=*/nullptr,
      /*.prepare=*/v2::Prepare,
      /*.invoke=*/v2::Eval,
      /*.profiling_string=*/nullptr,
      /*.builtin_code=*/tflite::BuiltinOperator_CUSTOM,
      /*.custom_name=*/"TransformTensorBilinear",
      /*.version=*/2,
  };
  return &reg;
}

}  // namespace tflite_operations
}  // namespace mediapipe
