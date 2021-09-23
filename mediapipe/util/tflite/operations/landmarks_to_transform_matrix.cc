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

#include "mediapipe/util/tflite/operations/landmarks_to_transform_matrix.h"

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/mediapipe/landmarks_to_transform_matrix.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/schema/schema_generated.h"

using ::tflite::gpu::BHWC;
using ::tflite::gpu::float2;
using ::tflite::gpu::float3;
using ::tflite::gpu::int2;
using ::tflite::gpu::int3;
using ::tflite::gpu::LandmarksToTransformMatrixV1Attributes;
using ::tflite::gpu::LandmarksToTransformMatrixV2Attributes;

using ::tflite::GetInput;
using ::tflite::GetOutput;
using ::tflite::GetTensorData;
using ::tflite::GetTensorShape;
using ::tflite::NumDimensions;
using ::tflite::NumInputs;
using ::tflite::NumOutputs;
using ::tflite::RuntimeShape;

namespace mediapipe {
namespace tflite_operations {
namespace {

constexpr int kDataInputTensor = 0;
constexpr int kOutputTensor = 0;
constexpr int3 kTensformMatrixShape(1, 4, 4);

float2 Read3DLandmarkXY(const float* data, int idx) {
  float2 result;
  result.x = data[idx * 3];
  result.y = data[idx * 3 + 1];
  return result;
}

float3 Read3DLandmarkXYZ(const float* data, int idx) {
  float3 result;
  result.x = data[idx * 3];
  result.y = data[idx * 3 + 1];
  result.z = data[idx * 3 + 2];
  return result;
}

struct Mat3 {
  Mat3() { data.resize(9); }
  Mat3(float x00, float x01, float x02, float x10, float x11, float x12,
       float x20, float x21, float x22)
      : data{x00, x01, x02, x10, x11, x12, x20, x21, x22} {}

  Mat3 operator*(const Mat3& other) {
    Mat3 result;
    for (int r = 0; r < 3; r++) {
      for (int c = 0; c < 3; c++) {
        float sum = 0;
        for (int k = 0; k < 3; k++) {
          sum += this->Get(r, k) * other.Get(k, c);
        }
        result.Set(r, c, sum);
      }
    }
    return result;
  }
  float3 operator*(const float3& vec) const {
    float3 result;
    for (int r = 0; r < 3; r++) {
      float sum = 0;
      for (int k = 0; k < 3; k++) {
        sum += this->Get(r, k) * vec[k];
      }
      result[r] = sum;
    }
    return result;
  }
  float Get(int x, int y) const { return data[x * 3 + y]; }
  void Set(int x, int y, float val) { data[x * 3 + y] = val; }

  std::vector<float> data;
};

struct Mat4 {
  Mat4() { data.resize(16); }
  Mat4(float x00, float x01, float x02, float x03, float x10, float x11,
       float x12, float x13, float x20, float x21, float x22, float x23,
       float x30, float x31, float x32, float x33)
      : data{x00, x01, x02, x03, x10, x11, x12, x13,
             x20, x21, x22, x23, x30, x31, x32, x33} {}
  void operator*=(const Mat4& other) {
    Mat4 result;
    for (int r = 0; r < 4; r++) {
      for (int c = 0; c < 4; c++) {
        float sum = 0;
        for (int k = 0; k < 4; k++) {
          sum += this->Get(r, k) * other.Get(k, c);
        }
        result.Set(r, c, sum);
      }
    }
    std::memcpy(this->data.data(), result.data.data(),
                result.data.size() * sizeof(float));
  }
  float Get(int x, int y) const { return data[x * 4 + y]; }
  void Set(int x, int y, float val) { data[x * 4 + y] = val; }

  std::vector<float> data;
};

namespace v1 {

inline void LandmarksToTransformMatrixV1(
    const LandmarksToTransformMatrixV1Attributes& params,
    const RuntimeShape& input0_shape, const float* landmarks,
    const RuntimeShape& output_shape, float* output_data) {
  TFLITE_CHECK_EQ(input0_shape.DimensionsCount(), 4);
  TFLITE_CHECK_EQ(output_shape.DimensionsCount(), 3);
  TFLITE_CHECK_EQ(input0_shape.Dims(0), 1);
  TFLITE_CHECK_EQ(input0_shape.Dims(1), 1);
  TFLITE_CHECK_EQ(input0_shape.Dims(2), 1);

  float2 left_landmark = Read3DLandmarkXY(landmarks, params.left_rotation_idx);
  float2 right_landmark =
      Read3DLandmarkXY(landmarks, params.right_rotation_idx);

  float alpha = -std::atan((right_landmark.y - left_landmark.y) /
                           (right_landmark.x - left_landmark.x));

  float2 max_value(-100000, -100000);
  float2 min_value(100000, 100000);
  for (int i = 0; i < params.subset.size(); i++) {
    for (int j = 0; j < 2; j++) {
      float2 landmark_current =
          Read3DLandmarkXY(landmarks, params.subset[i][j]);
      float2 rotated(
          landmark_current.x * cos(alpha) - landmark_current.y * sin(alpha),
          landmark_current.x * sin(alpha) + landmark_current.y * cos(alpha));
      max_value = float2(std::max(max_value.x, rotated.x),
                         std::max(max_value.y, rotated.y));
      min_value = float2(std::min(min_value.x, rotated.x),
                         std::min(min_value.y, rotated.y));
    }
  }

  float2 bbox_size((max_value.x - min_value.x) * params.bbox_size_multiplier,
                   (max_value.y - min_value.y) * params.bbox_size_multiplier);

  Mat3 scale_matrix(
      bbox_size.x / params.landmarks_range, 0.0, 0.0,  // first row
      0.0, bbox_size.y / params.landmarks_range, 0.0,  // second row
      0.0, 0.0, 1.0);                                  // third row

  float2 middle((max_value.x + min_value.x) / 2.0,
                (max_value.y + min_value.y) / 2.0);

  float2 rotated_middle(middle.x * cos(-alpha) - middle.y * sin(-alpha),
                        middle.x * sin(-alpha) + middle.y * cos(-alpha));

  Mat3 rotation_matrix(
      cos(-alpha), -sin(-alpha),
      (rotated_middle.x / params.landmarks_range) * 2.0 - 1.0,  // first row
      sin(-alpha), cos(-alpha),
      (rotated_middle.y / params.landmarks_range) * 2.0 - 1.0,  // second row
      0, 0, 1);                                                 // third row

  Mat3 to_relative(2.0 / (params.output_hw.w - 1.0), 0.0, -1.0,  // first row
                   0.0, 2.0 / (params.output_hw.h - 1.0), -1.0,  // second row
                   0.0, 0.0, 1.0);                               // third row

  Mat3 to_absolute((params.input_hw.w - 1.0) / 2.0, 0.0,
                   (params.input_hw.w - 1.0) / 2.0,  // first row
                   0.0, (params.input_hw.h - 1.0) / 2.0,
                   (params.input_hw.h - 1.0) / 2.0,  // second row
                   0.0, 0.0, 1.0);                   // third row

  // Inverse Transformstion Matrix
  Mat3 itm = to_absolute * rotation_matrix * scale_matrix * to_relative;

  output_data[0] = itm.Get(0, 0);
  output_data[1] = itm.Get(0, 1);
  output_data[2] = 0.0;
  output_data[3] = itm.Get(0, 2);

  output_data[4] = itm.Get(1, 0);
  output_data[5] = itm.Get(1, 1);
  output_data[6] = 0.0;
  output_data[7] = itm.Get(1, 2);

  output_data[8] = itm.Get(2, 0);
  output_data[9] = itm.Get(2, 1);
  output_data[10] = itm.Get(2, 2);
  output_data[11] = 0.0;

  output_data[12] = 0.0;
  output_data[13] = 0.0;
  output_data[14] = 0.0;
  output_data[15] = 1.0;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, kDataInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteFloat32);

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(3);
  output_size->data[0] = kTensformMatrixShape.x;
  output_size->data[1] = kTensformMatrixShape.y;
  output_size->data[2] = kTensformMatrixShape.z;

  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  LandmarksToTransformMatrixV1Attributes op_params;
  BHWC output_shape;
  auto status = tflite::gpu::ParseLandmarksToTransformMatrixV1Attributes(
      node->custom_initial_data, node->custom_initial_data_size, &op_params,
      &output_shape);
  if (!status.ok()) {
    context->ReportError(context, status.message().data());
    return kTfLiteError;
  }

  if (op_params.bbox_size_multiplier == 0) {
    context->ReportError(context, "Incorrect bbox_size_multiplier: %d",
                         op_params.bbox_size_multiplier);
    return kTfLiteError;
  }

  if (op_params.dimensions != 3) {
    context->ReportError(context, "Incorrect dimensions: %d",
                         op_params.dimensions);
    return kTfLiteError;
  }

  if (op_params.input_hw.h <= 0 || op_params.input_hw.w <= 0) {
    context->ReportError(context, "Incorrect input_hw: h = %d w = %d",
                         op_params.input_hw.h, op_params.input_hw.w);
    return kTfLiteError;
  }

  if (op_params.output_hw.h <= 0 || op_params.output_hw.w <= 0) {
    context->ReportError(context, "Incorrect output_hw: h = %d w = %d",
                         op_params.output_hw.h, op_params.output_hw.w);
    return kTfLiteError;
  }

  if (op_params.landmarks_range <= 0) {
    context->ReportError(context, "Incorrect landmarks_range: %d",
                         op_params.landmarks_range);
    return kTfLiteError;
  }

  if (op_params.left_rotation_idx < 0) {
    context->ReportError(context, "Incorrect left_rotation_idx: %d",
                         op_params.left_rotation_idx);
    return kTfLiteError;
  }

  if (op_params.right_rotation_idx < 0) {
    context->ReportError(context, "Incorrect right_rotation_idx: %d",
                         op_params.right_rotation_idx);
    return kTfLiteError;
  }

  if (op_params.subset.empty()) {
    context->ReportError(context, "Subset parameter is empty");
    return kTfLiteError;
  }

  int counter = 0;
  for (auto& val : op_params.subset) {
    for (int i = 0; i < 2; i++) {
      if (val[i] < 0) {
        context->ReportError(context,
                             "Incorrect subset value: index = %d, value = %d",
                             counter, val[i]);
        return kTfLiteError;
      }
      counter++;
    }
  }

  const TfLiteTensor* input0 = GetInput(context, node, kDataInputTensor);
  TF_LITE_ENSURE(context, input0 != nullptr);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  LandmarksToTransformMatrixV1(
      op_params, GetTensorShape(input0), GetTensorData<float>(input0),
      GetTensorShape(output), GetTensorData<float>(output));
  return kTfLiteOk;
}
}  // namespace v1

namespace v2 {

void EstimateRotationRadians(const float* input_data_0, int left_rotation_idx,
                             int right_rotation_idx,
                             float target_rotation_radians,
                             float* rotation_radians) {
  const float3 left_landmark =
      Read3DLandmarkXYZ(input_data_0, left_rotation_idx);
  const float3 right_landmark =
      Read3DLandmarkXYZ(input_data_0, right_rotation_idx);
  const float left_x = left_landmark[0];
  const float left_y = left_landmark[1];
  const float right_x = right_landmark[0];
  const float right_y = right_landmark[1];
  float rotation = std::atan2(right_y - left_y, right_x - left_x);
  rotation = target_rotation_radians - rotation;
  *rotation_radians = rotation;
}

void EstimateCenterAndSize(const float* input_data_0,
                           std::vector<tflite::gpu::int2> subset_idxs,
                           float rotation_radians, float* crop_x, float* crop_y,
                           float* crop_width, float* crop_height) {
  std::vector<float3> landmarks;
  landmarks.reserve(subset_idxs.size() * 2);
  for (int i = 0; i < subset_idxs.size(); i++) {
    landmarks.push_back(Read3DLandmarkXYZ(input_data_0, subset_idxs[i][0]));
    landmarks.push_back(Read3DLandmarkXYZ(input_data_0, subset_idxs[i][1]));
  }
  for (int i = 0; i < landmarks.size(); i++) {
    landmarks[i].z = 1.0;
  }
  const float& r = rotation_radians;
  // clang-format off
  const Mat3 t_rotation = Mat3(std::cos(r),  -std::sin(r), 0.0,
                               std::sin(r),   std::cos(r), 0.0,
                                       0.0,           0.0, 1.0);
  const Mat3 t_rotation_inverse =
                          Mat3(std::cos(-r), -std::sin(-r), 0.0,
                               std::sin(-r),  std::cos(-r), 0.0,
                                        0.0,           0.0, 1.0);
  // clang-format on
  for (int i = 0; i < landmarks.size(); i++) {
    landmarks[i] = t_rotation * landmarks[i];
  }
  float3 xy1_max = landmarks[0], xy1_min = landmarks[0];
  for (int i = 1; i < landmarks.size(); i++) {
    if (xy1_max.x < landmarks[i].x) xy1_max.x = landmarks[i].x;
    if (xy1_max.y < landmarks[i].y) xy1_max.y = landmarks[i].y;

    if (xy1_min.x > landmarks[i].x) xy1_min.x = landmarks[i].x;
    if (xy1_min.y > landmarks[i].y) xy1_min.y = landmarks[i].y;
  }
  *crop_width = xy1_max.x - xy1_min.x;
  *crop_height = xy1_max.y - xy1_min.y;
  float3 crop_xy1 = xy1_min;
  crop_xy1.x += xy1_max.x;
  crop_xy1.y += xy1_max.y;
  crop_xy1.x /= 2;
  crop_xy1.y /= 2;
  crop_xy1 = t_rotation_inverse * crop_xy1;
  *crop_x = crop_xy1.x;
  *crop_y = crop_xy1.y;
}

inline void LandmarksToTransformMatrixV2(
    const LandmarksToTransformMatrixV2Attributes& params,
    const RuntimeShape& input0_shape, const float* landmarks,
    const RuntimeShape& output_shape, float* output_data) {
  float rotation_radians = 0.0;
  EstimateRotationRadians(landmarks, params.left_rotation_idx,
                          params.right_rotation_idx,
                          params.target_rotation_radians, &rotation_radians);
  float crop_x = 0.0, crop_y = 0.0, crop_width = 0.0, crop_height = 0.0;
  EstimateCenterAndSize(landmarks, params.subset_idxs, rotation_radians,
                        &crop_x, &crop_y, &crop_width, &crop_height);
  // Turn off clang formatting to make matrices initialization more readable.
  // clang-format off
  Mat4 t = Mat4(1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0);
  const Mat4 t_shift = Mat4(1.0, 0.0, 0.0, crop_x,
                            0.0, 1.0, 0.0, crop_y,
                            0.0, 0.0, 1.0,    0.0,
                            0.0, 0.0, 0.0,    1.0);
  t *= t_shift;
  const float& r = -rotation_radians;
  const Mat4 t_rotation = Mat4(std::cos(r), -std::sin(r), 0.0, 0.0,
                               std::sin(r),  std::cos(r), 0.0, 0.0,
                                       0.0,          0.0, 1.0, 0.0,
                                       0.0,          0.0, 0.0, 1.0);
  t *= t_rotation;
  const float scale_x = params.scale_x * crop_width / params.output_width;
  const float scale_y = params.scale_y * crop_height / params.output_height;
  const Mat4 t_scale = Mat4(scale_x,     0.0, 0.0, 0.0,
                                0.0, scale_y, 0.0, 0.0,
                                0.0,     0.0, 1.0, 0.0,
                                0.0,     0.0, 0.0, 1.0);
  t *= t_scale;
  const float shift_x = -1.0 * (params.output_width / 2.0);
  const float shift_y = -1.0 * (params.output_height / 2.0);
  const Mat4 t_shift2 = Mat4(1.0, 0.0, 0.0, shift_x,
                             0.0, 1.0, 0.0, shift_y,
                             0.0, 0.0, 1.0,     0.0,
                             0.0, 0.0, 0.0,     1.0);
  t *= t_shift2;
  std::memcpy(output_data, t.data.data(), 16 * sizeof(float));
  // clang-format on
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, kDataInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 3);
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteFloat32);

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(3);
  output_size->data[0] = kTensformMatrixShape.x;
  output_size->data[1] = kTensformMatrixShape.y;
  output_size->data[2] = kTensformMatrixShape.z;

  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  LandmarksToTransformMatrixV2Attributes op_params;
  BHWC output_shape;
  auto status = tflite::gpu::ParseLandmarksToTransformMatrixV2Attributes(
      node->custom_initial_data, node->custom_initial_data_size, &op_params,
      &output_shape);
  if (!status.ok()) {
    context->ReportError(context, status.message().data());
    return kTfLiteError;
  }

  if (op_params.left_rotation_idx < 0) {
    context->ReportError(context, "Incorrect left_rotation_idx: %d",
                         op_params.left_rotation_idx);
    return kTfLiteError;
  }

  if (op_params.right_rotation_idx < 0) {
    context->ReportError(context, "Incorrect right_rotation_idx: %d",
                         op_params.right_rotation_idx);
    return kTfLiteError;
  }

  if (op_params.output_height <= 0) {
    context->ReportError(context, "Incorrect output_height: %d",
                         op_params.output_height);
    return kTfLiteError;
  }

  if (op_params.output_width <= 0) {
    context->ReportError(context, "Incorrect output_width: %d",
                         op_params.output_width);
    return kTfLiteError;
  }

  if (op_params.scale_x <= 0) {
    context->ReportError(context, "Incorrect scale_x: %d", op_params.scale_x);
    return kTfLiteError;
  }

  if (op_params.scale_y <= 0) {
    context->ReportError(context, "Incorrect scale_y: %d", op_params.scale_y);
    return kTfLiteError;
  }

  int counter = 0;
  for (auto& val : op_params.subset_idxs) {
    for (int i = 0; i < 2; i++) {
      if (val[i] < 0) {
        context->ReportError(context,
                             "Incorrect subset value: index = %d, value = %d",
                             counter, val[i]);
        return kTfLiteError;
      }
      counter++;
    }
  }

  const TfLiteTensor* input0 = GetInput(context, node, kDataInputTensor);
  TF_LITE_ENSURE(context, input0 != nullptr);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  LandmarksToTransformMatrixV2(
      op_params, GetTensorShape(input0), GetTensorData<float>(input0),
      GetTensorShape(output), GetTensorData<float>(output));
  return kTfLiteOk;
}

}  // namespace v2

}  // namespace

TfLiteRegistration* RegisterLandmarksToTransformMatrixV1() {
  static TfLiteRegistration reg = {
      /*.init=*/nullptr,
      /*.free=*/nullptr,
      /*.prepare=*/v1::Prepare,
      /*.invoke=*/v1::Eval,
      /*.profiling_string=*/nullptr,
      /*.builtin_code=*/tflite::BuiltinOperator_CUSTOM,
      /*.custom_name=*/"Landmarks2TransformMatrix",
      /*.version=*/1,
  };
  return &reg;
}
TfLiteRegistration* RegisterLandmarksToTransformMatrixV2() {
  static TfLiteRegistration reg = {
      /*.init=*/nullptr,
      /*.free=*/nullptr,
      /*.prepare=*/v2::Prepare,
      /*.invoke=*/v2::Eval,
      /*.profiling_string=*/nullptr,
      /*.builtin_code=*/tflite::BuiltinOperator_CUSTOM,
      /*.custom_name=*/"Landmarks2TransformMatrix",
      /*.version=*/2,
  };
  return &reg;
}

}  // namespace tflite_operations
}  // namespace mediapipe
