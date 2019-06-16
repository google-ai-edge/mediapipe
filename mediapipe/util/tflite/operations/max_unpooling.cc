// Copyright 2019 The MediaPipe Authors.
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

#include "mediapipe/util/tflite/operations/max_unpooling.h"

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/padding.h"

namespace mediapipe {
namespace tflite_operations {
namespace {

constexpr int kDataInputTensor = 0;
constexpr int kIndicesTensor = 1;
constexpr int kOutputTensor = 0;

inline void MaxUnpooling(const ::tflite::PoolParams& params,
                         const ::tflite::RuntimeShape& input_shape,
                         const float* input_data, const float* indices_data,
                         const ::tflite::RuntimeShape& output_shape,
                         float* output_data) {
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  std::memset(output_data, 0, output_shape.FlatSize() * sizeof(float));
  for (int batch = 0; batch < batches; ++batch) {
    for (int in_y = 0; in_y < input_height; ++in_y) {
      for (int in_x = 0; in_x < input_width; ++in_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const auto input_offset =
              Offset(input_shape, batch, in_y, in_x, channel);
          int idx = indices_data[input_offset];
          const int max_x = idx % params.filter_width;
          const int max_y = idx / params.filter_width;
          const int out_x =
              in_x * stride_width - params.padding_values.width + max_x;
          const int out_y =
              in_y * stride_height - params.padding_values.height + max_y;
          output_data[Offset(output_shape, batch, out_y, out_x, channel)] =
              input_data[input_offset];
        }
      }
    }
  }
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<const TfLitePoolParams*>(node->custom_initial_data);
  TfLitePaddingValues* data_padding =
      reinterpret_cast<TfLitePaddingValues*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, ::tflite::NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, ::tflite::NumOutputs(node), 1);
  TfLiteTensor* output = ::tflite::GetOutput(context, node, kOutputTensor);
  const TfLiteTensor* input =
      ::tflite::GetInput(context, node, kDataInputTensor);
  const TfLiteTensor* indices =
      ::tflite::GetInput(context, node, kIndicesTensor);
  TF_LITE_ENSURE_EQ(context, ::tflite::NumDimensions(indices), 4);
  TF_LITE_ENSURE_EQ(context, ::tflite::NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, indices->type, kTfLiteFloat32);

  int batches = input->dims->data[0];
  int height = input->dims->data[1];
  int width = input->dims->data[2];
  int channels_out = input->dims->data[3];

  int out_width = width * params->filter_width;
  int out_height = height * params->filter_height;
  data_padding->height = ::tflite::ComputePadding(
      params->stride_height, 1, out_height, params->filter_height, height);
  data_padding->width = ::tflite::ComputePadding(
      params->stride_width, 1, out_width, params->filter_width, width);

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = batches;
  output_size->data[1] = out_height;
  output_size->data[2] = out_width;
  output_size->data[3] = channels_out;
  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<const TfLitePoolParams*>(node->custom_initial_data);
  TfLitePaddingValues* data_padding =
      reinterpret_cast<TfLitePaddingValues*>(node->user_data);

  TfLiteTensor* output = ::tflite::GetOutput(context, node, kOutputTensor);
  const TfLiteTensor* input =
      ::tflite::GetInput(context, node, kDataInputTensor);
  const TfLiteTensor* indices =
      ::tflite::GetInput(context, node, kIndicesTensor);

  float activation_min, activation_max;
  ::tflite::CalculateActivationRange(params->activation, &activation_min,
                                     &activation_max);
  ::tflite::PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data_padding->height;
  op_params.padding_values.width = data_padding->width;
  op_params.float_activation_min = activation_min;
  op_params.float_activation_max = activation_max;
  MaxUnpooling(op_params, ::tflite::GetTensorShape(input),
               ::tflite::GetTensorData<float>(input),
               ::tflite::GetTensorData<float>(indices),
               ::tflite::GetTensorShape(output),
               ::tflite::GetTensorData<float>(output));
  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration* RegisterMaxUnpooling2D() {
  static TfLiteRegistration reg = {
      [](TfLiteContext*, const char*, size_t) -> void* {
        return new TfLitePaddingValues();
      },
      [](TfLiteContext*, void* buffer) -> void {
        delete reinterpret_cast<TfLitePaddingValues*>(buffer);
      },
      Prepare, Eval};
  return &reg;
}

}  // namespace tflite_operations
}  // namespace mediapipe
