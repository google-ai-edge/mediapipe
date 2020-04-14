// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
//
// This version has been modified by MediaPipe authors to support argmax
// indices. Details of the modification is marked below in the code.
#include "mediapipe/util/tflite/operations/max_pool_argmax.h"

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/padding.h"

namespace mediapipe {
namespace tflite_operations {
namespace {

constexpr int kDataInputTensor = 0;
constexpr int kOutputTensor = 0;
constexpr int kIndicesTensor = 1;

// These functions were copied from the following places:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/reference/reference_ops.h
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/pooling.cc

inline void MaxPoolArgmax(const ::tflite::PoolParams& params,
                          const ::tflite::RuntimeShape& input_shape,
                          const float* input_data,
                          const ::tflite::RuntimeShape& output_shape,
                          float* output_data, float* indices_data) {
  // Start of copy from
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/reference/reference_ops.h
  // Start of MediaPipe modificiation.
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          float max = std::numeric_limits<float>::lowest();
          int max_x = 0;
          int max_y = 0;
          for (int filter_y = filter_y_start; filter_y < filter_y_end;
               ++filter_y) {
            for (int filter_x = filter_x_start; filter_x < filter_x_end;
                 ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;
              float cur =
                  input_data[Offset(input_shape, batch, in_y, in_x, channel)];
              if (cur > max) {
                max = cur;
                max_x = filter_x;
                max_y = filter_y;
              }
            }
          }
          output_data[Offset(output_shape, batch, out_y, out_x, channel)] =
              ::tflite::ActivationFunctionWithMinMax(
                  max, params.float_activation_min,
                  params.float_activation_max);
          if (indices_data) {
            indices_data[Offset(output_shape, batch, out_y, out_x, channel)] =
                max_y * params.filter_width + max_x + 0.1f;
          }
        }
      }
    }
  }
  // End of MediaPipe modification.
  // End of copy.
}

// Start of copy from
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/pooling.cc
// Start of MediaPipe modificiation.
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<const TfLitePoolParams*>(node->custom_initial_data);
  TfLitePaddingValues* data_padding =
      reinterpret_cast<TfLitePaddingValues*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, ::tflite::NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, ::tflite::NumOutputs(node), 2);
  TfLiteTensor* output = ::tflite::GetOutput(context, node, kOutputTensor);
  TfLiteTensor* indices = ::tflite::GetOutput(context, node, kIndicesTensor);
  const TfLiteTensor* input =
      ::tflite::GetInput(context, node, kDataInputTensor);
  TF_LITE_ENSURE_EQ(context, ::tflite::NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, indices->type, kTfLiteFloat32);

  int batches = input->dims->data[0];
  int height = input->dims->data[1];
  int width = input->dims->data[2];
  int channels_out = input->dims->data[3];

  // Matching GetWindowedOutputSize in TensorFlow.
  auto padding = params->padding;
  auto compute_out_size = [padding](int image_size, int filter_size,
                                    int stride) -> int {
    return padding == kTfLitePaddingSame
               ? (image_size + stride - 1) / stride
               : padding == kTfLitePaddingValid
                     ? (image_size - filter_size + stride) / stride
                     : 0;
  };

  int out_width =
      compute_out_size(width, params->filter_width, params->stride_width);
  int out_height =
      compute_out_size(height, params->filter_height, params->stride_height);

  data_padding->height = ::tflite::ComputePadding(
      params->stride_height, 1, height, params->filter_height, out_height);
  data_padding->width = ::tflite::ComputePadding(
      params->stride_width, 1, width, params->filter_width, out_width);

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = batches;
  output_size->data[1] = out_height;
  output_size->data[2] = out_width;
  output_size->data[3] = channels_out;
  TfLiteIntArray* indices_size = TfLiteIntArrayCopy(output_size);
  if (context->ResizeTensor(context, output, output_size) != kTfLiteOk) {
    return kTfLiteError;
  }
  if (context->ResizeTensor(context, indices, indices_size) != kTfLiteOk) {
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<const TfLitePoolParams*>(node->custom_initial_data);
  TfLitePaddingValues* data_padding =
      reinterpret_cast<TfLitePaddingValues*>(node->user_data);

  TfLiteTensor* output = ::tflite::GetOutput(context, node, kOutputTensor);
  TfLiteTensor* indices = ::tflite::GetOutput(context, node, kIndicesTensor);
  const TfLiteTensor* input =
      ::tflite::GetInput(context, node, kDataInputTensor);

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
  MaxPoolArgmax(op_params, ::tflite::GetTensorShape(input),
                ::tflite::GetTensorData<float>(input),
                ::tflite::GetTensorShape(output),
                ::tflite::GetTensorData<float>(output),
                ::tflite::GetTensorData<float>(indices));
  return kTfLiteOk;
}
// End of MediaPipe modification.
// End of copy.

}  // namespace

TfLiteRegistration* RegisterMaxPoolingWithArgmax2D() {
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
