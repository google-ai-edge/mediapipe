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
// This version has been modified by MediaPipe authors to support bias. Details
// of the modification is marked below in the code.

#include "mediapipe/util/tflite/operations/transpose_conv_bias.h"

#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/padding.h"

namespace mediapipe {
namespace tflite_operations {
namespace {

constexpr int kWeightsTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kDataInputTensor = 0;
constexpr int kOutputTensor = 0;

// These functions were copied from the following places:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/reference/reference_ops.h
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/transpose_conv.cc

inline void TransposeConvBias(
    const ::tflite::ConvParams& params,
    const ::tflite::RuntimeShape& input_shape, const float* input_data,
    const ::tflite::RuntimeShape& filter_shape, const float* filter_data,
    const ::tflite::RuntimeShape& bias_shape, const float* bias_data,
    const ::tflite::RuntimeShape& output_shape, float* output_data,
    const ::tflite::RuntimeShape& im2col_shape, float* im2col_data) {
  // Start of copy from
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/reference/reference_ops.h
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;

  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(bias_shape.DimensionsCount(), 1);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  (void)im2col_data;   // only used in optimized code.
  (void)im2col_shape;  // only used in optimized code.

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  // Start of MediaPipe modificiation.

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; out_y++) {
      for (int out_x = 0; out_x < output_width; out_x++) {
        for (int out_channel = 0; out_channel < output_depth; out_channel++) {
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              bias_data[out_channel];
        }
      }
    }

    for (int in_y = 0; in_y < input_height; ++in_y) {
      for (int in_x = 0; in_x < input_width; ++in_x) {
        for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
          // Loop through the output elements it will influence
          const int out_x_origin = (in_x * stride_width) - pad_width;
          const int out_y_origin = (in_y * stride_height) - pad_height;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              for (int out_channel = 0; out_channel < output_depth;
                   ++out_channel) {
                // Compute output element location
                const int out_x = out_x_origin + filter_x;
                const int out_y = out_y_origin + filter_y;
                // We cannot accumulate out of bounds
                if ((out_x >= 0) && (out_x < output_width) && (out_y >= 0) &&
                    (out_y < output_height)) {
                  float input_value = input_data[Offset(
                      input_shape, batch, in_y, in_x, in_channel)];
                  float filter_value =
                      filter_data[Offset(filter_shape, out_channel, filter_y,
                                         filter_x, in_channel)];
                  output_data[Offset(output_shape, batch, out_y, out_x,
                                     out_channel)] +=
                      input_value * filter_value;
                }
              }
            }
          }
        }
      }
    }
  }
  // End of MediaPipe modification.
  // End of copy.
}

// Start of copy from
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/transpose_conv.cc
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, ::tflite::NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, ::tflite::NumOutputs(node), 1);

  const TfLiteTensor* weights =
      ::tflite::GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* bias = ::tflite::GetInput(context, node, kBiasTensor);
  const TfLiteTensor* input =
      ::tflite::GetInput(context, node, kDataInputTensor);
  TfLiteTensor* output = ::tflite::GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_EQ(context, ::tflite::NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, ::tflite::NumDimensions(weights), 4);
  TF_LITE_ENSURE_EQ(context, ::tflite::NumDimensions(bias), 1);

  // Start of MediaPipe modificiation.
  TF_LITE_ENSURE_EQ(context, ::tflite::SizeOfDimension(weights, 0),
                    ::tflite::SizeOfDimension(bias, 0));

  // Currently only supports float32.
  const TfLiteType data_type = input->type;
  TF_LITE_ENSURE(context, data_type == kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, output->type, data_type);
  TF_LITE_ENSURE_EQ(context, weights->type, data_type);
  TF_LITE_ENSURE_EQ(context, bias->type, data_type);

  // Ensure that weights and inputs have the same channel dimension.
  // Note: TOCO will reorder weights in the following format: OHWI.
  TF_LITE_ENSURE_EQ(context, ::tflite::SizeOfDimension(input, 3),
                    ::tflite::SizeOfDimension(weights, 3));

  // Ensure that weights and bias have the same output channel dimension.
  TF_LITE_ENSURE_EQ(context, ::tflite::SizeOfDimension(weights, 0),
                    ::tflite::SizeOfDimension(bias, 0));

  const auto* params = reinterpret_cast<const TfLiteTransposeConvParams*>(
      node->custom_initial_data);
  const int filter_width = ::tflite::SizeOfDimension(weights, 2);
  const int filter_height = ::tflite::SizeOfDimension(weights, 1);
  const int stride_width = params->stride_width;
  const int stride_height = params->stride_height;
  const int in_width = ::tflite::SizeOfDimension(input, 2);
  const int in_height = ::tflite::SizeOfDimension(input, 1);

  // Get height and width of the output image.
  TfLiteIntArray* output_shape_array = TfLiteIntArrayCreate(4);
  output_shape_array->data[0] = ::tflite::SizeOfDimension(input, 0);
  output_shape_array->data[3] = ::tflite::SizeOfDimension(weights, 0);

  TfLitePaddingValues padding_size{0, 0};
  if (params->padding == kTfLitePaddingSame) {
    padding_size.height =
        std::max(0, filter_height - (in_height - 1) % stride_height - 1);
    padding_size.width =
        std::max(0, filter_width - (in_width - 1) % stride_width - 1);
  }
  output_shape_array->data[1] =
      stride_height * (in_height - 1) + filter_height - padding_size.height;
  output_shape_array->data[2] =
      stride_width * (in_width - 1) + filter_width - padding_size.width;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_shape_array));
  return kTfLiteOk;
  // End of MediaPipe modification.
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* weights =
      ::tflite::GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* bias = ::tflite::GetInput(context, node, kBiasTensor);
  const TfLiteTensor* input =
      ::tflite::GetInput(context, node, kDataInputTensor);
  TfLiteTensor* output = ::tflite::GetOutput(context, node, kOutputTensor);

  const auto* params = reinterpret_cast<const TfLiteTransposeConvParams*>(
      node->custom_initial_data);

  const int filter_width = ::tflite::SizeOfDimension(weights, 2);
  const int filter_height = ::tflite::SizeOfDimension(weights, 1);
  const int stride_width = params->stride_width;
  const int stride_height = params->stride_height;
  const int in_width = ::tflite::SizeOfDimension(input, 2);
  const int in_height = ::tflite::SizeOfDimension(input, 1);

  TfLitePaddingValues padding_size{0, 0};
  if (params->padding == kTfLitePaddingSame) {
    padding_size.height =
        std::max(0, filter_height - (in_height - 1) % stride_height - 1);
    padding_size.width =
        std::max(0, filter_width - (in_width - 1) % stride_width - 1);
  }

  // Start of MediaPipe modificiation.

  // Currently only support float32.
  switch (input->type) {
    case kTfLiteFloat32: {
      ::tflite::ConvParams op_params;
      op_params.padding_type = ::tflite::PaddingType::kSame;
      op_params.padding_values.width = padding_size.width / 2;
      op_params.padding_values.height = padding_size.height / 2;
      op_params.stride_width = stride_width;
      op_params.stride_height = stride_height;

      TransposeConvBias(
          op_params, ::tflite::GetTensorShape(input),
          ::tflite::GetTensorData<float>(input),
          ::tflite::GetTensorShape(weights),
          ::tflite::GetTensorData<float>(weights),
          ::tflite::GetTensorShape(bias), ::tflite::GetTensorData<float>(bias),
          ::tflite::GetTensorShape(output),
          ::tflite::GetTensorData<float>(output),
          // Last two args specify im2col which reference_ops ignores.
          // (Note this does not lead to a performance regression, as the
          // previous optimized version was just a copy of the reference code.)
          // TODO: Allocate im2col tensors and switch to
          // optimized_ops.
          ::tflite::GetTensorShape(output),
          ::tflite::GetTensorData<float>(output));
      break;
    }
    default:
      context->ReportError(context, "Type %d, not currently supported.",
                           input->type);
      return kTfLiteError;
  }

  // End of MediaPipe modification.

  return kTfLiteOk;
}
// End of copy.

}  // namespace

TfLiteRegistration* RegisterConvolution2DTransposeBias() {
  static TfLiteRegistration reg = {nullptr, nullptr, Prepare, Eval};
  return &reg;
}

}  // namespace tflite_operations
}  // namespace mediapipe
