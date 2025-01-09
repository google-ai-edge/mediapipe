/* Copyright 2023 The MediaPipe Authors.

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

#include "mediapipe/tasks/cc/vision/custom_ops/fused_batch_norm.h"

#include <stddef.h>

#include "Eigen/Core"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace mediapipe::tflite_operations {
namespace vision::batch_norm {
namespace {

using tflite::GetTensorData;

constexpr int kInputIndex = 0;
constexpr int kInputScaleIndex = 1;
constexpr int kInputOffsetIndex = 2;
constexpr int kInputEstimatedMeanIndex = 3;
constexpr int kInputEstimatedVarIndex = 4;

constexpr int kOutputIndex = 0;
constexpr int kOutputBatchMeanIndex = 1;
constexpr int kOutputBatchVarIndex = 2;
constexpr int kOutputSavedMeanIndex = 3;
constexpr int kOutputSavedVarIndex = 4;

template <typename T, int NDIMS = 1, typename IndexType = Eigen::DenseIndex>
struct TTypes {
  // Rank-<NDIMS> tensor of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, IndexType>>
      Tensor;

  // Rank-1 tensor (vector) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType>> Vec;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType>>
      ConstVec;
};

template <typename T, typename U>
void FusedBarchNorm(TfLiteContext* context, TfLiteTensor* x_input,
                    TfLiteTensor* scale_input, TfLiteTensor* offset_input,
                    TfLiteTensor* running_mean_input,
                    TfLiteTensor* running_variance_input,
                    TfLiteTensor* y_output, TfLiteTensor* running_mean_output,
                    TfLiteTensor* running_var_output,
                    TfLiteTensor* saved_batch_mean_output,
                    TfLiteTensor* saved_batch_var_output,
                    U exponential_avg_factor, U epsilon) {
  const int batches = x_input->dims->data[0];
  const int height = x_input->dims->data[1];
  const int width = x_input->dims->data[2];
  const int depth = x_input->dims->data[3];

  Eigen::array<Eigen::DenseIndex, 4> x_dims = {batches, height, width, depth};
  Eigen::array<Eigen::DenseIndex, 1> depth_dims = {depth};

  const int rest_size = batches * height * width;

  typename TTypes<T, 4>::Tensor x(GetTensorData<T>(x_input), x_dims);
  typename TTypes<U>::ConstVec scale(GetTensorData<U>(scale_input), depth_dims);
  typename TTypes<U>::ConstVec offset(GetTensorData<U>(offset_input),
                                      depth_dims);
  typename TTypes<U>::ConstVec old_mean(GetTensorData<U>(running_mean_input),
                                        depth_dims);
  typename TTypes<U>::ConstVec old_variance(
      GetTensorData<U>(running_variance_input), depth_dims);
  typename TTypes<T, 4>::Tensor y(GetTensorData<T>(y_output), x_dims);
  typename TTypes<U>::Vec new_mean(GetTensorData<U>(running_mean_output),
                                   depth_dims);
  typename TTypes<U>::Vec new_variance(GetTensorData<U>(running_var_output),
                                       depth_dims);
  typename TTypes<U>::Vec saved_batch_mean(
      GetTensorData<U>(saved_batch_mean_output), depth_dims);
  typename TTypes<U>::Vec saved_batch_var(
      GetTensorData<U>(saved_batch_var_output), depth_dims);

  Eigen::DSizes<Eigen::Index, 2> rest_by_depth(rest_size, depth);
  Eigen::DSizes<Eigen::Index, 4> tensor_shape(batches, height, width, depth);

  Eigen::IndexList<Eigen::type2index<1>, Eigen::Index> one_by_depth;
  one_by_depth.set(1, depth);
  Eigen::IndexList<Eigen::type2index<0>> reduce_dims;
  Eigen::IndexList<Eigen::Index, Eigen::type2index<1>> bcast_spec;
  bcast_spec.set(0, rest_size);

  auto x_rest_by_depth = x.reshape(rest_by_depth).template cast<U>();
  const int rest_size_minus_one = (rest_size > 1) ? (rest_size - 1) : 1;
  U rest_size_inv = static_cast<U>(1.0f / static_cast<U>(rest_size));
  // This adjustment is for Bessel's correction
  U rest_size_adjust =
      static_cast<U>(rest_size) / static_cast<U>(rest_size_minus_one);

  Eigen::Tensor<U, 1, Eigen::RowMajor> batch_mean(depth);
  Eigen::Tensor<U, 1, Eigen::RowMajor> batch_variance(depth);

  batch_mean = (x_rest_by_depth.sum(reduce_dims) * rest_size_inv);
  auto x_centered =
      x_rest_by_depth - batch_mean.reshape(one_by_depth).broadcast(bcast_spec);

  batch_variance = x_centered.square().sum(reduce_dims) * rest_size_inv;
  auto scaling_factor = ((batch_variance + epsilon).rsqrt() * scale)
                            .eval()
                            .reshape(one_by_depth)
                            .broadcast(bcast_spec);
  auto x_scaled = x_centered * scaling_factor;
  auto x_shifted =
      (x_scaled + offset.reshape(one_by_depth).broadcast(bcast_spec))
          .template cast<T>();

  y.reshape(rest_by_depth) = x_shifted;
  if (exponential_avg_factor == U(1.0)) {
    saved_batch_var = batch_variance;
    saved_batch_mean = batch_mean;
    new_variance = batch_variance * rest_size_adjust;
    new_mean = batch_mean;
  } else {
    U one_minus_factor = U(1) - exponential_avg_factor;
    saved_batch_var = batch_variance;
    saved_batch_mean = batch_mean;
    new_variance = one_minus_factor * old_variance +
                   (exponential_avg_factor * rest_size_adjust) * batch_variance;
    new_mean =
        one_minus_factor * old_mean + exponential_avg_factor * batch_mean;
  }
}

}  // namespace

// Initializes FusedBatchNorm object from serialized parameters.
void* Initialize(TfLiteContext* /*context*/, const char* /*buffer*/,
                 size_t /*length*/) {
  return nullptr;
}

void Free(TfLiteContext* /*context*/, void* /*buffer*/) {}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, tflite::NumInputs(node), 5);
  TF_LITE_ENSURE_EQ(context, tflite::NumOutputs(node), 6);

  TfLiteTensor* output = tflite::GetOutput(context, node, kOutputIndex);
  TF_LITE_ENSURE(context, output != nullptr);
  TfLiteTensor* batch_mean =
      tflite::GetOutput(context, node, kOutputBatchMeanIndex);
  TF_LITE_ENSURE(context, batch_mean != nullptr);
  TfLiteTensor* batch_var =
      tflite::GetOutput(context, node, kOutputBatchVarIndex);
  TF_LITE_ENSURE(context, batch_var != nullptr);
  TfLiteTensor* saved_mean =
      tflite::GetOutput(context, node, kOutputSavedMeanIndex);
  TF_LITE_ENSURE(context, saved_mean != nullptr);
  TfLiteTensor* saved_var =
      tflite::GetOutput(context, node, kOutputSavedVarIndex);
  TF_LITE_ENSURE(context, saved_var != nullptr);
  TfLiteTensor* dummy_reserve_space = tflite::GetOutput(context, node, 5);
  TF_LITE_ENSURE(context, dummy_reserve_space != nullptr);

  const TfLiteTensor* input = tflite::GetInput(context, node, kInputIndex);
  TF_LITE_ENSURE(context, input != nullptr);
  const TfLiteTensor* scale = tflite::GetInput(context, node, kInputScaleIndex);
  TF_LITE_ENSURE(context, scale != nullptr);
  const TfLiteTensor* offset =
      tflite::GetInput(context, node, kInputOffsetIndex);
  TF_LITE_ENSURE(context, offset != nullptr);
  const TfLiteTensor* estimated_mean =
      tflite::GetInput(context, node, kInputEstimatedMeanIndex);
  TF_LITE_ENSURE(context, estimated_mean != nullptr);
  const TfLiteTensor* estimated_var =
      tflite::GetInput(context, node, kInputEstimatedVarIndex);
  TF_LITE_ENSURE(context, estimated_var != nullptr);

  TF_LITE_ENSURE_EQ(context, tflite::NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, tflite::NumDimensions(scale), 1);
  TF_LITE_ENSURE_EQ(context, tflite::NumDimensions(offset), 1);
  TF_LITE_ENSURE_EQ(context, tflite::NumDimensions(estimated_mean), 1);
  TF_LITE_ENSURE_EQ(context, tflite::NumDimensions(estimated_var), 1);
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, scale->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, offset->type, kTfLiteFloat32);

  int batches = input->dims->data[0];
  int height = input->dims->data[1];
  int width = input->dims->data[2];
  int depth = input->dims->data[3];
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = batches;
  output_size->data[1] = height;
  output_size->data[2] = width;
  output_size->data[3] = depth;
  if (context->ResizeTensor(context, output, output_size) != kTfLiteOk) {
    return kTfLiteError;
  }
  TfLiteIntArray* batch_mean_size = TfLiteIntArrayCreate(1);
  batch_mean_size->data[0] = depth;
  if (context->ResizeTensor(context, batch_mean, batch_mean_size) !=
      kTfLiteOk) {
    return kTfLiteError;
  }
  TfLiteIntArray* batch_var_size = TfLiteIntArrayCreate(1);
  batch_var_size->data[0] = depth;
  if (context->ResizeTensor(context, batch_var, batch_var_size) != kTfLiteOk) {
    return kTfLiteError;
  }
  TfLiteIntArray* saved_mean_size = TfLiteIntArrayCreate(1);
  saved_mean_size->data[0] = depth;
  if (context->ResizeTensor(context, saved_mean, saved_mean_size) !=
      kTfLiteOk) {
    return kTfLiteError;
  }
  TfLiteIntArray* saved_var_size = TfLiteIntArrayCreate(1);
  saved_var_size->data[0] = depth;
  if (context->ResizeTensor(context, saved_var, saved_var_size) != kTfLiteOk) {
    return kTfLiteError;
  }
  TfLiteIntArray* dummy_reserve_size = TfLiteIntArrayCreate(1);
  dummy_reserve_size->data[0] = 1;
  if (context->ResizeTensor(context, dummy_reserve_space, dummy_reserve_size) !=
      kTfLiteOk) {
    return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = tflite::GetInput(context, node, kInputIndex);
  TF_LITE_ENSURE(context, input != nullptr);
  const TfLiteTensor* scale = tflite::GetInput(context, node, kInputScaleIndex);
  TF_LITE_ENSURE(context, scale != nullptr);
  const TfLiteTensor* offset =
      tflite::GetInput(context, node, kInputOffsetIndex);
  TF_LITE_ENSURE(context, offset != nullptr);
  const TfLiteTensor* estimated_mean =
      tflite::GetInput(context, node, kInputEstimatedMeanIndex);
  TF_LITE_ENSURE(context, estimated_mean != nullptr);
  const TfLiteTensor* estimated_var =
      tflite::GetInput(context, node, kInputEstimatedVarIndex);
  TF_LITE_ENSURE(context, estimated_var != nullptr);

  TfLiteTensor* output = tflite::GetOutput(context, node, kOutputIndex);
  TF_LITE_ENSURE(context, output != nullptr);
  TfLiteTensor* batch_mean =
      tflite::GetOutput(context, node, kOutputBatchMeanIndex);
  TF_LITE_ENSURE(context, batch_mean != nullptr);
  TfLiteTensor* batch_var =
      tflite::GetOutput(context, node, kOutputBatchVarIndex);
  TF_LITE_ENSURE(context, batch_var != nullptr);
  TfLiteTensor* saved_mean =
      tflite::GetOutput(context, node, kOutputSavedMeanIndex);
  TF_LITE_ENSURE(context, saved_mean != nullptr);
  TfLiteTensor* saved_var =
      tflite::GetOutput(context, node, kOutputSavedVarIndex);
  TF_LITE_ENSURE(context, saved_var != nullptr);

  FusedBarchNorm<float, float>(
      context, const_cast<TfLiteTensor*>(input),
      const_cast<TfLiteTensor*>(scale), const_cast<TfLiteTensor*>(offset),
      const_cast<TfLiteTensor*>(estimated_mean),
      const_cast<TfLiteTensor*>(estimated_var), output, batch_mean, batch_var,
      saved_mean, saved_var, /*exponential_avg_factor=*/0.001f,
      /*epsilon=*/0.001f);

  return kTfLiteOk;
}
}  // namespace vision::batch_norm

TfLiteRegistration* Register_FusedBatchNorm() {
  static TfLiteRegistration r = {
      vision::batch_norm::Initialize, vision::batch_norm::Free,
      vision::batch_norm::Prepare, vision::batch_norm::Eval};
  return &r;
}

}  // namespace mediapipe::tflite_operations
