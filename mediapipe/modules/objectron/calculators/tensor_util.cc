// Copyright 2020 The MediaPipe Authors.
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

#include "mediapipe/modules/objectron/calculators/tensor_util.h"

#include "absl/log/absl_check.h"
#include "mediapipe/framework/port/logging.h"

namespace mediapipe {

cv::Mat ConvertTfliteTensorToCvMat(const TfLiteTensor& tensor) {
  // Check tensor is BxCxWxH (size = 4) and the batch size is one(data[0] = 1)
  ABSL_CHECK(tensor.dims->size == 4 && tensor.dims->data[0] == 1);
  ABSL_CHECK_EQ(kTfLiteFloat32, tensor.type)
      << "tflite_tensor type is not float";

  const size_t num_output_channels = tensor.dims->data[3];
  const int dims = 2;
  const int sizes[] = {tensor.dims->data[1], tensor.dims->data[2]};
  const int type = CV_MAKETYPE(CV_32F, num_output_channels);
  return cv::Mat(dims, sizes, type, reinterpret_cast<void*>(tensor.data.f));
}

cv::Mat ConvertTensorToCvMat(const mediapipe::Tensor& tensor) {
  // Check tensor is BxCxWxH (size = 4) and the batch size is one(data[0] = 1)
  ABSL_CHECK(tensor.shape().dims.size() == 4 && tensor.shape().dims[0] == 1);
  ABSL_CHECK_EQ(
      mediapipe::Tensor::ElementType::kFloat32 == tensor.element_type(), true)
      << "tensor type is not float";

  const size_t num_output_channels = tensor.shape().dims[3];
  const int dims = 2;
  const int sizes[] = {tensor.shape().dims[1], tensor.shape().dims[2]};
  const int type = CV_MAKETYPE(CV_32F, num_output_channels);
  auto cpu_view = tensor.GetCpuReadView();
  return cv::Mat(dims, sizes, type, const_cast<void*>(cpu_view.buffer<void>()));
}

}  // namespace mediapipe
