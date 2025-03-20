// Copyright 2023 The MediaPipe Authors.
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

#include "mediapipe/calculators/tensor/tensor_converter_cpu.h"

#include <algorithm>
#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"

namespace mediapipe {
namespace {

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    RowMajorMatrixXf;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
    ColMajorMatrixXf;

template <class T>
absl::Status NormalizeImage(const ImageFrame& image_frame, bool flip_vertically,
                            const std::pair<float, float>& output_range,
                            int max_num_channels, float* tensor_ptr) {
  const int height = image_frame.Height();
  const int width = image_frame.Width();
  const int channels = image_frame.NumberOfChannels();
  const int channels_preserved = std::min(channels, max_num_channels);
  const int channels_ignored = channels - channels_preserved;

  RET_CHECK_NE(output_range.first, output_range.second);
  const float scale = (output_range.second - output_range.first) / 255.0f;
  const float bias = output_range.first;

  for (int i = 0; i < height; ++i) {
    const T* image_ptr = reinterpret_cast<const T*>(
        image_frame.PixelData() +
        (flip_vertically ? height - 1 - i : i) * image_frame.WidthStep());
    for (int j = 0; j < width; ++j) {
      for (int c = 0; c < channels_preserved; ++c) {
        *tensor_ptr++ = *image_ptr++ * scale + bias;
      }
      image_ptr += channels_ignored;
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status NormalizeUInt8Image(const ImageFrame& image_frame,
                                 bool flip_vertically,
                                 const std::pair<float, float>& output_range,
                                 int max_num_channels, float* tensor_ptr) {
  return NormalizeImage<uint8_t>(image_frame, flip_vertically, output_range,
                                 max_num_channels, tensor_ptr);
}

absl::Status NormalizeFloatImage(const ImageFrame& image_frame,
                                 bool flip_vertically,
                                 const std::pair<float, float>& output_range,
                                 int max_num_channels, float* tensor_ptr) {
  return NormalizeImage<float>(image_frame, flip_vertically, output_range,
                               max_num_channels, tensor_ptr);
}

absl::Status CopyMatrixToTensor(const Matrix& matrix, bool is_row_major_matrix,
                                float* tensor_ptr) {
  if (is_row_major_matrix) {
    auto matrix_map =
        Eigen::Map<RowMajorMatrixXf>(tensor_ptr, matrix.rows(), matrix.cols());
    matrix_map = matrix;
  } else {
    auto matrix_map =
        Eigen::Map<ColMajorMatrixXf>(tensor_ptr, matrix.rows(), matrix.cols());
    matrix_map = matrix;
  }
  return absl::OkStatus();
}

absl::StatusOr<Tensor> ConvertImageFrameToTensorOnCpu(
    const ImageFrame& image_frame, const std::pair<float, float>& output_range,
    bool flip_vertically, int max_num_channels, MemoryManager* memory_manager) {
  const int height = image_frame.Height();
  const int width = image_frame.Width();
  const int channels = image_frame.NumberOfChannels();
  const int channels_preserved = std::min(channels, max_num_channels);
  const mediapipe::ImageFormat::Format format = image_frame.Format();

  if (!(format == mediapipe::ImageFormat::SRGBA ||
        format == mediapipe::ImageFormat::SRGB ||
        format == mediapipe::ImageFormat::GRAY8 ||
        format == mediapipe::ImageFormat::VEC32F1))
    RET_CHECK_FAIL() << "Unsupported CPU input format.";

  Tensor output_tensor(Tensor::ElementType::kFloat32,
                       Tensor::Shape{1, height, width, channels_preserved},
                       memory_manager);
  auto cpu_view = output_tensor.GetCpuWriteView();

  // Copy image data into tensor.
  if (image_frame.ByteDepth() == 1) {
    MP_RETURN_IF_ERROR(NormalizeUInt8Image(image_frame, flip_vertically,
                                           output_range, max_num_channels,
                                           cpu_view.buffer<float>()));
  } else if (image_frame.ByteDepth() == 4) {
    MP_RETURN_IF_ERROR(NormalizeFloatImage(image_frame, flip_vertically,
                                           output_range, max_num_channels,
                                           cpu_view.buffer<float>()));
  } else {
    return absl::InternalError(
        "Only byte-based (8 bit) and float (32 bit) images supported.");
  }
  return output_tensor;
}

absl::StatusOr<Tensor> ConvertMatrixToTensorOnCpu(
    const Matrix& matrix, bool row_major_matrix,
    MemoryManager* memory_manager) {
  const int height = matrix.rows();
  const int width = matrix.cols();
  const int channels = 1;
  Tensor output_tensor(Tensor::ElementType::kFloat32,
                       Tensor::Shape{1, height, width, channels},
                       memory_manager);
  MP_RETURN_IF_ERROR(
      CopyMatrixToTensor(matrix, row_major_matrix,
                         output_tensor.GetCpuWriteView().buffer<float>()));
  return output_tensor;
}

}  // namespace mediapipe
