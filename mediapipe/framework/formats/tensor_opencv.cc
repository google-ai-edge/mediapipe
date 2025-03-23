// Copyright 2022 The MediaPipe Authors.
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

#include "mediapipe/framework/formats/tensor_opencv.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"

namespace mediapipe::formats {
namespace {

// Maps the Tensor element type to OpenCV Mat type.
absl::StatusOr<int> GetMatType(Tensor::ElementType element_type, int channels) {
  switch (element_type) {
    case Tensor::ElementType::kFloat32:
      return CV_32FC(channels);
    case Tensor::ElementType::kUInt8:
      return CV_8UC(channels);
    case Tensor::ElementType::kInt8:
      return CV_8SC(channels);
    case Tensor::ElementType::kInt32:
      return CV_32SC(channels);
    case Tensor::ElementType::kChar:
      return CV_8SC(channels);
    case Tensor::ElementType::kBool:
      return CV_8UC(channels);
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported Tensor element type: ", static_cast<int>(element_type)));
  }
}

}  // namespace

absl::StatusOr<cv::Mat> MatView(const Tensor& tensor,
                                const Tensor::CpuReadView& view,
                                std::vector<int> slice) {
  // By default, don't slice.
  if (slice.empty()) {
    slice.insert(slice.end(), tensor.shape().dims.size(), -1);
  }

  RET_CHECK_EQ(slice.size(), tensor.shape().dims.size())
          .SetCode(absl::StatusCode::kInvalidArgument)
      << "Slice must have the same number of elements as the number of "
         "dimensions in the tensor.";

  // Unfortunately, OpenCV does not support slicing in the last dimension. The
  // last dimension always has to be consecutive.
  RET_CHECK_EQ(slice.back(), -1).SetCode(absl::StatusCode::kInvalidArgument)
      << "cv::Mat does not support slicing the last dimension.";

  // The mat dimensions is determined by the number of -1s in the slice.
  int num_mat_dims = std::count(slice.begin(), slice.end(), -1);

  // Compute the offset and strides for the sliced mat.
  int offset = 0;
  std::vector<int> mat_dims(num_mat_dims);
  std::vector<size_t> mat_steps(num_mat_dims);
  int curr_mat_dim = num_mat_dims - 1;
  int curr_stride = tensor.element_size();
  for (int n = tensor.shape().dims.size() - 1; n >= 0; --n) {
    if (slice[n] == -1) {
      // Create an additional mat dimension.
      mat_dims[curr_mat_dim] = tensor.shape().dims[n];
      mat_steps[curr_mat_dim] = curr_stride;
      curr_mat_dim--;
    } else {
      // Address a fixed position in this dimension.
      if (slice[n] < 0 || slice[n] >= tensor.shape().dims[n]) {
        return absl::InvalidArgumentError(
            absl::StrCat("Slice ", slice[n], " is out of bounds for dimension ",
                         n, " of shape ", tensor.shape().dims[n]));
      }
      offset += slice[n] * curr_stride;
    }
    curr_stride *= tensor.shape().dims[n];
  }

  // The last mat dimension is technically not a dimension, but rather the
  // number of channels.
  ABSL_CHECK(!mat_dims.empty() && !mat_steps.empty());
  int mat_channels = mat_channels = mat_dims.back();
  mat_dims.pop_back();
  mat_steps.pop_back();

  MP_ASSIGN_OR_RETURN(int mat_type,
                      GetMatType(tensor.element_type(), mat_channels));
  uint8_t* mat_buffer = const_cast<uint8_t*>(view.buffer<uint8_t>() + offset);
  return cv::Mat(mat_dims.size(), mat_dims.data(), mat_type, mat_buffer,
                 mat_steps.data());
}

}  // namespace mediapipe::formats
