// Copyright 2025 The MediaPipe Authors.
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
// Helper functions for working with ImageFrame and OpenCV.
#ifndef MEDIAPIPE_FRAMEWORK_FORMATS_TENSOR_OPENCV_H_
#define MEDIAPIPE_FRAMEWORK_FORMATS_TENSOR_OPENCV_H_

#include <vector>

#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/opencv_core_inc.h"

namespace mediapipe::formats {

// Creates a cv::Mat view into a Tensor (zero copy).
// The passed read view needs to outlive the returned cv::Mat.
//
// If slice is specified, the view is sliced by the specified dimensions. A
// value of -1 means the full dimension is used. For instance, if the tensor
// has shape [2, 64, 128, 3] and slice is [1, -1, -1, 3], the mat will have
// dims [64, 128] and will contain the Tensor[1, :, :, 3]]. If set, slice must
// have the same number of elements as the number of dimensions in the tensor.
//
// When converting a const Tensor into a cv::Mat, the const modifier is lost.
// The caller must be careful not to use the returned object to modify the
// data in a const Tensor, even though the returned data is mutable.
//
// Usage:
//   Tensor tensor = ... tensor of shape [10, 20, 3] ...;
//   MP_ASSIGN_OR_RETURN(TensorMatView view, TensorMatView::Create(tensor));
//   // view.mat is a view into the data in tensor with 2 dims and 3 channels.
//   // view.mat.data points into tensor.GetCpuReadView().buffer<T>().
absl::StatusOr<cv::Mat> MatView(const Tensor& tensor,
                                const Tensor::CpuReadView& view,
                                std::vector<int> slice = {});

// Prevent passing in the view as an rvalue reference, e.g.
// MatView(tensor, tensor.GetCpuReadView());
// The view needs to stay in scope while the mat is in scope.
absl::StatusOr<cv::Mat> MatView(const Tensor& tensor,
                                Tensor::CpuReadView&& view,
                                std::vector<int> slice = {}) = delete;

}  // namespace mediapipe::formats

#endif  // MEDIAPIPE_FRAMEWORK_FORMATS_IMAGE_FRAME_OPENCV_H_
