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

#ifndef MEDIAPIPE_FRAMEWORK_DEBUG_LOGGING_H_
#define MEDIAPIPE_FRAMEWORK_DEBUG_LOGGING_H_

#include "HalideBuffer.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/opencv_core_inc.h"

namespace mediapipe::debug {

// Logs the given channel (=last dimension) of the float tensor as a color or
// ASCII image, depending on terminal capabilities. The values are clamped to
// [min_range, max_range].
void LogTensorChannel(const mediapipe::Tensor& tensor, int channel,
                      absl::string_view name = "tensor", float min_range = 0.0f,
                      float max_range = 1.0f);

// Logs the given float tensor as a color or ASCII image, depending on terminal
// capabilities. Assumes a float tensor with dimensions [1, h, w, c]. The values
// are clamped to [min_range, max_range]. A one-channel tensor is printed as a
// grayscale image, two-channel and three-channel tensors are printed as a color
// image, and a tensor with more channels is averaged over all channels and
// printed as a grayscale image.
void LogTensor(const mediapipe::Tensor& tensor,
               absl::string_view name = "tensor", float min_range = 0.0f,
               float max_range = 1.0f);

// Logs the given image as a color or ASCII image, depending on terminal
// capabilities.
void LogImage(const mediapipe::ImageFrame& image,
              absl::string_view name = "image");

// Logs the given mat as a color or ASCII image, depending on terminal
// capabilities.
void LogMat(const cv::Mat& mat, absl::string_view name = "mat");

// Logs the given Halide buffer as a color or ASCII image, depending on terminal
// capabilities.
void LogHalideBuffer(Halide::Runtime::Buffer<const uint8_t> buffer,
                     absl::string_view name = "buffer");

}  // namespace mediapipe::debug

#endif  // MEDIAPIPE_FRAMEWORK_DEBUG_LOGGING_H_
