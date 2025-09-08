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

#ifndef MEDIAPIPE_CALCULATORS_TENSOR_IMAGE_TO_TENSOR_CALCULATOR_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_IMAGE_TO_TENSOR_CALCULATOR_H_

#include <array>
#include <vector>

#include "absl/strings/string_view.h"
#include "mediapipe/calculators/tensor/image_to_tensor_calculator.pb.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/api3/one_of.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/gpu/gpu_buffer.h"

namespace mediapipe::api3 {

inline constexpr absl::string_view kImageToTensorNodeName =
    "ImageToTensorCalculator";

// Converts image into Tensor, possibly with cropping, resizing and
// normalization, according to specified inputs and options.
//
// NOTE:
//   - One and only one of IMAGE and IMAGE_GPU should be specified.
//   - IMAGE input of type Image is processed on GPU if the data is already on
//     GPU (i.e., Image::UsesGpu() returns true), or otherwise processed on CPU.
//   - IMAGE input of type ImageFrame is always processed on CPU.
//   - IMAGE_GPU input (of type GpuBuffer) is always processed on GPU.
//
// Example:
//   node {
//     calculator: "ImageToTensorCalculator"
//     input_stream: "IMAGE:image"  # or "IMAGE_GPU:image"
//     input_stream: "NORM_RECT:roi"
//     output_stream: "TENSORS:tensors"
//     output_stream: "MATRIX:matrix"
//     options {
//       [mediapipe.ImageToTensorCalculatorOptions.ext] {
//         output_tensor_width: 256
//         output_tensor_height: 256
//         keep_aspect_ratio: false
//         output_tensor_float_range {
//           min: 0.0
//           max: 1.0
//         }
//         # gpu_origin: CONVENTIONAL # or TOP_LEFT
//       }
//     }
//   }
struct ImageToTensorNode : Node<kImageToTensorNodeName> {
  template <typename S>
  struct Contract {
    // Image[ImageFormat::SRGB / SRGBA, GpuBufferFormat::kBGRA32] or
    // ImageFrame [ImageFormat::SRGB/SRGBA] to extract from.
    //
    // NOTE: Either "IMAGE" or "IMAGE_GPU" must be specified.
    Optional<Input<S, OneOf<Image, mediapipe::ImageFrame>>> in{"IMAGE"};

    // GpuBuffer [GpuBufferFormat::kBGRA32] to extract from.
    //
    // NOTE: Either "IMAGE" or "IMAGE_GPU" must be specified.
    Optional<Input<S, GpuBuffer>> in_gpu{"IMAGE_GPU"};

    // Describes region of image to extract.
    // If not specified - rect covering the whole image is used.
    Optional<Input<S, NormalizedRect>> in_norm_rect{"NORM_RECT"};

    // Vector containing a single Tensor populated with an extracted RGB image.
    // NOTE: Either "TENSORS" or "TENSOR" must be used.
    Optional<Output<S, std::vector<Tensor>>> out_tensors{"TENSORS"};

    // Individual output tensor.
    // NOTE: Either "TENSORS" or "TENSOR" must be used.
    Optional<Output<S, Tensor>> out_tensor{"TENSOR"};

    // An std::array<float, 16> representing a 4x4 row-major-order matrix that
    // maps a point on the input image to a point on the output tensor, and
    // can be used to reverse the mapping by inverting the matrix.
    Optional<Output<S, std::array<float, 16>>> out_matrix{"MATRIX"};

    // An std::array<float, 4> representing the letterbox padding from the 4
    // sides ([left, top, right, bottom]) of the output image, normalized to
    // [0.f, 1.f] by the output dimensions. The padding values are non-zero
    // only when the "keep_aspect_ratio" is true.
    //
    // For instance, when the input image is 10x10 (width x height) and the
    // output dimensions specified in the calculator option are 20x40 and
    // "keep_aspect_ratio" is true, the calculator scales the input image to
    // 20x20 and places it in the middle of the output image with an equal
    // padding of 10 pixels at the top and the bottom. The resulting array
    // is therefore [0.f, 0.25f, 0.f, 0.25f] (10/40 = 0.25f).
    //
    // DEPRECATED: use MATRIX instead.
    Optional<Output<S, std::array<float, 4>>> out_letterbox_padding{
        "LETTERBOX_PADDING"};

    // Node options.
    Options<S, mediapipe::ImageToTensorCalculatorOptions> options;
  };
};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_IMAGE_TO_TENSOR_CALCULATOR_H_
