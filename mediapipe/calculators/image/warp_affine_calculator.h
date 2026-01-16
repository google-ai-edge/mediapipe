// Copyright 2021 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_CALCULATORS_IMAGE_WARP_AFFINE_CALCULATOR_H_
#define MEDIAPIPE_CALCULATORS_IMAGE_WARP_AFFINE_CALCULATOR_H_

#include <array>
#include <utility>

#include "absl/strings/string_view.h"
#include "mediapipe/calculators/image/warp_affine_calculator.pb.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"

#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gpu_buffer.h"
#endif  // !MEDIAPIPE_DISABLE_GPU

namespace mediapipe::api3 {

template <typename ImageT>
inline constexpr absl::string_view kWarpAffineNodeName;

// Runs affine transformation.
//
// Input:
//   IMAGE - Image/ImageFrame/GpuBuffer
//
//   MATRIX - std::array<float, 16>
//     Used as following:
//       output(x, y) = input(matrix[0] * x + matrix[1] * y + matrix[3],
//                            matrix[4] * x + matrix[5] * y + matrix[7])
//     where x and y ranges are defined by @OUTPUT_SIZE.
//
//   OUTPUT_SIZE - std::pair<int, int>
//     Size of the output image.
//
// Output:
//   IMAGE - Image/ImageFrame/GpuBuffer
//
//   Note:
//   - Output image type and format are the same as the input one.
//
// Usage example:
//   node {
//     calculator: "WarpAffineCalculator(Cpu|Gpu)"
//     input_stream: "IMAGE:image"
//     input_stream: "MATRIX:matrix"
//     input_stream: "OUTPUT_SIZE:size"
//     output_stream: "IMAGE:transformed_image"
//     options: {
//       [mediapipe.WarpAffineCalculatorOptions.ext] {
//         border_mode: BORDER_ZERO
//       }
//     }
//   }
template <typename ImageT>
struct WarpAffineNode : Node<kWarpAffineNodeName<ImageT>> {
  template <typename S>
  struct Contract {
    Input<S, ImageT> in_image{"IMAGE"};
    Input<S, std::array<float, 16>> matrix{"MATRIX"};
    Input<S, std::pair<int, int>> output_size{"OUTPUT_SIZE"};

    Output<S, ImageT> out_image{"IMAGE"};

    Options<S, mediapipe::WarpAffineCalculatorOptions> options;
  };
};

#if !MEDIAPIPE_DISABLE_OPENCV

template <>
inline constexpr absl::string_view kWarpAffineNodeName<ImageFrame> =
    "WarpAffineCalculatorCpu";

#endif  // !MEDIAPIPE_DISABLE_OPENCV

#if !MEDIAPIPE_DISABLE_GPU

template <>
inline constexpr absl::string_view kWarpAffineNodeName<GpuBuffer> =
    "WarpAffineCalculatorGpu";

#endif  // !MEDIAPIPE_DISABLE_GPU

template <>
inline constexpr absl::string_view kWarpAffineNodeName<Image> =
    "WarpAffineCalculator";

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_CALCULATORS_IMAGE_WARP_AFFINE_CALCULATOR_H_
