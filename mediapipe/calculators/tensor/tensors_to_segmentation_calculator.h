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

#ifndef MEDIAPIPE_CALCULATORS_TENSOR_TENSORS_TO_SEGMENTATION_CALCULATOR_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_TENSORS_TO_SEGMENTATION_CALCULATOR_H_

#include <utility>
#include <vector>

#include "mediapipe/calculators/tensor/tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/tensor.h"

namespace mediapipe::api3 {

// Converts provided tensor(s) to a segmentation mask.
//
// Performs optional upscale to OUTPUT_SIZE dimensions if provided,
// otherwise the mask is the same size as input tensor.
//
// If at least one input tensor is already on GPU, processing happens on GPU and
// the output mask is also stored on GPU. Otherwise, processing and the output
// mask are both on CPU.
//
// On GPU, the mask is an RGBA image, in both the R & A channels, scaled 0-1.
// On CPU, the mask is a ImageFormat::VEC32F1 image, with values scaled 0-1.
//
// Usage example:
// node {
//   calculator: "TensorsToSegmentationCalculator"
//   input_stream: "TENSORS:tensors"
//   input_stream: "OUTPUT_SIZE:size"
//   output_stream: "MASK:hair_mask"
//   node_options: {
//     [mediapipe.TensorsToSegmentationCalculatorOptions] {
//       output_layer_index: 1
//       # gpu_origin: CONVENTIONAL # or TOP_LEFT
//     }
//   }
// }
struct TensorsToSegmentationNode : Node<"TensorsToSegmentationCalculator"> {
  template <typename S>
  struct Contract {
    // Vector of Tensors of type kFloat32. Only the first tensor will be used.
    //
    // NOTE: Either TENSORS or TENSOR must be specified.
    Optional<Input<S, std::vector<Tensor>>> tensors_in{"TENSORS"};

    // Tensor of type kFloat32. Use this instead of TENSORS when the
    // tensors are available as individual Tensor streams, not as a stream
    // of vector of Tensors.
    //
    // NOTE: Either TENSOR or TENSORS must be specified.
    Optional<Input<S, Tensor>> tensor_in{"TENSOR"};

    // The size to scale output segmentation mask to.
    Optional<Input<S, std::pair<int, int>>> output_size_in{"OUTPUT_SIZE"};

    // Output mask, RGBA(GPU) / VEC32F1(CPU).
    Output<S, Image> mask_out{"MASK"};

    // Node options (e.g. activation).
    Options<S, mediapipe::TensorsToSegmentationCalculatorOptions> options;
  };
};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_TENSORS_TO_SEGMENTATION_CALCULATOR_H_
