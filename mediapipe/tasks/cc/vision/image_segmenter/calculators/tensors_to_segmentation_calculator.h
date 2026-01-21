/* Copyright 2025 The MediaPipe Authors.

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

#ifndef MEDIAPIPE_TASKS_CC_VISION_IMAGE_SEGMENTER_CALCULATORS_TENSORS_TO_SEGMENTATION_CALCULATOR_H_
#define MEDIAPIPE_TASKS_CC_VISION_IMAGE_SEGMENTER_CALCULATORS_TENSORS_TO_SEGMENTATION_CALCULATOR_H_

#include <utility>
#include <vector>

#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/calculators/tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/proto/segmenter_options.pb.h"

namespace mediapipe {
namespace tasks {

// Converts Tensors from a vector of Tensor to Segmentation masks. The
// calculator can output optional confidence masks if CONFIDENCE_MASK is
// connected, and an optional category mask if CATEGORY_MASK is connected. At
// least one of CONFIDENCE_MASK and CATEGORY_MASK must be connected.
//
// Performs optional resizing to OUTPUT_SIZE dimension if provided,
// otherwise the segmented masks is the same size as input tensor.
//
// Usage example:
//  node {
//    calculator: "TensorsToSegmentationCalculator"
//    input_stream: "TENSORS:tensors"
//    input_stream: "OUTPUT_SIZE:size"
//    output_stream: "CONFIDENCE_MASK:0:confidence_mask"
//    output_stream: "CONFIDENCE_MASK:1:confidence_mask"
//    output_stream: "CATEGORY_MASK:category_mask"
//    options {
//      [mediapipe.tasks.TensorsToSegmentationCalculatorOptions.ext] {
//        segmenter_options {
//          activation: SOFTMAX
//        }
//      }
//    }
//  }
struct TensorsToSegmentationNode
    : api3::Node<"::mediapipe::tasks::TensorsToSegmentationCalculator"> {
  template <typename S>
  struct Contract {
    // Vector containing a single KTfLiteFloat32 Tensor to be converted
    // to segmentation masks.
    api3::Input<S, std::vector<Tensor>> tensors_in{"TENSORS"};

    // Height and Width of the output masks, in form of std::pair<int, int>, if
    // provided, the size to resize masks to.
    api3::Optional<api3::Input<S, std::pair<int, int>>> output_size_in{
        "OUTPUT_SIZE"};

    // The calculator can output optional segmentation masks if SEGMENTATION is
    // connected. If CONFIDENCE_MASK is specified as output_type in
    // segmenter_options, the number of segmentation masks must match number of
    // channels in input tensor. If CATEGORY_MASK is specified, then only one
    // segmentation mask is output.
    api3::Repeated<api3::Output<S, Image>> segmentation_out{"SEGMENTATION"};

    // The calculator can output optional confidence masks if CONFIDENCE_MASK is
    // connected, and an optional category mask if CATEGORY_MASK is connected.
    // At least one of CONFIDENCE_MASK and CATEGORY_MASK must be connected.
    api3::Repeated<api3::Output<S, Image>> confidence_mask_out{
        "CONFIDENCE_MASK"};

    // A category mask of uint8 image where each pixel represents the class
    // which the pixel in the original image was predicted to belong to.
    api3::Optional<api3::Output<S, Image>> category_mask_out{"CATEGORY_MASK"};

    // Quality scores for each channel. This is only used when the input tensor
    // has 2 channels. Set to default value 1.0f if the input tensor has 1
    // channel.
    api3::Optional<api3::Output<S, std::vector<float>>> quality_scores_out{
        "QUALITY_SCORES"};

    // Check tensors_to_segmentation_calculator.proto
    api3::Options<S, TensorsToSegmentationCalculatorOptions> options;
  };
};

}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_IMAGE_SEGMENTER_CALCULATORS_TENSORS_TO_SEGMENTATION_CALCULATOR_H_
