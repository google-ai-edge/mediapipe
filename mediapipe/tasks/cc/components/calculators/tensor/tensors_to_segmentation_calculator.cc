/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

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

// TODO consolidate TensorsToSegmentationCalculator.
#include <algorithm>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/components/calculators/tensor/tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/tasks/cc/components/segmenter_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"
#include "mediapipe/util/label_map.pb.h"

namespace mediapipe {
namespace api2 {

namespace {

using ::mediapipe::Image;
using ::mediapipe::ImageFrameSharedPtr;
using ::mediapipe::tasks::SegmenterOptions;
using ::mediapipe::tasks::TensorsToSegmentationCalculatorOptions;
using ::mediapipe::tasks::vision::GetImageLikeTensorShape;
using ::mediapipe::tasks::vision::Shape;

void StableSoftmax(absl::Span<const float> values,
                   absl::Span<float> activated_values) {
  float max_value = *std::max_element(values.begin(), values.end());
  float denominator = 0.f;
  std::transform(values.begin(), values.end(), activated_values.begin(),
                 [&](float val) {
                   float exp_val = std::exp(val - max_value);
                   denominator += exp_val;
                   return exp_val;
                 });
  std::transform(activated_values.begin(), activated_values.end(),
                 activated_values.begin(),
                 [&denominator](float val) { return val / denominator; });
}

void Sigmoid(absl::Span<const float> values,
             absl::Span<float> activated_values) {
  std::transform(values.begin(), values.end(), activated_values.begin(),
                 [](float value) { return 1. / (1 + std::exp(-value)); });
}

}  // namespace

// Converts Tensors from a vector of Tensor to Segmentation.
//
// Performs optional resizing to OUTPUT_SIZE dimension if provided,
// otherwise the segmented masks is the same size as input tensor.
//
// Inputs:
//   TENSORS: Vector containing a single KTfLiteFloat32 Tensor to be converted
//            to segmentation masks.
//   OUTPUT_SIZE(optional): std::pair<int, int>. Height and Width, if provided,
//            the size to resize masks to.
//
// Output:
//   Segmentation: Segmenation proto.
//
// Options:
//   See tensors_to_segmentation_calculator.proto
//
// Usage example:
//  node {
//    calculator: "TensorsToSegmentationCalculator"
//    input_stream: "TENSORS:tensors"
//    input_stream: "OUTPUT_SIZE:size"
//    output_stream: "SEGMENTATION:0:segmentation"
//    output_stream: "SEGMENTATION:1:segmentation"
//    options {
//      [mediapipe.tasks.TensorsToSegmentationCalculatorOptions.ext] {
//        segmenter_options {
//          activation: SOFTMAX
//          output_type: CONFIDENCE_MASK
//        }
//      }
//    }
//  }
class TensorsToSegmentationCalculator : public Node {
 public:
  static constexpr Input<std::vector<Tensor>> kTensorsIn{"TENSORS"};
  static constexpr Input<std::pair<int, int>>::Optional kOutputSizeIn{
      "OUTPUT_SIZE"};
  static constexpr Output<Image>::Multiple kSegmentationOut{"SEGMENTATION"};
  MEDIAPIPE_NODE_CONTRACT(kTensorsIn, kOutputSizeIn, kSegmentationOut);

  absl::Status Open(CalculatorContext* cc);
  absl::Status Process(CalculatorContext* cc);

 private:
  std::vector<Image> GetSegmentationResult(const Shape& input_shape,
                                           const Shape& output_shape,
                                           const float* tensors_buffer);

  TensorsToSegmentationCalculatorOptions options_;
};

absl::Status TensorsToSegmentationCalculator::Open(
    mediapipe::CalculatorContext* cc) {
  options_ =
      cc->Options<mediapipe::tasks::TensorsToSegmentationCalculatorOptions>();
  RET_CHECK_NE(options_.segmenter_options().output_type(),
               SegmenterOptions::UNSPECIFIED)
      << "Must specify output_type as one of [CONFIDENCE_MASK|CATEGORY_MASK].";
  return absl::OkStatus();
}

absl::Status TensorsToSegmentationCalculator::Process(
    mediapipe::CalculatorContext* cc) {
  RET_CHECK_EQ(kTensorsIn(cc).Get().size(), 1)
      << "Expect a vector of single Tensor.";
  const auto& input_tensor = kTensorsIn(cc).Get()[0];
  ASSIGN_OR_RETURN(const Shape input_shape,
                   GetImageLikeTensorShape(input_tensor));

  // Category mask does not require activation function.
  if (options_.segmenter_options().output_type() ==
          SegmenterOptions::CONFIDENCE_MASK &&
      options_.segmenter_options().activation() == SegmenterOptions::SOFTMAX) {
    RET_CHECK_GT(input_shape.channels, 1)
        << "SOFTMAX activation requires channels > 1.";
  }

  int output_height = input_shape.height;
  int output_width = input_shape.width;
  if (cc->Inputs().HasTag("OUTPUT_SIZE")) {
    std::tie(output_width, output_height) = kOutputSizeIn(cc).Get();
  }
  Shape output_shape = {
      /* height= */ output_height,
      /* width= */ output_width,
      /* channels= */ options_.segmenter_options().output_type() ==
              SegmenterOptions::CATEGORY_MASK
          ? 1
          : input_shape.channels};

  std::vector<Image> segmented_masks = GetSegmentationResult(
      input_shape, output_shape, input_tensor.GetCpuReadView().buffer<float>());
  for (int i = 0; i < segmented_masks.size(); ++i) {
    kSegmentationOut(cc)[i].Send(std::move(segmented_masks[i]));
  }
  return absl::OkStatus();
}

std::vector<Image> TensorsToSegmentationCalculator::GetSegmentationResult(
    const Shape& input_shape, const Shape& output_shape,
    const float* tensors_buffer) {
  std::function<void(absl::Span<const float> values,
                     absl::Span<float> activated_values)>
      activation_fn;
  switch (options_.segmenter_options().activation()) {
    case SegmenterOptions::SIGMOID:
      activation_fn = &Sigmoid;
      break;
    case SegmenterOptions::SOFTMAX:
      activation_fn = &StableSoftmax;
      break;
    case SegmenterOptions::NONE:
      // Just copying for NONE activation.
      activation_fn = [](absl::Span<const float> values,
                         absl::Span<float> activated_values) {
        std::copy(values.begin(), values.end(), activated_values.begin());
      };
      break;
  }

  const bool is_category_mask = options_.segmenter_options().output_type() ==
                                SegmenterOptions::CATEGORY_MASK;
  const int cv_mat_type = is_category_mask ? CV_8UC1 : CV_32FC1;
  const int output_masks_num = output_shape.channels;

  // TODO Use libyuv for resizing instead.
  std::vector<cv::Mat> segmented_mask_mats;
  segmented_mask_mats.reserve(output_masks_num);
  for (int i = 0; i < output_masks_num; ++i) {
    segmented_mask_mats.push_back(
        cv::Mat(input_shape.height, input_shape.width, cv_mat_type));
  }

  // Applies activation function.
  const int tensor_size = input_shape.height * input_shape.width;
  if (is_category_mask) {
    for (int i = 0; i < tensor_size; ++i) {
      absl::Span<const float> confidence_scores(
          &tensors_buffer[i * input_shape.channels], input_shape.channels);
      const int maximum_category_idx =
          std::max_element(confidence_scores.begin(), confidence_scores.end()) -
          confidence_scores.begin();
      segmented_mask_mats[0].at<uint8_t>(
          i / input_shape.width, i % input_shape.width) = maximum_category_idx;
    }
  } else {
    std::vector<float> activated_values(input_shape.channels);
    absl::Span<float> activated_values_span(activated_values);
    for (int i = 0; i < tensor_size; ++i) {
      activation_fn(
          absl::MakeConstSpan(&tensors_buffer[i * input_shape.channels],
                              input_shape.channels),
          activated_values_span);
      for (int j = 0; j < input_shape.channels; ++j) {
        segmented_mask_mats[j].at<float>(
            i / input_shape.width, i % input_shape.width) = activated_values[j];
      }
    }
  }

  std::vector<Image> segmented_masks;
  segmented_masks.reserve(output_masks_num);
  // Resizes segmented masks to required output size.
  for (int i = 0; i < segmented_mask_mats.size(); i++) {
    // Pre-allocates ImageFrame memory to avoid copying from cv::Mat afterward.
    ImageFrameSharedPtr image_frame_ptr = std::make_shared<ImageFrame>(
        is_category_mask ? ImageFormat::GRAY8 : ImageFormat::VEC32F1,
        output_shape.width, output_shape.height, 1);
    cv::Mat resized_mask_mat_view =
        mediapipe::formats::MatView(image_frame_ptr.get());
    cv::resize(segmented_mask_mats[i], resized_mask_mat_view,
               resized_mask_mat_view.size(), 0, 0,
               cv_mat_type == CV_8UC1 ? cv::INTER_NEAREST : cv::INTER_LINEAR);
    segmented_masks.push_back(Image(image_frame_ptr));
  }
  return segmented_masks;
}

MEDIAPIPE_REGISTER_NODE(TensorsToSegmentationCalculator);

}  // namespace api2
}  // namespace mediapipe
