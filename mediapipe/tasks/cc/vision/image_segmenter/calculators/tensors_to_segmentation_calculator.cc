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

#include <algorithm>
#include <cstdint>
#include <limits>
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
#include "mediapipe/tasks/cc/vision/image_segmenter/calculators/tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/proto/segmenter_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"
#include "mediapipe/util/label_map.pb.h"

#ifdef __EMSCRIPTEN__
#include "mediapipe/tasks/cc/vision/image_segmenter/calculators/segmentation_postprocessor_gl.h"
#endif  // __EMSCRIPTEN__

// TODO: consolidate TensorToSegmentationCalculator.
namespace mediapipe {
namespace tasks {
namespace {

using ::mediapipe::Image;
using ::mediapipe::ImageFrameSharedPtr;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Node;
using ::mediapipe::api2::Output;
using ::mediapipe::tasks::TensorsToSegmentationCalculatorOptions;
using ::mediapipe::tasks::vision::GetImageLikeTensorShape;
using ::mediapipe::tasks::vision::Shape;
using ::mediapipe::tasks::vision::image_segmenter::proto::SegmenterOptions;

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

std::vector<Image> ProcessForCategoryMaskCpu(const Shape& input_shape,
                                             const Shape& output_shape,
                                             const SegmenterOptions& options,
                                             const float* tensors_buffer) {
  cv::Mat resized_tensors_mat;
  cv::Mat tensors_mat_view(
      input_shape.height, input_shape.width, CV_32FC(input_shape.channels),
      reinterpret_cast<void*>(const_cast<float*>(tensors_buffer)));
  if (output_shape.height == input_shape.height &&
      output_shape.width == input_shape.width) {
    resized_tensors_mat = tensors_mat_view;
  } else {
    // Resize input tensors to output size.
    // TOOD(b/273633027) Use an efficient way to find values for category mask
    // instead of resizing the whole tensor .
    cv::resize(tensors_mat_view, resized_tensors_mat,
               {output_shape.width, output_shape.height}, 0, 0,
               cv::INTER_LINEAR);
  }

  // Category mask Image.
  ImageFrameSharedPtr image_frame_ptr = std::make_shared<ImageFrame>(
      ImageFormat::GRAY8, output_shape.width, output_shape.height, 1);
  Image category_mask(image_frame_ptr);

  // Fill in the maximum category in the category mask image.
  cv::Mat category_mask_mat_view =
      mediapipe::formats::MatView(image_frame_ptr.get());
  const int input_channels = input_shape.channels;
  category_mask_mat_view.forEach<uint8_t>(
      [&resized_tensors_mat, &input_channels, &options](uint8_t& pixel,
                                                        const int position[]) {
        float* tensors_buffer =
            resized_tensors_mat.ptr<float>(position[0], position[1]);
        absl::Span<float> confidence_scores(tensors_buffer, input_channels);
        // Only process the activation function if it is SIGMOID. If NONE,
        // we do nothing for activation, If SOFTMAX, it is required
        // to have input_channels > 1, and for input_channels > 1, we don't need
        // activation to find the maximum value.
        if (options.activation() == SegmenterOptions::SIGMOID) {
          Sigmoid(confidence_scores, confidence_scores);
        }
        if (input_channels == 1) {
          // if the input tensor is a single mask, it is assumed to be a binary
          // foreground segmentation mask. For such a mask, we make foreground
          // category 1, and background category 0.
          pixel = static_cast<uint8_t>(*tensors_buffer > 0.5f);
        } else {
          const int maximum_category_idx =
              std::max_element(confidence_scores.begin(),
                               confidence_scores.end()) -
              confidence_scores.begin();
          pixel = maximum_category_idx;
        }
      });
  return {category_mask};
}

std::vector<Image> ProcessForConfidenceMaskCpu(const Shape& input_shape,
                                               const Shape& output_shape,
                                               const SegmenterOptions& options,
                                               const float* tensors_buffer) {
  std::function<void(absl::Span<const float> values,
                     absl::Span<float> activated_values)>
      activation_fn;
  switch (options.activation()) {
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

  // TODO Use libyuv for resizing instead.
  std::vector<Image> confidence_masks;
  std::vector<cv::Mat> confidence_mask_mats;
  confidence_masks.reserve(input_shape.channels);
  confidence_mask_mats.reserve(input_shape.channels);
  for (int i = 0; i < input_shape.channels; ++i) {
    confidence_masks.push_back(Image(std::make_shared<ImageFrame>(
        ImageFormat::VEC32F1, input_shape.width, input_shape.height, 1)));
    confidence_mask_mats.push_back(mediapipe::formats::MatView(
        confidence_masks.back().GetImageFrameSharedPtr().get()));
  }

  // Applies activation function.
  const int tensor_size = input_shape.height * input_shape.width;
  std::vector<float> activated_values(input_shape.channels);
  absl::Span<float> activated_values_span(activated_values);
  for (int i = 0; i < tensor_size; ++i) {
    activation_fn(absl::MakeConstSpan(&tensors_buffer[i * input_shape.channels],
                                      input_shape.channels),
                  activated_values_span);
    for (int j = 0; j < input_shape.channels; ++j) {
      confidence_mask_mats[j].at<float>(
          i / input_shape.width, i % input_shape.width) = activated_values[j];
    }
  }
  if (output_shape.height == input_shape.height &&
      output_shape.width == input_shape.width) {
    return confidence_masks;
  }
  std::vector<Image> resized_confidence_masks;
  resized_confidence_masks.reserve(confidence_mask_mats.size());
  // Resizes segmented masks to required output size.
  for (int i = 0; i < confidence_mask_mats.size(); i++) {
    // Pre-allocates ImageFrame memory to avoid copying from cv::Mat
    // afterward.
    ImageFrameSharedPtr image_frame_ptr = std::make_shared<ImageFrame>(
        ImageFormat::VEC32F1, output_shape.width, output_shape.height, 1);
    cv::Mat resized_mask_mat_view =
        mediapipe::formats::MatView(image_frame_ptr.get());
    cv::resize(confidence_mask_mats[i], resized_mask_mat_view,
               resized_mask_mat_view.size(), 0, 0, cv::INTER_LINEAR);
    resized_confidence_masks.push_back(Image(image_frame_ptr));
  }
  return resized_confidence_masks;
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
//   Segmentation: Segmentation proto.
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

  static absl::Status UpdateContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc);
  absl::Status Process(CalculatorContext* cc);

 private:
  std::vector<Image> GetSegmentationResultCpu(const Shape& input_shape,
                                              const Shape& output_shape,
                                              const float* tensors_buffer);
  TensorsToSegmentationCalculatorOptions options_;

#ifdef __EMSCRIPTEN__
  SegmentationPostprocessorGl postprocessor_;
#endif  // __EMSCRIPTEN__
};

// static
absl::Status TensorsToSegmentationCalculator::UpdateContract(
    CalculatorContract* cc) {
#ifdef __EMSCRIPTEN__
  return SegmentationPostprocessorGl::UpdateContract(cc);
#else
  return absl::OkStatus();
#endif  // __EMSCRIPTEN__
}

absl::Status TensorsToSegmentationCalculator::Open(
    mediapipe::CalculatorContext* cc) {
  options_ = cc->Options<TensorsToSegmentationCalculatorOptions>();
  RET_CHECK_NE(options_.segmenter_options().output_type(),
               SegmenterOptions::UNSPECIFIED)
      << "Must specify output_type as one of [CONFIDENCE_MASK|CATEGORY_MASK].";
#ifdef __EMSCRIPTEN__
  MP_RETURN_IF_ERROR(postprocessor_.Initialize(cc, options_));
#endif  // __EMSCRIPTEN__
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

  // Use GPU postprocessing on web when Tensor is there already and has <= 12
  // categories.
#ifdef __EMSCRIPTEN__
  if (input_tensor.ready_as_opengl_texture_2d() && input_shape.channels <= 12) {
    std::vector<std::unique_ptr<Image>> segmented_masks =
        postprocessor_.GetSegmentationResultGpu(input_shape, output_shape,
                                                input_tensor);
    for (int i = 0; i < segmented_masks.size(); ++i) {
      kSegmentationOut(cc)[i].Send(std::move(segmented_masks[i]));
    }
    return absl::OkStatus();
  }
#endif  // __EMSCRIPTEN__

  // Otherwise, use CPU postprocessing.
  std::vector<Image> segmented_masks = GetSegmentationResultCpu(
      input_shape, output_shape, input_tensor.GetCpuReadView().buffer<float>());
  for (int i = 0; i < segmented_masks.size(); ++i) {
    kSegmentationOut(cc)[i].Send(std::move(segmented_masks[i]));
  }
  return absl::OkStatus();
}

std::vector<Image> TensorsToSegmentationCalculator::GetSegmentationResultCpu(
    const Shape& input_shape, const Shape& output_shape,
    const float* tensors_buffer) {
  if (options_.segmenter_options().output_type() ==
      SegmenterOptions::CATEGORY_MASK) {
    return ProcessForCategoryMaskCpu(input_shape, output_shape,
                                     options_.segmenter_options(),
                                     tensors_buffer);
  } else {
    return ProcessForConfidenceMaskCpu(input_shape, output_shape,
                                       options_.segmenter_options(),
                                       tensors_buffer);
  }
}

MEDIAPIPE_REGISTER_NODE(::mediapipe::tasks::TensorsToSegmentationCalculator);

}  // namespace tasks
}  // namespace mediapipe
