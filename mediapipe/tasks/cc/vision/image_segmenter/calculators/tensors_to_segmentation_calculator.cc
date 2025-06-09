/* Copyright 2022 The MediaPipe Authors.

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
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/calculators/tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/proto/segmenter_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"
#include "mediapipe/util/label_map.pb.h"

#ifdef __EMSCRIPTEN__
#define TASK_SEGMENTATION_USE_GL_POSTPROCESSING 1
#elif MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31 && \
    !MEDIAPIPE_USING_LEGACY_SWIFTSHADER && defined(MEDIAPIPE_ANDROID)
#define TASK_SEGMENTATION_USE_GL_POSTPROCESSING 1
#else
#undef TASK_SEGMENTATION_USE_GL_POSTPROCESSING
#endif  // __EMSCRIPTEN__

#ifdef TASK_SEGMENTATION_USE_GL_POSTPROCESSING
#include "mediapipe/tasks/cc/vision/image_segmenter/calculators/segmentation_postprocessor_gl.h"
#endif  // TASK_SEGMENTATION_USE_GL_POSTPROCESSING

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

constexpr uint8_t kUnLabeledPixelValue = 255;

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

// Linearly interpolate the value between v0 and v1. Assume 0 <= t <= 1.
float LinearInterpolate(float v0, float v1, float t) {
  return v0 + (v1 - v0) * t;
}

// Bilinearly interpolate the value between 4 points. Assume 0 <= t0, t1 <= 1.
float BilinearInterpolate(float v00, float v10, float v01, float v11, float t0,
                          float t1) {
  return LinearInterpolate(LinearInterpolate(v00, v10, t0),
                           LinearInterpolate(v01, v11, t0), t1);
}

float GetTensorElement(const Shape& input_shape, const float* tensors_buffer,
                       int x, int y, int c) {
  return tensors_buffer[y * input_shape.channels * input_shape.width +
                        x * input_shape.channels + c];
}

Image ProcessForCategoryMaskCpu(const Shape& input_shape,
                                const Shape& output_shape,
                                const SegmenterOptions& options,
                                const float* tensors_buffer) {
  const float width_scale =
      (input_shape.width - 1) / static_cast<float>(output_shape.width - 1);
  const float height_scale =
      (input_shape.height - 1) / static_cast<float>(output_shape.height - 1);

  // Category mask Image.
  ImageFrameSharedPtr image_frame_ptr = std::make_shared<ImageFrame>(
      ImageFormat::GRAY8, output_shape.width, output_shape.height, 1);
  Image category_mask(image_frame_ptr);

  // Fill in the maximum category in the category mask image.
  cv::Mat category_mask_mat_view =
      mediapipe::formats::MatView(image_frame_ptr.get());
  const int input_channels = input_shape.channels;
  category_mask_mat_view.forEach<uint8_t>([&tensors_buffer, &input_shape,
                                           &width_scale, &height_scale,
                                           &input_channels,
                                           &options](uint8_t& pixel,
                                                     const int position[]) {
    std::vector<float> confidence_scores(input_channels);
    int y0 =
        static_cast<int>(std::max(std::floor(position[0] * height_scale), 0.f));
    int x0 =
        static_cast<int>(std::max(std::floor(position[1] * width_scale), 0.f));
    int y1 = static_cast<int>(std::min(std::ceil(position[0] * height_scale),
                                       input_shape.height - 1.f));
    int x1 = static_cast<int>(std ::min(std::ceil(position[1] * width_scale),
                                        input_shape.width - 1.f));
    float t0 = std::max(std::min(position[0] * height_scale - y0, 1.f), 0.f);
    float t1 = std::max(std::min(position[1] * width_scale - x0, 1.f), 0.f);
    for (int i = 0; i < input_channels; ++i) {
      confidence_scores[i] = BilinearInterpolate(
          GetTensorElement(input_shape, tensors_buffer, x0, y0, i),
          GetTensorElement(input_shape, tensors_buffer, x0, y1, i),
          GetTensorElement(input_shape, tensors_buffer, x1, y0, i),
          GetTensorElement(input_shape, tensors_buffer, x1, y1, i), t0, t1);
    }
    absl::Span<float> confidence_scores_span(confidence_scores.data(),
                                             confidence_scores.size());

    // Only process the activation function if it is SIGMOID. If NONE,
    // we do nothing for activation, If SOFTMAX, it is required
    // to have input_channels > 1, and for input_channels > 1, we don't need
    // activation to find the maximum value.
    if (options.activation() == SegmenterOptions::SIGMOID) {
      Sigmoid(confidence_scores_span, confidence_scores_span);
    }
    if (input_channels == 1) {
      // if the input tensor is a single mask, it is assumed to be a binary
      // foreground segmentation mask. For such a mask, instead of a true
      // argmax, we simply use 0.5 as the cutoff, assigning 0 (foreground) or
      // 255 (background) based on whether the confidence value reaches this
      // cutoff or not, respectively.
      pixel = confidence_scores[0] > 0.5f ? 0 : kUnLabeledPixelValue;
    } else {
      const int maximum_category_idx =
          std::max_element(confidence_scores.begin(), confidence_scores.end()) -
          confidence_scores.begin();
      pixel = maximum_category_idx;
    }
  });
  return category_mask;
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

// Converts Tensors from a vector of Tensor to Segmentation masks. The
// calculator can output optional confidence masks if CONFIDENCE_MASK is
// connected, and an optional category mask if CATEGORY_MASK is connected. At
// least one of CONFIDENCE_MASK and CATEGORY_MASK must be connected.
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
//   CONFIDENCE_MASK @Multiple: Multiple masks of float image where, for each
//   mask, each pixel represents the prediction confidence, usually in the [0,
//   1] range.
//   CATEGORY_MASK @Optional: A category mask of uint8 image where each pixel
//   represents the class which the pixel in the original image was predicted to
//   belong to.
//
// Options:
//   See tensors_to_segmentation_calculator.proto
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
class TensorsToSegmentationCalculator : public Node {
 public:
  static constexpr Input<std::vector<Tensor>> kTensorsIn{"TENSORS"};
  static constexpr Input<std::pair<int, int>>::Optional kOutputSizeIn{
      "OUTPUT_SIZE"};
  static constexpr Output<Image>::Multiple kSegmentationOut{"SEGMENTATION"};
  static constexpr Output<Image>::Multiple kConfidenceMaskOut{
      "CONFIDENCE_MASK"};
  static constexpr Output<Image>::Optional kCategoryMaskOut{"CATEGORY_MASK"};
  static constexpr Output<std::vector<float>>::Optional kQualityScoresOut{
      "QUALITY_SCORES"};
  MEDIAPIPE_NODE_CONTRACT(kTensorsIn, kOutputSizeIn, kSegmentationOut,
                          kConfidenceMaskOut, kCategoryMaskOut,
                          kQualityScoresOut);

  static absl::Status UpdateContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc);

 private:
  std::vector<Image> GetSegmentationResultCpu(const Shape& input_shape,
                                              const Shape& output_shape,
                                              const float* tensors_buffer);
  TensorsToSegmentationCalculatorOptions options_;

#ifdef TASK_SEGMENTATION_USE_GL_POSTPROCESSING
  SegmentationPostprocessorGl postprocessor_;
#endif  // TASK_SEGMENTATION_USE_GL_POSTPROCESSING
};

// static
absl::Status TensorsToSegmentationCalculator::UpdateContract(
    CalculatorContract* cc) {
#ifdef TASK_SEGMENTATION_USE_GL_POSTPROCESSING
  return SegmentationPostprocessorGl::UpdateContract(cc);
#else
  return absl::OkStatus();
#endif  // TASK_SEGMENTATION_USE_GL_POSTPROCESSING
}

absl::Status TensorsToSegmentationCalculator::Open(
    mediapipe::CalculatorContext* cc) {
  options_ = cc->Options<TensorsToSegmentationCalculatorOptions>();
  // TODO: remove deprecated output type support.
  if (options_.segmenter_options().has_output_type()) {
    RET_CHECK_NE(options_.segmenter_options().output_type(),
                 SegmenterOptions::UNSPECIFIED)
        << "Must specify output_type as one of "
           "[CONFIDENCE_MASK|CATEGORY_MASK].";
  } else {
    if (!cc->Outputs().HasTag("CONFIDENCE_MASK") &&
        !cc->Outputs().HasTag("CATEGORY_MASK")) {
      return absl::InvalidArgumentError(
          "At least one of CONFIDENCE_MASK and CATEGORY_MASK must be "
          "connected.");
    }
  }
#ifdef TASK_SEGMENTATION_USE_GL_POSTPROCESSING
  MP_RETURN_IF_ERROR(postprocessor_.Initialize(cc, options_));
#endif  // TASK_SEGMENTATION_USE_GL_POSTPROCESSING
  return absl::OkStatus();
}

absl::Status TensorsToSegmentationCalculator::Process(
    mediapipe::CalculatorContext* cc) {
  const auto& input_tensors = kTensorsIn(cc).Get();
  if (input_tensors.size() != 1 && input_tensors.size() != 2) {
    return absl::InvalidArgumentError(
        "Expect input tensor vector of size 1 or 2.");
  }
  const auto& input_tensor =
      input_tensors.size() == 1 ? input_tensors[0] : input_tensors[1];
  MP_ASSIGN_OR_RETURN(const Shape input_shape,
                      GetImageLikeTensorShape(input_tensor));

  // TODO: should use tensor signature to get the correct output
  // tensor.
  if (input_tensors.size() == 2) {
    const auto& quality_tensor = input_tensors[0];
    const float* quality_score_buffer =
        quality_tensor.GetCpuReadView().buffer<float>();
    const std::vector<float> quality_scores(
        quality_score_buffer,
        quality_score_buffer +
            (quality_tensor.bytes() / quality_tensor.element_size()));
    kQualityScoresOut(cc).Send(quality_scores);
  } else {
    // If the input_tensors don't contain quality scores, send the default
    // quality scores as 1.
    const std::vector<float> quality_scores(input_shape.channels, 1.0f);
    kQualityScoresOut(cc).Send(quality_scores);
  }

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

  // Use GPU postprocessing on web when Tensor is there already.
#ifdef TASK_SEGMENTATION_USE_GL_POSTPROCESSING
  Shape output_shape = {/* height= */ output_height,
                        /* width= */ output_width,
                        /* channels= */ input_shape.channels};
  if (input_tensor.ready_on_gpu()) {
    bool produce_category_mask = options_.segmenter_options().output_type() ==
                                     SegmenterOptions::CATEGORY_MASK ||
                                 cc->Outputs().HasTag("CATEGORY_MASK");
    bool produce_confidence_masks =
        options_.segmenter_options().output_type() ==
            SegmenterOptions::CONFIDENCE_MASK ||
        cc->Outputs().HasTag("CONFIDENCE_MASK");
    std::vector<std::unique_ptr<Image>> segmented_masks =
        postprocessor_.GetSegmentationResultGpu(
            input_shape, output_shape, input_tensor, produce_confidence_masks,
            produce_category_mask);
    bool new_style = cc->Outputs().HasTag("CATEGORY_MASK") ||
                     cc->Outputs().HasTag("CONFIDENCE_MASK");
    if (new_style) {
      if (produce_confidence_masks) {
        for (int i = 0; i < input_shape.channels; ++i) {
          kConfidenceMaskOut(cc)[i].Send(std::move(segmented_masks[i]));
        }
      }
      if (produce_category_mask) {
        int category_mask_index =
            produce_confidence_masks ? input_shape.channels : 0;
        kCategoryMaskOut(cc).Send(
            std::move(segmented_masks[category_mask_index]));
      }
    } else {
      // TODO: remove deprecated output type support.
      for (int i = 0; i < segmented_masks.size(); ++i) {
        kSegmentationOut(cc)[i].Send(std::move(segmented_masks[i]));
      }
    }
    return absl::OkStatus();
  }
#endif  // TASK_SEGMENTATION_USE_GL_POSTPROCESSING

  // Otherwise, use CPU postprocessing.
  const float* tensors_buffer = input_tensor.GetCpuReadView().buffer<float>();

  // TODO: remove deprecated output type support.
  if (options_.segmenter_options().has_output_type()) {
    std::vector<Image> segmented_masks = GetSegmentationResultCpu(
        input_shape,
        {/* height= */ output_height,
         /* width= */ output_width,
         /* channels= */ options_.segmenter_options().output_type() ==
                 SegmenterOptions::CATEGORY_MASK
             ? 1
             : input_shape.channels},
        input_tensor.GetCpuReadView().buffer<float>());
    for (int i = 0; i < segmented_masks.size(); ++i) {
      kSegmentationOut(cc)[i].Send(std::move(segmented_masks[i]));
    }
    return absl::OkStatus();
  }

  if (cc->Outputs().HasTag("CONFIDENCE_MASK")) {
    std::vector<Image> confidence_masks = ProcessForConfidenceMaskCpu(
        input_shape,
        {/* height= */ output_height,
         /* width= */ output_width,
         /* channels= */ input_shape.channels},
        options_.segmenter_options(), tensors_buffer);
    for (int i = 0; i < confidence_masks.size(); ++i) {
      kConfidenceMaskOut(cc)[i].Send(std::move(confidence_masks[i]));
    }
  }
  if (cc->Outputs().HasTag("CATEGORY_MASK")) {
    kCategoryMaskOut(cc).Send(ProcessForCategoryMaskCpu(
        input_shape,
        {/* height= */ output_height,
         /* width= */ output_width,
         /* channels= */ 1},
        options_.segmenter_options(), tensors_buffer));
  }
  return absl::OkStatus();
}

std::vector<Image> TensorsToSegmentationCalculator::GetSegmentationResultCpu(
    const Shape& input_shape, const Shape& output_shape,
    const float* tensors_buffer) {
  if (options_.segmenter_options().output_type() ==
      SegmenterOptions::CATEGORY_MASK) {
    return {ProcessForCategoryMaskCpu(input_shape, output_shape,
                                      options_.segmenter_options(),
                                      tensors_buffer)};
  } else {
    return ProcessForConfidenceMaskCpu(input_shape, output_shape,
                                       options_.segmenter_options(),
                                       tensors_buffer);
  }
}

MEDIAPIPE_REGISTER_NODE(::mediapipe::tasks::TensorsToSegmentationCalculator);

}  // namespace tasks
}  // namespace mediapipe
