// Copyright 2023 The MediaPipe Authors.
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

#include "mediapipe/calculators/tensor/tensors_to_segmentation_converter_opencv.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_segmentation_converter.h"
#include "mediapipe/calculators/tensor/tensors_to_segmentation_utils.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_opencv.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"

namespace mediapipe {
namespace {

using ::mediapipe::tensors_to_segmentation_utils::GetHwcFromDims;

class TensorsToSegmentationOpenCvConverter
    : public TensorsToSegmentationConverter {
 public:
  absl::Status Init(const TensorsToSegmentationCalculatorOptions& options) {
    options_ = options;
    return absl::OkStatus();
  }

  absl::StatusOr<std::unique_ptr<Image>> Convert(
      const std::vector<Tensor>& input_tensors, int output_width,
      int output_height) override;

 private:
  template <class T>
  absl::Status ApplyActivation(cv::Mat& tensor_mat, cv::Mat* small_mask_mat);

  TensorsToSegmentationCalculatorOptions options_;
};

absl::StatusOr<std::unique_ptr<Image>>
TensorsToSegmentationOpenCvConverter::Convert(
    const std::vector<Tensor>& input_tensors, int output_width,
    int output_height) {
  if (input_tensors.empty()) {
    return absl::InvalidArgumentError("input_tensors vector is empty.");
  }
  MP_ASSIGN_OR_RETURN(auto hwc, GetHwcFromDims(input_tensors[0].shape().dims));
  auto [tensor_height, tensor_width, tensor_channels] = hwc;
  // Create initial working mask.
  cv::Mat small_mask_mat(cv::Size(tensor_width, tensor_height), CV_32FC1);

  // Wrap input tensor.
  auto raw_input_tensor = &input_tensors[0];
  auto raw_input_view = raw_input_tensor->GetCpuReadView();
  const float* raw_input_data = raw_input_view.buffer<float>();
  cv::Mat tensor_mat(cv::Size(tensor_width, tensor_height),
                     CV_MAKETYPE(CV_32F, tensor_channels),
                     const_cast<float*>(raw_input_data));

  // Process mask tensor and apply activation function.
  if (tensor_channels == 2) {
    MP_RETURN_IF_ERROR(ApplyActivation<cv::Vec2f>(tensor_mat, &small_mask_mat));
  } else if (tensor_channels == 1) {
    RET_CHECK(mediapipe::TensorsToSegmentationCalculatorOptions::SOFTMAX !=
              options_.activation());  // Requires 2 channels.
    if (mediapipe::TensorsToSegmentationCalculatorOptions::NONE ==
        options_.activation())  // Pass-through optimization.
      tensor_mat.copyTo(small_mask_mat);
    else
      MP_RETURN_IF_ERROR(ApplyActivation<float>(tensor_mat, &small_mask_mat));
  } else {
    RET_CHECK_FAIL() << "Unsupported number of tensor channels "
                     << tensor_channels;
  }

  // Send out image as CPU packet.
  std::shared_ptr<ImageFrame> mask_frame = std::make_shared<ImageFrame>(
      ImageFormat::VEC32F1, output_width, output_height);
  auto output_mask = std::make_unique<Image>(mask_frame);
  auto output_mat = formats::MatView(output_mask.get());
  // Upsample small mask into output.
  cv::resize(small_mask_mat, *output_mat,
             cv::Size(output_width, output_height));
  return output_mask;
}

template <class T>
absl::Status TensorsToSegmentationOpenCvConverter::ApplyActivation(
    cv::Mat& tensor_mat, cv::Mat* small_mask_mat) {
  // Configure activation function.
  const int output_layer_index = options_.output_layer_index();
  using Options = ::mediapipe::TensorsToSegmentationCalculatorOptions;
  const auto activation_fn = [&](const cv::Vec2f& mask_value) {
    float new_mask_value = 0;
    // TODO consider moving switch out of the loop,
    // and also avoid float/Vec2f casting.
    switch (options_.activation()) {
      case Options::NONE: {
        new_mask_value = mask_value[0];
        break;
      }
      case Options::SIGMOID: {
        const float pixel0 = mask_value[0];
        new_mask_value = 1.0 / (std::exp(-pixel0) + 1.0);
        break;
      }
      case Options::SOFTMAX: {
        const float pixel0 = mask_value[0];
        const float pixel1 = mask_value[1];
        const float max_pixel = std::max(pixel0, pixel1);
        const float min_pixel = std::min(pixel0, pixel1);
        const float softmax_denom =
            /*exp(max_pixel - max_pixel)=*/1.0f +
            std::exp(min_pixel - max_pixel);
        new_mask_value = std::exp(mask_value[output_layer_index] - max_pixel) /
                         softmax_denom;
        break;
      }
    }
    return new_mask_value;
  };

  // Process mask tensor.
  for (int i = 0; i < tensor_mat.rows; ++i) {
    for (int j = 0; j < tensor_mat.cols; ++j) {
      const T& input_pix = tensor_mat.at<T>(i, j);
      const float mask_value = activation_fn(input_pix);
      small_mask_mat->at<float>(i, j) = mask_value;
    }
  }

  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<TensorsToSegmentationConverter>>
CreateOpenCvConverter(const TensorsToSegmentationCalculatorOptions& options) {
  auto converter = std::make_unique<TensorsToSegmentationOpenCvConverter>();
  MP_RETURN_IF_ERROR(converter->Init(options));
  return converter;
}

}  // namespace mediapipe
