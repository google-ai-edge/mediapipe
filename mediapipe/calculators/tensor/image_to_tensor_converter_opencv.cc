// Copyright 2020 The MediaPipe Authors.
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

#include "mediapipe/calculators/tensor/image_to_tensor_converter_opencv.h"

#include <cmath>
#include <memory>

#include "mediapipe/calculators/tensor/image_to_tensor_converter.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_opencv.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/statusor.h"

namespace mediapipe {

namespace {

class OpenCvProcessor : public ImageToTensorConverter {
 public:
  OpenCvProcessor(BorderMode border_mode, Tensor::ElementType tensor_type)
      : tensor_type_(tensor_type) {
    switch (border_mode) {
      case BorderMode::kReplicate:
        border_mode_ = cv::BORDER_REPLICATE;
        break;
      case BorderMode::kZero:
        border_mode_ = cv::BORDER_CONSTANT;
        break;
    }
    switch (tensor_type_) {
      case Tensor::ElementType::kInt8:
        mat_type_ = CV_8SC3;
        break;
      case Tensor::ElementType::kFloat32:
        mat_type_ = CV_32FC3;
        break;
      case Tensor::ElementType::kUInt8:
        mat_type_ = CV_8UC3;
        break;
      default:
        mat_type_ = -1;
    }
  }

  absl::StatusOr<Tensor> Convert(const mediapipe::Image& input,
                                 const RotatedRect& roi,
                                 const Size& output_dims, float range_min,
                                 float range_max) override {
    if (input.image_format() != mediapipe::ImageFormat::SRGB &&
        input.image_format() != mediapipe::ImageFormat::SRGBA) {
      return InvalidArgumentError(
          absl::StrCat("Only RGBA/RGB formats are supported, passed format: ",
                       static_cast<uint32_t>(input.image_format())));
    }
    auto src = mediapipe::formats::MatView(&input);

    constexpr int kNumChannels = 3;
    Tensor tensor(tensor_type_, Tensor::Shape{1, output_dims.height,
                                              output_dims.width, kNumChannels});
    auto buffer_view = tensor.GetCpuWriteView();
    cv::Mat dst;
    switch (tensor_type_) {
      case Tensor::ElementType::kInt8:
        dst = cv::Mat(output_dims.height, output_dims.width, mat_type_,
                      buffer_view.buffer<int8>());
        break;
      case Tensor::ElementType::kFloat32:
        dst = cv::Mat(output_dims.height, output_dims.width, mat_type_,
                      buffer_view.buffer<float>());
        break;
      case Tensor::ElementType::kUInt8:
        dst = cv::Mat(output_dims.height, output_dims.width, mat_type_,
                      buffer_view.buffer<uint8>());
        break;
      default:
        return InvalidArgumentError(
            absl::StrCat("Unsupported tensor type: ", tensor_type_));
    }

    const cv::RotatedRect rotated_rect(cv::Point2f(roi.center_x, roi.center_y),
                                       cv::Size2f(roi.width, roi.height),
                                       roi.rotation * 180.f / M_PI);
    cv::Mat src_points;
    cv::boxPoints(rotated_rect, src_points);

    const float dst_width = output_dims.width;
    const float dst_height = output_dims.height;
    /* clang-format off */
    float dst_corners[8] = {0.0f,      dst_height,
                            0.0f,      0.0f,
                            dst_width, 0.0f,
                            dst_width, dst_height};
    /* clang-format on */

    cv::Mat dst_points = cv::Mat(4, 2, CV_32F, dst_corners);
    cv::Mat projection_matrix =
        cv::getPerspectiveTransform(src_points, dst_points);
    cv::Mat transformed;
    cv::warpPerspective(*src, transformed, projection_matrix,
                        cv::Size(dst_width, dst_height),
                        /*flags=*/cv::INTER_LINEAR,
                        /*borderMode=*/border_mode_);

    if (transformed.channels() > kNumChannels) {
      cv::Mat proper_channels_mat;
      cv::cvtColor(transformed, proper_channels_mat, cv::COLOR_RGBA2RGB);
      transformed = proper_channels_mat;
    }

    constexpr float kInputImageRangeMin = 0.0f;
    constexpr float kInputImageRangeMax = 255.0f;
    ASSIGN_OR_RETURN(
        auto transform,
        GetValueRangeTransformation(kInputImageRangeMin, kInputImageRangeMax,
                                    range_min, range_max));
    transformed.convertTo(dst, mat_type_, transform.scale, transform.offset);
    return tensor;
  }

 private:
  enum cv::BorderTypes border_mode_;
  Tensor::ElementType tensor_type_;
  int mat_type_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<ImageToTensorConverter>> CreateOpenCvConverter(
    CalculatorContext* cc, BorderMode border_mode,
    Tensor::ElementType tensor_type) {
  if (tensor_type != Tensor::ElementType::kInt8 &&
      tensor_type != Tensor::ElementType::kFloat32 &&
      tensor_type != Tensor::ElementType::kUInt8) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Tensor type is currently not supported by OpenCvProcessor, type: ",
        tensor_type));
  }
  return absl::make_unique<OpenCvProcessor>(border_mode, tensor_type);
}

}  // namespace mediapipe
