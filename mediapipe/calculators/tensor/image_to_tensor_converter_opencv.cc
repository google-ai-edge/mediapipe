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

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
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

class ImageToTensorOpenCvConverter : public ImageToTensorConverter {
 public:
  ImageToTensorOpenCvConverter(BorderMode border_mode,
                               Tensor::ElementType tensor_type,
                               cv::InterpolationFlags flags)
      : tensor_type_(tensor_type), flags_(flags) {
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
        mat_gray_type_ = CV_8SC1;
        break;
      case Tensor::ElementType::kFloat32:
        mat_type_ = CV_32FC3;
        mat_gray_type_ = CV_32FC1;
        break;
      case Tensor::ElementType::kUInt8:
        mat_type_ = CV_8UC3;
        mat_gray_type_ = CV_8UC1;
        break;
      default:
        mat_type_ = -1;
        mat_gray_type_ = -1;
    }
  }

  absl::Status Convert(const mediapipe::Image& input, const RotatedRect& roi,
                       float range_min, float range_max,
                       int tensor_buffer_offset,
                       Tensor& output_tensor) override {
    const bool is_supported_format =
        input.image_format() == mediapipe::ImageFormat::SRGB ||
        input.image_format() == mediapipe::ImageFormat::SRGBA ||
        input.image_format() == mediapipe::ImageFormat::GRAY8;
    if (!is_supported_format) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported format: ", static_cast<uint32_t>(input.image_format())));
    }

    RET_CHECK_GE(tensor_buffer_offset, 0)
        << "The input tensor_buffer_offset needs to be non-negative.";
    const auto& output_shape = output_tensor.shape();
    MP_RETURN_IF_ERROR(ValidateTensorShape(output_shape));

    const int output_height = output_shape.dims[1];
    const int output_width = output_shape.dims[2];
    const int output_channels = output_shape.dims[3];
    const int num_elements_per_img =
        output_height * output_width * output_channels;
    auto buffer_view = output_tensor.GetCpuWriteView();
    cv::Mat dst;
    const int dst_data_type = output_channels == 1 ? mat_gray_type_ : mat_type_;
    switch (tensor_type_) {
      case Tensor::ElementType::kInt8:
        RET_CHECK_GE(
            output_shape.num_elements(),
            tensor_buffer_offset / sizeof(int8_t) + num_elements_per_img)
            << "The buffer offset + the input image size is larger than the "
               "allocated tensor buffer.";
        dst = cv::Mat(output_height, output_width, dst_data_type,
                      buffer_view.buffer<int8_t>() +
                          tensor_buffer_offset / sizeof(int8_t));
        break;
      case Tensor::ElementType::kFloat32:
        RET_CHECK_GE(
            output_shape.num_elements(),
            tensor_buffer_offset / sizeof(float) + num_elements_per_img)
            << "The buffer offset + the input image size is larger than the "
               "allocated tensor buffer.";
        dst = cv::Mat(
            output_height, output_width, dst_data_type,
            buffer_view.buffer<float>() + tensor_buffer_offset / sizeof(float));
        break;
      case Tensor::ElementType::kUInt8:
        RET_CHECK_GE(
            output_shape.num_elements(),
            tensor_buffer_offset / sizeof(uint8_t) + num_elements_per_img)
            << "The buffer offset + the input image size is larger than the "
               "allocated tensor buffer.";
        dst = cv::Mat(output_height, output_width, dst_data_type,
                      buffer_view.buffer<uint8_t>() +
                          tensor_buffer_offset / sizeof(uint8_t));
        break;
      default:
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported tensor type: ", tensor_type_));
    }

    const cv::RotatedRect rotated_rect(cv::Point2f(roi.center_x, roi.center_y),
                                       cv::Size2f(roi.width, roi.height),
                                       roi.rotation * 180.f / M_PI);
    cv::Mat src_points;
    cv::boxPoints(rotated_rect, src_points);

    const float dst_width = output_width;
    const float dst_height = output_height;
    /* clang-format off */
    float dst_corners[8] = {0.0f,      dst_height,
                            0.0f,      0.0f,
                            dst_width, 0.0f,
                            dst_width, dst_height};
    /* clang-format on */

    auto src = mediapipe::formats::MatView(&input);
    cv::Mat dst_points = cv::Mat(4, 2, CV_32F, dst_corners);
    cv::Mat projection_matrix =
        cv::getPerspectiveTransform(src_points, dst_points);
    cv::Mat transformed;
    cv::warpPerspective(*src, transformed, projection_matrix,
                        cv::Size(dst_width, dst_height),
                        /*flags=*/flags_,
                        /*borderMode=*/border_mode_);

    if (transformed.channels() > output_channels) {
      cv::Mat proper_channels_mat;
      cv::cvtColor(transformed, proper_channels_mat, cv::COLOR_RGBA2RGB);
      transformed = proper_channels_mat;
    }

    constexpr float kInputImageRangeMin = 0.0f;
    constexpr float kInputImageRangeMax = 255.0f;
    MP_ASSIGN_OR_RETURN(
        auto transform,
        GetValueRangeTransformation(kInputImageRangeMin, kInputImageRangeMax,
                                    range_min, range_max));
    transformed.convertTo(dst, dst_data_type, transform.scale,
                          transform.offset);
    return absl::OkStatus();
  }

 private:
  absl::Status ValidateTensorShape(const Tensor::Shape& output_shape) {
    RET_CHECK_EQ(output_shape.dims.size(), 4)
        << "Wrong output dims size: " << output_shape.dims.size();
    RET_CHECK_GE(output_shape.dims[0], 1)
        << "The batch dimension needs to be equal or larger than 1.";
    RET_CHECK(output_shape.dims[3] == 3 || output_shape.dims[3] == 1)
        << "Wrong output channel: " << output_shape.dims[3];
    return absl::OkStatus();
  }

  enum cv::BorderTypes border_mode_;
  Tensor::ElementType tensor_type_;
  cv::InterpolationFlags flags_;
  int mat_type_;
  int mat_gray_type_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<ImageToTensorConverter>> CreateOpenCvConverter(
    CalculatorContext* cc, BorderMode border_mode,
    Tensor::ElementType tensor_type, cv::InterpolationFlags flags) {
  if (tensor_type != Tensor::ElementType::kInt8 &&
      tensor_type != Tensor::ElementType::kFloat32 &&
      tensor_type != Tensor::ElementType::kUInt8) {
    return absl::InvalidArgumentError(
        absl::StrCat("Tensor type is currently not supported by "
                     "ImageToTensorOpenCvConverter, type: ",
                     tensor_type));
  }
  return std::make_unique<ImageToTensorOpenCvConverter>(border_mode,
                                                        tensor_type, flags);
}

}  // namespace mediapipe
