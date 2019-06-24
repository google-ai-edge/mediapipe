// Copyright 2019 The MediaPipe Authors.
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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

// Crops the input texture to the given rectangle region. The rectangle can
// be at arbitrary location on the image with rotation. If there's rotation, the
// output texture will have the size of the input rectangle. The rotation should
// be in radian, see rect.proto for detail.
// Currently it only works for CPU.
//
// Input:
//   IMAGE: ImageFrame representing the input image.
//   One of the following two tags:
//   RECT - A Rect proto specifying the width/height and location of the
//          cropping rectangle.
//   NORM_RECT - A NormalizedRect proto specifying the width/height and location
//          of the cropping rectangle in normalized coordinates.
//
// Output:
//   IMAGE - Cropped frames.
class ImageCroppingCalculator : public CalculatorBase {
 public:
  ImageCroppingCalculator() = default;
  ~ImageCroppingCalculator() override = default;

  static ::mediapipe::Status GetContract(CalculatorContract* cc);
  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  ::mediapipe::Status RenderCpu(CalculatorContext* cc);
  ::mediapipe::Status RenderGpu(CalculatorContext* cc);

  // TODO: Merge with GlCroppingCalculator to have GPU support.
  bool use_gpu_{};
};
REGISTER_CALCULATOR(ImageCroppingCalculator);

::mediapipe::Status ImageCroppingCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag("IMAGE"));
  RET_CHECK(cc->Outputs().HasTag("IMAGE"));

  cc->Inputs().Tag("IMAGE").Set<ImageFrame>();

  if (cc->Inputs().HasTag("RECT")) {
    cc->Inputs().Tag("RECT").Set<Rect>();
  }
  if (cc->Inputs().HasTag("NORM_RECT")) {
    cc->Inputs().Tag("NORM_RECT").Set<NormalizedRect>();
  }

  cc->Outputs().Tag("IMAGE").Set<ImageFrame>();

  return ::mediapipe::OkStatus();
}

::mediapipe::Status ImageCroppingCalculator::Process(CalculatorContext* cc) {
  if (use_gpu_) {
    RETURN_IF_ERROR(RenderGpu(cc));
  } else {
    RETURN_IF_ERROR(RenderCpu(cc));
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status ImageCroppingCalculator::RenderCpu(CalculatorContext* cc) {
  const auto& input_img = cc->Inputs().Tag("IMAGE").Get<ImageFrame>();
  cv::Mat input_mat = formats::MatView(&input_img);

  float rect_center_x = input_img.Width() / 2.0f;
  float rect_center_y = input_img.Height() / 2.0f;
  float rotation = 0.0f;
  int target_width = input_img.Width();
  int target_height = input_img.Height();
  if (cc->Inputs().HasTag("RECT")) {
    const auto& rect = cc->Inputs().Tag("RECT").Get<Rect>();
    if (rect.width() > 0 && rect.height() > 0 && rect.x_center() >= 0 &&
        rect.y_center() >= 0) {
      rotation = rect.rotation();
      rect_center_x = rect.x_center();
      rect_center_y = rect.y_center();
      target_width = rect.width();
      target_height = rect.height();
    }
  } else if (cc->Inputs().HasTag("NORM_RECT")) {
    const auto& rect = cc->Inputs().Tag("NORM_RECT").Get<NormalizedRect>();
    if (rect.width() > 0.0 && rect.height() > 0.0 && rect.x_center() >= 0.0 &&
        rect.y_center() >= 0.0) {
      rotation = rect.rotation();
      rect_center_x = std::round(rect.x_center() * input_img.Width());
      rect_center_y = std::round(rect.y_center() * input_img.Height());
      target_width = std::round(rect.width() * input_img.Width());
      target_height = std::round(rect.height() * input_img.Height());
    }
  }

  cv::Mat rotated_mat;
  if (std::abs(rotation) > 1e-5) {
    // TODO: Use open source common math library.
    const float pi = 3.1415926f;
    rotation = rotation * 180.0 / pi;

    // First rotation the image.
    cv::Point2f src_center(rect_center_x, rect_center_y);
    cv::Mat rotation_mat = cv::getRotationMatrix2D(src_center, rotation, 1.0);
    cv::warpAffine(input_mat, rotated_mat, rotation_mat, input_mat.size());
  } else {
    input_mat.copyTo(rotated_mat);
  }

  // Then crop the requested area.
  const cv::Rect cropping_rect(rect_center_x - target_width / 2,
                               rect_center_y - target_height / 2, target_width,
                               target_height);
  cv::Mat cropped_image = cv::Mat(rotated_mat, cropping_rect);

  std::unique_ptr<ImageFrame> output_frame(new ImageFrame(
      input_img.Format(), cropped_image.cols, cropped_image.rows));
  cv::Mat output_mat = formats::MatView(output_frame.get());
  cropped_image.copyTo(output_mat);
  cc->Outputs().Tag("IMAGE").Add(output_frame.release(), cc->InputTimestamp());
  return ::mediapipe::OkStatus();
}

::mediapipe::Status ImageCroppingCalculator::RenderGpu(CalculatorContext* cc) {
  return ::mediapipe::UnimplementedError("GPU support is not implemented yet.");
}

}  // namespace mediapipe
