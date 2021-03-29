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

#include "mediapipe/examples/desktop/autoflip/quality/padding_effect_generator.h"

#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace autoflip {

PaddingEffectGenerator::PaddingEffectGenerator(const int input_width,
                                               const int input_height,
                                               const double target_aspect_ratio,
                                               bool scale_to_multiple_of_two) {
  target_aspect_ratio_ = target_aspect_ratio;
  const double input_aspect_ratio =
      static_cast<double>(input_width) / static_cast<double>(input_height);
  input_width_ = input_width;
  input_height_ = input_height;
  is_vertical_padding_ = input_aspect_ratio > target_aspect_ratio;
  output_width_ = is_vertical_padding_
                      ? std::round(target_aspect_ratio * input_height)
                      : input_width;
  output_height_ = is_vertical_padding_
                       ? input_height
                       : std::round(input_width / target_aspect_ratio);
  if (scale_to_multiple_of_two) {
    output_width_ = output_width_ / 2 * 2;
    output_height_ = output_height_ / 2 * 2;
  }
}

absl::Status PaddingEffectGenerator::Process(
    const ImageFrame& input_frame, const float background_contrast,
    const int blur_cv_size, const float overlay_opacity,
    ImageFrame* output_frame, const cv::Scalar* background_color_in_rgb) {
  RET_CHECK_EQ(input_frame.Width(), input_width_);
  RET_CHECK_EQ(input_frame.Height(), input_height_);
  RET_CHECK(output_frame);

  cv::Mat original_image = formats::MatView(&input_frame);
  // This is the canvas that we are going to draw the padding effect on to.
  cv::Mat canvas(output_height_, output_width_, original_image.type());

  const int effective_input_width =
      is_vertical_padding_ ? input_width_ : input_height_;
  const int effective_input_height =
      is_vertical_padding_ ? input_height_ : input_width_;
  const int effective_output_width =
      is_vertical_padding_ ? output_width_ : output_height_;
  const int effective_output_height =
      is_vertical_padding_ ? output_height_ : output_width_;

  if (!is_vertical_padding_) {
    original_image = original_image.t();
    canvas = canvas.t();
  }

  const int foreground_height =
      effective_input_height * effective_output_width / effective_input_width;
  int x = -1, y = -1, width = -1, height = -1;

  // The following steps does the padding operation, with several steps.
  // #1, we prepare the background. If a solid background color is given, we use
  //     it directly. Otherwise, we first crop a region of size "output_width_ *
  //     output_height_" off of the original frame to become the background of
  //     the final frame, and then we blur it and adjust contrast and opacity.
  if (background_color_in_rgb != nullptr) {
    canvas = *background_color_in_rgb;
  } else {
    // Copy the original image to the background.
    x = 0.5 * (effective_input_width - effective_output_width);
    y = 0;
    width = effective_output_width;
    height = effective_output_height;
    cv::Rect crop_window_for_background(x, y, width, height);
    original_image(crop_window_for_background).copyTo(canvas);

    // Blur.
    const int cv_size =
        blur_cv_size % 2 == 1 ? blur_cv_size : (blur_cv_size + 1);
    const cv::Size kernel(cv_size, cv_size);
    // TODO: the larger the kernel size, the slower the blurring
    // operation is. Consider running multiple sequential blurs with smaller
    // sizes to simulate the effect of using a large size. This might be able to
    // speed up the process.
    x = 0;
    width = effective_output_width;
    const cv::Rect canvas_rect(0, 0, canvas.cols, canvas.rows);
    // Blur the top region (above foreground).
    y = 0;
    height = (effective_output_height - foreground_height) / 2 + cv_size;
    const cv::Rect top_blur_region =
        cv::Rect(x, y, width, height) & canvas_rect;
    if (top_blur_region.area() > 0) {
      cv::Mat top_blurred = canvas(top_blur_region);
      cv::GaussianBlur(top_blurred, top_blurred, kernel, 0, 0);
    }
    // Blur the bottom region (below foreground).
    y = height + foreground_height - cv_size;
    height = effective_output_height - y;
    const cv::Rect bottom_blur_region =
        cv::Rect(x, y, width, height) & canvas_rect;
    if (bottom_blur_region.area() > 0) {
      cv::Mat bottom_blurred = canvas(bottom_blur_region);
      cv::GaussianBlur(bottom_blurred, bottom_blurred, kernel, 0, 0);
    }

    const float kEqualThreshold = 0.0001f;
    // Background contrast adjustment.
    if (std::abs(background_contrast - 1.0f) > kEqualThreshold) {
      canvas *= background_contrast;
    }

    // Alpha blend a translucent black layer.
    if (std::abs(overlay_opacity - 0.0f) > kEqualThreshold) {
      cv::Mat overlay = cv::Mat::zeros(canvas.size(), canvas.type());
      cv::addWeighted(overlay, overlay_opacity, canvas, 1 - overlay_opacity, 0,
                      canvas);
    }
  }

  // #2, we crop the entire region off of the original frame. This will become
  //     the foreground in the final frame.
  x = 0;
  y = 0;
  width = effective_input_width;
  height = effective_input_height;

  cv::Rect crop_window_for_foreground(x, y, width, height);

  // #3, we specify a region of size computed as below in the final frame to
  //     embed the foreground that we obtained in #2. The aspect ratio of
  //     this region should be the same as the foreground, but with a
  //     smaller size. Therefore, the height and width are derived using
  //     the ratio of the sizes.
  //     - embed size: output_width_ * height (to be computed)
  //     - foreground: input_width * input_height
  //
  //     The location of this region is horizontally centralized in the
  //     frame, and saturated in horizontal dimension.
  x = 0;
  y = (effective_output_height - foreground_height) / 2;
  width = effective_output_width;
  height = foreground_height;

  cv::Rect region_to_embed_foreground(x, y, width, height);
  cv::Mat dst = canvas(region_to_embed_foreground);
  cv::resize(original_image(crop_window_for_foreground), dst, dst.size());

  if (!is_vertical_padding_) {
    canvas = canvas.t();
  }

  output_frame->CopyPixelData(input_frame.Format(), canvas.cols, canvas.rows,
                              canvas.data,
                              ImageFrame::kDefaultAlignmentBoundary);
  return absl::OkStatus();
}

cv::Rect PaddingEffectGenerator::ComputeOutputLocation() {
  const int effective_input_width =
      is_vertical_padding_ ? input_width_ : input_height_;
  const int effective_input_height =
      is_vertical_padding_ ? input_height_ : input_width_;
  const int effective_output_width =
      is_vertical_padding_ ? output_width_ : output_height_;
  const int effective_output_height =
      is_vertical_padding_ ? output_height_ : output_width_;

  // Step 3 from "process" call above, compute foreground location.
  const int foreground_height =
      effective_input_height * effective_output_width / effective_input_width;
  const int x = 0;
  const int y = (effective_output_height - foreground_height) / 2;
  const int width = effective_output_width;
  const int height = foreground_height;

  cv::Rect region_to_embed_foreground(x, y, width, height);

  return region_to_embed_foreground;
}

}  // namespace autoflip
}  // namespace mediapipe
