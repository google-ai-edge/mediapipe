#ifndef MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_QUALITY_PADDING_EFFECT_GENERATOR_H_
#define MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_QUALITY_PADDING_EFFECT_GENERATOR_H_

#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace autoflip {

// Generates padding effects given input frames. Depending on where the padded
// contents are added, there are two cases:
// 1) Pad on the top and bottom of the input frame, aka vertical padding, i.e.
//    input_aspect_ratio > target_aspect_ratio. In this case, output frames will
//    have the same height as input frames, and the width will be adjusted to
//    match the target aspect ratio.
// 2) Pad on the left and right of the input frame, aka horizontal padding, i.e.
//    input_aspect_ratio < target_aspect_ratio. In this case, output frames will
//    have the same width as original frames, and the height will be adjusted to
//    match the target aspect ratio.
// If a background color is given, the background of the output frame will be
// filled with this solid color; otherwise, it is a blurred version of the input
// frame.
//
// Note: in both horizontal and vertical padding effects, the output frame size
// will be at most as large as the input frame size, with one dimension the
// same as the input (horizontal padding: width, vertical padding: height). If
// you intented to have the output frame be larger, you could add a
// ScaleImageCalculator as an upstream node before calling this calculator in
// your MediaPipe graph (not as a downstream node, because visual details may
// lose after appling the padding effect).
class PaddingEffectGenerator {
 public:
  // Always outputs width and height that are divisible by 2 if
  // scale_to_multiple_of_two is set to true.
  PaddingEffectGenerator(const int input_width, const int input_height,
                         const double target_aspect_ratio,
                         bool scale_to_multiple_of_two = false);

  // Apply the padding effect on the input frame.
  // - blur_cv_size: The cv::Size() parameter used in creating blurry effects
  //   for padding backgrounds.
  // - background_contrast: Contrast adjustment for padding background. This
  //   value should between 0 and 1, and the smaller the value, the darker the
  //   background.
  // - overlay_opacity: In addition to adjusting the contrast, a translucent
  //   black layer will be alpha blended with the background. This value defines
  //   the opacity of the black layer.
  // - background_color_in_rgb: If not null, uses this solid color as background
  //   instead of blurring the image, and does not adjust contrast or opacity.
  ::mediapipe::Status Process(
      const ImageFrame& input_frame, const float background_contrast,
      const int blur_cv_size, const float overlay_opacity,
      ImageFrame* output_frame,
      const cv::Scalar* background_color_in_rgb = nullptr);

  // Compute the "render location" on the output frame where the "crop from"
  // location is to be placed.  For use with external rendering soutions.
  cv::Rect ComputeOutputLocation();

 private:
  double target_aspect_ratio_;
  int input_width_ = -1;
  int input_height_ = -1;
  int output_width_ = -1;
  int output_height_ = -1;
  bool is_vertical_padding_;
};

}  // namespace autoflip
}  // namespace mediapipe

#endif  // MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_QUALITY_PADDING_EFFECT_GENERATOR_H_
