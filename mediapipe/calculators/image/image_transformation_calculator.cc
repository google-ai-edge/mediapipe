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

#include "mediapipe/calculators/image/image_transformation_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/scale_mode.pb.h"

#if defined(__ANDROID__) || defined(__APPLE__) && !TARGET_OS_OSX
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_quad_renderer.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/shader_util.h"
#endif  // __ANDROID__ || iOS

#if defined(__ANDROID__)
// The size of Java arrays is dynamic, which makes it difficult to
// generate the right packet type with a fixed size. Therefore, we
// are using unsized arrays on Android.
typedef int DimensionsPacketType[];
#else
typedef int DimensionsPacketType[2];
#endif  // __ANDROID__

#define DEFAULT_SCALE_MODE mediapipe::ScaleMode_Mode_STRETCH

namespace mediapipe {

#if defined(__ANDROID__) || defined(__APPLE__) && !TARGET_OS_OSX

#endif  // __ANDROID__ || iOS

namespace {
int RotationModeToDegrees(mediapipe::RotationMode_Mode rotation) {
  switch (rotation) {
    case mediapipe::RotationMode_Mode_UNKNOWN:
    case mediapipe::RotationMode_Mode_ROTATION_0:
      return 0;
    case mediapipe::RotationMode_Mode_ROTATION_90:
      return 90;
    case mediapipe::RotationMode_Mode_ROTATION_180:
      return 180;
    case mediapipe::RotationMode_Mode_ROTATION_270:
      return 270;
  }
}
mediapipe::RotationMode_Mode DegreesToRotationMode(int degrees) {
  switch (degrees) {
    case 0:
      return mediapipe::RotationMode_Mode_ROTATION_0;
    case 90:
      return mediapipe::RotationMode_Mode_ROTATION_90;
    case 180:
      return mediapipe::RotationMode_Mode_ROTATION_180;
    case 270:
      return mediapipe::RotationMode_Mode_ROTATION_270;
    default:
      return mediapipe::RotationMode_Mode_UNKNOWN;
  }
}
mediapipe::ScaleMode_Mode ParseScaleMode(
    mediapipe::ScaleMode_Mode scale_mode,
    mediapipe::ScaleMode_Mode default_mode) {
  switch (scale_mode) {
    case mediapipe::ScaleMode_Mode_DEFAULT:
      return default_mode;
    case mediapipe::ScaleMode_Mode_STRETCH:
      return scale_mode;
    case mediapipe::ScaleMode_Mode_FIT:
      return scale_mode;
    case mediapipe::ScaleMode_Mode_FILL_AND_CROP:
      return scale_mode;
    default:
      return default_mode;
  }
}
}  // namespace

// Scales, rotates, and flips images horizontally or vertically.
//
// Input:
//   One of the following two tags:
//   IMAGE: ImageFrame representing the input image.
//   IMAGE_GPU: GpuBuffer representing the input image.
//
//   ROTATION_DEGREES (optional): The counterclockwise rotation angle in
//   degrees. This allows different rotation angles for different frames. It has
//   to be a multiple of 90 degrees. If provided, it overrides the
//   ROTATION_DEGREES input side packet.
//
// Output:
//   One of the following two tags:
//   IMAGE - ImageFrame representing the output image.
//   IMAGE_GPU - GpuBuffer representing the output image.
//
//   LETTERBOX_PADDING (optional): An std::array<float, 4> representing the
//   letterbox padding from the 4 sides ([left, top, right, bottom]) of the
//   output image, normalized to [0.f, 1.f] by the output dimensions. The
//   padding values are non-zero only when the scale mode specified in the
//   calculator options is FIT. For instance, when the input image is 10x10
//   (width x height) and the output dimensions specified in the calculator
//   option are 20x40 and scale mode is FIT, the calculator scales the input
//   image to 20x20 and places it in the middle of the output image with an
//   equal padding of 10 pixels at the top and the bottom. The resulting array
//   is therefore [0.f, 0.25f, 0.f, 0.25f] (10/40 = 0.25f).
//
// Input side packet:
//   OUTPUT_DIMENSIONS (optional): The output width and height in pixels as the
//   first two elements in an integer array. It overrides the corresponding
//   field in the calculator options.
//
//   ROTATION_DEGREES (optional): The counterclockwise rotation angle in
//   degrees. It has to be a multiple of 90 degrees. It overrides the
//   corresponding field in the calculator options.
//
// Calculator options (see image_transformation_calculator.proto):
//   output_width, output_height - (optional) Desired scaled image size.
//   rotation_mode - (optional) Rotation in multiples of 90 degrees.
//   flip_vertically, flip_horizontally - (optional) flip about x or y axis.
//   scale_mode - (optional) Stretch, Fit, or Fill and Crop
//
// Note: To enable horizontal or vertical flipping, specify them in the
// calculator options. Flipping is applied after rotation.
//
// Note: Only scale mode STRETCH is currently supported on CPU,
// and flipping is not yet supported either.
//
class ImageTransformationCalculator : public CalculatorBase {
 public:
  ImageTransformationCalculator() = default;
  ~ImageTransformationCalculator() override = default;

  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;
  ::mediapipe::Status Close(CalculatorContext* cc) override;

 private:
  ::mediapipe::Status RenderCpu(CalculatorContext* cc);
  ::mediapipe::Status RenderGpu(CalculatorContext* cc);
  ::mediapipe::Status GlSetup();

  void ComputeOutputDimensions(int input_width, int input_height,
                               int* output_width, int* output_height);
  void ComputeOutputLetterboxPadding(int input_width, int input_height,
                                     int output_width, int output_height,
                                     std::array<float, 4>* padding);

  ImageTransformationCalculatorOptions options_;
  int output_width_ = 0;
  int output_height_ = 0;
  mediapipe::RotationMode_Mode rotation_;
  mediapipe::ScaleMode_Mode scale_mode_;

  bool use_gpu_ = false;
#if defined(__ANDROID__) || defined(__APPLE__) && !TARGET_OS_OSX
  GlCalculatorHelper helper_;
  std::unique_ptr<QuadRenderer> rgb_renderer_;
  std::unique_ptr<QuadRenderer> yuv_renderer_;
  std::unique_ptr<QuadRenderer> ext_rgb_renderer_;
#endif  // __ANDROID__ || iOS
};
REGISTER_CALCULATOR(ImageTransformationCalculator);

// static
::mediapipe::Status ImageTransformationCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag("IMAGE") ^ cc->Inputs().HasTag("IMAGE_GPU"));
  RET_CHECK(cc->Outputs().HasTag("IMAGE") ^ cc->Outputs().HasTag("IMAGE_GPU"));

  if (cc->Inputs().HasTag("IMAGE")) {
    RET_CHECK(cc->Outputs().HasTag("IMAGE"));
    cc->Inputs().Tag("IMAGE").Set<ImageFrame>();
    cc->Outputs().Tag("IMAGE").Set<ImageFrame>();
  }
#if defined(__ANDROID__) || defined(__APPLE__) && !TARGET_OS_OSX
  if (cc->Inputs().HasTag("IMAGE_GPU")) {
    RET_CHECK(cc->Outputs().HasTag("IMAGE_GPU"));
    cc->Inputs().Tag("IMAGE_GPU").Set<GpuBuffer>();
    cc->Outputs().Tag("IMAGE_GPU").Set<GpuBuffer>();
  }
#endif  // __ANDROID__ || iOS
  if (cc->Inputs().HasTag("ROTATION_DEGREES")) {
    cc->Inputs().Tag("ROTATION_DEGREES").Set<int>();
  }

  if (cc->InputSidePackets().HasTag("OUTPUT_DIMENSIONS")) {
    cc->InputSidePackets().Tag("OUTPUT_DIMENSIONS").Set<DimensionsPacketType>();
  }
  if (cc->InputSidePackets().HasTag("ROTATION_DEGREES")) {
    cc->InputSidePackets().Tag("ROTATION_DEGREES").Set<int>();
  }

  if (cc->Outputs().HasTag("LETTERBOX_PADDING")) {
    cc->Outputs().Tag("LETTERBOX_PADDING").Set<std::array<float, 4>>();
  }

#if defined(__ANDROID__) || defined(__APPLE__) && !TARGET_OS_OSX
  MP_RETURN_IF_ERROR(GlCalculatorHelper::UpdateContract(cc));
#endif  // __ANDROID__ || iOS

  return ::mediapipe::OkStatus();
}

::mediapipe::Status ImageTransformationCalculator::Open(CalculatorContext* cc) {
  // Inform the framework that we always output at the same timestamp
  // as we receive a packet at.
  cc->SetOffset(TimestampDiff(0));

  options_ = cc->Options<ImageTransformationCalculatorOptions>();

  if (cc->Inputs().HasTag("IMAGE_GPU")) {
    use_gpu_ = true;
  }

  if (cc->InputSidePackets().HasTag("OUTPUT_DIMENSIONS")) {
    const auto& dimensions = cc->InputSidePackets()
                                 .Tag("OUTPUT_DIMENSIONS")
                                 .Get<DimensionsPacketType>();
    output_width_ = dimensions[0];
    output_height_ = dimensions[1];
  } else {
    output_width_ = options_.output_width();
    output_height_ = options_.output_height();
  }
  if (cc->InputSidePackets().HasTag("ROTATION_DEGREES")) {
    rotation_ = DegreesToRotationMode(
        cc->InputSidePackets().Tag("ROTATION_DEGREES").Get<int>());
  } else {
    rotation_ = options_.rotation_mode();
  }

  scale_mode_ = ParseScaleMode(options_.scale_mode(), DEFAULT_SCALE_MODE);

  if (use_gpu_) {
#if defined(__ANDROID__) || defined(__APPLE__) && !TARGET_OS_OSX
    // Let the helper access the GL context information.
    MP_RETURN_IF_ERROR(helper_.Open(cc));
#else
    RET_CHECK_FAIL() << "GPU processing is for Android and iOS only.";
#endif  // __ANDROID__ || iOS
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status ImageTransformationCalculator::Process(
    CalculatorContext* cc) {
  if (use_gpu_) {
#if defined(__ANDROID__) || defined(__APPLE__) && !TARGET_OS_OSX
    return helper_.RunInGlContext(
        [this, cc]() -> ::mediapipe::Status { return RenderGpu(cc); });
#endif  // __ANDROID__ || iOS
  } else {
    return RenderCpu(cc);
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status ImageTransformationCalculator::Close(
    CalculatorContext* cc) {
  if (use_gpu_) {
#if defined(__ANDROID__) || defined(__APPLE__) && !TARGET_OS_OSX
    QuadRenderer* rgb_renderer = rgb_renderer_.release();
    QuadRenderer* yuv_renderer = yuv_renderer_.release();
    QuadRenderer* ext_rgb_renderer = ext_rgb_renderer_.release();
    helper_.RunInGlContext([rgb_renderer, yuv_renderer, ext_rgb_renderer] {
      if (rgb_renderer) {
        rgb_renderer->GlTeardown();
        delete rgb_renderer;
      }
      if (ext_rgb_renderer) {
        ext_rgb_renderer->GlTeardown();
        delete ext_rgb_renderer;
      }
      if (yuv_renderer) {
        yuv_renderer->GlTeardown();
        delete yuv_renderer;
      }
    });
#endif  // __ANDROID__ || iOS
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status ImageTransformationCalculator::RenderCpu(
    CalculatorContext* cc) {
  int input_width = cc->Inputs().Tag("IMAGE").Get<ImageFrame>().Width();
  int input_height = cc->Inputs().Tag("IMAGE").Get<ImageFrame>().Height();

  const auto& input_img = cc->Inputs().Tag("IMAGE").Get<ImageFrame>();
  cv::Mat input_mat = formats::MatView(&input_img);
  cv::Mat scaled_mat;

  if (scale_mode_ == mediapipe::ScaleMode_Mode_STRETCH) {
    cv::resize(input_mat, scaled_mat, cv::Size(output_width_, output_height_));
  } else {
    const float scale =
        std::min(static_cast<float>(output_width_) / input_width,
                 static_cast<float>(output_height_) / input_height);
    const int target_width = std::round(input_width * scale);
    const int target_height = std::round(input_height * scale);

    if (scale_mode_ == mediapipe::ScaleMode_Mode_FIT) {
      cv::Mat intermediate_mat;
      cv::resize(input_mat, intermediate_mat,
                 cv::Size(target_width, target_height));
      const int top = (output_height_ - target_height) / 2;
      const int bottom = output_height_ - target_height - top;
      const int left = (output_width_ - target_width) / 2;
      const int right = output_width_ - target_width - left;
      cv::copyMakeBorder(intermediate_mat, scaled_mat, top, bottom, left, right,
                         options_.constant_padding() ? cv::BORDER_CONSTANT
                                                     : cv::BORDER_REPLICATE);
    } else {
      cv::resize(input_mat, scaled_mat, cv::Size(target_width, target_height));
      output_width_ = target_width;
      output_height_ = target_height;
    }
  }

  int output_width;
  int output_height;
  ComputeOutputDimensions(input_width, input_height, &output_width,
                          &output_height);
  if (cc->Outputs().HasTag("LETTERBOX_PADDING")) {
    auto padding = absl::make_unique<std::array<float, 4>>();
    ComputeOutputLetterboxPadding(input_width, input_height, output_width,
                                  output_height, padding.get());
    cc->Outputs()
        .Tag("LETTERBOX_PADDING")
        .Add(padding.release(), cc->InputTimestamp());
  }

  if (cc->InputSidePackets().HasTag("ROTATION_DEGREES")) {
    rotation_ = DegreesToRotationMode(
        cc->InputSidePackets().Tag("ROTATION_DEGREES").Get<int>());
  }

  cv::Mat rotated_mat;
  const int angle = RotationModeToDegrees(rotation_);
  cv::Point2f src_center(scaled_mat.cols / 2.0, scaled_mat.rows / 2.0);
  cv::Mat rotation_mat = cv::getRotationMatrix2D(src_center, angle, 1.0);
  cv::warpAffine(scaled_mat, rotated_mat, rotation_mat, scaled_mat.size());

  std::unique_ptr<ImageFrame> output_frame(
      new ImageFrame(input_img.Format(), output_width, output_height));
  cv::Mat output_mat = formats::MatView(output_frame.get());
  rotated_mat.copyTo(output_mat);
  cc->Outputs().Tag("IMAGE").Add(output_frame.release(), cc->InputTimestamp());

  return ::mediapipe::OkStatus();
}

::mediapipe::Status ImageTransformationCalculator::RenderGpu(
    CalculatorContext* cc) {
#if defined(__ANDROID__) || defined(__APPLE__) && !TARGET_OS_OSX
  int input_width = cc->Inputs().Tag("IMAGE_GPU").Get<GpuBuffer>().width();
  int input_height = cc->Inputs().Tag("IMAGE_GPU").Get<GpuBuffer>().height();

  int output_width;
  int output_height;
  ComputeOutputDimensions(input_width, input_height, &output_width,
                          &output_height);

  if (cc->Outputs().HasTag("LETTERBOX_PADDING")) {
    auto padding = absl::make_unique<std::array<float, 4>>();
    ComputeOutputLetterboxPadding(input_width, input_height, output_width,
                                  output_height, padding.get());
    cc->Outputs()
        .Tag("LETTERBOX_PADDING")
        .Add(padding.release(), cc->InputTimestamp());
  }

  const auto& input = cc->Inputs().Tag("IMAGE_GPU").Get<GpuBuffer>();
  QuadRenderer* renderer = nullptr;
  GlTexture src1;

#if defined(__APPLE__) && !TARGET_OS_OSX
  if (input.format() == GpuBufferFormat::kBiPlanar420YpCbCr8VideoRange ||
      input.format() == GpuBufferFormat::kBiPlanar420YpCbCr8FullRange) {
    if (!yuv_renderer_) {
      yuv_renderer_ = absl::make_unique<QuadRenderer>();
      MP_RETURN_IF_ERROR(
          yuv_renderer_->GlSetup(::mediapipe::kYUV2TexToRGBFragmentShader,
                                 {"video_frame_y", "video_frame_uv"}));
    }
    renderer = yuv_renderer_.get();
    src1 = helper_.CreateSourceTexture(input, 0);
  } else  // NOLINT(readability/braces)
#endif    // iOS
  {
    src1 = helper_.CreateSourceTexture(input);
#if defined(__ANDROID__)
    if (src1.target() == GL_TEXTURE_EXTERNAL_OES) {
      if (!ext_rgb_renderer_) {
        ext_rgb_renderer_ = absl::make_unique<QuadRenderer>();
        MP_RETURN_IF_ERROR(ext_rgb_renderer_->GlSetup(
            ::mediapipe::kBasicTexturedFragmentShaderOES, {"video_frame"}));
      }
      renderer = ext_rgb_renderer_.get();
    } else  // NOLINT(readability/braces)
#endif      // __ANDROID__
    {
      if (!rgb_renderer_) {
        rgb_renderer_ = absl::make_unique<QuadRenderer>();
        MP_RETURN_IF_ERROR(rgb_renderer_->GlSetup());
      }
      renderer = rgb_renderer_.get();
    }
  }
  RET_CHECK(renderer) << "Unsupported input texture type";

  if (cc->InputSidePackets().HasTag("ROTATION_DEGREES")) {
    rotation_ = DegreesToRotationMode(
        cc->InputSidePackets().Tag("ROTATION_DEGREES").Get<int>());
  }

  static mediapipe::FrameScaleMode scale_mode =
      mediapipe::FrameScaleModeFromProto(scale_mode_,
                                         mediapipe::FrameScaleMode::kStretch);
  mediapipe::FrameRotation rotation =
      mediapipe::FrameRotationFromDegrees(RotationModeToDegrees(rotation_));

  auto dst = helper_.CreateDestinationTexture(output_width, output_height,
                                              input.format());

  helper_.BindFramebuffer(dst);  // GL_TEXTURE0
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(src1.target(), src1.name());

  MP_RETURN_IF_ERROR(renderer->GlRender(
      src1.width(), src1.height(), dst.width(), dst.height(), scale_mode,
      rotation, options_.flip_horizontally(), options_.flip_vertically(),
      /*flip_texture=*/false));

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(src1.target(), 0);

  // Execute GL commands, before getting result.
  glFlush();

  auto output = dst.GetFrame<GpuBuffer>();
  cc->Outputs().Tag("IMAGE_GPU").Add(output.release(), cc->InputTimestamp());

#endif  // __ANDROID__ || iOS

  return ::mediapipe::OkStatus();
}

void ImageTransformationCalculator::ComputeOutputDimensions(
    int input_width, int input_height, int* output_width, int* output_height) {
  if (output_width_ > 0 && output_height_ > 0) {
    *output_width = output_width_;
    *output_height = output_height_;
  } else if (rotation_ == mediapipe::RotationMode_Mode_ROTATION_90 ||
             rotation_ == mediapipe::RotationMode_Mode_ROTATION_270) {
    *output_width = input_height;
    *output_height = input_width;
  } else {
    *output_width = input_width;
    *output_height = input_height;
  }
}

void ImageTransformationCalculator::ComputeOutputLetterboxPadding(
    int input_width, int input_height, int output_width, int output_height,
    std::array<float, 4>* padding) {
  if (scale_mode_ == mediapipe::ScaleMode_Mode_FIT) {
    if (rotation_ == mediapipe::RotationMode_Mode_ROTATION_90 ||
        rotation_ == mediapipe::RotationMode_Mode_ROTATION_270) {
      std::swap(input_width, input_height);
    }
    const float input_aspect_ratio =
        static_cast<float>(input_width) / input_height;
    const float output_aspect_ratio =
        static_cast<float>(output_width) / output_height;
    if (input_aspect_ratio < output_aspect_ratio) {
      // Compute left and right padding.
      (*padding)[0] = (1.f - input_aspect_ratio / output_aspect_ratio) / 2.f;
      (*padding)[2] = (*padding)[0];
    } else if (output_aspect_ratio < input_aspect_ratio) {
      // Compute top and bottom padding.
      (*padding)[1] = (1.f - output_aspect_ratio / input_aspect_ratio) / 2.f;
      (*padding)[3] = (*padding)[1];
    }
  }
}

}  // namespace mediapipe
