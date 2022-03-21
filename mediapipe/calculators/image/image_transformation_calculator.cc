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
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/gpu/scale_mode.pb.h"

#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_quad_renderer.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/shader_util.h"
#endif  // !MEDIAPIPE_DISABLE_GPU

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

#if !MEDIAPIPE_DISABLE_GPU

#endif  // !MEDIAPIPE_DISABLE_GPU
#if defined(MEDIAPIPE_IOS)

#endif  // defined(MEDIAPIPE_IOS)

namespace {
constexpr char kImageFrameTag[] = "IMAGE";
constexpr char kGpuBufferTag[] = "IMAGE_GPU";
constexpr char kVideoPrestreamTag[] = "VIDEO_PRESTREAM";

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
//   One of the following tags:
//   IMAGE: ImageFrame representing the input image.
//   IMAGE_GPU: GpuBuffer representing the input image.
//
//   OUTPUT_DIMENSIONS (optional): The output width and height in pixels as
//   pair<int, int>. If set, it will override corresponding field in calculator
//   options and input side packet.
//
//   ROTATION_DEGREES (optional): The counterclockwise rotation angle in
//   degrees. This allows different rotation angles for different frames. It has
//   to be a multiple of 90 degrees. If provided, it overrides the
//   ROTATION_DEGREES input side packet.
//
//   FLIP_HORIZONTALLY (optional): Whether to flip image horizontally or not. If
//   provided, it overrides the FLIP_HORIZONTALLY input side packet and/or
//   corresponding field in the calculator options.
//
//   FLIP_VERTICALLY (optional): Whether to flip image vertically or not. If
//   provided, it overrides the FLIP_VERTICALLY input side packet and/or
//   corresponding field in the calculator options.
//
//   VIDEO_PRESTREAM (optional): VideoHeader for the input ImageFrames, if
//   rotating or scaling the frames, the header width and height will be updated
//   appropriately. Note the header is updated only based on dimensions and
//   rotations specified as side packets or options, input_stream
//   transformations will not update the header.
//
// Output:
//   One of the following tags:
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
//   FLIP_HORIZONTALLY (optional): Whether to flip image horizontally or not.
//   It overrides the corresponding field in the calculator options.
//
//   FLIP_VERTICALLY (optional): Whether to flip image vertically or not.
//   It overrides the corresponding field in the calculator options.
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
// Note: Input defines output, so only matchig types supported:
// IMAGE -> IMAGE  or  IMAGE_GPU -> IMAGE_GPU
//
class ImageTransformationCalculator : public CalculatorBase {
 public:
  ImageTransformationCalculator() = default;
  ~ImageTransformationCalculator() override = default;

  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status RenderCpu(CalculatorContext* cc);
  absl::Status RenderGpu(CalculatorContext* cc);
  absl::Status GlSetup();

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
  bool flip_horizontally_ = false;
  bool flip_vertically_ = false;

  bool use_gpu_ = false;
#if !MEDIAPIPE_DISABLE_GPU
  GlCalculatorHelper gpu_helper_;
  std::unique_ptr<QuadRenderer> rgb_renderer_;
  std::unique_ptr<QuadRenderer> yuv_renderer_;
  std::unique_ptr<QuadRenderer> ext_rgb_renderer_;
#endif  // !MEDIAPIPE_DISABLE_GPU
};
REGISTER_CALCULATOR(ImageTransformationCalculator);

// static
absl::Status ImageTransformationCalculator::GetContract(
    CalculatorContract* cc) {
  // Only one input can be set, and the output type must match.
  RET_CHECK(cc->Inputs().HasTag(kImageFrameTag) ^
            cc->Inputs().HasTag(kGpuBufferTag));

  bool use_gpu = false;

  if (cc->Inputs().HasTag(kImageFrameTag)) {
    RET_CHECK(cc->Outputs().HasTag(kImageFrameTag));
    cc->Inputs().Tag(kImageFrameTag).Set<ImageFrame>();
    cc->Outputs().Tag(kImageFrameTag).Set<ImageFrame>();
  }
#if !MEDIAPIPE_DISABLE_GPU
  if (cc->Inputs().HasTag(kGpuBufferTag)) {
    RET_CHECK(cc->Outputs().HasTag(kGpuBufferTag));
    cc->Inputs().Tag(kGpuBufferTag).Set<GpuBuffer>();
    cc->Outputs().Tag(kGpuBufferTag).Set<GpuBuffer>();
    use_gpu |= true;
  }
#endif  // !MEDIAPIPE_DISABLE_GPU

  if (cc->Inputs().HasTag("OUTPUT_DIMENSIONS")) {
    cc->Inputs().Tag("OUTPUT_DIMENSIONS").Set<std::pair<int, int>>();
  }

  if (cc->Inputs().HasTag("ROTATION_DEGREES")) {
    cc->Inputs().Tag("ROTATION_DEGREES").Set<int>();
  }
  if (cc->Inputs().HasTag("FLIP_HORIZONTALLY")) {
    cc->Inputs().Tag("FLIP_HORIZONTALLY").Set<bool>();
  }
  if (cc->Inputs().HasTag("FLIP_VERTICALLY")) {
    cc->Inputs().Tag("FLIP_VERTICALLY").Set<bool>();
  }

  RET_CHECK(cc->Inputs().HasTag(kVideoPrestreamTag) ==
            cc->Outputs().HasTag(kVideoPrestreamTag))
      << "If VIDEO_PRESTREAM is provided, it must be provided both as an "
         "inputs and output stream.";
  if (cc->Inputs().HasTag(kVideoPrestreamTag)) {
    RET_CHECK(!(cc->Inputs().HasTag("OUTPUT_DIMENSIONS") ||
                cc->Inputs().HasTag("ROTATION_DEGREES")))
        << "If specifying VIDEO_PRESTREAM, the transformations that affect the "
           "dimensions of the frames (OUTPUT_DIMENSIONS and ROTATION_DEGREES) "
           "need to be constant for every frame, meaning they can only be "
           "provided in the calculator options or side packets.";
    cc->Inputs().Tag(kVideoPrestreamTag).Set<mediapipe::VideoHeader>();
    cc->Outputs().Tag(kVideoPrestreamTag).Set<mediapipe::VideoHeader>();
  }

  if (cc->InputSidePackets().HasTag("OUTPUT_DIMENSIONS")) {
    cc->InputSidePackets().Tag("OUTPUT_DIMENSIONS").Set<DimensionsPacketType>();
  }
  if (cc->InputSidePackets().HasTag("ROTATION_DEGREES")) {
    cc->InputSidePackets().Tag("ROTATION_DEGREES").Set<int>();
  }
  if (cc->InputSidePackets().HasTag("FLIP_HORIZONTALLY")) {
    cc->InputSidePackets().Tag("FLIP_HORIZONTALLY").Set<bool>();
  }
  if (cc->InputSidePackets().HasTag("FLIP_VERTICALLY")) {
    cc->InputSidePackets().Tag("FLIP_VERTICALLY").Set<bool>();
  }

  if (cc->Outputs().HasTag("LETTERBOX_PADDING")) {
    cc->Outputs().Tag("LETTERBOX_PADDING").Set<std::array<float, 4>>();
  }

  if (use_gpu) {
#if !MEDIAPIPE_DISABLE_GPU
    MP_RETURN_IF_ERROR(GlCalculatorHelper::UpdateContract(cc));
#endif  // !MEDIAPIPE_DISABLE_GPU
  }

  return absl::OkStatus();
}

absl::Status ImageTransformationCalculator::Open(CalculatorContext* cc) {
  // Inform the framework that we always output at the same timestamp
  // as we receive a packet at.
  cc->SetOffset(TimestampDiff(0));

  options_ = cc->Options<ImageTransformationCalculatorOptions>();

  if (cc->Inputs().HasTag(kGpuBufferTag)) {
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

  if (cc->InputSidePackets().HasTag("FLIP_HORIZONTALLY")) {
    flip_horizontally_ =
        cc->InputSidePackets().Tag("FLIP_HORIZONTALLY").Get<bool>();
  } else {
    flip_horizontally_ = options_.flip_horizontally();
  }

  if (cc->InputSidePackets().HasTag("FLIP_VERTICALLY")) {
    flip_vertically_ =
        cc->InputSidePackets().Tag("FLIP_VERTICALLY").Get<bool>();
  } else {
    flip_vertically_ = options_.flip_vertically();
  }

  scale_mode_ = ParseScaleMode(options_.scale_mode(), DEFAULT_SCALE_MODE);

  if (use_gpu_) {
#if !MEDIAPIPE_DISABLE_GPU
    // Let the helper access the GL context information.
    MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
#else
    RET_CHECK_FAIL() << "GPU processing not enabled.";
#endif  // !MEDIAPIPE_DISABLE_GPU
  }

  return absl::OkStatus();
}

absl::Status ImageTransformationCalculator::Process(CalculatorContext* cc) {
  // First update the video header if it is given, based on the rotation and
  // dimensions specified as side packets or options. This will only be done
  // once, so streaming transformation changes will not be reflected in
  // the header.
  if (cc->Inputs().HasTag(kVideoPrestreamTag) &&
      !cc->Inputs().Tag(kVideoPrestreamTag).IsEmpty() &&
      cc->Outputs().HasTag(kVideoPrestreamTag)) {
    mediapipe::VideoHeader header =
        cc->Inputs().Tag(kVideoPrestreamTag).Get<mediapipe::VideoHeader>();
    // Update the header's width and height if needed.
    ComputeOutputDimensions(header.width, header.height, &header.width,
                            &header.height);
    cc->Outputs()
        .Tag(kVideoPrestreamTag)
        .AddPacket(mediapipe::MakePacket<mediapipe::VideoHeader>(header).At(
            mediapipe::Timestamp::PreStream()));
  }

  // Override values if specified so.
  if (cc->Inputs().HasTag("ROTATION_DEGREES") &&
      !cc->Inputs().Tag("ROTATION_DEGREES").IsEmpty()) {
    rotation_ =
        DegreesToRotationMode(cc->Inputs().Tag("ROTATION_DEGREES").Get<int>());
  }
  if (cc->Inputs().HasTag("FLIP_HORIZONTALLY") &&
      !cc->Inputs().Tag("FLIP_HORIZONTALLY").IsEmpty()) {
    flip_horizontally_ = cc->Inputs().Tag("FLIP_HORIZONTALLY").Get<bool>();
  }
  if (cc->Inputs().HasTag("FLIP_VERTICALLY") &&
      !cc->Inputs().Tag("FLIP_VERTICALLY").IsEmpty()) {
    flip_vertically_ = cc->Inputs().Tag("FLIP_VERTICALLY").Get<bool>();
  }
  if (cc->Inputs().HasTag("OUTPUT_DIMENSIONS")) {
    if (cc->Inputs().Tag("OUTPUT_DIMENSIONS").IsEmpty()) {
      return absl::OkStatus();
    } else {
      const auto& image_size =
          cc->Inputs().Tag("OUTPUT_DIMENSIONS").Get<std::pair<int, int>>();
      output_width_ = image_size.first;
      output_height_ = image_size.second;
    }
  }

  if (use_gpu_) {
#if !MEDIAPIPE_DISABLE_GPU
    if (cc->Inputs().Tag(kGpuBufferTag).IsEmpty()) {
      return absl::OkStatus();
    }
    return gpu_helper_.RunInGlContext(
        [this, cc]() -> absl::Status { return RenderGpu(cc); });
#endif  // !MEDIAPIPE_DISABLE_GPU
  } else {
    if (cc->Inputs().Tag(kImageFrameTag).IsEmpty()) {
      return absl::OkStatus();
    }
    return RenderCpu(cc);
  }
  return absl::OkStatus();
}

absl::Status ImageTransformationCalculator::Close(CalculatorContext* cc) {
  if (use_gpu_) {
#if !MEDIAPIPE_DISABLE_GPU
    QuadRenderer* rgb_renderer = rgb_renderer_.release();
    QuadRenderer* yuv_renderer = yuv_renderer_.release();
    QuadRenderer* ext_rgb_renderer = ext_rgb_renderer_.release();
    gpu_helper_.RunInGlContext([rgb_renderer, yuv_renderer, ext_rgb_renderer] {
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
#endif  // !MEDIAPIPE_DISABLE_GPU
  }

  return absl::OkStatus();
}

absl::Status ImageTransformationCalculator::RenderCpu(CalculatorContext* cc) {
  cv::Mat input_mat;
  mediapipe::ImageFormat::Format format;

  const auto& input = cc->Inputs().Tag(kImageFrameTag).Get<ImageFrame>();
  input_mat = formats::MatView(&input);
  format = input.Format();

  const int input_width = input_mat.cols;
  const int input_height = input_mat.rows;
  int output_width;
  int output_height;
  ComputeOutputDimensions(input_width, input_height, &output_width,
                          &output_height);

  if (output_width_ > 0 && output_height_ > 0) {
    cv::Mat scaled_mat;
    if (scale_mode_ == mediapipe::ScaleMode_Mode_STRETCH) {
      int scale_flag =
          input_mat.cols > output_width_ && input_mat.rows > output_height_
              ? cv::INTER_AREA
              : cv::INTER_LINEAR;
      cv::resize(input_mat, scaled_mat, cv::Size(output_width_, output_height_),
                 0, 0, scale_flag);
    } else {
      const float scale =
          std::min(static_cast<float>(output_width_) / input_width,
                   static_cast<float>(output_height_) / input_height);
      const int target_width = std::round(input_width * scale);
      const int target_height = std::round(input_height * scale);
      int scale_flag = scale < 1.0f ? cv::INTER_AREA : cv::INTER_LINEAR;
      if (scale_mode_ == mediapipe::ScaleMode_Mode_FIT) {
        cv::Mat intermediate_mat;
        cv::resize(input_mat, intermediate_mat,
                   cv::Size(target_width, target_height), 0, 0, scale_flag);
        const int top = (output_height_ - target_height) / 2;
        const int bottom = output_height_ - target_height - top;
        const int left = (output_width_ - target_width) / 2;
        const int right = output_width_ - target_width - left;
        cv::copyMakeBorder(intermediate_mat, scaled_mat, top, bottom, left,
                           right,
                           options_.constant_padding() ? cv::BORDER_CONSTANT
                                                       : cv::BORDER_REPLICATE);
      } else {
        cv::resize(input_mat, scaled_mat, cv::Size(target_width, target_height),
                   0, 0, scale_flag);
        output_width = target_width;
        output_height = target_height;
      }
    }
    input_mat = scaled_mat;
  }

  if (cc->Outputs().HasTag("LETTERBOX_PADDING")) {
    auto padding = absl::make_unique<std::array<float, 4>>();
    ComputeOutputLetterboxPadding(input_width, input_height, output_width,
                                  output_height, padding.get());
    cc->Outputs()
        .Tag("LETTERBOX_PADDING")
        .Add(padding.release(), cc->InputTimestamp());
  }

  cv::Mat rotated_mat;
  cv::Size rotated_size(output_width, output_height);
  if (input_mat.size() == rotated_size) {
    const int angle = RotationModeToDegrees(rotation_);
    cv::Point2f src_center(input_mat.cols / 2.0, input_mat.rows / 2.0);
    cv::Mat rotation_mat = cv::getRotationMatrix2D(src_center, angle, 1.0);
    cv::warpAffine(input_mat, rotated_mat, rotation_mat, rotated_size);
  } else {
    switch (rotation_) {
      case mediapipe::RotationMode_Mode_UNKNOWN:
      case mediapipe::RotationMode_Mode_ROTATION_0:
        rotated_mat = input_mat;
        break;
      case mediapipe::RotationMode_Mode_ROTATION_90:
        cv::rotate(input_mat, rotated_mat, cv::ROTATE_90_COUNTERCLOCKWISE);
        break;
      case mediapipe::RotationMode_Mode_ROTATION_180:
        cv::rotate(input_mat, rotated_mat, cv::ROTATE_180);
        break;
      case mediapipe::RotationMode_Mode_ROTATION_270:
        cv::rotate(input_mat, rotated_mat, cv::ROTATE_90_CLOCKWISE);
        break;
    }
  }

  cv::Mat flipped_mat;
  if (flip_horizontally_ || flip_vertically_) {
    const int flip_code =
        flip_horizontally_ && flip_vertically_ ? -1 : flip_horizontally_;
    cv::flip(rotated_mat, flipped_mat, flip_code);
  } else {
    flipped_mat = rotated_mat;
  }

  std::unique_ptr<ImageFrame> output_frame(
      new ImageFrame(format, output_width, output_height));
  cv::Mat output_mat = formats::MatView(output_frame.get());
  flipped_mat.copyTo(output_mat);
  cc->Outputs()
      .Tag(kImageFrameTag)
      .Add(output_frame.release(), cc->InputTimestamp());

  return absl::OkStatus();
}

absl::Status ImageTransformationCalculator::RenderGpu(CalculatorContext* cc) {
#if !MEDIAPIPE_DISABLE_GPU
  const auto& input = cc->Inputs().Tag(kGpuBufferTag).Get<GpuBuffer>();
  const int input_width = input.width();
  const int input_height = input.height();

  int output_width;
  int output_height;
  ComputeOutputDimensions(input_width, input_height, &output_width,
                          &output_height);

  if (scale_mode_ == mediapipe::ScaleMode_Mode_FILL_AND_CROP) {
    const float scale =
        std::min(static_cast<float>(output_width_) / input_width,
                 static_cast<float>(output_height_) / input_height);
    output_width = std::round(input_width * scale);
    output_height = std::round(input_height * scale);
  }

  if (cc->Outputs().HasTag("LETTERBOX_PADDING")) {
    auto padding = absl::make_unique<std::array<float, 4>>();
    ComputeOutputLetterboxPadding(input_width, input_height, output_width,
                                  output_height, padding.get());
    cc->Outputs()
        .Tag("LETTERBOX_PADDING")
        .Add(padding.release(), cc->InputTimestamp());
  }

  QuadRenderer* renderer = nullptr;
  GlTexture src1;

#if defined(MEDIAPIPE_IOS)
  if (input.format() == GpuBufferFormat::kBiPlanar420YpCbCr8VideoRange ||
      input.format() == GpuBufferFormat::kBiPlanar420YpCbCr8FullRange) {
    if (!yuv_renderer_) {
      yuv_renderer_ = absl::make_unique<QuadRenderer>();
      MP_RETURN_IF_ERROR(
          yuv_renderer_->GlSetup(::mediapipe::kYUV2TexToRGBFragmentShader,
                                 {"video_frame_y", "video_frame_uv"}));
    }
    renderer = yuv_renderer_.get();
    src1 = gpu_helper_.CreateSourceTexture(input, 0);
  } else  // NOLINT(readability/braces)
#endif    // iOS
  {
    src1 = gpu_helper_.CreateSourceTexture(input);
#if defined(TEXTURE_EXTERNAL_OES)
    if (src1.target() == GL_TEXTURE_EXTERNAL_OES) {
      if (!ext_rgb_renderer_) {
        ext_rgb_renderer_ = absl::make_unique<QuadRenderer>();
        MP_RETURN_IF_ERROR(ext_rgb_renderer_->GlSetup(
            ::mediapipe::kBasicTexturedFragmentShaderOES, {"video_frame"}));
      }
      renderer = ext_rgb_renderer_.get();
    } else  // NOLINT(readability/braces)
#endif      // TEXTURE_EXTERNAL_OES
    {
      if (!rgb_renderer_) {
        rgb_renderer_ = absl::make_unique<QuadRenderer>();
        MP_RETURN_IF_ERROR(rgb_renderer_->GlSetup());
      }
      renderer = rgb_renderer_.get();
    }
  }
  RET_CHECK(renderer) << "Unsupported input texture type";

  mediapipe::FrameScaleMode scale_mode = mediapipe::FrameScaleModeFromProto(
      scale_mode_, mediapipe::FrameScaleMode::kStretch);
  mediapipe::FrameRotation rotation =
      mediapipe::FrameRotationFromDegrees(RotationModeToDegrees(rotation_));

  auto dst = gpu_helper_.CreateDestinationTexture(output_width, output_height,
                                                  input.format());

  gpu_helper_.BindFramebuffer(dst);
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(src1.target(), src1.name());

  MP_RETURN_IF_ERROR(renderer->GlRender(
      src1.width(), src1.height(), dst.width(), dst.height(), scale_mode,
      rotation, flip_horizontally_, flip_vertically_,
      /*flip_texture=*/false));

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(src1.target(), 0);

  // Execute GL commands, before getting result.
  glFlush();

  auto output = dst.template GetFrame<GpuBuffer>();
  cc->Outputs().Tag(kGpuBufferTag).Add(output.release(), cc->InputTimestamp());

#endif  // !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
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
  padding->fill(0.f);
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
