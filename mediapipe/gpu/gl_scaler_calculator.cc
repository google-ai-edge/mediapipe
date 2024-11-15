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

#include "absl/status/statusor.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/tool/options_util.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_quad_renderer.h"
#include "mediapipe/gpu/gl_scaler_calculator.pb.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_buffer_format.h"
#include "mediapipe/gpu/shader_util.h"

#ifdef __ANDROID__
// The size of Java arrays is dynamic, which makes it difficult to
// generate the right packet type with a fixed size. Therefore, we
// are using unsized arrays on Android.
typedef int DimensionsPacketType[];
#else
typedef int DimensionsPacketType[2];
#endif

namespace mediapipe {

constexpr char kLeftRightPaddingTag[] = "LEFT_RIGHT_PADDING";
constexpr char kTopBottomPaddingTag[] = "TOP_BOTTOM_PADDING";
constexpr char kOptionsTag[] = "OPTIONS";
constexpr char kOutputDimensionsTag[] = "OUTPUT_DIMENSIONS";
constexpr char kRotationTag[] = "ROTATION";
constexpr char kImageTag[] = "IMAGE";

using Image = mediapipe::Image;

// Scales, rotates, horizontal or vertical flips the image.
// See GlSimpleCalculatorBase for inputs, outputs and input side packets.
// Additional input streams:
//   ROTATION: the counterclockwise rotation angle in degrees. This allows
//   user to specify different rotation angles for different frames. If this
//   stream is provided, it will override the ROTATION input side packet.
//   OUTPUT_DIMENSIONS: the output width and height in pixels.
// Additional output streams:
//   TOP_BOTTOM_PADDING: If use FIT scale mode, this stream outputs the padding
//   size of the input image in normalized value [0, 1] for top and bottom
//   sides with equal padding. E.g. Using FIT scale mode, if the input images
//   size is 10x10 and the required output size is 20x40, then the top and
//   bottom side of the image will both having padding of 10 pixels. So the
//   value of output stream is 10 / 40 = 0.25.
//   LEFT_RIGHT_PADDING: If use FIT scale mode, this stream outputs the padding
//   size of the input image in normalized value [0, 1] for left and right side.
//   E.g. Using FIT scale mode, if the input images size is 10x10 and the
//   required output size is 6x5, then the left and right side of the image will
//   both having padding of 1 pixels. So the value of output stream is 1 / 5 =
//   0.2.
// Additional input side packets:
//   OPTIONS: the GlScalerCalculatorOptions to use. Will replace or merge with
//   existing calculator options, depending on field merge_fields.
//   OUTPUT_DIMENSIONS: the output width and height in pixels.
//   ROTATION: the counterclockwise rotation angle in degrees.
// These can also be specified as options.
// To enable horizontal or vertical flip, specify them in options.
// The flipping is applied after rotation.
class GlScalerCalculator : public CalculatorBase {
 public:
  GlScalerCalculator() {}
  ~GlScalerCalculator();

  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

  absl::Status GlSetup();
  absl::Status GlRender(const GlTexture& src, const GlTexture& dst);
  void GetOutputDimensions(int src_width, int src_height, int* dst_width,
                           int* dst_height);
  void GetOutputPadding(int src_width, int src_height, int dst_width,
                        int dst_height, float* top_bottom_padding,
                        float* left_right_padding);
  GpuBufferFormat GetOutputFormat(GpuBufferFormat input_format) {
    return use_input_format_for_output_ ? input_format
                                        : GpuBufferFormat::kBGRA32;
  }

 private:
  // Returns the input GpuBuffer, or fails if it's empty.
  absl::StatusOr<GpuBuffer> GetInputGpuBuffer(CalculatorContext* cc);

  GlCalculatorHelper helper_;
  int dst_width_ = 0;
  int dst_height_ = 0;
  float dst_scale_ = -1.f;
  FrameRotation rotation_;
  std::unique_ptr<QuadRenderer> rgb_renderer_;
  std::unique_ptr<QuadRenderer> yuv_renderer_;
#ifdef __ANDROID__
  std::unique_ptr<QuadRenderer> ext_rgb_renderer_;
#endif
  bool vertical_flip_output_;
  bool horizontal_flip_output_;
  FrameScaleMode scale_mode_ = FrameScaleMode::kStretch;
  bool use_nearest_neighbor_interpolation_ = false;
  bool use_input_format_for_output_ = false;
};
REGISTER_CALCULATOR(GlScalerCalculator);

// static
absl::Status GlScalerCalculator::GetContract(CalculatorContract* cc) {
  if (cc->Inputs().HasTag(kImageTag)) {
    cc->Inputs().Tag(kImageTag).Set<Image>();
  } else {
    TagOrIndex(&cc->Inputs(), "VIDEO", 0).Set<GpuBuffer>();
  }
  if (cc->Outputs().HasTag(kImageTag)) {
    cc->Outputs().Tag(kImageTag).Set<Image>();
  } else {
    TagOrIndex(&cc->Outputs(), "VIDEO", 0).Set<GpuBuffer>();
  }

  if (cc->Inputs().HasTag(kRotationTag)) {
    cc->Inputs().Tag(kRotationTag).Set<int>();
  }
  if (cc->Inputs().HasTag(kOutputDimensionsTag)) {
    cc->Inputs().Tag(kOutputDimensionsTag).Set<DimensionsPacketType>();
  }
  MP_RETURN_IF_ERROR(GlCalculatorHelper::UpdateContract(cc));

  if (cc->InputSidePackets().HasTag(kOptionsTag)) {
    cc->InputSidePackets().Tag(kOptionsTag).Set<GlScalerCalculatorOptions>();
  }
  if (HasTagOrIndex(&cc->InputSidePackets(), "OUTPUT_DIMENSIONS", 1)) {
    TagOrIndex(&cc->InputSidePackets(), "OUTPUT_DIMENSIONS", 1)
        .Set<DimensionsPacketType>();
  }
  if (cc->InputSidePackets().HasTag(kRotationTag)) {
    // Counterclockwise rotation.
    cc->InputSidePackets().Tag(kRotationTag).Set<int>();
  }

  if (cc->Outputs().HasTag(kTopBottomPaddingTag) &&
      cc->Outputs().HasTag(kLeftRightPaddingTag)) {
    cc->Outputs().Tag(kTopBottomPaddingTag).Set<float>();
    cc->Outputs().Tag(kLeftRightPaddingTag).Set<float>();
  }
  return absl::OkStatus();
}

absl::Status GlScalerCalculator::Open(CalculatorContext* cc) {
  // Inform the framework that we always output at the same timestamp
  // as we receive a packet at.
  cc->SetOffset(mediapipe::TimestampDiff(0));

  // Let the helper access the GL context information.
  MP_RETURN_IF_ERROR(helper_.Open(cc));

  int rotation_ccw = 0;
  const auto& options =
      tool::RetrieveOptions(cc->Options<GlScalerCalculatorOptions>(),
                            cc->InputSidePackets(), "OPTIONS");
  if (options.has_output_width()) {
    dst_width_ = options.output_width();
  }
  if (options.has_output_height()) {
    dst_height_ = options.output_height();
  }
  if (options.has_output_scale()) {
    dst_scale_ = options.output_scale();
  }
  if (options.has_rotation()) {
    rotation_ccw = options.rotation();
  }
  if (options.has_flip_vertical()) {
    vertical_flip_output_ = options.flip_vertical();
  } else {
    vertical_flip_output_ = false;
  }
  if (options.has_flip_horizontal()) {
    horizontal_flip_output_ = options.flip_horizontal();
  } else {
    horizontal_flip_output_ = false;
  }
  if (options.has_scale_mode()) {
    scale_mode_ =
        FrameScaleModeFromProto(options.scale_mode(), FrameScaleMode::kStretch);
  }
  use_nearest_neighbor_interpolation_ =
      options.use_nearest_neighbor_interpolation();
  use_input_format_for_output_ = options.use_input_format_for_output();
  if (HasTagOrIndex(cc->InputSidePackets(), "OUTPUT_DIMENSIONS", 1)) {
    const auto& dimensions =
        TagOrIndex(cc->InputSidePackets(), "OUTPUT_DIMENSIONS", 1)
            .Get<DimensionsPacketType>();
    dst_width_ = dimensions[0];
    dst_height_ = dimensions[1];
  }
  if (cc->InputSidePackets().HasTag(kRotationTag)) {
    rotation_ccw = cc->InputSidePackets().Tag(kRotationTag).Get<int>();
  }

  MP_RETURN_IF_ERROR(FrameRotationFromInt(&rotation_, rotation_ccw));

  return absl::OkStatus();
}

absl::StatusOr<GpuBuffer> GlScalerCalculator::GetInputGpuBuffer(
    CalculatorContext* cc) {
  if (cc->Inputs().HasTag(kImageTag)) {
    auto& input = cc->Inputs().Tag(kImageTag);
    RET_CHECK(!input.IsEmpty());
    return input.Get<Image>().GetGpuBuffer();
  }
  auto& input = TagOrIndex(cc->Inputs(), "VIDEO", 0);
  RET_CHECK(!input.IsEmpty());
  return input.Get<GpuBuffer>();
}

absl::Status GlScalerCalculator::Process(CalculatorContext* cc) {
  if (cc->Inputs().HasTag(kOutputDimensionsTag)) {
    if (cc->Inputs().Tag(kOutputDimensionsTag).IsEmpty()) {
      // OUTPUT_DIMENSIONS input stream is specified, but value is missing.
      return absl::OkStatus();
    }

    const auto& dimensions =
        cc->Inputs().Tag(kOutputDimensionsTag).Get<DimensionsPacketType>();
    dst_width_ = dimensions[0];
    dst_height_ = dimensions[1];
  }

  return helper_.RunInGlContext([this, cc]() -> absl::Status {
    MP_ASSIGN_OR_RETURN(GpuBuffer input, GetInputGpuBuffer(cc));
    QuadRenderer* renderer = nullptr;
    GlTexture src1;
    GlTexture src2;

#ifdef __APPLE__
    if (input.format() == GpuBufferFormat::kBiPlanar420YpCbCr8VideoRange ||
        input.format() == GpuBufferFormat::kBiPlanar420YpCbCr8FullRange) {
      if (!yuv_renderer_) {
        yuv_renderer_ = absl::make_unique<QuadRenderer>();
        MP_RETURN_IF_ERROR(yuv_renderer_->GlSetup(
            kYUV2TexToRGBFragmentShader, {"video_frame_y", "video_frame_uv"}));
      }
      renderer = yuv_renderer_.get();
      src1 = helper_.CreateSourceTexture(input, 0);
      src2 = helper_.CreateSourceTexture(input, 1);
    } else  // NOLINT(readability/braces)
#endif      // __APPLE__
    {
      src1 = helper_.CreateSourceTexture(input);
#ifdef __ANDROID__
      if (src1.target() == GL_TEXTURE_EXTERNAL_OES) {
        if (!ext_rgb_renderer_) {
          ext_rgb_renderer_ = absl::make_unique<QuadRenderer>();
          MP_RETURN_IF_ERROR(ext_rgb_renderer_->GlSetup(
              kBasicTexturedFragmentShaderOES, {"video_frame"}));
        }
        renderer = ext_rgb_renderer_.get();
      } else  // NOLINT(readability/braces)
#endif        // __ANDROID__
      {
        if (!rgb_renderer_) {
          rgb_renderer_ = absl::make_unique<QuadRenderer>();
          MP_RETURN_IF_ERROR(rgb_renderer_->GlSetup());
        }
        renderer = rgb_renderer_.get();
      }
    }
    RET_CHECK(renderer) << "Unsupported input texture type";

    // Override input side packet if ROTATION input packet is provided.
    if (cc->Inputs().HasTag(kRotationTag)) {
      int rotation_ccw = cc->Inputs().Tag(kRotationTag).Get<int>();
      MP_RETURN_IF_ERROR(FrameRotationFromInt(&rotation_, rotation_ccw));
    }

    int dst_width;
    int dst_height;
    GetOutputDimensions(src1.width(), src1.height(), &dst_width, &dst_height);

    if (cc->Outputs().HasTag(kTopBottomPaddingTag) &&
        cc->Outputs().HasTag(kLeftRightPaddingTag)) {
      float top_bottom_padding;
      float left_right_padding;
      GetOutputPadding(src1.width(), src1.height(), dst_width, dst_height,
                       &top_bottom_padding, &left_right_padding);
      cc->Outputs()
          .Tag(kTopBottomPaddingTag)
          .AddPacket(
              MakePacket<float>(top_bottom_padding).At(cc->InputTimestamp()));
      cc->Outputs()
          .Tag(kLeftRightPaddingTag)
          .AddPacket(
              MakePacket<float>(left_right_padding).At(cc->InputTimestamp()));
    }

    auto dst = helper_.CreateDestinationTexture(
        dst_width, dst_height, GetOutputFormat(input.format()));

    helper_.BindFramebuffer(dst);

    if (scale_mode_ == FrameScaleMode::kFit) {
      // In kFit scale mode, the rendered quad does not fill the whole
      // framebuffer, so clear it beforehand.
      glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
      glClear(GL_COLOR_BUFFER_BIT);
    }

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(src1.target(), src1.name());
    if (src2.name()) {
      glActiveTexture(GL_TEXTURE2);
      glBindTexture(src2.target(), src2.name());
    }

    if (use_nearest_neighbor_interpolation_) {
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    }

    MP_RETURN_IF_ERROR(renderer->GlRender(
        src1.width(), src1.height(), dst.width(), dst.height(), scale_mode_,
        rotation_, horizontal_flip_output_, vertical_flip_output_,
        /*flip_texture*/ false));

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(src1.target(), 0);
    if (src2.name()) {
      glActiveTexture(GL_TEXTURE2);
      glBindTexture(src2.target(), 0);
    }

    glFlush();

    if (cc->Outputs().HasTag(kImageTag)) {
      auto output = dst.GetFrame<Image>();
      cc->Outputs().Tag(kImageTag).Add(output.release(), cc->InputTimestamp());
    } else {
      auto output = dst.GetFrame<GpuBuffer>();
      TagOrIndex(&cc->Outputs(), "VIDEO", 0)
          .Add(output.release(), cc->InputTimestamp());
    }

    return absl::OkStatus();
  });
}

void GlScalerCalculator::GetOutputDimensions(int src_width, int src_height,
                                             int* dst_width, int* dst_height) {
  if (dst_width_ > 0 && dst_height_ > 0) {
    *dst_width = dst_width_;
    *dst_height = dst_height_;
    return;
  }
  if (dst_scale_ > 0) {
    // Scales the destination size, but just uses src size as a temporary for
    // calculations.
    src_width = static_cast<int>(src_width * dst_scale_);
    src_height = static_cast<int>(src_height * dst_scale_);
    // Round to nearest multiply of 4 for better memory alignment.
    src_width = ((src_width + 2) >> 2) << 2;
    src_height = ((src_height + 2) >> 2) << 2;
  }
  if (rotation_ == FrameRotation::k90 || rotation_ == FrameRotation::k270) {
    *dst_width = src_height;
    *dst_height = src_width;
  } else {
    *dst_width = src_width;
    *dst_height = src_height;
  }
}

void GlScalerCalculator::GetOutputPadding(int src_width, int src_height,
                                          int dst_width, int dst_height,
                                          float* top_bottom_padding,
                                          float* left_right_padding) {
  *top_bottom_padding = 0.0f;
  *left_right_padding = 0.0f;
  if (rotation_ == FrameRotation::k90 || rotation_ == FrameRotation::k270) {
    const int tmp = src_width;
    src_width = src_height;
    src_height = tmp;
  }
  if (scale_mode_ == FrameScaleMode::kFit) {
    const float src_scale = 1.0f * src_width / src_height;
    const float dst_scale = 1.0f * dst_width / dst_height;
    if (src_scale - dst_scale > 1e-5) {
      // Total padding on top and bottom sides.
      *top_bottom_padding =
          1.0f - 1.0f * dst_width / src_width * src_height / dst_height;
      // Get padding on each side.
      *top_bottom_padding /= 2.0f;

    } else if (dst_scale - src_scale > 1e-5) {
      // Total padding on left and right sides.
      *left_right_padding =
          1.0f - 1.0f / dst_width * src_width / src_height * dst_height;
      // Get padding on each side.
      *left_right_padding /= 2.0f;
    }
  }
}

GlScalerCalculator::~GlScalerCalculator() {
  // TODO: use move capture when we have C++14 or better.
  QuadRenderer* rgb_renderer = rgb_renderer_.release();
  QuadRenderer* yuv_renderer = yuv_renderer_.release();
  if (rgb_renderer || yuv_renderer) {
    helper_.RunInGlContext([rgb_renderer, yuv_renderer] {
      if (rgb_renderer) {
        rgb_renderer->GlTeardown();
        delete rgb_renderer;
      }
      if (yuv_renderer) {
        yuv_renderer->GlTeardown();
        delete yuv_renderer;
      }
    });
  }
}

}  // namespace mediapipe
