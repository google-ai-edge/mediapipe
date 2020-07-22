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
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/options_util.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_quad_renderer.h"
#include "mediapipe/gpu/gl_scaler_calculator.pb.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
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

  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;

  ::mediapipe::Status GlSetup();
  ::mediapipe::Status GlRender(const GlTexture& src, const GlTexture& dst);
  void GetOutputDimensions(int src_width, int src_height, int* dst_width,
                           int* dst_height);
  void GetOutputPadding(int src_width, int src_height, int dst_width,
                        int dst_height, float* top_bottom_padding,
                        float* left_right_padding);
  GpuBufferFormat GetOutputFormat() { return GpuBufferFormat::kBGRA32; }

 private:
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
};
REGISTER_CALCULATOR(GlScalerCalculator);

// static
::mediapipe::Status GlScalerCalculator::GetContract(CalculatorContract* cc) {
  TagOrIndex(&cc->Inputs(), "VIDEO", 0).Set<GpuBuffer>();
  TagOrIndex(&cc->Outputs(), "VIDEO", 0).Set<GpuBuffer>();
  if (cc->Inputs().HasTag("ROTATION")) {
    cc->Inputs().Tag("ROTATION").Set<int>();
  }
  if (cc->Inputs().HasTag("OUTPUT_DIMENSIONS")) {
    cc->Inputs().Tag("OUTPUT_DIMENSIONS").Set<DimensionsPacketType>();
  }
  MP_RETURN_IF_ERROR(GlCalculatorHelper::UpdateContract(cc));

  if (cc->InputSidePackets().HasTag("OPTIONS")) {
    cc->InputSidePackets().Tag("OPTIONS").Set<GlScalerCalculatorOptions>();
  }
  if (HasTagOrIndex(&cc->InputSidePackets(), "OUTPUT_DIMENSIONS", 1)) {
    TagOrIndex(&cc->InputSidePackets(), "OUTPUT_DIMENSIONS", 1)
        .Set<DimensionsPacketType>();
  }
  if (cc->InputSidePackets().HasTag("ROTATION")) {
    // Counterclockwise rotation.
    cc->InputSidePackets().Tag("ROTATION").Set<int>();
  }

  if (cc->Outputs().HasTag("TOP_BOTTOM_PADDING") &&
      cc->Outputs().HasTag("LEFT_RIGHT_PADDING")) {
    cc->Outputs().Tag("TOP_BOTTOM_PADDING").Set<float>();
    cc->Outputs().Tag("LEFT_RIGHT_PADDING").Set<float>();
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status GlScalerCalculator::Open(CalculatorContext* cc) {
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

  if (HasTagOrIndex(cc->InputSidePackets(), "OUTPUT_DIMENSIONS", 1)) {
    const auto& dimensions =
        TagOrIndex(cc->InputSidePackets(), "OUTPUT_DIMENSIONS", 1)
            .Get<DimensionsPacketType>();
    dst_width_ = dimensions[0];
    dst_height_ = dimensions[1];
  }
  if (cc->InputSidePackets().HasTag("ROTATION")) {
    rotation_ccw = cc->InputSidePackets().Tag("ROTATION").Get<int>();
  }

  MP_RETURN_IF_ERROR(FrameRotationFromInt(&rotation_, rotation_ccw));

  return ::mediapipe::OkStatus();
}

::mediapipe::Status GlScalerCalculator::Process(CalculatorContext* cc) {
  if (cc->Inputs().HasTag("OUTPUT_DIMENSIONS")) {
    if (cc->Inputs().Tag("OUTPUT_DIMENSIONS").IsEmpty()) {
      // OUTPUT_DIMENSIONS input stream is specified, but value is missing.
      return ::mediapipe::OkStatus();
    }

    const auto& dimensions =
        cc->Inputs().Tag("OUTPUT_DIMENSIONS").Get<DimensionsPacketType>();
    dst_width_ = dimensions[0];
    dst_height_ = dimensions[1];
  }

  return helper_.RunInGlContext([this, cc]() -> ::mediapipe::Status {
    const auto& input = TagOrIndex(cc->Inputs(), "VIDEO", 0).Get<GpuBuffer>();
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
#endif  // __APPLE__
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
#endif  // __ANDROID__
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
    if (cc->Inputs().HasTag("ROTATION")) {
      int rotation_ccw = cc->Inputs().Tag("ROTATION").Get<int>();
      MP_RETURN_IF_ERROR(FrameRotationFromInt(&rotation_, rotation_ccw));
    }

    int dst_width;
    int dst_height;
    GetOutputDimensions(src1.width(), src1.height(), &dst_width, &dst_height);

    if (cc->Outputs().HasTag("TOP_BOTTOM_PADDING") &&
        cc->Outputs().HasTag("LEFT_RIGHT_PADDING")) {
      float top_bottom_padding;
      float left_right_padding;
      GetOutputPadding(src1.width(), src1.height(), dst_width, dst_height,
                       &top_bottom_padding, &left_right_padding);
      cc->Outputs()
          .Tag("TOP_BOTTOM_PADDING")
          .AddPacket(
              MakePacket<float>(top_bottom_padding).At(cc->InputTimestamp()));
      cc->Outputs()
          .Tag("LEFT_RIGHT_PADDING")
          .AddPacket(
              MakePacket<float>(left_right_padding).At(cc->InputTimestamp()));
    }

    auto dst = helper_.CreateDestinationTexture(dst_width, dst_height,
                                                GetOutputFormat());

    helper_.BindFramebuffer(dst);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(src1.target(), src1.name());
    if (src2.name()) {
      glActiveTexture(GL_TEXTURE2);
      glBindTexture(src2.target(), src2.name());
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

    auto output = dst.GetFrame<GpuBuffer>();

    TagOrIndex(&cc->Outputs(), "VIDEO", 0)
        .Add(output.release(), cc->InputTimestamp());

    return ::mediapipe::OkStatus();
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
