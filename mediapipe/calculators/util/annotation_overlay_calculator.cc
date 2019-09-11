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

#include <memory>

#include "mediapipe/calculators/util/annotation_overlay_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/vector.h"
#include "mediapipe/util/annotation_renderer.h"
#include "mediapipe/util/color.pb.h"

#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/shader_util.h"
#endif  // __ANDROID__ or iOS

namespace mediapipe {

namespace {

constexpr char kInputFrameTag[] = "INPUT_FRAME";
constexpr char kOutputFrameTag[] = "OUTPUT_FRAME";

constexpr char kInputFrameTagGpu[] = "INPUT_FRAME_GPU";
constexpr char kOutputFrameTagGpu[] = "OUTPUT_FRAME_GPU";

enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };

// Round up n to next multiple of m.
size_t RoundUp(size_t n, size_t m) { return ((n + m - 1) / m) * m; }  // NOLINT

// When using GPU, this color will become transparent when the calculator
// merges the annotation overlay with the image frame. As a result, drawing in
// this color is not supported and it should be set to something unlikely used.
constexpr int kAnnotationBackgroundColor[] = {100, 101, 102};
}  // namespace

// A calculator for rendering data on images.
//
// Inputs:
//  1. INPUT_FRAME or INPUT_FRAME_GPU (optional): An ImageFrame (or GpuBuffer)
//     containing the input image.
//     If output is CPU, and input isn't provided, the renderer creates a
//     blank canvas with the width, height and color provided in the options.
//  2. RenderData proto on variable number of input streams. All the RenderData
//     at a particular timestamp is drawn on the image in the order of their
//     input streams. No tags required.
//
// Output:
//  1. OUTPUT_FRAME or OUTPUT_FRAME_GPU: A rendered ImageFrame (or GpuBuffer).
//
// For CPU input frames, only SRGBA, SRGB and GRAY8 format are supported. The
// output format is the same as input except for GRAY8 where the output is in
// SRGB to support annotations in color.
//
// For GPU input frames, only 4-channel images are supported.
//
// Note: When using GPU, drawing with color kAnnotationBackgroundColor (defined
// above) is not supported.
//
// Example config (CPU):
// node {
//   calculator: "AnnotationOverlayCalculator"
//   input_stream: "INPUT_FRAME:image_frames"
//   input_stream: "render_data_1"
//   input_stream: "render_data_2"
//   input_stream: "render_data_3"
//   output_stream: "OUTPUT_FRAME:decorated_frames"
//   options {
//     [mediapipe.AnnotationOverlayCalculatorOptions.ext] {
//     }
//   }
// }
//
// Example config (GPU):
// node {
//   calculator: "AnnotationOverlayCalculator"
//   input_stream: "INPUT_FRAME_GPU:image_frames"
//   input_stream: "render_data_1"
//   input_stream: "render_data_2"
//   input_stream: "render_data_3"
//   output_stream: "OUTPUT_FRAME_GPU:decorated_frames"
//   options {
//     [mediapipe.AnnotationOverlayCalculatorOptions.ext] {
//     }
//   }
// }
//
class AnnotationOverlayCalculator : public CalculatorBase {
 public:
  AnnotationOverlayCalculator() = default;
  ~AnnotationOverlayCalculator() override = default;

  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  // From Calculator.
  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;
  ::mediapipe::Status Close(CalculatorContext* cc) override;

 private:
  ::mediapipe::Status CreateRenderTargetCpu(CalculatorContext* cc,
                                            std::unique_ptr<cv::Mat>& image_mat,
                                            ImageFormat::Format* target_format);
  ::mediapipe::Status CreateRenderTargetGpu(
      CalculatorContext* cc, std::unique_ptr<cv::Mat>& image_mat);
  ::mediapipe::Status RenderToGpu(CalculatorContext* cc, uchar* overlay_image);
  ::mediapipe::Status RenderToCpu(CalculatorContext* cc,
                                  const ImageFormat::Format& target_format,
                                  uchar* data_image);

  ::mediapipe::Status GlRender(CalculatorContext* cc);
  ::mediapipe::Status GlSetup(CalculatorContext* cc);

  // Options for the calculator.
  AnnotationOverlayCalculatorOptions options_;

  // Underlying helper renderer library.
  std::unique_ptr<AnnotationRenderer> renderer_;

  // Number of input streams with render data.
  int num_render_streams_;

  // Indicates if image frame is available as input.
  bool image_frame_available_ = false;

  bool use_gpu_ = false;
  bool gpu_initialized_ = false;
#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
  mediapipe::GlCalculatorHelper gpu_helper_;
  GLuint program_ = 0;
  GLuint image_mat_tex_ = 0;  // Overlay drawing image for GPU.
  int width_ = 0;
  int height_ = 0;
#endif  // __ANDROID__ or iOS
};
REGISTER_CALCULATOR(AnnotationOverlayCalculator);

::mediapipe::Status AnnotationOverlayCalculator::GetContract(
    CalculatorContract* cc) {
  CHECK_GE(cc->Inputs().NumEntries(), 1);

  if (cc->Inputs().HasTag(kInputFrameTag) &&
      cc->Inputs().HasTag(kInputFrameTagGpu)) {
    return ::mediapipe::InternalError("Cannot have multiple input images.");
  }
  if (cc->Inputs().HasTag(kInputFrameTagGpu) !=
      cc->Outputs().HasTag(kOutputFrameTagGpu)) {
    return ::mediapipe::InternalError("GPU output must have GPU input.");
  }

  // Assume all inputs are render streams; adjust below.
  int num_render_streams = cc->Inputs().NumEntries();

  // Input image to render onto copy of.
#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
  if (cc->Inputs().HasTag(kInputFrameTagGpu)) {
    cc->Inputs().Tag(kInputFrameTagGpu).Set<mediapipe::GpuBuffer>();
    num_render_streams = cc->Inputs().NumEntries() - 1;
  }
#endif  // __ANDROID__ or iOS
  if (cc->Inputs().HasTag(kInputFrameTag)) {
    cc->Inputs().Tag(kInputFrameTag).Set<ImageFrame>();
    num_render_streams = cc->Inputs().NumEntries() - 1;
  }

  // Data streams to render.
  for (int i = 0; i < num_render_streams; ++i) {
    cc->Inputs().Index(i).Set<RenderData>();
  }

  // Rendered image.
#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
  if (cc->Outputs().HasTag(kOutputFrameTagGpu)) {
    cc->Outputs().Tag(kOutputFrameTagGpu).Set<mediapipe::GpuBuffer>();
  }
#endif  // __ANDROID__ or iOS
  if (cc->Outputs().HasTag(kOutputFrameTag)) {
    cc->Outputs().Tag(kOutputFrameTag).Set<ImageFrame>();
  }

#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
  MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
#endif  // __ANDROID__ or iOS

  return ::mediapipe::OkStatus();
}

::mediapipe::Status AnnotationOverlayCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  options_ = cc->Options<AnnotationOverlayCalculatorOptions>();
  if (cc->Inputs().HasTag(kInputFrameTagGpu) &&
      cc->Outputs().HasTag(kOutputFrameTagGpu)) {
#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
    use_gpu_ = true;
#else
    RET_CHECK_FAIL() << "GPU processing is for Android and iOS only.";
#endif  // __ANDROID__ or iOS
  }

  if (cc->Inputs().HasTag(kInputFrameTagGpu) ||
      cc->Inputs().HasTag(kInputFrameTag)) {
    image_frame_available_ = true;
    num_render_streams_ = cc->Inputs().NumEntries() - 1;
  } else {
    image_frame_available_ = false;
    RET_CHECK(options_.has_canvas_width_px());
    RET_CHECK(options_.has_canvas_height_px());
    num_render_streams_ = cc->Inputs().NumEntries();
  }

  // Initialize the helper renderer library.
  renderer_ = absl::make_unique<AnnotationRenderer>();
  renderer_->SetFlipTextVertically(options_.flip_text_vertically());

  // Set the output header based on the input header (if present).
  const char* input_tag = use_gpu_ ? kInputFrameTagGpu : kInputFrameTag;
  const char* output_tag = use_gpu_ ? kOutputFrameTagGpu : kOutputFrameTag;
  if (image_frame_available_ &&
      !cc->Inputs().Tag(input_tag).Header().IsEmpty()) {
    const auto& input_header =
        cc->Inputs().Tag(input_tag).Header().Get<VideoHeader>();
    auto* output_video_header = new VideoHeader(input_header);
    cc->Outputs().Tag(output_tag).SetHeader(Adopt(output_video_header));
  }

  if (use_gpu_) {
#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
    MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
#endif  // __ANDROID__ or iOS
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status AnnotationOverlayCalculator::Process(
    CalculatorContext* cc) {
  // Initialize render target, drawn with OpenCV.
  std::unique_ptr<cv::Mat> image_mat;
  ImageFormat::Format target_format;
  if (use_gpu_) {
#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
    if (!gpu_initialized_) {
      MP_RETURN_IF_ERROR(
          gpu_helper_.RunInGlContext([this, cc]() -> ::mediapipe::Status {
            MP_RETURN_IF_ERROR(GlSetup(cc));
            return ::mediapipe::OkStatus();
          }));
      gpu_initialized_ = true;
    }
#endif  // __ANDROID__ or iOS
    MP_RETURN_IF_ERROR(CreateRenderTargetGpu(cc, image_mat));
  } else {
    MP_RETURN_IF_ERROR(CreateRenderTargetCpu(cc, image_mat, &target_format));
  }

  // Reset the renderer with the image_mat. No copy here.
  renderer_->AdoptImage(image_mat.get());

  // Render streams onto render target.
  for (int i = 0; i < num_render_streams_; ++i) {
    if (cc->Inputs().Index(i).IsEmpty()) {
      continue;
    }
    const RenderData& render_data = cc->Inputs().Index(i).Get<RenderData>();
    renderer_->RenderDataOnImage(render_data);
  }

  if (use_gpu_) {
#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
    // Overlay rendered image in OpenGL, onto a copy of input.
    uchar* image_mat_ptr = image_mat->data;
    MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext(
        [this, cc, image_mat_ptr]() -> ::mediapipe::Status {
          MP_RETURN_IF_ERROR(RenderToGpu(cc, image_mat_ptr));
          return ::mediapipe::OkStatus();
        }));
#endif  // __ANDROID__ or iOS
  } else {
    // Copy the rendered image to output.
    uchar* image_mat_ptr = image_mat->data;
    MP_RETURN_IF_ERROR(RenderToCpu(cc, target_format, image_mat_ptr));
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status AnnotationOverlayCalculator::Close(CalculatorContext* cc) {
#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
  gpu_helper_.RunInGlContext([this] {
    if (program_) glDeleteProgram(program_);
    program_ = 0;
    if (image_mat_tex_) glDeleteTextures(1, &image_mat_tex_);
    image_mat_tex_ = 0;
  });
#endif  // __ANDROID__ or iOS

  return ::mediapipe::OkStatus();
}

::mediapipe::Status AnnotationOverlayCalculator::RenderToCpu(
    CalculatorContext* cc, const ImageFormat::Format& target_format,
    uchar* data_image) {
  auto output_frame = absl::make_unique<ImageFrame>(
      target_format, renderer_->GetImageWidth(), renderer_->GetImageHeight());

#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
  output_frame->CopyPixelData(target_format, renderer_->GetImageWidth(),
                              renderer_->GetImageHeight(), data_image,
                              ImageFrame::kGlDefaultAlignmentBoundary);
#else
  output_frame->CopyPixelData(target_format, renderer_->GetImageWidth(),
                              renderer_->GetImageHeight(), data_image,
                              ImageFrame::kDefaultAlignmentBoundary);
#endif  // __ANDROID__ or iOS

  cc->Outputs()
      .Tag(kOutputFrameTag)
      .Add(output_frame.release(), cc->InputTimestamp());

  return ::mediapipe::OkStatus();
}

::mediapipe::Status AnnotationOverlayCalculator::RenderToGpu(
    CalculatorContext* cc, uchar* overlay_image) {
#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
  // Source and destination textures.
  const auto& input_frame =
      cc->Inputs().Tag(kInputFrameTagGpu).Get<mediapipe::GpuBuffer>();
  auto input_texture = gpu_helper_.CreateSourceTexture(input_frame);

  auto output_texture = gpu_helper_.CreateDestinationTexture(
      width_, height_, mediapipe::GpuBufferFormat::kBGRA32);

  // Upload render target to GPU.
  {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBindTexture(GL_TEXTURE_2D, image_mat_tex_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGB,
                    GL_UNSIGNED_BYTE, overlay_image);
    glBindTexture(GL_TEXTURE_2D, 0);
  }

  // Blend overlay image in GPU shader.
  {
    gpu_helper_.BindFramebuffer(output_texture);  // GL_TEXTURE0

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, input_texture.name());
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, image_mat_tex_);

    MP_RETURN_IF_ERROR(GlRender(cc));

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glFlush();
  }

  // Send out blended image as GPU packet.
  auto output_frame = output_texture.GetFrame<mediapipe::GpuBuffer>();
  cc->Outputs()
      .Tag(kOutputFrameTagGpu)
      .Add(output_frame.release(), cc->InputTimestamp());

  // Cleanup
  input_texture.Release();
  output_texture.Release();
#endif  // __ANDROID__ or iOS

  return ::mediapipe::OkStatus();
}

::mediapipe::Status AnnotationOverlayCalculator::CreateRenderTargetCpu(
    CalculatorContext* cc, std::unique_ptr<cv::Mat>& image_mat,
    ImageFormat::Format* target_format) {
  if (image_frame_available_) {
    const auto& input_frame =
        cc->Inputs().Tag(kInputFrameTag).Get<ImageFrame>();

    int target_mat_type;
    switch (input_frame.Format()) {
      case ImageFormat::SRGBA:
        *target_format = ImageFormat::SRGBA;
        target_mat_type = CV_8UC4;
        break;
      case ImageFormat::SRGB:
        *target_format = ImageFormat::SRGB;
        target_mat_type = CV_8UC3;
        break;
      case ImageFormat::GRAY8:
        *target_format = ImageFormat::SRGB;
        target_mat_type = CV_8UC3;
        break;
      default:
        return ::mediapipe::UnknownError("Unexpected image frame format.");
        break;
    }

    image_mat = absl::make_unique<cv::Mat>(
        input_frame.Height(), input_frame.Width(), target_mat_type);
    if (input_frame.Format() == ImageFormat::GRAY8) {
      const int target_num_channels =
          ImageFrame::NumberOfChannelsForFormat(*target_format);
      for (int i = 0; i < input_frame.PixelDataSize(); i++) {
        const auto& pix = input_frame.PixelData()[i];
        for (int c = 0; c < target_num_channels; c++) {
          image_mat->data[i * target_num_channels + c] = pix;
        }
      }
    } else {
      // Make of a copy since the input frame may be consumed by other nodes.
      const int buffer_size =
          input_frame.Height() * input_frame.Width() *
          ImageFrame::NumberOfChannelsForFormat(*target_format);
      input_frame.CopyToBuffer(image_mat->data, buffer_size);
    }
  } else {
    image_mat = absl::make_unique<cv::Mat>(
        options_.canvas_height_px(), options_.canvas_width_px(), CV_8UC3,
        cv::Scalar(options_.canvas_color().r(), options_.canvas_color().g(),
                   options_.canvas_color().b()));
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status AnnotationOverlayCalculator::CreateRenderTargetGpu(
    CalculatorContext* cc, std::unique_ptr<cv::Mat>& image_mat) {
#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
  if (image_frame_available_) {
    const auto& input_frame =
        cc->Inputs().Tag(kInputFrameTagGpu).Get<mediapipe::GpuBuffer>();

    const mediapipe::ImageFormat::Format format =
        mediapipe::ImageFormatForGpuBufferFormat(input_frame.format());
    if (format != mediapipe::ImageFormat::SRGBA)
      RET_CHECK_FAIL() << "Unsupported GPU input format.";

    image_mat = absl::make_unique<cv::Mat>(
        height_, width_, CV_8UC3,
        cv::Scalar(kAnnotationBackgroundColor[0], kAnnotationBackgroundColor[1],
                   kAnnotationBackgroundColor[2]));
  } else {
    image_mat = absl::make_unique<cv::Mat>(
        options_.canvas_height_px(), options_.canvas_width_px(), CV_8UC3,
        cv::Scalar(options_.canvas_color().r(), options_.canvas_color().g(),
                   options_.canvas_color().b()));
  }
#endif  // __ANDROID__ or iOS

  return ::mediapipe::OkStatus();
}

::mediapipe::Status AnnotationOverlayCalculator::GlRender(
    CalculatorContext* cc) {
#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
  static const GLfloat square_vertices[] = {
      -1.0f, -1.0f,  // bottom left
      1.0f,  -1.0f,  // bottom right
      -1.0f, 1.0f,   // top left
      1.0f,  1.0f,   // top right
  };
  static const GLfloat texture_vertices[] = {
      0.0f, 0.0f,  // bottom left
      1.0f, 0.0f,  // bottom right
      0.0f, 1.0f,  // top left
      1.0f, 1.0f,  // top right
  };

  // program
  glUseProgram(program_);

  // vertex storage
  GLuint vbo[2];
  glGenBuffers(2, vbo);
  GLuint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  // vbo 0
  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(GLfloat), square_vertices,
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(ATTRIB_VERTEX);
  glVertexAttribPointer(ATTRIB_VERTEX, 2, GL_FLOAT, 0, 0, nullptr);

  // vbo 1
  glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
  glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(GLfloat), texture_vertices,
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(ATTRIB_TEXTURE_POSITION);
  glVertexAttribPointer(ATTRIB_TEXTURE_POSITION, 2, GL_FLOAT, 0, 0, nullptr);

  // draw
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  // cleanup
  glDisableVertexAttribArray(ATTRIB_VERTEX);
  glDisableVertexAttribArray(ATTRIB_TEXTURE_POSITION);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
  glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(2, vbo);
#endif  // __ANDROID__ or iOS

  return ::mediapipe::OkStatus();
}

::mediapipe::Status AnnotationOverlayCalculator::GlSetup(
    CalculatorContext* cc) {
#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
  const GLint attr_location[NUM_ATTRIBUTES] = {
      ATTRIB_VERTEX,
      ATTRIB_TEXTURE_POSITION,
  };
  const GLchar* attr_name[NUM_ATTRIBUTES] = {
      "position",
      "texture_coordinate",
  };

  // Shader to overlay a texture onto another when overlay is non-zero.
  const GLchar* frag_src = GLES_VERSION_COMPAT
      R"(
  #if __VERSION__ < 130
    #define in varying
  #endif  // __VERSION__ < 130

  #ifdef GL_ES
    #define fragColor gl_FragColor
    precision highp float;
  #else
    #define lowp
    #define mediump
    #define highp
    #define texture2D texture
    out vec4 fragColor;
  #endif  // defined(GL_ES)

    in vec2 sample_coordinate;
    uniform sampler2D input_frame;
    uniform sampler2D overlay;
    uniform vec3 transparent_color;

    void main() {
      vec3 image_pix = texture2D(input_frame, sample_coordinate).rgb;
      vec3 overlay_pix = texture2D(overlay, sample_coordinate).rgb;
      vec3 out_pix = image_pix;
      float dist = distance(overlay_pix.rgb, transparent_color);
      if (dist > 0.001) out_pix = overlay_pix;
      fragColor.rgb = out_pix;
      fragColor.a = 1.0;
    }
  )";

  // Create shader program and set parameters
  mediapipe::GlhCreateProgram(mediapipe::kBasicVertexShader, frag_src,
                              NUM_ATTRIBUTES, (const GLchar**)&attr_name[0],
                              attr_location, &program_);
  RET_CHECK(program_) << "Problem initializing the program.";
  glUseProgram(program_);
  glUniform1i(glGetUniformLocation(program_, "input_frame"), 1);
  glUniform1i(glGetUniformLocation(program_, "overlay"), 2);
  glUniform3f(glGetUniformLocation(program_, "transparent_color"),
              kAnnotationBackgroundColor[0] / 255.0,
              kAnnotationBackgroundColor[1] / 255.0,
              kAnnotationBackgroundColor[2] / 255.0);

  // Init texture for opencv rendered frame.
  const auto& input_frame =
      cc->Inputs().Tag(kInputFrameTagGpu).Get<mediapipe::GpuBuffer>();
  // Ensure GPU texture is divisible by 4. See b/138751944 for more info.
  width_ =
      RoundUp(input_frame.width(), ImageFrame::kGlDefaultAlignmentBoundary);
  height_ =
      RoundUp(input_frame.height(), ImageFrame::kGlDefaultAlignmentBoundary);
  {
    glGenTextures(1, &image_mat_tex_);
    glBindTexture(GL_TEXTURE_2D, image_mat_tex_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width_, height_, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
  }
#endif  // __ANDROID__ or iOS

  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
