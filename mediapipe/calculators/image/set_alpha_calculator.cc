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

#include "absl/log/absl_log.h"
#include "mediapipe/calculators/image/set_alpha_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/vector.h"

#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/shader_util.h"
#endif  // !MEDIAPIPE_DISABLE_GPU

namespace mediapipe {

namespace {

constexpr char kInputFrameTag[] = "IMAGE";
constexpr char kInputAlphaTag[] = "ALPHA";
constexpr char kOutputFrameTag[] = "IMAGE";

constexpr char kInputFrameTagGpu[] = "IMAGE_GPU";
constexpr char kInputAlphaTagGpu[] = "ALPHA_GPU";
constexpr char kOutputFrameTagGpu[] = "IMAGE_GPU";

constexpr int kNumChannelsRGBA = 4;

enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };

// Combines an RGB cv::Mat and an alpha cv::Mat of the same dimensions into an
// RGBA cv::Mat. Alpha may be read as uint8 or as another numeric type; in the
// latter case, it is upscaled to values between 0 and 255 from an assumed input
// range of [0, 1). Only the first channel of Alpha is used. Input & output Mat
// must be uchar.
template <typename AlphaType>
absl::Status CopyAlphaImage(const cv::Mat& alpha_mat, cv::Mat& output_mat) {
  RET_CHECK_EQ(output_mat.rows, alpha_mat.rows);
  RET_CHECK_EQ(output_mat.cols, alpha_mat.cols);

  for (int i = 0; i < output_mat.rows; ++i) {
    const AlphaType* alpha_ptr = alpha_mat.ptr<AlphaType>(i);
    uchar* out_ptr = output_mat.ptr<uchar>(i);
    for (int j = 0; j < output_mat.cols; ++j) {
      const int out_idx = j * kNumChannelsRGBA;
      const int alpha_idx = j * alpha_mat.channels();
      if constexpr (std::is_same<AlphaType, uchar>::value) {
        out_ptr[out_idx + 3] = alpha_ptr[alpha_idx + 0];  // channel 0 of mask
      } else {
        const AlphaType alpha = alpha_ptr[alpha_idx + 0];  // channel 0 of mask
        out_ptr[out_idx + 3] = static_cast<uchar>(round(alpha * 255.0f));
      }
    }
  }
  return absl::OkStatus();
}
}  // namespace

// A calculator for setting the alpha channel of an RGBA image.
//
// The alpha channel can be set to a single value, or come from an image mask.
// If the input image has an alpha channel, it will be updated.
// If the input image doesn't have an alpha channel, one will be added.
// Adding alpha channel to a Grayscale (single channel) input is not supported.
//
// Inputs:
//   One of the following two IMAGE tags:
//   IMAGE: ImageFrame containing input image - RGB or RGBA.
//   IMAGE_GPU: GpuBuffer containing input image - RGB or RGBA.
//
//   ALPHA (optional): ImageFrame alpha mask to apply,
//                     can be any # of channels, only first channel used,
//                     must be same format as input
//   ALPHA_GPU (optional): GpuBuffer alpha mask to apply,
//                         can be any # of channels, only first channel used,
//                         must be same format as input
//   If ALPHA* input tag is not set, the 'alpha_value' option must be used.
//
// Output:
//   One of the following two tags:
//   IMAGE:    An ImageFrame with alpha channel set - RGBA only.
//   IMAGE_GPU:  A GpuBuffer with alpha channel set - RGBA only.
//
// Options:
//   alpha_value (optional): The alpha value to set to input image, [0-255],
//                           takes precedence over input mask.
//   If alpha_value is not set, the ALPHA* input tag must be used.
//
// Notes:
//   Either alpha_value option or ALPHA (or ALPHA_GPU) must be set.
//   All CPU inputs must have the same image dimensions and data type.
//
class SetAlphaCalculator : public CalculatorBase {
 public:
  SetAlphaCalculator() = default;
  ~SetAlphaCalculator() override = default;

  static absl::Status GetContract(CalculatorContract* cc);

  // From Calculator.
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status RenderGpu(CalculatorContext* cc);
  absl::Status RenderCpu(CalculatorContext* cc);

  absl::Status GlSetup(CalculatorContext* cc);
  void GlRender(CalculatorContext* cc);

  mediapipe::SetAlphaCalculatorOptions options_;
  float alpha_value_ = -1.f;

  bool use_gpu_ = false;
  bool gpu_initialized_ = false;
#if !MEDIAPIPE_DISABLE_GPU
  mediapipe::GlCalculatorHelper gpu_helper_;
  GLuint program_ = 0;
#endif  // !MEDIAPIPE_DISABLE_GPU
};
REGISTER_CALCULATOR(SetAlphaCalculator);

absl::Status SetAlphaCalculator::GetContract(CalculatorContract* cc) {
  RET_CHECK_GE(cc->Inputs().NumEntries(), 1);

  bool use_gpu = false;

  if (cc->Inputs().HasTag(kInputFrameTag) &&
      cc->Inputs().HasTag(kInputFrameTagGpu)) {
    return absl::InternalError("Cannot have multiple input images.");
  }
  if (cc->Inputs().HasTag(kInputFrameTagGpu) !=
      cc->Outputs().HasTag(kOutputFrameTagGpu)) {
    return absl::InternalError("GPU output must have GPU input.");
  }

  // Input image to add/edit alpha channel.
#if !MEDIAPIPE_DISABLE_GPU
  if (cc->Inputs().HasTag(kInputFrameTagGpu)) {
    cc->Inputs().Tag(kInputFrameTagGpu).Set<mediapipe::GpuBuffer>();
    use_gpu |= true;
  }
#endif  // !MEDIAPIPE_DISABLE_GPU
  if (cc->Inputs().HasTag(kInputFrameTag)) {
    cc->Inputs().Tag(kInputFrameTag).Set<ImageFrame>();
  }

  // Input alpha image mask (optional)
#if !MEDIAPIPE_DISABLE_GPU
  if (cc->Inputs().HasTag(kInputAlphaTagGpu)) {
    cc->Inputs().Tag(kInputAlphaTagGpu).Set<mediapipe::GpuBuffer>();
    use_gpu |= true;
  }
#endif  // !MEDIAPIPE_DISABLE_GPU
  if (cc->Inputs().HasTag(kInputAlphaTag)) {
    cc->Inputs().Tag(kInputAlphaTag).Set<ImageFrame>();
  }

  // RGBA output image.
#if !MEDIAPIPE_DISABLE_GPU
  if (cc->Outputs().HasTag(kOutputFrameTagGpu)) {
    cc->Outputs().Tag(kOutputFrameTagGpu).Set<mediapipe::GpuBuffer>();
    use_gpu |= true;
  }
#endif  // !MEDIAPIPE_DISABLE_GPU
  if (cc->Outputs().HasTag(kOutputFrameTag)) {
    cc->Outputs().Tag(kOutputFrameTag).Set<ImageFrame>();
  }

  if (use_gpu) {
#if !MEDIAPIPE_DISABLE_GPU
    MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
#endif  // !MEDIAPIPE_DISABLE_GPU
  }

  return absl::OkStatus();
}

absl::Status SetAlphaCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  options_ = cc->Options<mediapipe::SetAlphaCalculatorOptions>();

  if (cc->Inputs().HasTag(kInputFrameTagGpu) &&
      cc->Outputs().HasTag(kOutputFrameTagGpu)) {
#if !MEDIAPIPE_DISABLE_GPU
    use_gpu_ = true;
#else
    RET_CHECK_FAIL() << "GPU processing not enabled.";
#endif  // !MEDIAPIPE_DISABLE_GPU
  }

  // Get global value from options (-1 if not set).
  alpha_value_ = options_.alpha_value();
  if (use_gpu_) alpha_value_ /= 255.0;

  const bool use_image_mask = cc->Inputs().HasTag(kInputAlphaTag) ||
                              cc->Inputs().HasTag(kInputAlphaTagGpu);
  if (!((alpha_value_ >= 0) ^ use_image_mask))
    RET_CHECK_FAIL() << "Must use either image mask or options alpha value.";

  if (use_gpu_) {
#if !MEDIAPIPE_DISABLE_GPU
    MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
#endif
  }  //  !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
}

absl::Status SetAlphaCalculator::Process(CalculatorContext* cc) {
  if (use_gpu_) {
#if !MEDIAPIPE_DISABLE_GPU
    MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([this, cc]() -> absl::Status {
      if (!gpu_initialized_) {
        MP_RETURN_IF_ERROR(GlSetup(cc));
        gpu_initialized_ = true;
      }
      MP_RETURN_IF_ERROR(RenderGpu(cc));
      return absl::OkStatus();
    }));
#endif  // !MEDIAPIPE_DISABLE_GPU
  } else {
    MP_RETURN_IF_ERROR(RenderCpu(cc));
  }

  return absl::OkStatus();
}

absl::Status SetAlphaCalculator::Close(CalculatorContext* cc) {
#if !MEDIAPIPE_DISABLE_GPU
  gpu_helper_.RunInGlContext([this] {
    if (program_) glDeleteProgram(program_);
    program_ = 0;
  });
#endif  // !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
}

absl::Status SetAlphaCalculator::RenderCpu(CalculatorContext* cc) {
  if (cc->Inputs().Tag(kInputFrameTag).IsEmpty()) {
    return absl::OkStatus();
  }

  // Setup source image
  const auto& input_frame = cc->Inputs().Tag(kInputFrameTag).Get<ImageFrame>();
  const cv::Mat input_mat = formats::MatView(&input_frame);
  if (!(input_mat.type() == CV_8UC3 || input_mat.type() == CV_8UC4)) {
    ABSL_LOG(ERROR) << "Only 3 or 4 channel 8-bit input image supported";
  }

  // Setup destination image
  auto output_frame = absl::make_unique<ImageFrame>(
      ImageFormat::SRGBA, input_mat.cols, input_mat.rows);
  cv::Mat output_mat = formats::MatView(output_frame.get());

  const bool has_alpha_mask = cc->Inputs().HasTag(kInputAlphaTag) &&
                              !cc->Inputs().Tag(kInputAlphaTag).IsEmpty();
  const bool use_alpha_mask = alpha_value_ < 0 && has_alpha_mask;

  // Copy rgb part of the image in CPU
  if (input_mat.channels() == 3) {
    cv::cvtColor(input_mat, output_mat, cv::COLOR_RGB2RGBA);
  } else {
    input_mat.copyTo(output_mat);
  }

  // Setup alpha image in CPU.
  if (use_alpha_mask) {
    const auto& alpha_mask = cc->Inputs().Tag(kInputAlphaTag).Get<ImageFrame>();
    cv::Mat alpha_mat = formats::MatView(&alpha_mask);

    const bool alpha_is_float = CV_MAT_DEPTH(alpha_mat.type()) == CV_32F;
    RET_CHECK(alpha_is_float || CV_MAT_DEPTH(alpha_mat.type()) == CV_8U);

    if (alpha_is_float) {
      MP_RETURN_IF_ERROR(CopyAlphaImage<float>(alpha_mat, output_mat));
    } else {
      MP_RETURN_IF_ERROR(CopyAlphaImage<uchar>(alpha_mat, output_mat));
    }
  } else {
    const uchar alpha_value = std::min(std::max(0.0f, alpha_value_), 255.0f);
    for (int i = 0; i < output_mat.rows; ++i) {
      uchar* out_ptr = output_mat.ptr<uchar>(i);
      for (int j = 0; j < output_mat.cols; ++j) {
        const int out_idx = j * kNumChannelsRGBA;
        out_ptr[out_idx + 3] = alpha_value;  // use value from options
      }
    }
  }

  cc->Outputs()
      .Tag(kOutputFrameTag)
      .Add(output_frame.release(), cc->InputTimestamp());

  return absl::OkStatus();
}

absl::Status SetAlphaCalculator::RenderGpu(CalculatorContext* cc) {
  if (cc->Inputs().Tag(kInputFrameTagGpu).IsEmpty()) {
    return absl::OkStatus();
  }
#if !MEDIAPIPE_DISABLE_GPU
  // Setup source texture.
  const auto& input_frame =
      cc->Inputs().Tag(kInputFrameTagGpu).Get<mediapipe::GpuBuffer>();
  if (!(input_frame.format() == mediapipe::GpuBufferFormat::kBGRA32 ||
        input_frame.format() == mediapipe::GpuBufferFormat::kRGB24)) {
    ABSL_LOG(ERROR) << "Only RGB or RGBA input image supported";
  }
  auto input_texture = gpu_helper_.CreateSourceTexture(input_frame);

  // Setup destination texture.
  const int width = input_frame.width(), height = input_frame.height();
  auto output_texture = gpu_helper_.CreateDestinationTexture(
      width, height, mediapipe::GpuBufferFormat::kBGRA32);

  const bool has_alpha_mask = cc->Inputs().HasTag(kInputAlphaTagGpu) &&
                              !cc->Inputs().Tag(kInputAlphaTagGpu).IsEmpty();

  // Setup alpha texture and Update image in GPU shader.
  if (has_alpha_mask) {
    const auto& alpha_mask =
        cc->Inputs().Tag(kInputAlphaTagGpu).Get<mediapipe::GpuBuffer>();
    auto alpha_texture = gpu_helper_.CreateSourceTexture(alpha_mask);
    gpu_helper_.BindFramebuffer(output_texture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, input_texture.name());
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, alpha_texture.name());
    GlRender(cc);  // use channel 0 of mask
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    alpha_texture.Release();
  } else {
    gpu_helper_.BindFramebuffer(output_texture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, input_texture.name());
    GlRender(cc);  // use value from options
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
  }
  glFlush();

  // Send out image as GPU packet.
  auto output_frame = output_texture.GetFrame<mediapipe::GpuBuffer>();
  cc->Outputs()
      .Tag(kOutputFrameTagGpu)
      .Add(output_frame.release(), cc->InputTimestamp());

  // Cleanup
  input_texture.Release();
  output_texture.Release();
#endif  // !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
}

void SetAlphaCalculator::GlRender(CalculatorContext* cc) {
#if !MEDIAPIPE_DISABLE_GPU
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

#endif  // !MEDIAPIPE_DISABLE_GPU
}

absl::Status SetAlphaCalculator::GlSetup(CalculatorContext* cc) {
#if !MEDIAPIPE_DISABLE_GPU
  const GLint attr_location[NUM_ATTRIBUTES] = {
      ATTRIB_VERTEX,
      ATTRIB_TEXTURE_POSITION,
  };
  const GLchar* attr_name[NUM_ATTRIBUTES] = {
      "position",
      "texture_coordinate",
  };

  // Shader to overlay a texture onto another when overlay is non-zero.
  // TODO split into two shaders to handle alpha_value<0 separately
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
    uniform sampler2D alpha_mask;
    uniform float alpha_value;

    void main() {
      vec3 image_pix = texture2D(input_frame, sample_coordinate).rgb;
      float alpha = alpha_value;
      if (alpha_value < 0.0) alpha = texture2D(alpha_mask, sample_coordinate).r;
      vec4 out_pix = vec4(image_pix, alpha);
      fragColor = out_pix;
    }
  )";

  // Create shader program and set parameters.
  mediapipe::GlhCreateProgram(mediapipe::kBasicVertexShader, frag_src,
                              NUM_ATTRIBUTES, (const GLchar**)&attr_name[0],
                              attr_location, &program_);
  RET_CHECK(program_) << "Problem initializing the program.";
  glUseProgram(program_);
  glUniform1i(glGetUniformLocation(program_, "input_frame"), 1);
  glUniform1i(glGetUniformLocation(program_, "alpha_mask"), 2);
  glUniform1f(glGetUniformLocation(program_, "alpha_value"), alpha_value_);

#endif  // !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
}

}  // namespace mediapipe
