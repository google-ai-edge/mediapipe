// Copyright 2021 The MediaPipe Authors.
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

#include <algorithm>
#include <memory>

#include "absl/log/absl_log.h"
#include "mediapipe/calculators/image/segmentation_smoothing_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/vector.h"

#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/shader_util.h"
#endif  // !MEDIAPIPE_DISABLE_GPU

#if !MEDIAPIPE_DISABLE_OPENCV
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/image_opencv.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#endif  // !MEDIAPIPE_DISABLE_OPENCV

namespace mediapipe {

namespace {
constexpr char kCurrentMaskTag[] = "MASK";
constexpr char kPreviousMaskTag[] = "MASK_PREVIOUS";
constexpr char kOutputMaskTag[] = "MASK_SMOOTHED";

enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };
}  // namespace

// A calculator for mixing two segmentation masks together,
// based on an uncertantity probability estimate.
//
// Inputs:
//   MASK - Image containing the new/current mask.
//          [ImageFormat::VEC32F1, or
//           GpuBufferFormat::kBGRA32/kRGB24/kGrayHalf16/kGrayFloat32]
//   MASK_PREVIOUS - Image containing previous mask.
//                   [Same format as MASK_CURRENT]
//   * If input channels is >1, only the first channel (R) is used as the mask.
//
// Output:
//   MASK_SMOOTHED - Blended mask.
//                   [Same format as MASK_CURRENT]
//   * The resulting filtered mask will be stored in R channel,
//     and duplicated in A if 4 channels.
//
// Options:
//   combine_with_previous_ratio - Amount of previous to blend with current.
//
// Example:
//  node {
//    calculator: "SegmentationSmoothingCalculator"
//    input_stream: "MASK:mask"
//    input_stream: "MASK_PREVIOUS:mask_previous"
//    output_stream: "MASK_SMOOTHED:mask_smoothed"
//    options: {
//      [mediapipe.SegmentationSmoothingCalculatorOptions.ext] {
//        combine_with_previous_ratio: 0.9
//      }
//    }
//  }
//
class SegmentationSmoothingCalculator : public CalculatorBase {
 public:
  SegmentationSmoothingCalculator() = default;

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

  float combine_with_previous_ratio_;

  bool gpu_initialized_ = false;
#if !MEDIAPIPE_DISABLE_GPU
  mediapipe::GlCalculatorHelper gpu_helper_;
  GLuint program_ = 0;
#endif  // !MEDIAPIPE_DISABLE_GPU
};
REGISTER_CALCULATOR(SegmentationSmoothingCalculator);

absl::Status SegmentationSmoothingCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK_GE(cc->Inputs().NumEntries(), 1);

  cc->Inputs().Tag(kCurrentMaskTag).Set<Image>();
  cc->Inputs().Tag(kPreviousMaskTag).Set<Image>();
  cc->Outputs().Tag(kOutputMaskTag).Set<Image>();

#if !MEDIAPIPE_DISABLE_GPU
  MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(
      cc, /*request_gpu_as_optional=*/true));
#endif  // !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
}

absl::Status SegmentationSmoothingCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  auto options =
      cc->Options<mediapipe::SegmentationSmoothingCalculatorOptions>();
  combine_with_previous_ratio_ = options.combine_with_previous_ratio();

  return absl::OkStatus();
}

absl::Status SegmentationSmoothingCalculator::Process(CalculatorContext* cc) {
  if (cc->Inputs().Tag(kCurrentMaskTag).IsEmpty()) {
    return absl::OkStatus();
  }
  if (cc->Inputs().Tag(kPreviousMaskTag).IsEmpty()) {
    // Pass through current image if previous is not available.
    cc->Outputs()
        .Tag(kOutputMaskTag)
        .AddPacket(cc->Inputs().Tag(kCurrentMaskTag).Value());
    return absl::OkStatus();
  }

  // Run on GPU if incoming data is on GPU.
  const bool use_gpu = cc->Inputs().Tag(kCurrentMaskTag).Get<Image>().UsesGpu();

  if (use_gpu) {
#if !MEDIAPIPE_DISABLE_GPU
    if (!gpu_initialized_) {
      MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
    }
    MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([this, cc]() -> absl::Status {
      if (!gpu_initialized_) {
        MP_RETURN_IF_ERROR(GlSetup(cc));
        gpu_initialized_ = true;
      }
      MP_RETURN_IF_ERROR(RenderGpu(cc));
      return absl::OkStatus();
    }));
#else
    return absl::InternalError("GPU processing is disabled.");
#endif  // !MEDIAPIPE_DISABLE_GPU
  } else {
#if !MEDIAPIPE_DISABLE_OPENCV
    MP_RETURN_IF_ERROR(RenderCpu(cc));
#else
    return absl::InternalError("OpenCV processing is disabled.");
#endif  // !MEDIAPIPE_DISABLE_OPENCV
  }

  return absl::OkStatus();
}

absl::Status SegmentationSmoothingCalculator::Close(CalculatorContext* cc) {
#if !MEDIAPIPE_DISABLE_GPU
  if (gpu_initialized_) {
    gpu_helper_.RunInGlContext([this] {
      if (program_) glDeleteProgram(program_);
      program_ = 0;
    });
  }
#endif  // !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
}

absl::Status SegmentationSmoothingCalculator::RenderCpu(CalculatorContext* cc) {
#if !MEDIAPIPE_DISABLE_OPENCV
  // Setup source images.
  const auto& current_frame = cc->Inputs().Tag(kCurrentMaskTag).Get<Image>();
  auto current_mat = mediapipe::formats::MatView(&current_frame);
  RET_CHECK_EQ(current_mat->type(), CV_32FC1)
      << "Only 1-channel float input image is supported.";

  const auto& previous_frame = cc->Inputs().Tag(kPreviousMaskTag).Get<Image>();
  auto previous_mat = mediapipe::formats::MatView(&previous_frame);
  RET_CHECK_EQ(previous_mat->type(), current_mat->type())
      << "Warning: mixing input format types: " << previous_mat->type()
      << " != " << previous_mat->type();

  RET_CHECK_EQ(current_mat->rows, previous_mat->rows);
  RET_CHECK_EQ(current_mat->cols, previous_mat->cols);

  // Setup destination image.
  auto output_frame = std::make_shared<ImageFrame>(
      current_frame.image_format(), current_mat->cols, current_mat->rows);
  cv::Mat output_mat = mediapipe::formats::MatView(output_frame.get());
  output_mat.setTo(cv::Scalar(0));

  // Blending function.
  const auto blending_fn = [&](const float prev_mask_value,
                               const float new_mask_value) {
    /*
     * Assume p := new_mask_value
     * H(p) := 1 + (p * log(p) + (1-p) * log(1-p)) / log(2)
     * uncertainty alpha(p) =
     *   Clamp(1 - (1 - H(p)) * (1 - H(p)), 0, 1) [squaring the uncertainty]
     *
     * The following polynomial approximates uncertainty alpha as a function
     * of (p + 0.5):
     */
    const float c1 = 5.68842;
    const float c2 = -0.748699;
    const float c3 = -57.8051;
    const float c4 = 291.309;
    const float c5 = -624.717;
    const float t = new_mask_value - 0.5f;
    const float x = t * t;

    const float uncertainty =
        1.0f -
        std::min(1.0f, x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * c5)))));

    return new_mask_value + (prev_mask_value - new_mask_value) *
                                (uncertainty * combine_with_previous_ratio_);
  };

  // Write directly to the first channel of output.
  for (int i = 0; i < output_mat.rows; ++i) {
    float* out_ptr = output_mat.ptr<float>(i);
    const float* curr_ptr = current_mat->ptr<float>(i);
    const float* prev_ptr = previous_mat->ptr<float>(i);
    for (int j = 0; j < output_mat.cols; ++j) {
      const float new_mask_value = curr_ptr[j];
      const float prev_mask_value = prev_ptr[j];
      out_ptr[j] = blending_fn(prev_mask_value, new_mask_value);
    }
  }

  cc->Outputs()
      .Tag(kOutputMaskTag)
      .AddPacket(MakePacket<Image>(output_frame).At(cc->InputTimestamp()));
#endif  // !MEDIAPIPE_DISABLE_OPENCV

  return absl::OkStatus();
}

absl::Status SegmentationSmoothingCalculator::RenderGpu(CalculatorContext* cc) {
#if !MEDIAPIPE_DISABLE_GPU
  // Setup source textures.
  const auto& current_frame = cc->Inputs().Tag(kCurrentMaskTag).Get<Image>();
  RET_CHECK(
      (current_frame.format() == mediapipe::GpuBufferFormat::kBGRA32 ||
       current_frame.format() == mediapipe::GpuBufferFormat::kGrayHalf16 ||
       current_frame.format() == mediapipe::GpuBufferFormat::kGrayFloat32 ||
       current_frame.format() == mediapipe::GpuBufferFormat::kRGB24))
      << "Only RGBA, RGB, or 1-channel Float input image supported.";

  auto current_texture = gpu_helper_.CreateSourceTexture(current_frame);

  const auto& previous_frame = cc->Inputs().Tag(kPreviousMaskTag).Get<Image>();
  if (previous_frame.format() != current_frame.format()) {
    ABSL_LOG(ERROR) << "Warning: mixing input format types. ";
  }
  auto previous_texture = gpu_helper_.CreateSourceTexture(previous_frame);

  // Setup destination texture.
  const int width = current_frame.width(), height = current_frame.height();
  auto output_texture = gpu_helper_.CreateDestinationTexture(
      width, height, current_frame.format());

  // Process shader.
  {
    gpu_helper_.BindFramebuffer(output_texture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, current_texture.name());
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, previous_texture.name());
    GlRender(cc);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
  }
  glFlush();

  // Send out image as GPU packet.
  auto output_frame = output_texture.GetFrame<Image>();
  cc->Outputs()
      .Tag(kOutputMaskTag)
      .Add(output_frame.release(), cc->InputTimestamp());
#endif  // !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
}

void SegmentationSmoothingCalculator::GlRender(CalculatorContext* cc) {
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

absl::Status SegmentationSmoothingCalculator::GlSetup(CalculatorContext* cc) {
#if !MEDIAPIPE_DISABLE_GPU
  const GLint attr_location[NUM_ATTRIBUTES] = {
      ATTRIB_VERTEX,
      ATTRIB_TEXTURE_POSITION,
  };
  const GLchar* attr_name[NUM_ATTRIBUTES] = {
      "position",
      "texture_coordinate",
  };

  // Shader to blend in previous mask based on computed uncertainty probability.
  const std::string frag_src =
      absl::StrCat(std::string(mediapipe::kMediaPipeFragmentShaderPreamble),
                   R"(
    DEFAULT_PRECISION(mediump, float)

    #ifdef GL_ES
    #define fragColor gl_FragColor
    #else
    out vec4 fragColor;
    #endif  // defined(GL_ES);

    in vec2 sample_coordinate;
    uniform sampler2D current_mask;
    uniform sampler2D previous_mask;
    uniform float combine_with_previous_ratio;

    void main() {
      vec4 current_pix = texture2D(current_mask, sample_coordinate);
      vec4 previous_pix = texture2D(previous_mask, sample_coordinate);
      float new_mask_value = current_pix.r;
      float prev_mask_value = previous_pix.r;

      // Assume p := new_mask_value
      // H(p) := 1 + (p * log(p) + (1-p) * log(1-p)) / log(2)
      // uncertainty alpha(p) =
      //   Clamp(1 - (1 - H(p)) * (1 - H(p)), 0, 1) [squaring the uncertainty]
      //
      // The following polynomial approximates uncertainty alpha as a function
      // of (p + 0.5):
      const float c1 = 5.68842;
      const float c2 = -0.748699;
      const float c3 = -57.8051;
      const float c4 = 291.309;
      const float c5 = -624.717;
      float t = new_mask_value - 0.5;
      float x = t * t;

      float uncertainty =
        1.0 - min(1.0, x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * c5)))));

      new_mask_value +=
        (prev_mask_value - new_mask_value) * (uncertainty * combine_with_previous_ratio);

      fragColor = vec4(new_mask_value, 0.0, 0.0, new_mask_value);
    }
  )");

  // Create shader program and set parameters.
  mediapipe::GlhCreateProgram(mediapipe::kBasicVertexShader, frag_src.c_str(),
                              NUM_ATTRIBUTES, (const GLchar**)&attr_name[0],
                              attr_location, &program_);
  RET_CHECK(program_) << "Problem initializing the program.";
  glUseProgram(program_);
  glUniform1i(glGetUniformLocation(program_, "current_mask"), 1);
  glUniform1i(glGetUniformLocation(program_, "previous_mask"), 2);
  glUniform1f(glGetUniformLocation(program_, "combine_with_previous_ratio"),
              combine_with_previous_ratio_);

#endif  // !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
}

}  // namespace mediapipe
