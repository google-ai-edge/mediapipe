// Copyright 2020 The MediaPipe Authors.
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

#include "mediapipe/calculators/tensor/image_to_tensor_converter_gl_texture.h"

#include "mediapipe/framework/port.h"

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30

#include <array>
#include <memory>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/calculators/tensor/image_to_tensor_converter.h"
#include "mediapipe/calculators/tensor/image_to_tensor_converter_gl_utils.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/shader_util.h"

namespace mediapipe {

namespace {

constexpr int kAttribVertex = 0;
constexpr int kAttribTexturePosition = 1;
constexpr int kNumAttributes = 2;

class ImageToTensorGlTextureConverter : public ImageToTensorConverter {
 public:
  absl::Status Init(CalculatorContext* cc, bool input_starts_at_bottom,
                    BorderMode border_mode) {
    MP_RETURN_IF_ERROR(gl_helper_.Open(cc));
    return gl_helper_.RunInGlContext([this, input_starts_at_bottom,
                                      border_mode]() -> absl::Status {
      use_custom_zero_border_ =
          border_mode == BorderMode::kZero &&
          !IsGlClampToBorderSupported(gl_helper_.GetGlContext());
      border_mode_ = border_mode;

      const GLint attr_location[kNumAttributes] = {
          kAttribVertex,
          kAttribTexturePosition,
      };
      const GLchar* attr_name[kNumAttributes] = {
          "position",
          "texture_coordinate",
      };

      constexpr GLchar kExtractSubRectVertexShader[] = R"(
            in vec4 position;
            in highp vec4 texture_coordinate;
            out highp vec2 sample_coordinate;
            uniform mat4 transform_matrix;

            void main() {
              gl_Position = position;
              // Apply transformation from roi coordinates to original image coordinates.
              vec4 tc = transform_matrix * texture_coordinate;
          #ifdef INPUT_STARTS_AT_BOTTOM
              // Opengl texture sampler has origin in lower left corner,
              // so we invert y coordinate.
              tc.y = 1.0 - tc.y;
          #endif  // defined(INPUT_STARTS_AT_BOTTOM)
              sample_coordinate = tc.xy;
            }
          )";

      constexpr GLchar kExtractSubRectFragBody[] = R"(
            DEFAULT_PRECISION(highp, float)

            // Provided by kExtractSubRectVertexShader.
            in vec2 sample_coordinate;

            uniform sampler2D input_texture;
            uniform float alpha;
            uniform float beta;

            #ifdef GL_ES
              #define fragColor gl_FragColor
            #else
              out vec4 fragColor;
            #endif  // defined(GL_ES);

            void main() {
              vec4 color = texture2D(input_texture, sample_coordinate);
            #ifdef CUSTOM_ZERO_BORDER_MODE
              float out_of_bounds =
                  float(sample_coordinate.x < 0.0 || sample_coordinate.x > 1.0 ||
                        sample_coordinate.y < 0.0 || sample_coordinate.y > 1.0);
              color = mix(color, vec4(0.0, 0.0, 0.0, 0.0), out_of_bounds);
            #endif  // defined(CUSTOM_ZERO_BORDER_MODE)
              fragColor = alpha * color + beta;
            }
          )";

      std::string starts_at_bottom_def;
      if (input_starts_at_bottom) {
        starts_at_bottom_def = R"(
          #define INPUT_STARTS_AT_BOTTOM
        )";
      }

      // Create program and set parameters.
      const std::string extract_sub_rect_vertex_src =
          absl::StrCat(mediapipe::kMediaPipeVertexShaderPreamble,
                       starts_at_bottom_def, kExtractSubRectVertexShader);

      std::string custom_zero_border_mode_def;
      if (use_custom_zero_border_) {
        custom_zero_border_mode_def = R"(
          #define CUSTOM_ZERO_BORDER_MODE
        )";
      }
      const std::string extract_sub_rect_frag_src =
          absl::StrCat(mediapipe::kMediaPipeFragmentShaderPreamble,
                       custom_zero_border_mode_def, kExtractSubRectFragBody);
      mediapipe::GlhCreateProgram(extract_sub_rect_vertex_src.c_str(),
                                  extract_sub_rect_frag_src.c_str(),
                                  kNumAttributes, &attr_name[0], attr_location,
                                  &program_);

      RET_CHECK(program_) << "Problem initializing image to tensor program.";
      glUseProgram(program_);
      glUniform1i(glGetUniformLocation(program_, "input_texture"), 1);
      alpha_id_ = glGetUniformLocation(program_, "alpha");
      beta_id_ = glGetUniformLocation(program_, "beta");
      matrix_id_ = glGetUniformLocation(program_, "transform_matrix");

      glGenFramebuffers(1, &framebuffer_);

      // vertex storage
      glGenBuffers(2, vbo_);
      glGenVertexArrays(1, &vao_);

      // vbo 0
      glBindBuffer(GL_ARRAY_BUFFER, vbo_[0]);
      glBufferData(GL_ARRAY_BUFFER, sizeof(mediapipe::kBasicSquareVertices),
                   mediapipe::kBasicSquareVertices, GL_STATIC_DRAW);

      // vbo 1
      glBindBuffer(GL_ARRAY_BUFFER, vbo_[1]);
      glBufferData(GL_ARRAY_BUFFER, sizeof(mediapipe::kBasicTextureVertices),
                   mediapipe::kBasicTextureVertices, GL_STATIC_DRAW);

      glBindBuffer(GL_ARRAY_BUFFER, 0);

      return absl::OkStatus();
    });
  }

  absl::Status Convert(const mediapipe::Image& input, const RotatedRect& roi,
                       float range_min, float range_max,
                       int tensor_buffer_offset,
                       Tensor& output_tensor) override {
    if (input.format() != mediapipe::GpuBufferFormat::kBGRA32 &&
        input.format() != mediapipe::GpuBufferFormat::kRGBAHalf64 &&
        input.format() != mediapipe::GpuBufferFormat::kRGBAFloat128 &&
        input.format() != mediapipe::GpuBufferFormat::kRGB24) {
      return InvalidArgumentError(absl::StrCat(
          "Unsupported format: ", static_cast<uint32_t>(input.format())));
    }
    // TODO: support tensor_buffer_offset > 0 scenario.
    RET_CHECK_EQ(tensor_buffer_offset, 0)
        << "The non-zero tensor_buffer_offset input is not supported yet.";
    const auto& output_shape = output_tensor.shape();
    MP_RETURN_IF_ERROR(ValidateTensorShape(output_shape));

    MP_RETURN_IF_ERROR(gl_helper_.RunInGlContext(
        [this, &output_tensor, &input, &roi, &output_shape, range_min,
         range_max]() -> absl::Status {
          auto input_texture = gl_helper_.CreateSourceTexture(input);

          constexpr float kInputImageRangeMin = 0.0f;
          constexpr float kInputImageRangeMax = 1.0f;
          MP_ASSIGN_OR_RETURN(auto transform,
                              GetValueRangeTransformation(
                                  kInputImageRangeMin, kInputImageRangeMax,
                                  range_min, range_max));
          auto tensor_view = output_tensor.GetOpenGlTexture2dWriteView();
          MP_RETURN_IF_ERROR(ExtractSubRect(input_texture, roi,
                                            /*flip_horizontally=*/false,
                                            transform.scale, transform.offset,
                                            output_shape, &tensor_view));
          return absl::OkStatus();
        }));

    return absl::OkStatus();
  }

  absl::Status ExtractSubRect(const mediapipe::GlTexture& texture,
                              const RotatedRect& sub_rect,
                              bool flip_horizontally, float alpha, float beta,
                              const Tensor::Shape& output_shape,
                              Tensor::OpenGlTexture2dView* output) {
    const int output_height = output_shape.dims[1];
    const int output_width = output_shape.dims[2];
    std::array<float, 16> transform_mat;

    glDisable(GL_DEPTH_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);
    glViewport(0, 0, output_width, output_height);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, output->name());
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           output->name(), 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(texture.target(), texture.name());

    // a) Filtering.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // b) Clamping.
    switch (border_mode_) {
      case BorderMode::kReplicate: {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        break;
      }
      case BorderMode::kZero: {
        if (!use_custom_zero_border_) {
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
          glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR,
                           std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f}.data());
        }
        break;
      }
    }

    glUseProgram(program_);
    glUniform1f(alpha_id_, alpha);
    glUniform1f(beta_id_, beta);

    // If our context is ES2, then we must use GL_FALSE for our 'transpose'
    // GLboolean in glUniformMatrix4fv, or else we'll get an INVALID_VALUE
    // error. So in that case, we'll grab the transpose of our original matrix
    // and send that instead.
    const auto gl_context = mediapipe::GlContext::GetCurrent();
    ABSL_LOG_IF(FATAL, !gl_context) << "GlContext is not bound to the thread.";
    if (gl_context->GetGlVersion() == mediapipe::GlVersion::kGLES2) {
      GetTransposedRotatedSubRectToRectTransformMatrix(
          sub_rect, texture.width(), texture.height(), flip_horizontally,
          &transform_mat);
      glUniformMatrix4fv(matrix_id_, 1, GL_FALSE, transform_mat.data());
    } else {
      GetRotatedSubRectToRectTransformMatrix(sub_rect, texture.width(),
                                             texture.height(),
                                             flip_horizontally, &transform_mat);
      glUniformMatrix4fv(matrix_id_, 1, GL_TRUE, transform_mat.data());
    }

    // vao
    glBindVertexArray(vao_);

    // vbo 0
    glBindBuffer(GL_ARRAY_BUFFER, vbo_[0]);
    glEnableVertexAttribArray(kAttribVertex);
    glVertexAttribPointer(kAttribVertex, 2, GL_FLOAT, 0, 0, nullptr);

    // vbo 1
    glBindBuffer(GL_ARRAY_BUFFER, vbo_[1]);
    glEnableVertexAttribArray(kAttribTexturePosition);
    glVertexAttribPointer(kAttribTexturePosition, 2, GL_FLOAT, 0, 0, nullptr);

    // draw
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    // Resetting to MediaPipe texture param defaults.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glDisableVertexAttribArray(kAttribVertex);
    glDisableVertexAttribArray(kAttribTexturePosition);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glFlush();

    return absl::OkStatus();
  }

  ~ImageToTensorGlTextureConverter() override {
    gl_helper_.RunInGlContext([this]() {
      // Release OpenGL resources.
      if (framebuffer_ != 0) glDeleteFramebuffers(1, &framebuffer_);
      if (program_ != 0) glDeleteProgram(program_);
      if (vao_ != 0) glDeleteVertexArrays(1, &vao_);
      glDeleteBuffers(2, vbo_);
    });
  }

 private:
  absl::Status ValidateTensorShape(const Tensor::Shape& output_shape) {
    RET_CHECK_EQ(output_shape.dims.size(), 4)
        << "Wrong output dims size: " << output_shape.dims.size();
    RET_CHECK_EQ(output_shape.dims[0], 1)
        << "Handling batch dimension not equal to 1 is not implemented in this "
           "converter.";
    RET_CHECK_EQ(output_shape.dims[3], 3)
        << "Wrong output channel: " << output_shape.dims[3];
    return absl::OkStatus();
  }

  mediapipe::GlCalculatorHelper gl_helper_;
  bool use_custom_zero_border_ = false;
  BorderMode border_mode_ = BorderMode::kReplicate;
  GLuint vao_ = 0;
  GLuint vbo_[2] = {0, 0};
  GLuint program_ = 0;
  GLuint framebuffer_ = 0;
  GLint alpha_id_ = 0;
  GLint beta_id_ = 0;
  GLint matrix_id_ = 0;
};

}  // namespace

absl::StatusOr<std::unique_ptr<ImageToTensorConverter>>
CreateImageToGlTextureTensorConverter(CalculatorContext* cc,
                                      bool input_starts_at_bottom,
                                      BorderMode border_mode) {
  auto result = std::make_unique<ImageToTensorGlTextureConverter>();
  MP_RETURN_IF_ERROR(result->Init(cc, input_starts_at_bottom, border_mode));
  return result;
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
