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

#include "mediapipe/calculators/image/affine_transformation_runner_gl.h"

#include <memory>
#include <optional>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "Eigen/LU"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/calculators/image/affine_transformation.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_origin.pb.h"
#include "mediapipe/gpu/shader_util.h"

namespace mediapipe {

namespace {

using mediapipe::GlCalculatorHelper;
using mediapipe::GlhCreateProgram;
using mediapipe::GlTexture;
using mediapipe::GpuBuffer;
using mediapipe::GpuOrigin;

bool IsMatrixVerticalFlipNeeded(GpuOrigin::Mode gpu_origin) {
  switch (gpu_origin) {
    case GpuOrigin::DEFAULT:
    case GpuOrigin::CONVENTIONAL:
#ifdef __APPLE__
      return false;
#else
      return true;
#endif  //  __APPLE__
    case GpuOrigin::TOP_LEFT:
      return false;
  }
}

#ifdef __APPLE__
#define GL_CLAMP_TO_BORDER_MAY_BE_SUPPORTED 0
#else
#define GL_CLAMP_TO_BORDER_MAY_BE_SUPPORTED 1
#endif  //  __APPLE__

bool IsGlClampToBorderSupported(const mediapipe::GlContext& gl_context) {
  return gl_context.gl_major_version() > 3 ||
         (gl_context.gl_major_version() == 3 &&
          gl_context.gl_minor_version() >= 2);
}

constexpr int kAttribVertex = 0;
constexpr int kAttribTexturePosition = 1;
constexpr int kNumAttributes = 2;

class GlTextureWarpAffineRunner
    : public AffineTransformation::Runner<GpuBuffer,
                                          std::unique_ptr<GpuBuffer>> {
 public:
  GlTextureWarpAffineRunner(std::shared_ptr<GlCalculatorHelper> gl_helper,
                            GpuOrigin::Mode gpu_origin)
      : gl_helper_(gl_helper), gpu_origin_(gpu_origin) {}
  absl::Status Init() {
    return gl_helper_->RunInGlContext([this]() -> absl::Status {
      const GLint attr_location[kNumAttributes] = {
          kAttribVertex,
          kAttribTexturePosition,
      };
      const GLchar* attr_name[kNumAttributes] = {
          "position",
          "texture_coordinate",
      };

      constexpr GLchar kVertShader[] = R"(
            in vec4 position;
            in mediump vec4 texture_coordinate;
            out mediump vec2 sample_coordinate;
            uniform mat4 transform_matrix;

            void main() {
              gl_Position = position;
              vec4 tc = transform_matrix * texture_coordinate;
              sample_coordinate = tc.xy;
            }
          )";

      constexpr GLchar kFragShader[] = R"(
            DEFAULT_PRECISION(mediump, float)
            in vec2 sample_coordinate;
            uniform sampler2D input_texture;

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
              fragColor = color;
            }
          )";

      // Create program and set parameters.
      auto create_fn = [&](const std::string& vs,
                           const std::string& fs) -> absl::StatusOr<Program> {
        GLuint program = 0;
        GlhCreateProgram(vs.c_str(), fs.c_str(), kNumAttributes, &attr_name[0],
                         attr_location, &program);

        RET_CHECK(program) << "Problem initializing warp affine program.";
        glUseProgram(program);
        glUniform1i(glGetUniformLocation(program, "input_texture"), 1);
        GLint matrix_id = glGetUniformLocation(program, "transform_matrix");
        return Program{.id = program, .matrix_id = matrix_id};
      };

      const std::string vert_src =
          absl::StrCat(mediapipe::kMediaPipeVertexShaderPreamble, kVertShader);

      const std::string frag_src = absl::StrCat(
          mediapipe::kMediaPipeFragmentShaderPreamble, kFragShader);

      ASSIGN_OR_RETURN(program_, create_fn(vert_src, frag_src));

      auto create_custom_zero_fn = [&]() -> absl::StatusOr<Program> {
        std::string custom_zero_border_mode_def = R"(
          #define CUSTOM_ZERO_BORDER_MODE
        )";
        const std::string frag_custom_zero_src =
            absl::StrCat(mediapipe::kMediaPipeFragmentShaderPreamble,
                         custom_zero_border_mode_def, kFragShader);
        return create_fn(vert_src, frag_custom_zero_src);
      };
#if GL_CLAMP_TO_BORDER_MAY_BE_SUPPORTED
      if (!IsGlClampToBorderSupported(gl_helper_->GetGlContext())) {
        ASSIGN_OR_RETURN(program_custom_zero_, create_custom_zero_fn());
      }
#else
      ASSIGN_OR_RETURN(program_custom_zero_, create_custom_zero_fn());
#endif  // GL_CLAMP_TO_BORDER_MAY_BE_SUPPORTED

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

  absl::StatusOr<std::unique_ptr<GpuBuffer>> Run(
      const GpuBuffer& input, const std::array<float, 16>& matrix,
      const AffineTransformation::Size& size,
      AffineTransformation::BorderMode border_mode) override {
    std::unique_ptr<GpuBuffer> gpu_buffer;
    MP_RETURN_IF_ERROR(
        gl_helper_->RunInGlContext([this, &input, &matrix, &size, &border_mode,
                                    &gpu_buffer]() -> absl::Status {
          auto input_texture = gl_helper_->CreateSourceTexture(input);
          auto output_texture = gl_helper_->CreateDestinationTexture(
              size.width, size.height, input.format());

          MP_RETURN_IF_ERROR(
              RunInternal(input_texture, matrix, border_mode, &output_texture));
          gpu_buffer = output_texture.GetFrame<GpuBuffer>();
          return absl::OkStatus();
        }));

    return gpu_buffer;
  }

  absl::Status RunInternal(const GlTexture& texture,
                           const std::array<float, 16>& matrix,
                           AffineTransformation::BorderMode border_mode,
                           GlTexture* output) {
    glDisable(GL_DEPTH_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);
    glViewport(0, 0, output->width(), output->height());

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
    std::optional<Program> program = program_;
    switch (border_mode) {
      case AffineTransformation::BorderMode::kReplicate: {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        break;
      }
      case AffineTransformation::BorderMode::kZero: {
#if GL_CLAMP_TO_BORDER_MAY_BE_SUPPORTED
        if (program_custom_zero_) {
          program = program_custom_zero_;
        } else {
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
          glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR,
                           std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f}.data());
        }
#else
        RET_CHECK(program_custom_zero_)
            << "Program must have been initialized.";
        program = program_custom_zero_;
#endif  // GL_CLAMP_TO_BORDER_MAY_BE_SUPPORTED
        break;
      }
    }
    glUseProgram(program->id);

    Eigen::Matrix<float, 4, 4, Eigen::RowMajor> eigen_mat(matrix.data());
    if (IsMatrixVerticalFlipNeeded(gpu_origin_)) {
      // @matrix describes affine transformation in terms of TOP LEFT origin, so
      // in some cases/on some platforms an extra flipping should be done before
      // and after.
      const Eigen::Matrix<float, 4, 4, Eigen::RowMajor> flip_y(
          {{1.0f, 0.0f, 0.0f, 0.0f},
           {0.0f, -1.0f, 0.0f, 1.0f},
           {0.0f, 0.0f, 1.0f, 0.0f},
           {0.0f, 0.0f, 0.0f, 1.0f}});
      eigen_mat = flip_y * eigen_mat * flip_y;
    }

    // If GL context is ES2, then GL_FALSE must be used for 'transpose'
    // GLboolean in glUniformMatrix4fv, or else INVALID_VALUE error is reported.
    // Hence, transposing the matrix and always passing transposed.
    eigen_mat.transposeInPlace();
    glUniformMatrix4fv(program->matrix_id, 1, GL_FALSE, eigen_mat.data());

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

    return absl::OkStatus();
  }

  ~GlTextureWarpAffineRunner() override {
    gl_helper_->RunInGlContext([this]() {
      // Release OpenGL resources.
      if (framebuffer_ != 0) glDeleteFramebuffers(1, &framebuffer_);
      if (program_.id != 0) glDeleteProgram(program_.id);
      if (program_custom_zero_ && program_custom_zero_->id != 0) {
        glDeleteProgram(program_custom_zero_->id);
      }
      if (vao_ != 0) glDeleteVertexArrays(1, &vao_);
      glDeleteBuffers(2, vbo_);
    });
  }

 private:
  struct Program {
    GLuint id;
    GLint matrix_id;
  };
  std::shared_ptr<GlCalculatorHelper> gl_helper_;
  GpuOrigin::Mode gpu_origin_;
  GLuint vao_ = 0;
  GLuint vbo_[2] = {0, 0};
  Program program_;
  std::optional<Program> program_custom_zero_;
  GLuint framebuffer_ = 0;
};

#undef GL_CLAMP_TO_BORDER_MAY_BE_SUPPORTED

}  // namespace

absl::StatusOr<std::unique_ptr<
    AffineTransformation::Runner<GpuBuffer, std::unique_ptr<GpuBuffer>>>>
CreateAffineTransformationGlRunner(
    std::shared_ptr<GlCalculatorHelper> gl_helper, GpuOrigin::Mode gpu_origin) {
  auto runner =
      absl::make_unique<GlTextureWarpAffineRunner>(gl_helper, gpu_origin);
  MP_RETURN_IF_ERROR(runner->Init());
  return runner;
}

}  // namespace mediapipe
