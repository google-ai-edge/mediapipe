// Copyright 2024 The MediaPipe Authors.
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
#include "mediapipe/calculators/tensor/tensor_converter_gl30.h"

#include <optional>

#include "absl/status/statusor.h"
#include "mediapipe/framework/port.h"

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "mediapipe/calculators/tensor/tensor_converter_gpu.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/gpu/gl_base.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/gl_texture_view.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/shader_util.h"

namespace mediapipe {

namespace {

class TensorConverterGlImpl : public TensorConverterGpu {
 public:
  explicit TensorConverterGlImpl(GlCalculatorHelper& gpu_helper,
                                 MemoryManager* memory_manager)
      : gpu_helper_(gpu_helper), memory_manager_(memory_manager) {}

  ~TensorConverterGlImpl() override {
    glDeleteFramebuffers(1, &framebuffer_);
    glDeleteProgram(to_tex2d_program_);
  }

  // OpenGL ES 3.0 fragment shader Texture2d -> Texture2d conversion.
  absl::Status InitTensorConverterProgramGl30(
      bool include_alpha, bool single_channel,
      std::optional<std::pair<float, float>> output_range,
      bool flip_vertically) {
    const std::string shader_source = absl::Substitute(
        R"glsl(
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
          out $0 fragColor;
        #endif  // defined(GL_ES)

          in vec2 sample_coordinate;
          uniform sampler2D frame;

          void main() {
            vec2 coord = $1
            vec4 pixel = texture2D(frame, coord);
            $2  // normalize [-1,1]
            fragColor.r = pixel.r;  // r channel
            $3  // g & b channels
            $4  // alpha channel
          })glsl",
        /*$0=*/single_channel ? "vec1" : "vec4",
        /*$1=*/
        flip_vertically
            ? "vec2(sample_coordinate.x, 1.0 - sample_coordinate.y);"
            : "sample_coordinate;",
        /*$2=*/
        output_range.has_value()
            ? absl::Substitute("pixel = pixel * float($0) + float($1);",
                               (output_range->second - output_range->first),
                               output_range->first)
            : "",
        /*$3=*/single_channel ? "" : R"glsl(fragColor.g = pixel.g;
                                            fragColor.b = pixel.b;)glsl",
        /*$4=*/
        include_alpha ? "fragColor.a = pixel.a;"
                      : (single_channel ? "" : "fragColor.a = 1.0;"));

    const GLint attr_location[NUM_ATTRIBUTES] = {
        ATTRIB_VERTEX,
        ATTRIB_TEXTURE_POSITION,
    };
    const GLchar* attr_name[NUM_ATTRIBUTES] = {
        "position",
        "texture_coordinate",
    };

    // shader program and params
    mediapipe::GlhCreateProgram(
        mediapipe::kBasicVertexShader, shader_source.c_str(), NUM_ATTRIBUTES,
        &attr_name[0], attr_location, &to_tex2d_program_);
    RET_CHECK(to_tex2d_program_) << "Problem initializing the program.";
    glUseProgram(to_tex2d_program_);
    glUniform1i(glGetUniformLocation(to_tex2d_program_, "frame"), 1);
    glGenFramebuffers(1, &framebuffer_);
    return absl::OkStatus();
  }

  absl::Status Init(int input_width, int input_height,
                    std::optional<std::pair<float, float>> output_range,
                    bool include_alpha, bool single_channel,
                    bool flip_vertically, int num_output_channels) {
    width_ = input_width;
    height_ = input_height;
    num_output_channels_ = num_output_channels;
    return InitTensorConverterProgramGl30(include_alpha, single_channel,
                                          output_range, flip_vertically);
  }

  Tensor Convert(const GpuBuffer& input) override {
    const auto input_texture = gpu_helper_.CreateSourceTexture(input);
    Tensor output(Tensor::ElementType::kFloat32,
                  Tensor::Shape{1, height_, width_, num_output_channels_},
                  memory_manager_);
    glUseProgram(to_tex2d_program_);
    glDisable(GL_DEPTH_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);
    glViewport(0, 0, input_texture.width(), input_texture.height());
    glActiveTexture(GL_TEXTURE0);
    auto output_view = output.GetOpenGlTexture2dWriteView();
    glBindTexture(GL_TEXTURE_2D, output_view.name());
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           output_view.name(), 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(input_texture.target(), input_texture.name());
    glVertexAttribPointer(ATTRIB_VERTEX, 2, GL_FLOAT, 0, 0,
                          mediapipe::kBasicSquareVertices);
    glEnableVertexAttribArray(ATTRIB_VERTEX);
    glVertexAttribPointer(ATTRIB_TEXTURE_POSITION, 2, GL_FLOAT, 0, 0,
                          mediapipe::kBasicTextureVertices);
    glEnableVertexAttribArray(ATTRIB_TEXTURE_POSITION);

    // draw
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    // cleanup
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);

    glFlush();
    return output;
  }

 private:
  enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };
  GLuint to_tex2d_program_;
  GLuint framebuffer_;

  int width_ = 0;
  int height_ = 0;
  int num_output_channels_ = 0;

  GlCalculatorHelper& gpu_helper_;
  MemoryManager* memory_manager_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<TensorConverterGpu>> CreateTensorConverterGl30(
    GlCalculatorHelper& gpu_helper, MemoryManager* memory_manager,
    int input_width, int input_height,
    std::optional<std::pair<float, float>> output_range, bool include_alpha,
    bool single_channel, bool flip_vertically, int num_output_channels) {
  auto converter =
      std::make_unique<TensorConverterGlImpl>(gpu_helper, memory_manager);
  MP_RETURN_IF_ERROR(converter->Init(input_width, input_height, output_range,
                                     include_alpha, single_channel,
                                     flip_vertically, num_output_channels));
  return converter;
}

}  // namespace mediapipe
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
