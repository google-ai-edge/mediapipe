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
#include "mediapipe/calculators/tensor/tensor_converter_gl31.h"

#include "mediapipe/framework/port.h"

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
#include <memory>
#include <string>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "mediapipe/calculators/tensor/tensor_converter_gpu.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/gpu/gl_base.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"

namespace mediapipe {

namespace {

constexpr int kWorkgroupSize = 8;  // Block size for GPU shader.
// Commonly used to compute the number of blocks to launch in a kernel.
int NumGroups(const int size, const int group_size) {  // NOLINT
  return (size + group_size - 1) / group_size;
}

class TensorConverterGlImpl : public TensorConverterGpu {
 public:
  explicit TensorConverterGlImpl(GlCalculatorHelper& gpu_helper,
                                 MemoryManager* memory_manager)
      : gpu_helper_(gpu_helper), memory_manager_(memory_manager) {}

  ~TensorConverterGlImpl() override { glDeleteProgram(to_buffer_program_); }

  absl::Status InitTensorConverterProgramGl31(
      bool include_alpha, bool single_channel,
      std::optional<std::pair<float, float>> output_range,
      bool flip_vertically) {
    // Shader to convert GL Texture to Shader Storage Buffer Object
    // (SSBO), with normalization to either: [0,1] or [-1,1].
    const std::string shader_source = absl::Substitute(
        R"glsl( #version 310 es
          layout(local_size_x = $0, local_size_y = $0) in;
          layout(binding = 0) uniform sampler2D input_texture;
          layout(std430, binding = 1) buffer Output {float elements[];} output_data;
          ivec2 width_height = ivec2($1, $2);
          void main() {
            ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
            if (gid.x >= width_height.x || gid.y >= width_height.y) return;
            vec4 pixel = texelFetch(input_texture, gid, 0);
            $3  // normalize [-1,1]
            int linear_index = $7 * ($4 * width_height.x + gid.x);
            output_data.elements[linear_index + 0] = pixel.x;  // r channel
            $5  // g & b channels
            $6  // alpha channel
          })glsl",
        /*$0=*/kWorkgroupSize,
        /*$1=*/width_,
        /*$2=*/height_,
        /*$3=*/
        output_range.has_value()
            ? absl::Substitute("pixel = pixel * float($0) + float($1);",
                               (output_range->second - output_range->first),
                               output_range->first)
            : "",
        /*$4=*/flip_vertically ? "(width_height.y - 1 - gid.y)" : "gid.y",
        /*$5=*/
        single_channel
            ? ""
            : R"glsl(output_data.elements[linear_index + 1] = pixel.y;
                     output_data.elements[linear_index + 2] = pixel.z;)glsl",
        /*$6=*/
        include_alpha ? "output_data.elements[linear_index + 3] = pixel.w;"
                      : "",
        /*$7=*/num_output_channels_);
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    const GLchar* sources[] = {shader_source.c_str()};
    glShaderSource(shader, 1, sources, nullptr);
    glCompileShader(shader);
    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    RET_CHECK(compiled == GL_TRUE);
    to_buffer_program_ = glCreateProgram();
    glAttachShader(to_buffer_program_, shader);
    glDeleteShader(shader);
    glLinkProgram(to_buffer_program_);
    return absl::OkStatus();
  }

  absl::Status Init(int input_width, int input_height,
                    std::optional<std::pair<float, float>> output_range,
                    bool include_alpha, bool single_channel,
                    bool flip_vertically, int num_output_channels) {
    width_ = input_width;
    height_ = input_height;
    num_output_channels_ = num_output_channels;
    return InitTensorConverterProgramGl31(include_alpha, single_channel,
                                          output_range, flip_vertically);
  }

  Tensor Convert(const GpuBuffer& input) override {
    const auto input_texture = gpu_helper_.CreateSourceTexture(input);
    Tensor output(Tensor::ElementType::kFloat32,
                  Tensor::Shape{1, height_, width_, num_output_channels_},
                  memory_manager_);
    // Convert GL texture into SSBO.
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, input_texture.name());
    auto output_view = output.GetOpenGlBufferWriteView();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, output_view.name());
    glUseProgram(to_buffer_program_);
    glDispatchCompute(NumGroups(input_texture.width(), kWorkgroupSize),
                      NumGroups(input_texture.height(), kWorkgroupSize), 1);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glFlush();
    return output;
  }

 private:
  GLuint to_buffer_program_;

  int width_ = 0;
  int height_ = 0;
  int num_output_channels_ = 0;

  GlCalculatorHelper& gpu_helper_;
  MemoryManager* memory_manager_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<TensorConverterGpu>> CreateTensorConverterGl31(
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
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
