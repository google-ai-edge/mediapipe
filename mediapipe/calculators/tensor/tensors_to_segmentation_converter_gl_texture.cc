// Copyright 2023 The MediaPipe Authors.
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

#include "mediapipe/calculators/tensor/tensors_to_segmentation_converter_gl_texture.h"

#if !MEDIAPIPE_DISABLE_GPU

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/calculators/tensor/tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_segmentation_converter.h"
#include "mediapipe/calculators/tensor/tensors_to_segmentation_utils.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/gpu/gl_base.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/gpu_buffer_format.h"
#include "mediapipe/gpu/gpu_origin.pb.h"
#include "mediapipe/gpu/gpu_origin_utils.h"
#include "mediapipe/gpu/shader_util.h"

namespace mediapipe {
namespace {

enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };

using ::mediapipe::tensors_to_segmentation_utils::GetHwcFromDims;
using ::mediapipe::tensors_to_segmentation_utils::GlRender;

class TensorsToSegmentationGlTextureConverter
    : public TensorsToSegmentationConverter {
 public:
  ~TensorsToSegmentationGlTextureConverter() override;
  absl::Status Init(CalculatorContext* cc,
                    const TensorsToSegmentationCalculatorOptions& options);
  absl::StatusOr<std::unique_ptr<Image>> Convert(const Tensor& input_tensor,
                                                 int output_width,
                                                 int output_height) override;

 private:
  mediapipe::GlCalculatorHelper gpu_helper_;
  // TODO: Refactor upsample program out of the conversion.
  GLuint upsample_program_;
  bool gpu_initialized_ = false;
  GLuint mask_program_20_;
};

TensorsToSegmentationGlTextureConverter::
    ~TensorsToSegmentationGlTextureConverter() {
  if (gpu_initialized_) {
    gpu_helper_.RunInGlContext([this] {
      if (upsample_program_) glDeleteProgram(upsample_program_);
      upsample_program_ = 0;
      if (mask_program_20_) glDeleteProgram(mask_program_20_);
      mask_program_20_ = 0;
    });
  }
}

absl::Status TensorsToSegmentationGlTextureConverter::Init(
    CalculatorContext* cc,
    const TensorsToSegmentationCalculatorOptions& options) {
  MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
  MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([this,
                                                 &options]() -> absl::Status {
    // A shader to process a segmentation tensor into an output mask.
    // Currently uses 4 channels for output, and sets R+A channels as mask
    // value.
    const std::string shader_header = absl::StrCat(
        std::string(mediapipe::kMediaPipeFragmentShaderPreamble), R"(
DEFAULT_PRECISION(mediump, float)
)");
    /* Shader defines will be inserted here. */

    const std::string shader_src_main = R"(
in vec2 sample_coordinate;

uniform sampler2D input_texture;

#ifdef GL_ES
#define fragColor gl_FragColor
#else
out vec4 fragColor;
#endif  // defined(GL_ES);

void main() {
#ifdef FLIP_Y_COORD
  float y_coord = 1.0 - sample_coordinate.y;
#else
  float y_coord = sample_coordinate.y;
#endif  // defined(FLIP_Y_COORD)
  vec2 adjusted_coordinate = vec2(sample_coordinate.x, y_coord);
  vec4 input_value = texture2D(input_texture, adjusted_coordinate);

  // Run activation function.
  // One and only one of FN_SOFTMAX,FN_SIGMOID,FN_NONE will be defined.

#ifdef FN_SOFTMAX
  // Only two channel input tensor is supported.
  vec2 input_px = input_value.rg;
  float shift = max(input_px.r, input_px.g);
  float softmax_denom = exp(input_px.r - shift) + exp(input_px.g - shift);
  float new_mask_value =
      exp(mix(input_px.r, input_px.g, float(OUTPUT_LAYER_INDEX)) - shift) / softmax_denom;
#endif // FN_SOFTMAX

#ifdef FN_SIGMOID
  float new_mask_value = 1.0 / (exp(-input_value.r) + 1.0);
#endif // FN_SIGMOID

#ifdef FN_NONE
  float new_mask_value = input_value.r;
#endif // FN_NONE

  vec4 out_value = vec4(new_mask_value, 0.0, 0.0, new_mask_value);
  fragColor = out_value;
})";

    // Shader defines.
    using Options = ::mediapipe::TensorsToSegmentationCalculatorOptions;
    const std::string output_layer_index =
        "\n#define OUTPUT_LAYER_INDEX int(" +
        std::to_string(options.output_layer_index()) + ")";
    MP_ASSIGN_OR_RETURN(bool gpu_texture_starts_at_bottom,
                        IsGpuOriginAtBottom(options.gpu_origin()));
    const std::string flip_y_coord =
        gpu_texture_starts_at_bottom ? "\n#define FLIP_Y_COORD" : "";
    const std::string fn_none =
        options.activation() == Options::NONE ? "\n#define FN_NONE" : "";
    const std::string fn_sigmoid =
        options.activation() == Options::SIGMOID ? "\n#define FN_SIGMOID" : "";
    const std::string fn_softmax =
        options.activation() == Options::SOFTMAX ? "\n#define FN_SOFTMAX" : "";
    const std::string two_channel = options.activation() == Options::SOFTMAX
                                        ? "\n#define TWO_CHANNEL_INPUT"
                                        : "";
    const std::string shader_defines =
        absl::StrCat(output_layer_index, flip_y_coord, fn_softmax, fn_sigmoid,
                     fn_none, two_channel);

    // Build full shader.
    const std::string shader_src_no_previous =
        absl::StrCat(shader_header, shader_defines, shader_src_main);

    // Vertex shader attributes.
    const GLint attr_location[NUM_ATTRIBUTES] = {
        ATTRIB_VERTEX,
        ATTRIB_TEXTURE_POSITION,
    };
    const GLchar* attr_name[NUM_ATTRIBUTES] = {
        "position",
        "texture_coordinate",
    };

    // Main shader program & parameters
    mediapipe::GlhCreateProgram(
        mediapipe::kBasicVertexShader, shader_src_no_previous.c_str(),
        NUM_ATTRIBUTES, &attr_name[0], attr_location, &mask_program_20_);
    RET_CHECK(mask_program_20_) << "Problem initializing the program.";
    glUseProgram(mask_program_20_);
    glUniform1i(glGetUniformLocation(mask_program_20_, "input_texture"), 1);

    // Simple pass-through program, used for hardware upsampling.
    mediapipe::GlhCreateProgram(
        mediapipe::kBasicVertexShader, mediapipe::kBasicTexturedFragmentShader,
        NUM_ATTRIBUTES, &attr_name[0], attr_location, &upsample_program_);
    RET_CHECK(upsample_program_) << "Problem initializing the program.";
    glUseProgram(upsample_program_);
    glUniform1i(glGetUniformLocation(upsample_program_, "video_frame"), 1);

    return absl::OkStatus();
  }));

  gpu_initialized_ = true;

  return absl::OkStatus();
}

// Steps:
// 1. receive tensor
// 2. process segmentation tensor into small mask
// 3. upsample small mask into output mask to be same size as input image
absl::StatusOr<std::unique_ptr<Image>>
TensorsToSegmentationGlTextureConverter::Convert(const Tensor& input_tensor,
                                                 int output_width,
                                                 int output_height) {
  std::unique_ptr<Image> output_image_mask;
  MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext(
      [this, &input_tensor, output_width, output_height,
       &output_image_mask]() -> absl::Status {
        MP_ASSIGN_OR_RETURN(auto hwc,
                            GetHwcFromDims(input_tensor.shape().dims));
        auto [tensor_height, tensor_width, tensor_channels] = hwc;

        // Create initial working mask texture.
        mediapipe::GlTexture small_mask_texture;

        // Run shader, process mask tensor.
        {
          small_mask_texture = gpu_helper_.CreateDestinationTexture(
              tensor_width, tensor_height,
              mediapipe::GpuBufferFormat::kBGRA32);  // actually GL_RGBA8

          // Go through CPU if not already texture 2D (no direct conversion
          // yet). Tensor::GetOpenGlTexture2dReadView() doesn't automatically
          // convert types.
          if (!input_tensor.ready_as_opengl_texture_2d()) {
            (void)input_tensor.GetCpuReadView();
          }

          auto read_view = input_tensor.GetOpenGlTexture2dReadView();

          gpu_helper_.BindFramebuffer(small_mask_texture);
          glActiveTexture(GL_TEXTURE1);
          glBindTexture(GL_TEXTURE_2D, read_view.name());
          glUseProgram(mask_program_20_);
          GlRender();
          glBindTexture(GL_TEXTURE_2D, 0);
          glFlush();
        }

        // Upsample small mask into output.
        mediapipe::GlTexture output_texture =
            gpu_helper_.CreateDestinationTexture(
                output_width, output_height,
                mediapipe::GpuBufferFormat::kBGRA32);  // actually GL_RGBA8

        // Run shader, upsample result.
        {
          gpu_helper_.BindFramebuffer(output_texture);
          glActiveTexture(GL_TEXTURE1);
          glBindTexture(GL_TEXTURE_2D, small_mask_texture.name());
          glUseProgram(upsample_program_);
          GlRender();
          glBindTexture(GL_TEXTURE_2D, 0);
          glFlush();
        }

        // Send out image as GPU packet.
        output_image_mask = output_texture.GetFrame<Image>();

        // Cleanup
        output_texture.Release();
        return absl::OkStatus();
      }));

  return output_image_mask;
}

}  // namespace

absl::StatusOr<std::unique_ptr<TensorsToSegmentationConverter>>
CreateGlTextureConverter(
    CalculatorContext* cc,
    const mediapipe::TensorsToSegmentationCalculatorOptions& options) {
  auto converter = std::make_unique<TensorsToSegmentationGlTextureConverter>();
  MP_RETURN_IF_ERROR(converter->Init(cc, options));
  return converter;
}

}  // namespace mediapipe

#endif  // !MEDIAPIPE_DISABLE_GPU
