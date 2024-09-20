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

#include "mediapipe/framework/port.h"

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/calculators/tensor/tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_segmentation_converter.h"
#include "mediapipe/calculators/tensor/tensors_to_segmentation_converter_gl_buffer.h"
#include "mediapipe/calculators/tensor/tensors_to_segmentation_utils.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/gpu_buffer_format.h"
#include "mediapipe/gpu/gpu_origin.pb.h"
#include "mediapipe/gpu/gpu_origin_utils.h"
#include "mediapipe/gpu/shader_util.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/converters/util.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_program.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_shader.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_texture.h"

namespace mediapipe {
namespace {

using ::mediapipe::tensors_to_segmentation_utils::GetHwcFromDims;
using ::mediapipe::tensors_to_segmentation_utils::GlRender;
using ::mediapipe::tensors_to_segmentation_utils::NumGroups;
using ::tflite::gpu::gl::GlProgram;
using ::tflite::gpu::gl::GlShader;

constexpr int kWorkgroupSize = 8;  // Block size for GPU shader.
enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };

class TensorsToSegmentationGlBufferConverter
    : public TensorsToSegmentationConverter {
 public:
  ~TensorsToSegmentationGlBufferConverter() override;
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
  int cached_width_ = 0;
  int cached_height_ = 0;
  std::unique_ptr<tflite::gpu::gl::GlTexture> small_mask_texture_;
  std::unique_ptr<GlProgram> mask_program_31_;
};

TensorsToSegmentationGlBufferConverter::
    ~TensorsToSegmentationGlBufferConverter() {
  if (gpu_initialized_) {
    gpu_helper_.RunInGlContext([this] {
      if (upsample_program_) glDeleteProgram(upsample_program_);
      upsample_program_ = 0;
      mask_program_31_.reset();
      small_mask_texture_.reset();
    });
  }
}

absl::Status TensorsToSegmentationGlBufferConverter::Init(
    CalculatorContext* cc,
    const TensorsToSegmentationCalculatorOptions& options) {
  MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
  MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([this,
                                                 &options]() -> absl::Status {
    // A shader to process a segmentation tensor into an output mask.
    // Currently uses 4 channels for output, and sets R+A channels as mask
    // value.
    const tflite::gpu::uint3 workgroup_size = {kWorkgroupSize, kWorkgroupSize,
                                               1};
    const std::string shader_header =
        absl::StrCat(tflite::gpu::gl::GetShaderHeader(workgroup_size), R"(
precision highp float;

layout(rgba8, binding = 0) writeonly uniform highp image2D output_texture;

uniform ivec2 out_size;
)");
    /* Shader defines will be inserted here. */

    const std::string shader_src_main = R"(
layout(std430, binding = 2) readonly buffer B0 {
#ifdef TWO_CHANNEL_INPUT
  vec2 elements[];
#else
  float elements[];
#endif // TWO_CHANNEL_INPUT
} input_data;   // data tensor

void main() {
  int out_width = out_size.x;
  int out_height = out_size.y;

  ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
  if (gid.x >= out_width || gid.y >= out_height) { return; }
  int linear_index = gid.y * out_width + gid.x;

#ifdef TWO_CHANNEL_INPUT
  vec2 input_value = input_data.elements[linear_index];
#else
  vec2 input_value = vec2(input_data.elements[linear_index], 0.0);
#endif // TWO_CHANNEL_INPUT

// Run activation function.
// One and only one of FN_SOFTMAX,FN_SIGMOID,FN_NONE will be defined.
#ifdef FN_SOFTMAX
  // Only two channel input tensor is supported.
  vec2 input_px = input_value.rg;
  float shift = max(input_px.r, input_px.g);
  float softmax_denom = exp(input_px.r - shift) + exp(input_px.g - shift);
  float new_mask_value =
      exp(input_px[OUTPUT_LAYER_INDEX] - shift) / softmax_denom;
#endif // FN_SOFTMAX

#ifdef FN_SIGMOID
  float new_mask_value = 1.0 / (exp(-input_value.r) + 1.0);
#endif // FN_SIGMOID

#ifdef FN_NONE
  float new_mask_value = input_value.r;
#endif // FN_NONE

#ifdef FLIP_Y_COORD
  int y_coord = out_height - gid.y - 1;
#else
  int y_coord = gid.y;
#endif  // defined(FLIP_Y_COORD)
  ivec2 output_coordinate = ivec2(gid.x, y_coord);

  vec4 out_value = vec4(new_mask_value, 0.0, 0.0, new_mask_value);
  imageStore(output_texture, output_coordinate, out_value);
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
    GlShader shader_without_previous;
    MP_RETURN_IF_ERROR(GlShader::CompileShader(
        GL_COMPUTE_SHADER, shader_src_no_previous, &shader_without_previous));
    mask_program_31_ = absl::make_unique<GlProgram>();
    MP_RETURN_IF_ERROR(GlProgram::CreateWithShader(shader_without_previous,
                                                   mask_program_31_.get()));
    small_mask_texture_ = absl::make_unique<tflite::gpu::gl::GlTexture>();

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
TensorsToSegmentationGlBufferConverter::Convert(const Tensor& input_tensor,
                                                int output_width,
                                                int output_height) {
  std::unique_ptr<Image> output_image_mask;
  MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext(
      [this, &input_tensor, output_width, output_height,
       &output_image_mask]() -> absl::Status {
        MP_ASSIGN_OR_RETURN(auto hwc,
                            GetHwcFromDims(input_tensor.shape().dims));
        auto [tensor_height, tensor_width, tensor_channels] = hwc;
        {
          // Only recreate if the size has changed. See b/297809673 for more
          // details.
          if (tensor_width != cached_width_ ||
              tensor_height != cached_height_) {
            MP_RETURN_IF_ERROR(CreateReadWriteRgbaImageTexture(
                tflite::gpu::DataType::UINT8,  // GL_RGBA8
                {tensor_width, tensor_height}, small_mask_texture_.get()));
            cached_width_ = tensor_width;
            cached_height_ = tensor_height;
          }

          const int output_index = 0;
          glBindImageTexture(output_index, small_mask_texture_->id(), 0,
                             GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);

          auto read_view = input_tensor.GetOpenGlBufferReadView();
          glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, read_view.name());

          const tflite::gpu::uint3 workgroups = {
              NumGroups(tensor_width, kWorkgroupSize),
              NumGroups(tensor_height, kWorkgroupSize), 1};

          glUseProgram(mask_program_31_->id());
          glUniform2i(glGetUniformLocation(mask_program_31_->id(), "out_size"),
                      tensor_width, tensor_height);

          MP_RETURN_IF_ERROR(mask_program_31_->Dispatch(workgroups));
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
          glBindTexture(GL_TEXTURE_2D, small_mask_texture_->id());
          glUseProgram(upsample_program_);
          GlRender();
          glBindTexture(GL_TEXTURE_2D, 0);
          glFlush();
        }

        // Store the result into the output pointer.
        output_image_mask = output_texture.GetFrame<Image>();

        // Cleanup
        output_texture.Release();
        return absl::OkStatus();
      }));

  return output_image_mask;
}

}  // namespace

absl::StatusOr<std::unique_ptr<TensorsToSegmentationConverter>>
CreateGlBufferConverter(
    CalculatorContext* cc,
    const mediapipe::TensorsToSegmentationCalculatorOptions& options) {
  auto converter = std::make_unique<TensorsToSegmentationGlBufferConverter>();
  MP_RETURN_IF_ERROR(converter->Init(cc, options));
  return converter;
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
