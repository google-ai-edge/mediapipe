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

#include "mediapipe/calculators/tensor/image_to_tensor_converter_gl_buffer.h"

#include "mediapipe/framework/port.h"

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31

#include <array>
#include <memory>
#include <vector>

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
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/command_queue.h"
#include "tensorflow/lite/delegates/gpu/gl/converters/util.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_texture.h"
#include "tensorflow/lite/delegates/gpu/gl/request_gpu_info.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace mediapipe {

namespace {

// Implements a common pattern of extracting a subrect from RGBA input texture
// and resizing it into a buffer.
class SubRectExtractorGl {
 public:
  // Extracts a region defined by @sub_rect, removes A channel, transforms input
  // pixels as alpha * x + beta and resizes result into destination.
  absl::Status ExtractSubRectToBuffer(
      const tflite::gpu::gl::GlTexture& texture,
      const tflite::gpu::HW& texture_size, const RotatedRect& sub_rect,
      bool flip_horizontally, float alpha, float beta,
      const tflite::gpu::HW& destination_size,
      tflite::gpu::gl::CommandQueue* command_queue,
      tflite::gpu::gl::GlBuffer* destination);

  static absl::StatusOr<SubRectExtractorGl> Create(
      const mediapipe::GlContext& gl_context, bool input_starts_at_bottom,
      BorderMode border_mode);

 private:
  explicit SubRectExtractorGl(tflite::gpu::gl::GlProgram program,
                              tflite::gpu::uint3 workgroup_size,
                              bool use_custom_zero_border,
                              BorderMode border_mode)
      : program_(std::move(program)),
        workgroup_size_(workgroup_size),
        use_custom_zero_border_(use_custom_zero_border),
        border_mode_(border_mode) {}

  tflite::gpu::gl::GlProgram program_;
  tflite::gpu::uint3 workgroup_size_;
  bool use_custom_zero_border_ = false;
  BorderMode border_mode_ = BorderMode::kReplicate;
};

absl::Status SetMat4x4(const tflite::gpu::gl::GlProgram& program,
                       const std::string& name, float* data) {
  GLint uniform_id;
  MP_RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glGetUniformLocation, &uniform_id,
                                        program.id(), name.c_str()));
  return TFLITE_GPU_CALL_GL(glProgramUniformMatrix4fv, program.id(), uniform_id,
                            1, GL_TRUE, data);
}

constexpr char kShaderCode[] = R"(
layout(std430) buffer;

precision highp float;

// It is possible to use "vec3 elements[];" here, however due to alignment
// requirements it works only when "packed" layout is used. "packed" layout is
// determined by implementation and it's expected that OpenGL API is used to
// query the layout. Favoring float array over vec3, considering performance is
// comparable, layout is the same and no need for layout querying (even though
// it's not quite needed here as there's only one member).
layout(binding = 0) writeonly buffer B0 {
  float elements[];
} output_data;

uniform ivec2 out_size;
uniform float alpha;
uniform float beta;
uniform mat4 transform_matrix;
uniform mediump sampler2D input_data;

void main() {
    int out_width = out_size.x;
    int out_height = out_size.y;

    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    if (gid.x >= out_width || gid.y >= out_height) {
        return;
    }

    // transform from image.width, image.height range to [0, 1]
    float normal_x = (float(gid.x) + 0.5f) / float(out_width);
    float normal_y = (float(gid.y) + 0.5f) / float(out_height);
    vec4 tc = vec4(normal_x, normal_y, 0.0, 1.0);

    // Apply transformation from roi coordinates to original image coordinates.
    tc = transform_matrix * tc;
#ifdef INPUT_STARTS_AT_BOTTOM
    // Opengl texture sampler has origin in lower left corner,
    // so we invert y coordinate.
    tc.y = 1.0f - tc.y;
#endif  // INPUT_STARTS_AT_BOTTOM
    vec4 src_value = alpha * texture(input_data, tc.xy) + beta;

#ifdef CUSTOM_ZERO_BORDER_MODE
    float out_of_bounds =
      float(tc.x < 0.0 || tc.x > 1.0 || tc.y < 0.0 || tc.y > 1.0);
    src_value = mix(src_value, vec4(0.0, 0.0, 0.0, 0.0), out_of_bounds);
#endif

    int linear_index = gid.y * out_width + gid.x;

    // output_data.elements is populated as though it contains vec3 elements.
    int first_component_index = 3 * linear_index;
    output_data.elements[first_component_index] = src_value.r;
    output_data.elements[first_component_index + 1] = src_value.g;
    output_data.elements[first_component_index + 2] = src_value.b;
}
)";

absl::Status SubRectExtractorGl::ExtractSubRectToBuffer(
    const tflite::gpu::gl::GlTexture& texture,
    const tflite::gpu::HW& texture_size, const RotatedRect& texture_sub_rect,
    bool flip_horizontally, float alpha, float beta,
    const tflite::gpu::HW& destination_size,
    tflite::gpu::gl::CommandQueue* command_queue,
    tflite::gpu::gl::GlBuffer* destination) {
  std::array<float, 16> transform_mat;
  GetRotatedSubRectToRectTransformMatrix(texture_sub_rect, texture_size.w,
                                         texture_size.h, flip_horizontally,
                                         &transform_mat);
  MP_RETURN_IF_ERROR(texture.BindAsSampler2D(0));

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

  MP_RETURN_IF_ERROR(destination->BindToIndex(0));
  MP_RETURN_IF_ERROR(program_.SetParameter({"input_data", 0}));
  MP_RETURN_IF_ERROR(
      SetMat4x4(program_, "transform_matrix", transform_mat.data()));
  MP_RETURN_IF_ERROR(program_.SetParameter(
      {"out_size", tflite::gpu::int2(destination_size.w, destination_size.h)}));
  MP_RETURN_IF_ERROR(program_.SetParameter({"alpha", alpha}));
  MP_RETURN_IF_ERROR(program_.SetParameter({"beta", beta}));
  tflite::gpu::uint3 num_workgroups = tflite::gpu::DivideRoundUp(
      tflite::gpu::uint3{destination_size.w, destination_size.h, 1},
      workgroup_size_);
  MP_RETURN_IF_ERROR(command_queue->Dispatch(program_, num_workgroups));

  // Resetting to MediaPipe texture param defaults.
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  return absl::OkStatus();
}

absl::StatusOr<SubRectExtractorGl> SubRectExtractorGl::Create(
    const mediapipe::GlContext& gl_context, bool input_starts_at_bottom,
    BorderMode border_mode) {
  bool use_custom_zero_border = border_mode == BorderMode::kZero &&
                                !IsGlClampToBorderSupported(gl_context);

  const tflite::gpu::uint3 workgroup_size = {8, 8, 1};
  std::string starts_at_bottom_def;
  if (input_starts_at_bottom) {
    starts_at_bottom_def = R"(
      #define INPUT_STARTS_AT_BOTTOM;
    )";
  }
  std::string custom_zero_border_mode_def;
  if (use_custom_zero_border) {
    custom_zero_border_mode_def = R"(
      #define CUSTOM_ZERO_BORDER_MODE
    )";
  }
  const std::string full_shader_source = absl::StrCat(
      tflite::gpu::gl::GetShaderHeader(workgroup_size), starts_at_bottom_def,
      custom_zero_border_mode_def, kShaderCode);

  tflite::gpu::gl::GlShader shader;
  MP_RETURN_IF_ERROR(tflite::gpu::gl::GlShader::CompileShader(
      GL_COMPUTE_SHADER, full_shader_source, &shader));
  tflite::gpu::gl::GlProgram program;
  MP_RETURN_IF_ERROR(
      tflite::gpu::gl::GlProgram::CreateWithShader(shader, &program));

  return SubRectExtractorGl(std::move(program), workgroup_size,
                            use_custom_zero_border, border_mode);
}

class ImageToTensorGlBufferConverter : public ImageToTensorConverter {
 public:
  absl::Status Init(CalculatorContext* cc, bool input_starts_at_bottom,
                    BorderMode border_mode) {
    MP_RETURN_IF_ERROR(gl_helper_.Open(cc));
    return gl_helper_.RunInGlContext([this, input_starts_at_bottom,
                                      border_mode]() -> absl::Status {
      tflite::gpu::GpuInfo gpu_info;
      MP_RETURN_IF_ERROR(tflite::gpu::gl::RequestGpuInfo(&gpu_info));
      RET_CHECK(gpu_info.IsApiOpenGl31OrAbove())
          << "OpenGL ES 3.1 is required.";
      command_queue_ = tflite::gpu::gl::NewCommandQueue(gpu_info);

      MP_ASSIGN_OR_RETURN(
          auto extractor,
          SubRectExtractorGl::Create(gl_helper_.GetGlContext(),
                                     input_starts_at_bottom, border_mode));
      extractor_ = absl::make_unique<SubRectExtractorGl>(std::move(extractor));
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
    const auto& output_shape = output_tensor.shape();
    MP_RETURN_IF_ERROR(ValidateTensorShape(output_shape));

    MP_RETURN_IF_ERROR(gl_helper_.RunInGlContext(
        [this, &output_tensor, &input, &roi, &output_shape, range_min,
         range_max, tensor_buffer_offset]() -> absl::Status {
          const int input_num_channels = input.channels();
          auto source_texture = gl_helper_.CreateSourceTexture(input);
          tflite::gpu::gl::GlTexture input_texture(
              GL_TEXTURE_2D, source_texture.name(),
              input_num_channels == 4 ? GL_RGBA : GL_RGB,
              source_texture.width() * source_texture.height() *
                  input_num_channels * sizeof(uint8_t),
              /*layer=*/0,
              /*owned=*/false);

          constexpr float kInputImageRangeMin = 0.0f;
          constexpr float kInputImageRangeMax = 1.0f;
          MP_ASSIGN_OR_RETURN(auto transform,
                              GetValueRangeTransformation(
                                  kInputImageRangeMin, kInputImageRangeMax,
                                  range_min, range_max));

          const int output_size = output_tensor.bytes() / output_shape.dims[0];
          auto buffer_view = output_tensor.GetOpenGlBufferWriteView();
          tflite::gpu::gl::GlBuffer output(GL_SHADER_STORAGE_BUFFER,
                                           buffer_view.name(), output_size,
                                           /*offset=*/tensor_buffer_offset,
                                           /*has_ownership=*/false);
          MP_RETURN_IF_ERROR(extractor_->ExtractSubRectToBuffer(
              input_texture,
              tflite::gpu::HW(source_texture.height(), source_texture.width()),
              roi,
              /*flip_horizontally=*/false, transform.scale, transform.offset,
              tflite::gpu::HW(output_shape.dims[1], output_shape.dims[2]),
              command_queue_.get(), &output));

          return absl::OkStatus();
        }));

    return absl::OkStatus();
  }

  ~ImageToTensorGlBufferConverter() override {
    gl_helper_.RunInGlContext([this]() {
      // Release OpenGL resources.
      extractor_ = nullptr;
      command_queue_ = nullptr;
    });
  }

 private:
  absl::Status ValidateTensorShape(const Tensor::Shape& output_shape) {
    RET_CHECK_EQ(output_shape.dims.size(), 4)
        << "Wrong output dims size: " << output_shape.dims.size();
    RET_CHECK_GE(output_shape.dims[0], 1)
        << "The batch dimension needs to be greater or equal to 1.";
    RET_CHECK_EQ(output_shape.dims[3], 3)
        << "Wrong output channel: " << output_shape.dims[3];
    return absl::OkStatus();
  }

  std::unique_ptr<tflite::gpu::gl::CommandQueue> command_queue_;
  std::unique_ptr<SubRectExtractorGl> extractor_;
  mediapipe::GlCalculatorHelper gl_helper_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<ImageToTensorConverter>>
CreateImageToGlBufferTensorConverter(CalculatorContext* cc,
                                     bool input_starts_at_bottom,
                                     BorderMode border_mode) {
  auto result = absl::make_unique<ImageToTensorGlBufferConverter>();
  MP_RETURN_IF_ERROR(result->Init(cc, input_starts_at_bottom, border_mode));

  return result;
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
