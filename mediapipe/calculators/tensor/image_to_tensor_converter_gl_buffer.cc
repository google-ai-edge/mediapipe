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

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/absl_log.h"          // from @com_google_absl
#include "absl/status/status.h"         // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"       // from @com_google_absl
#include "absl/strings/str_cat.h"       // from @com_google_absl
#include "mediapipe/calculators/tensor/image_to_tensor_converter.h"
#include "mediapipe/calculators/tensor/image_to_tensor_converter_gl_utils.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "tflite/delegates/gpu/common/shape.h"         // from @litert
#include "tflite/delegates/gpu/common/types.h"         // from @litert
#include "tflite/delegates/gpu/gl/command_queue.h"     // from @litert
#include "tflite/delegates/gpu/gl/converters/util.h"   // from @litert
#include "tflite/delegates/gpu/gl/gl_buffer.h"         // from @litert
#include "tflite/delegates/gpu/gl/gl_call.h"           // from @litert
#include "tflite/delegates/gpu/gl/gl_texture.h"        // from @litert
#include "tflite/delegates/gpu/gl/request_gpu_info.h"  // from @litert
#include "tflite/delegates/gpu/gl/variable.h"          // from @litert

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

  // `tensor_type`: float32, uint8 or int8.
  static absl::StatusOr<SubRectExtractorGl> Create(
      const mediapipe::GlContext& gl_context, bool input_starts_at_bottom,
      BorderMode border_mode, Tensor::ElementType tensor_type);

 private:
  explicit SubRectExtractorGl(tflite::gpu::gl::GlProgram program,
                              tflite::gpu::uint3 workgroup_size,
                              bool use_custom_zero_border,
                              BorderMode border_mode,
                              Tensor::ElementType tensor_type)
      : program_(std::move(program)),
        workgroup_size_(workgroup_size),
        use_custom_zero_border_(use_custom_zero_border),
        border_mode_(border_mode),
        tensor_type_(tensor_type) {}

  tflite::gpu::gl::GlProgram program_;
  tflite::gpu::uint3 workgroup_size_;
  bool use_custom_zero_border_ = false;
  BorderMode border_mode_ = BorderMode::kReplicate;
  Tensor::ElementType tensor_type_ = Tensor::ElementType::kFloat32;
};

absl::Status SetMat4x4(const tflite::gpu::gl::GlProgram& program,
                       const std::string& name, float* data) {
  GLint uniform_id;
  ABSL_RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glGetUniformLocation, &uniform_id,
                                          program.id(), name.c_str()));
  return TFLITE_GPU_CALL_GL(glProgramUniformMatrix4fv, program.id(), uniform_id,
                            1, GL_TRUE, data);
}

// Returns the shader invocation count for an output of `destination_size`.
absl::StatusOr<tflite::gpu::uint3> GetWorkSize(
    Tensor::ElementType tensor_type, const tflite::gpu::HW& destination_size) {
  RET_CHECK_GT(destination_size.w, 0);
  RET_CHECK_GT(destination_size.h, 0);
  switch (tensor_type) {
    case Tensor::ElementType::kFloat32:
      return tflite::gpu::uint3{static_cast<uint32_t>(destination_size.w),
                                static_cast<uint32_t>(destination_size.h), 1u};
    case Tensor::ElementType::kUInt8:
    case Tensor::ElementType::kInt8: {
      // One invocation per 4 pixels. X wraps to stay within the minimum
      // GL_MAX_COMPUTE_WORK_GROUP_COUNT.
      constexpr uint64_t kInvocationsPerRow = 1024ULL * 64;
      const uint64_t num_pixels =
          static_cast<uint64_t>(destination_size.w) * destination_size.h;
      const uint64_t num_invocations = (num_pixels + 3) / 4;
      if (num_invocations <= kInvocationsPerRow) {
        return tflite::gpu::uint3{static_cast<uint32_t>(num_invocations), 1u,
                                  1u};
      }
      const uint64_t num_rows =
          (num_invocations + kInvocationsPerRow - 1) / kInvocationsPerRow;
      RET_CHECK_LE(num_rows, 65535ULL)
          << "Output tensor too large for GLES 3.1 compute dispatch.";
      return tflite::gpu::uint3{static_cast<uint32_t>(kInvocationsPerRow),
                                static_cast<uint32_t>(num_rows), 1u};
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported output tensor type: ",
                       Tensor::ElementTypeName(tensor_type)));
  }
}

constexpr char kShaderCode[] = R"(
layout(std430) buffer;

precision highp float;

uniform ivec2 out_size;
uniform float alpha;
uniform float beta;
uniform mat4 transform_matrix;
uniform mediump sampler2D input_data;

// Samples the input at output pixel `gid`, transformed as alpha * x + beta.
vec4 sample_output_pixel(ivec2 gid) {
    int out_width = out_size.x;
    int out_height = out_size.y;

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
    return src_value;
}

#ifdef OUTPUT_BYTES
// 8-bit output. GLSL ES has no 8-bit buffers, so each invocation writes 4 RGB
// pixels as 3 whole uints, avoiding races on shared uints. Values are rounded,
// clamped and truncated to 8 bits (two's complement for int8).
layout(binding = 0) writeonly buffer B0 {
  uint elements[];
} output_data;

uint to_byte(float value) {
#ifdef OUTPUT_SIGNED
    float clamped = clamp(floor(value + 0.5), -128.0, 127.0);
#else
    float clamped = clamp(floor(value + 0.5), 0.0, 255.0);
#endif  // OUTPUT_SIGNED
    return uint(int(clamped) & 0xFF);
}

uint pack_bytes(float b0, float b1, float b2, float b3) {
    return to_byte(b0) | (to_byte(b1) << 8) | (to_byte(b2) << 16) |
           (to_byte(b3) << 24);
}

void main() {
    uint num_pixels = uint(out_size.x) * uint(out_size.y);
    uint invocation =
        gl_GlobalInvocationID.y * (gl_NumWorkGroups.x * gl_WorkGroupSize.x) +
        gl_GlobalInvocationID.x;
    uint first_pixel = 4u * invocation;
    if (first_pixel >= num_pixels) {
        return;
    }
    vec3 p[4];
    for (uint i = 0u; i < 4u; ++i) {
        uint pixel = first_pixel + i;
        p[i] = vec3(0.0);
        if (pixel < num_pixels) {
            p[i] = sample_output_pixel(
                ivec2(int(pixel % uint(out_size.x)),
                      int(pixel / uint(out_size.x)))).rgb;
        }
    }
    uint num_elements = (3u * num_pixels + 3u) / 4u;
    uint first_element = 3u * invocation;
    output_data.elements[first_element] =
        pack_bytes(p[0].r, p[0].g, p[0].b, p[1].r);
    if (first_element + 1u < num_elements) {
        output_data.elements[first_element + 1u] =
            pack_bytes(p[1].g, p[1].b, p[2].r, p[2].g);
    }
    if (first_element + 2u < num_elements) {
        output_data.elements[first_element + 2u] =
            pack_bytes(p[2].b, p[3].r, p[3].g, p[3].b);
    }
}
#else
// It is possible to use "vec3 elements[];" here, however due to alignment
// requirements it works only when "packed" layout is used. "packed" layout is
// determined by implementation and it's expected that OpenGL API is used to
// query the layout. Favoring float array over vec3, considering performance is
// comparable, layout is the same and no need for layout querying (even though
// it's not quite needed here as there's only one member).
layout(binding = 0) writeonly buffer B0 {
  float elements[];
} output_data;

void main() {
    int out_width = out_size.x;
    int out_height = out_size.y;

    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    if (gid.x >= out_width || gid.y >= out_height) {
        return;
    }
    vec4 src_value = sample_output_pixel(gid);

    int linear_index = gid.y * out_width + gid.x;

    // output_data.elements is populated as though it contains vec3 elements.
    int first_component_index = 3 * linear_index;
    output_data.elements[first_component_index] = src_value.r;
    output_data.elements[first_component_index + 1] = src_value.g;
    output_data.elements[first_component_index + 2] = src_value.b;
}
#endif  // OUTPUT_BYTES
)";

absl::Status SubRectExtractorGl::ExtractSubRectToBuffer(
    const tflite::gpu::gl::GlTexture& texture,
    const tflite::gpu::HW& texture_size, const RotatedRect& texture_sub_rect,
    bool flip_horizontally, float alpha, float beta,
    const tflite::gpu::HW& destination_size,
    tflite::gpu::gl::CommandQueue* command_queue,
    tflite::gpu::gl::GlBuffer* destination) {
  ABSL_ASSIGN_OR_RETURN(const tflite::gpu::uint3 work_size,
                        GetWorkSize(tensor_type_, destination_size));
  std::array<float, 16> transform_mat;
  GetRotatedSubRectToRectTransformMatrix(texture_sub_rect, texture_size.w,
                                         texture_size.h, flip_horizontally,
                                         &transform_mat);
  ABSL_RETURN_IF_ERROR(texture.BindAsSampler2D(0));

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

  ABSL_RETURN_IF_ERROR(destination->BindToIndex(0));
  ABSL_RETURN_IF_ERROR(program_.SetParameter({"input_data", 0}));
  ABSL_RETURN_IF_ERROR(
      SetMat4x4(program_, "transform_matrix", transform_mat.data()));
  ABSL_RETURN_IF_ERROR(program_.SetParameter(
      {"out_size", tflite::gpu::int2(destination_size.w, destination_size.h)}));
  ABSL_RETURN_IF_ERROR(program_.SetParameter({"alpha", alpha}));
  ABSL_RETURN_IF_ERROR(program_.SetParameter({"beta", beta}));
  tflite::gpu::uint3 num_workgroups =
      tflite::gpu::DivideRoundUp(work_size, workgroup_size_);
  ABSL_RETURN_IF_ERROR(command_queue->Dispatch(program_, num_workgroups));

  // Resetting to MediaPipe texture param defaults.
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  return absl::OkStatus();
}

absl::StatusOr<SubRectExtractorGl> SubRectExtractorGl::Create(
    const mediapipe::GlContext& gl_context, bool input_starts_at_bottom,
    BorderMode border_mode, Tensor::ElementType tensor_type) {
  bool use_custom_zero_border = border_mode == BorderMode::kZero &&
                                !IsGlClampToBorderSupported(gl_context);
  const bool output_bytes = tensor_type != Tensor::ElementType::kFloat32;

  const tflite::gpu::uint3 workgroup_size =
      output_bytes ? tflite::gpu::uint3{64, 1, 1} : tflite::gpu::uint3{8, 8, 1};
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
  std::string output_bytes_def;
  if (output_bytes) {
    output_bytes_def = R"(
      #define OUTPUT_BYTES
    )";
  }
  if (tensor_type == Tensor::ElementType::kInt8) {
    absl::StrAppend(&output_bytes_def, R"(
      #define OUTPUT_SIGNED
    )");
  }
  const std::string full_shader_source = absl::StrCat(
      tflite::gpu::gl::GetShaderHeader(workgroup_size), starts_at_bottom_def,
      custom_zero_border_mode_def, output_bytes_def, kShaderCode);

  tflite::gpu::gl::GlShader shader;
  ABSL_RETURN_IF_ERROR(tflite::gpu::gl::GlShader::CompileShader(
      GL_COMPUTE_SHADER, full_shader_source, &shader));
  tflite::gpu::gl::GlProgram program;
  ABSL_RETURN_IF_ERROR(
      tflite::gpu::gl::GlProgram::CreateWithShader(shader, &program));

  return SubRectExtractorGl(std::move(program), workgroup_size,
                            use_custom_zero_border, border_mode, tensor_type);
}

class ImageToTensorGlBufferConverter : public ImageToTensorConverter {
 public:
  absl::Status Init(CalculatorContext* cc, bool input_starts_at_bottom,
                    BorderMode border_mode, Tensor::ElementType tensor_type) {
    switch (tensor_type) {
      case Tensor::ElementType::kFloat32:
      case Tensor::ElementType::kUInt8:
      case Tensor::ElementType::kInt8:
        break;
      default:
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported output tensor type: ",
                         Tensor::ElementTypeName(tensor_type)));
    }
    tensor_type_ = tensor_type;
    ABSL_RETURN_IF_ERROR(gl_helper_.Open(cc));
    return InitGl(input_starts_at_bottom, border_mode);
  }

  absl::Status Convert(const mediapipe::Image& input, const RotatedRect& roi,
                       float range_min, float range_max,
                       int tensor_buffer_offset,
                       Tensor& output_tensor) override {
    if (input.format() != mediapipe::GpuBufferFormat::kBGRA32 &&
        input.format() != mediapipe::GpuBufferFormat::kRGBA32 &&
        input.format() != mediapipe::GpuBufferFormat::kRGBAHalf64 &&
        input.format() != mediapipe::GpuBufferFormat::kRGBAFloat128 &&
        input.format() != mediapipe::GpuBufferFormat::kRGB24) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported format: ", static_cast<uint32_t>(input.format())));
    }
    const auto& output_shape = output_tensor.shape();
    ABSL_RETURN_IF_ERROR(ValidateTensorShape(output_shape));
    ABSL_RETURN_IF_ERROR(ValidateOutputTensorTypeAndRange(
        output_tensor, tensor_type_, range_min, range_max));
    ABSL_RETURN_IF_ERROR(gl_helper_.RunInGlContext(
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
          ABSL_ASSIGN_OR_RETURN(auto transform,
                                GetValueRangeTransformation(
                                    kInputImageRangeMin, kInputImageRangeMax,
                                    range_min, range_max));

          const int output_size = output_tensor.bytes() / output_shape.dims[0];
          auto buffer_view = output_tensor.GetOpenGlBufferWriteView();
          tflite::gpu::gl::GlBuffer output(GL_SHADER_STORAGE_BUFFER,
                                           buffer_view.name(), output_size,
                                           /*offset=*/tensor_buffer_offset,
                                           /*has_ownership=*/false);
          const tflite::gpu::HW texture_size(source_texture.height(),
                                             source_texture.width());
          const tflite::gpu::HW destination_size(output_shape.dims[1],
                                                 output_shape.dims[2]);

          // Write directly into the output buffer if it can be bound as an
          // SSBO (size a multiple of 4, aligned offset).
          if (output_size % 4 == 0 &&
              tensor_buffer_offset % ssbo_offset_alignment_ == 0) {
            return extractor_->ExtractSubRectToBuffer(
                input_texture, texture_size, roi,
                /*flip_horizontally=*/false, transform.scale, transform.offset,
                destination_size, command_queue_.get(), &output);
          }

          ABSL_LOG_FIRST_N(WARNING, 1)
              << "Cannot write directly into the output buffer: " << output_size
              << " bytes at offset " << tensor_buffer_offset << " ("
              << destination_size.w << "x" << destination_size.h << " pixels, "
              << output_tensor.element_size()
              << " bytes per element) with an SSBO offset alignment of "
              << ssbo_offset_alignment_
              << " bytes and 4-byte SSBO elements. Falling back to writing "
                 "into a scratch buffer and copying it into the output "
                 "buffer, which is slower.";
          const size_t scratch_size = (output_size + 3) / 4 * 4;
          if (scratch_buffer_.bytes_size() < scratch_size) {
            ABSL_RETURN_IF_ERROR(
                tflite::gpu::gl::CreateReadWriteShaderStorageBuffer<uint32_t>(
                    scratch_size / 4, &scratch_buffer_));
          }
          tflite::gpu::gl::GlBuffer scratch;
          ABSL_RETURN_IF_ERROR(
              scratch_buffer_.MakeView(/*offset=*/0, scratch_size, &scratch));
          ABSL_RETURN_IF_ERROR(extractor_->ExtractSubRectToBuffer(
              input_texture, texture_size, roi,
              /*flip_horizontally=*/false, transform.scale, transform.offset,
              destination_size, command_queue_.get(), &scratch));
          // glCopyBufferSubData has no alignment requirements.
          ABSL_RETURN_IF_ERROR(
              scratch_buffer_.MakeView(/*offset=*/0, output_size, &scratch));
          return tflite::gpu::gl::CopyBuffer(scratch, output);
        }));

    return absl::OkStatus();
  }

  ~ImageToTensorGlBufferConverter() override {
    gl_helper_.RunInGlContext([this]() {
      // Release OpenGL resources.
      extractor_ = nullptr;
      command_queue_ = nullptr;
      scratch_buffer_ = tflite::gpu::gl::GlBuffer();
    });
  }

 private:
  absl::Status InitGl(bool input_starts_at_bottom, BorderMode border_mode) {
    return gl_helper_.RunInGlContext([this, input_starts_at_bottom,
                                      border_mode]() -> absl::Status {
      tflite::gpu::GpuInfo gpu_info;
      ABSL_RETURN_IF_ERROR(tflite::gpu::gl::RequestGpuInfo(&gpu_info));
      RET_CHECK(gpu_info.IsApiOpenGl31OrAbove())
          << "OpenGL ES 3.1 is required.";
      command_queue_ = tflite::gpu::gl::NewCommandQueue(gpu_info);

      GLint ssbo_offset_alignment = 0;
      glGetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT,
                    &ssbo_offset_alignment);
      ssbo_offset_alignment_ = std::max(4, ssbo_offset_alignment);

      ABSL_ASSIGN_OR_RETURN(
          auto extractor, SubRectExtractorGl::Create(
                              gl_helper_.GetGlContext(), input_starts_at_bottom,
                              border_mode, tensor_type_));
      extractor_ = std::make_unique<SubRectExtractorGl>(std::move(extractor));
      return absl::OkStatus();
    });
  }

  absl::Status ValidateTensorShape(const Tensor::Shape& output_shape) {
    RET_CHECK_EQ(output_shape.dims.size(), 4)
        << "Wrong output dims size: " << output_shape.dims.size();
    RET_CHECK_GE(output_shape.dims[0], 1)
        << "The batch dimension needs to be greater or equal to 1.";
    RET_CHECK_EQ(output_shape.dims[3], 3)
        << "Wrong output channel: " << output_shape.dims[3];
    return absl::OkStatus();
  }

  Tensor::ElementType tensor_type_ = Tensor::ElementType::kFloat32;
  int ssbo_offset_alignment_ = 4;
  std::unique_ptr<tflite::gpu::gl::CommandQueue> command_queue_;
  std::unique_ptr<SubRectExtractorGl> extractor_;
  tflite::gpu::gl::GlBuffer scratch_buffer_;
  mediapipe::GlCalculatorHelper gl_helper_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<ImageToTensorConverter>>
CreateImageToGlBufferTensorConverter(CalculatorContext* cc,
                                     bool input_starts_at_bottom,
                                     BorderMode border_mode,
                                     Tensor::ElementType tensor_type) {
  auto result = absl::make_unique<ImageToTensorGlBufferConverter>();
  ABSL_RETURN_IF_ERROR(
      result->Init(cc, input_starts_at_bottom, border_mode, tensor_type));

  return result;
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
