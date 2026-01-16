#include "mediapipe/tasks/cc/vision/image_segmenter/calculators/ssbo_to_texture_converter.h"

#include "tensorflow/lite/delegates/gpu/gl/converters/util.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_shader.h"

// Quick compile-time warning to ensure usage on the proper platform.
#if !(MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31)
#warning "SsboToTextureConverter should be used with OpenGL ES 3.1 or above"
#endif

namespace mediapipe {
namespace tasks {
namespace {

using ::tflite::gpu::gl::GlProgram;
using ::tflite::gpu::gl::GlShader;

constexpr int kWorkgroupSize = 8;  // Block size for GPU shader.
const tflite::gpu::uint3 workgroup_size = {kWorkgroupSize, kWorkgroupSize, 1};

// "Delinearization" shader:
// Example data using n=5 channels: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 -->
// 0,1,2,3 | 4,X,X,X | 5,6,7,8 | 9,X,X,X | 10,11,12,13 | 14,X,X,X
const char delinearization_shader_source[] = R"(
precision highp float;
layout(rgba32f, binding = 0) writeonly uniform highp image2D output_texture;

uniform ivec2 out_size;
uniform int num_channels;
uniform int num_channels_padded;  // ^ rounded up to nearest multiple of 4

layout(std430, binding = 2) readonly buffer B0 {
  float elements[];
} input_data;   // data tensor

void main() {
  int out_width = out_size.x;
  int out_height = out_size.y;

  ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
  if (gid.x >= out_width || gid.y >= out_height) { return; }
  int linear_index_pixels = gid.y * out_width + gid.x;
  int linear_index = linear_index_pixels * 4;

  int num_completed_chunks = linear_index / num_channels_padded;
  int offset = linear_index % num_channels_padded;
  int data_index = num_completed_chunks * num_channels + offset;

  // Early exit if fully outside buffer
  int data_size = input_data.elements.length();
  if (data_index >= data_size) return;

  // We add some extra logic here just to ensure we don't overrun buffer and get
  // undefined behavior.  TODO: Come up with nicer way around this if
  // we end up needing this sort of patch more frequently.
  float x = input_data.elements[data_index];
  float y = 0.0;
  float z = 0.0;
  float w = 0.0;
  if (data_index + 3 < data_size) {
    w = input_data.elements[data_index + 3];
    z = input_data.elements[data_index + 2];
    y = input_data.elements[data_index + 1];
  } else if (data_index + 2 < data_size) {
    z = input_data.elements[data_index + 2];
    y = input_data.elements[data_index + 1];
  } else if (data_index + 1 < data_size) {
    y = input_data.elements[data_index + 1];
  }

  ivec2 output_coordinate = ivec2(gid.x, gid.y);
  vec4 out_value = vec4(x, y, z, w);
  imageStore(output_texture, output_coordinate, out_value);
})";

// Commonly used to compute the number of blocks to launch in a kernel.
int NumGroups(const int size, const int group_size) {  // NOLINT
  return (size + group_size - 1) / group_size;
}

}  // namespace

absl::Status SsboToTextureConverter::Init() {
  GlShader delinearization_shader;
  std::string delinearization_shader_source_with_headers =
      absl::StrCat(tflite::gpu::gl::GetShaderHeader(workgroup_size),
                   delinearization_shader_source);
  MP_RETURN_IF_ERROR(GlShader::CompileShader(
      GL_COMPUTE_SHADER, delinearization_shader_source_with_headers,
      &delinearization_shader));
  delinearization_program_ = absl::make_unique<GlProgram>();
  MP_RETURN_IF_ERROR(GlProgram::CreateWithShader(
      delinearization_shader, delinearization_program_.get()));
  return absl::OkStatus();
}

void SsboToTextureConverter::Close() { delinearization_program_.reset(); }

std::pair<const uint32_t, const uint32_t>
SsboToTextureConverter::GetTextureSize() {
  return std::make_pair(texture_width_, texture_height_);
}

absl::StatusOr<GLuint> SsboToTextureConverter::ConvertTensorToGlTexture(
    const Tensor& tensor, const uint32_t width, const uint32_t height,
    const uint32_t channels) {
  // The tflite::gpu:: namespace looks like it's much simpler and older-- it
  // doesn't tap into any memory pools, and doesn't allow linearF32 filtering
  // where available, for example. The key difference is that it uses
  // glTexStorage2D for allocation instead of glTexImage2D, which is necessary
  // in order to create an immutable format (as required by glBindImageTexture).
  // MP will automatically use this for RGBA16F but not RGBA32F textures
  // currently, oddly enough.  So options are:
  // (1) extend MP to similarly handle RGBA32F
  // (2) just make our own texture here and keep reusing, recreating if the size
  //     changes, which should generally not happen. (This is ok because we use
  //     the texture immediately and never output it from the calculator).
  // (3) Change glBindImageTexture call to alternative so we can just use
  //     existing MP glTexImage2D storage creation?  This seems less than
  //     ideal since it's rather nice to keep the above program in compute
  //     shader format.
  // TODO: To be safe for this initial implementation, we go with
  // option #2, as it's simplest/easiest, but this should be cleaned up later.
  const uint32_t num_pixels_per_element = ((channels + 3) / 4);
  const uint32_t padded_channels = 4 * num_pixels_per_element;
  const uint32_t texture_width = width * num_pixels_per_element;
  const uint32_t texture_height = height;
  if (texture_width != texture_width_ || texture_height != texture_height_) {
    // tflite::gpu::gl::GlTexture autoreleases, so we don't have to worry about
    // freeing memory.
    MP_RETURN_IF_ERROR(CreateReadWriteRgbaImageTexture(
        tflite::gpu::DataType::FLOAT32, {texture_width, texture_height},
        &out_texture_));
    texture_width_ = texture_width;
    texture_height_ = texture_height;
  }

  glBindImageTexture(0 /* output index */, out_texture_.id(), 0, GL_FALSE, 0,
                     GL_WRITE_ONLY, GL_RGBA32F);
  auto read_view = tensor.GetOpenGlBufferReadView();
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2 /* input index */,
                   read_view.name());

  glUseProgram(delinearization_program_->id());
  glUniform2i(glGetUniformLocation(delinearization_program_->id(), "out_size"),
              texture_width, texture_height);
  glUniform1i(
      glGetUniformLocation(delinearization_program_->id(), "num_channels"),
      channels);
  glUniform1i(glGetUniformLocation(delinearization_program_->id(),
                                   "num_channels_padded"),
              padded_channels);

  const tflite::gpu::uint3 workgroups = {
      NumGroups(texture_width, kWorkgroupSize),
      NumGroups(texture_height, kWorkgroupSize), 1};
  MP_RETURN_IF_ERROR(delinearization_program_->Dispatch(workgroups));
  return out_texture_.id();
}

}  // namespace tasks
}  // namespace mediapipe
