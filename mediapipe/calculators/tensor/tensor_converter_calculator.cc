// Copyright 2019 The MediaPipe Authors.
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

#include <string>
#include <vector>

#include "mediapipe/calculators/tensor/tensor_converter_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/resource_util.h"

#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gpu_buffer.h"
#if MEDIAPIPE_METAL_ENABLED
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#import "mediapipe/gpu/MPPMetalHelper.h"
#elif MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
#include "mediapipe/gpu/gl_calculator_helper.h"
#if MEDIAPIPE_OPENGL_ES_VERSION < MEDIAPIPE_OPENGL_ES_31
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/shader_util.h"
#endif  // MEDIAPIPE_OPENGL_ES_VERSION < MEDIAPIPE_OPENGL_ES_31
#endif  // MEDIAPIPE_METAL_ENABLED
#endif  // !MEDIAPIPE_DISABLE_GPU

namespace {
constexpr int kWorkgroupSize = 8;  // Block size for GPU shader.
// Commonly used to compute the number of blocks to launch in a kernel.
int NumGroups(const int size, const int group_size) {  // NOLINT
  return (size + group_size - 1) / group_size;
}

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    RowMajorMatrixXf;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
    ColMajorMatrixXf;

constexpr char kImageFrameTag[] = "IMAGE";
constexpr char kGpuBufferTag[] = "IMAGE_GPU";
constexpr char kTensorsTag[] = "TENSORS";
constexpr char kMatrixTag[] = "MATRIX";
}  // namespace

namespace mediapipe {

// Calculator for normalizing and converting an ImageFrame, GpuBuffer or Matrix
// into a Tensor.
//
// This calculator is designed to be used with the TfLiteInferenceCalcualtor,
// as a pre-processing step for calculator inputs.
//
// IMAGE and IMAGE_GPU inputs are normalized to [-1,1] (default) or [0,1],
// specified by options (unless outputting a quantized tensor).
//
// Input:
//  One of the following tags:
//  IMAGE - ImageFrame (assumed to be 8-bit or 32-bit data).
//  IMAGE_GPU - GpuBuffer (assumed to be RGBA or RGB GL texture).
//  MATRIX - Matrix.
//
// Output:
//  One of the following tags:
//  TENSORS - Vector of Tensors of type kFloat32. The resource type used:
//          - MTLBuffer if Metal API is available
//          - SSBO if Metal is unavailable and OpenGL ES 3.1 is available
//          - Texture2D if Metal and GLES 3.1 are not available and GLES 3.0 is.
//
// Example use:
// node {
//   calculator: "TensorConverterCalculator"
//   input_stream: "IMAGE:input_image"
//   output_stream: "TENSORS:image_tensor"
//   options: {
//     [mediapipe.TensorConverterCalculatorOptions.ext] {
//       zero_center: true
//     }
//   }
// }
//
// IMPORTANT Notes:
//  GPU tensors are currently only supported on mobile platforms.

class TensorConverterCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status InitGpu(CalculatorContext* cc);
  absl::Status LoadOptions(CalculatorContext* cc);
  template <class T>
  absl::Status NormalizeImage(const ImageFrame& image_frame,
                              bool flip_vertically, float* tensor_ptr);
  absl::Status CopyMatrixToTensor(const Matrix& matrix, float* tensor_ptr);
  absl::Status ProcessCPU(CalculatorContext* cc);
  absl::Status ProcessGPU(CalculatorContext* cc);

#if MEDIAPIPE_METAL_ENABLED
  MPPMetalHelper* gpu_helper_ = nullptr;
  id<MTLComputePipelineState> to_buffer_program_;
#elif MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
  mediapipe::GlCalculatorHelper gpu_helper_;
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
  GLuint to_buffer_program_;
#else
  enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };
  GLuint to_tex2d_program_;
  GLuint framebuffer_;
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
#endif  // MEDIAPIPE_METAL_ENABLED

  bool initialized_ = false;
  bool use_gpu_ = false;
  absl::optional<std::pair<float, float>> output_range_;
  bool flip_vertically_ = false;
  bool row_major_matrix_ = false;
  int max_num_channels_ = 3;
};
REGISTER_CALCULATOR(TensorConverterCalculator);

absl::Status TensorConverterCalculator::GetContract(CalculatorContract* cc) {
  // Confirm only one of the input streams is present.
  RET_CHECK(static_cast<int>(cc->Inputs().HasTag(kImageFrameTag)) +
                static_cast<int>(cc->Inputs().HasTag(kGpuBufferTag)) +
                static_cast<int>(cc->Inputs().HasTag(kMatrixTag)) ==
            1);

  if (cc->Inputs().HasTag(kImageFrameTag)) {
    cc->Inputs().Tag(kImageFrameTag).Set<ImageFrame>();
  }
  if (cc->Inputs().HasTag(kMatrixTag)) {
    cc->Inputs().Tag(kMatrixTag).Set<Matrix>();
  }

#if !MEDIAPIPE_DISABLE_GPU
  if (cc->Inputs().HasTag(kGpuBufferTag)) {
    cc->Inputs().Tag(kGpuBufferTag).Set<mediapipe::GpuBuffer>();
#if MEDIAPIPE_METAL_ENABLED
    MP_RETURN_IF_ERROR([MPPMetalHelper updateContract:cc]);
#elif MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
    MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
#endif  // MEDIAPIPE_METAL_ENABLED
  }
#endif  // !MEDIAPIPE_DISABLE_GPU

  RET_CHECK(cc->Outputs().HasTag(kTensorsTag));
  cc->Outputs().Tag(kTensorsTag).Set<std::vector<Tensor>>();
  return absl::OkStatus();
}

absl::Status TensorConverterCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  MP_RETURN_IF_ERROR(LoadOptions(cc));

#if !MEDIAPIPE_DISABLE_GPU
  if (cc->Inputs().HasTag(kGpuBufferTag)) {
    use_gpu_ = true;
#if MEDIAPIPE_METAL_ENABLED
    gpu_helper_ = [[MPPMetalHelper alloc] initWithCalculatorContext:cc];
    RET_CHECK(gpu_helper_);
#elif MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
    MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
#endif  // MEDIAPIPE_METAL_ENABLED
  }
#endif  // !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
}

absl::Status TensorConverterCalculator::Process(CalculatorContext* cc) {
  if (use_gpu_) {
    if (cc->Inputs().Tag(kGpuBufferTag).IsEmpty()) {
      return absl::OkStatus();
    }
    // Convert to GPU tensors type.
    MP_RETURN_IF_ERROR(ProcessGPU(cc));
  } else {
    // Convert to CPU tensors or Matrix type.
    MP_RETURN_IF_ERROR(ProcessCPU(cc));
  }
  return absl::OkStatus();
}

absl::Status TensorConverterCalculator::Close(CalculatorContext* cc) {
#if !MEDIAPIPE_DISABLE_GPU
  if (use_gpu_) {
#if MEDIAPIPE_METAL_ENABLED
    to_buffer_program_ = nil;
#elif MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
    gpu_helper_.RunInGlContext([this] {
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
      glDeleteProgram(to_buffer_program_);
#else
      glDeleteFramebuffers(1, &framebuffer_);
      glDeleteProgram(to_tex2d_program_);
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
    });
#endif  // MEDIAPIPE_METAL_ENABLED
  }
#endif  // !MEDIAPIPE_DISABLE_GPU
  return absl::OkStatus();
}

absl::Status TensorConverterCalculator::ProcessCPU(CalculatorContext* cc) {
  auto output_tensors = absl::make_unique<std::vector<Tensor>>();
  if (cc->Inputs().HasTag(kImageFrameTag)) {
    if (cc->Inputs().Tag(kImageFrameTag).IsEmpty()) {
      return absl::OkStatus();
    }
    const auto& image_frame =
        cc->Inputs().Tag(kImageFrameTag).Get<ImageFrame>();
    const int height = image_frame.Height();
    const int width = image_frame.Width();
    const int channels = image_frame.NumberOfChannels();
    const int channels_preserved = std::min(channels, max_num_channels_);
    const mediapipe::ImageFormat::Format format = image_frame.Format();

    if (!(format == mediapipe::ImageFormat::SRGBA ||
          format == mediapipe::ImageFormat::SRGB ||
          format == mediapipe::ImageFormat::GRAY8 ||
          format == mediapipe::ImageFormat::VEC32F1))
      RET_CHECK_FAIL() << "Unsupported CPU input format.";

    output_tensors->emplace_back(
        Tensor::ElementType::kFloat32,
        Tensor::Shape{1, height, width, channels_preserved});
    auto cpu_view = output_tensors->back().GetCpuWriteView();

    // Copy image data into tensor.
    if (image_frame.ByteDepth() == 1) {
      MP_RETURN_IF_ERROR(NormalizeImage<uint8>(image_frame, flip_vertically_,
                                               cpu_view.buffer<float>()));
    } else if (image_frame.ByteDepth() == 4) {
      MP_RETURN_IF_ERROR(NormalizeImage<float>(image_frame, flip_vertically_,
                                               cpu_view.buffer<float>()));
    } else {
      return absl::InternalError(
          "Only byte-based (8 bit) and float (32 bit) images supported.");
    }
  } else if (cc->Inputs().HasTag(kMatrixTag)) {
    if (cc->Inputs().Tag(kMatrixTag).IsEmpty()) {
      return absl::OkStatus();
    }
    const auto& matrix = cc->Inputs().Tag(kMatrixTag).Get<Matrix>();
    const int height = matrix.rows();
    const int width = matrix.cols();
    const int channels = 1;
    output_tensors->emplace_back(Tensor::ElementType::kFloat32,
                                 Tensor::Shape{1, height, width, channels});
    MP_RETURN_IF_ERROR(CopyMatrixToTensor(
        matrix, output_tensors->back().GetCpuWriteView().buffer<float>()));
  } else {
    return absl::OkStatus();
  }
  cc->Outputs()
      .Tag(kTensorsTag)
      .Add(output_tensors.release(), cc->InputTimestamp());

  return absl::OkStatus();
}

absl::Status TensorConverterCalculator::ProcessGPU(CalculatorContext* cc) {
#if !MEDIAPIPE_DISABLE_GPU
  if (!initialized_) {
    MP_RETURN_IF_ERROR(InitGpu(cc));
    initialized_ = true;
  }
  const auto& input =
      cc->Inputs().Tag(kGpuBufferTag).Get<mediapipe::GpuBuffer>();
  int width = input.width();
  int height = input.height();
  int channels = max_num_channels_;
  auto output_tensors = absl::make_unique<std::vector<Tensor>>();
  output_tensors->emplace_back(Tensor::ElementType::kFloat32,
                               Tensor::Shape{1, height, width, channels});
#if MEDIAPIPE_METAL_ENABLED
  id<MTLDevice> device = gpu_helper_.mtlDevice;
  id<MTLCommandBuffer> command_buffer = [gpu_helper_ commandBuffer];
  command_buffer.label = @"TensorConverterCalculatorConvert";
  id<MTLComputeCommandEncoder> compute_encoder =
      [command_buffer computeCommandEncoder];
  [compute_encoder setComputePipelineState:to_buffer_program_];
  id<MTLTexture> src_texture = [gpu_helper_ metalTextureWithGpuBuffer:input];
  [compute_encoder setTexture:src_texture atIndex:0];
  auto output_view =
      output_tensors->at(0).GetMtlBufferWriteView(command_buffer);
  [compute_encoder setBuffer:output_view.buffer() offset:0 atIndex:1];
  MTLSize threads_per_group = MTLSizeMake(kWorkgroupSize, kWorkgroupSize, 1);
  MTLSize threadgroups =
      MTLSizeMake(NumGroups(input.width(), kWorkgroupSize),
                  NumGroups(input.height(), kWorkgroupSize), 1);
  [compute_encoder dispatchThreadgroups:threadgroups
                  threadsPerThreadgroup:threads_per_group];
  [compute_encoder endEncoding];
  [command_buffer commit];
#elif MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
  MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext(
      [this, &output_tensors, &input]() -> absl::Status {
        auto src = gpu_helper_.CreateSourceTexture(input);
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
        // Convert GL texture into SSBO.
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, src.name());
        auto output_view = output_tensors->back().GetOpenGlBufferWriteView();
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, output_view.name());
        glUseProgram(to_buffer_program_);
        glDispatchCompute(NumGroups(input.width(), kWorkgroupSize),
                          NumGroups(input.height(), kWorkgroupSize), 1);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
#else
        // Texture2D -> Texture2D with OpenGL ES 3.0.
        glUseProgram(to_tex2d_program_);
        glDisable(GL_DEPTH_TEST);
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);
        glViewport(0, 0, src.width(), src.height());
        glActiveTexture(GL_TEXTURE0);
        auto output_view = output_tensors->back().GetOpenGlTexture2dWriteView();
        glBindTexture(GL_TEXTURE_2D, output_view.name());
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, output_view.name(), 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(src.target(), src.name());
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
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
        src.Release();
        return absl::OkStatus();
      }));
#endif  // MEDIAPIPE_METAL_ENABLED
  cc->Outputs()
      .Tag(kTensorsTag)
      .Add(output_tensors.release(), cc->InputTimestamp());
#else
  RET_CHECK_FAIL() << "GPU processing is not enabled.";
#endif  // !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
}

absl::Status TensorConverterCalculator::InitGpu(CalculatorContext* cc) {
#if !MEDIAPIPE_DISABLE_GPU
  // Get input image sizes.
  const auto& input =
      cc->Inputs().Tag(kGpuBufferTag).Get<mediapipe::GpuBuffer>();
  mediapipe::ImageFormat::Format format =
      mediapipe::ImageFormatForGpuBufferFormat(input.format());
  const bool include_alpha = (max_num_channels_ == 4);
  const bool single_channel = (max_num_channels_ == 1);
  if (!(format == mediapipe::ImageFormat::GRAY8 ||
        format == mediapipe::ImageFormat::SRGB ||
        format == mediapipe::ImageFormat::SRGBA))
    RET_CHECK_FAIL() << "Unsupported GPU input format.";
  if (include_alpha && (format != mediapipe::ImageFormat::SRGBA))
    RET_CHECK_FAIL() << "Num input channels is less than desired output.";

#if MEDIAPIPE_METAL_ENABLED
  id<MTLDevice> device = gpu_helper_.mtlDevice;
  // Shader to convert GL Texture to Metal Buffer,
  // with normalization to either: [0,1] or [-1,1].
  const std::string shader_source = absl::Substitute(
      R"(
  #include <metal_stdlib>

  using namespace metal;

  kernel void convertKernel(
      texture2d<half, access::sample> in_tex  [[ texture(0) ]],
      device float*                   out_buf [[ buffer(1) ]],
      uint2                           gid     [[ thread_position_in_grid ]]) {
    if (gid.x >= in_tex.get_width() || gid.y >= in_tex.get_height()) return;
    constexpr sampler texture_sampler(coord::pixel, address::clamp_to_edge);
    const float2 coord = float2(gid.x, gid.y);
    half4 pixel = in_tex.sample(texture_sampler, coord);
    $0   // normalize [-1,1]
    const int linear_index = $1 * ($2 * in_tex.get_width() + gid.x);
    out_buf[linear_index + 0] = pixel.x;
    $3  // g & b channels
    $4  // alpha channel
  }
      )",
      /*$0=*/
      output_range_.has_value()
          ? absl::Substitute("pixel = pixel * half($0) + half($1);",
                             (output_range_->second - output_range_->first),
                             output_range_->first)
          : "",
      /*$1=*/max_num_channels_,
      /*$2=*/flip_vertically_ ? "(in_tex.get_height() - 1 - gid.y)" : "gid.y",
      /*$3=*/
      single_channel ? "" : R"(out_buf[linear_index + 1] = pixel.y;
               out_buf[linear_index + 2] = pixel.z;)",
      /*$4=*/include_alpha ? "out_buf[linear_index + 3] = pixel.w;" : "");

  NSString* library_source =
      [NSString stringWithUTF8String:shader_source.c_str()];
  NSError* error = nil;
  id<MTLLibrary> library =
      [device newLibraryWithSource:library_source options:nullptr error:&error];
  RET_CHECK(library != nil) << "Couldn't create shader library "
                            << [[error localizedDescription] UTF8String];
  id<MTLFunction> kernel_func = nil;
  kernel_func = [library newFunctionWithName:@"convertKernel"];
  RET_CHECK(kernel_func != nil) << "Couldn't create kernel function.";
  to_buffer_program_ =
      [device newComputePipelineStateWithFunction:kernel_func error:&error];
  RET_CHECK(to_buffer_program_ != nil) << "Couldn't create pipeline state " <<
      [[error localizedDescription] UTF8String];
#elif MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
  MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([this, &include_alpha,
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
                                                 &input,
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
                                                 &single_channel]()
                                                    -> absl::Status {
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
    // Shader to convert GL Texture to Shader Storage Buffer Object (SSBO),
    // with normalization to either: [0,1] or [-1,1].
    const std::string shader_source = absl::Substitute(
        R"( #version 310 es
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
          })",
        /*$0=*/kWorkgroupSize, /*$1=*/input.width(), /*$2=*/input.height(),
        /*$3=*/
        output_range_.has_value()
            ? absl::Substitute("pixel = pixel * float($0) + float($1);",
                               (output_range_->second - output_range_->first),
                               output_range_->first)
            : "",
        /*$4=*/flip_vertically_ ? "(width_height.y - 1 - gid.y)" : "gid.y",
        /*$5=*/
        single_channel ? ""
                       : R"(output_data.elements[linear_index + 1] = pixel.y;
                            output_data.elements[linear_index + 2] = pixel.z;)",
        /*$6=*/
        include_alpha ? "output_data.elements[linear_index + 3] = pixel.w;"
                      : "",
        /*$7=*/max_num_channels_);
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    const GLchar* sources[] = {shader_source.c_str()};
    glShaderSource(shader, 1, sources, NULL);
    glCompileShader(shader);
    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    RET_CHECK(compiled == GL_TRUE);
    to_buffer_program_ = glCreateProgram();
    glAttachShader(to_buffer_program_, shader);
    glDeleteShader(shader);
    glLinkProgram(to_buffer_program_);
#else
    // OpenGL ES 3.0 fragment shader Texture2d -> Texture2d conversion.
    const std::string shader_source = absl::Substitute(
        R"(
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
          })",
        /*$0=*/single_channel ? "vec1" : "vec4",
        /*$1=*/
        flip_vertically_
            ? "vec2(sample_coordinate.x, 1.0 - sample_coordinate.y);"
            : "sample_coordinate;",
        /*$2=*/output_range_.has_value()
            ? absl::Substitute("pixel = pixel * float($0) + float($1);",
                               (output_range_->second - output_range_->first),
                               output_range_->first)
            : "",
        /*$3=*/single_channel ? "" : R"(fragColor.g = pixel.g;
                            fragColor.b = pixel.b;)",
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

#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
    return absl::OkStatus();
  }));
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
#endif  // !MEDIAPIPE_DISABLE_GPU
  return absl::OkStatus();
}

absl::Status TensorConverterCalculator::LoadOptions(CalculatorContext* cc) {
  // Get calculator options specified in the graph.
  const auto& options =
      cc->Options<::mediapipe::TensorConverterCalculatorOptions>();

  // if zero_center, set output float range to match [-1, 1] as specified in
  // calculator proto.
  if (options.zero_center()) {
    output_range_.emplace(std::pair<float, float>(-1.0, 1.0));
  }

  // Custom output_tensor_float_range values.
  // If the float range is specified in pb text, use the specified values
  // instead.
  if (options.has_output_tensor_float_range()) {
    output_range_.emplace(options.output_tensor_float_range().min(),
                          options.output_tensor_float_range().max());
    CHECK_GT(output_range_->second, output_range_->first);
  }

  // Custom div and sub values.
  if (options.use_custom_normalization()) {
    output_range_.emplace(std::pair<float, float>(
        -options.custom_sub(),
        -options.custom_sub() + 255.0 / options.custom_div()));
  }

  // Get y-flip mode.
  flip_vertically_ = options.flip_vertically();

  // Get row_major_matrix mode.
  row_major_matrix_ = options.row_major_matrix();

  // Get desired way to handle input channels.
  max_num_channels_ = options.max_num_channels();
  CHECK_GE(max_num_channels_, 1);
  CHECK_LE(max_num_channels_, 4);
  CHECK_NE(max_num_channels_, 2);
  return absl::OkStatus();
}

template <class T>
absl::Status TensorConverterCalculator::NormalizeImage(
    const ImageFrame& image_frame, bool flip_vertically, float* tensor_ptr) {
  const int height = image_frame.Height();
  const int width = image_frame.Width();
  const int channels = image_frame.NumberOfChannels();
  const int channels_preserved = std::min(channels, max_num_channels_);
  const int channels_ignored = channels - channels_preserved;

  if (output_range_.has_value()) {
    // If the output float range is set and we are not using custom
    // normalization, normalize the pixel values from [0, 255] to the specified
    // output range.
    RET_CHECK_NE(output_range_->first, output_range_->second);
    const float scale = (output_range_->second - output_range_->first) / 255.0f;
    const float bias = output_range_->first;

    for (int i = 0; i < height; ++i) {
      const T* image_ptr = reinterpret_cast<const T*>(
          image_frame.PixelData() +
          (flip_vertically ? height - 1 - i : i) * image_frame.WidthStep());
      for (int j = 0; j < width; ++j) {
        for (int c = 0; c < channels_preserved; ++c) {
          *tensor_ptr++ = *image_ptr++ * scale + bias;
        }
        image_ptr += channels_ignored;
      }
    }
  } else {
    // [0,1], scale only (bias == 0)
    // Verified that there are no precision issues with 1.0f / 255.0f expression
    const float scale = 1.0f / 255.0f;
    for (int i = 0; i < height; ++i) {
      const T* image_ptr = reinterpret_cast<const T*>(
          image_frame.PixelData() +
          (flip_vertically ? height - 1 - i : i) * image_frame.WidthStep());
      for (int j = 0; j < width; ++j) {
        for (int c = 0; c < channels_preserved; ++c) {
          *tensor_ptr++ = *image_ptr++ * scale;
        }
        image_ptr += channels_ignored;
      }
    }
  }

  return absl::OkStatus();
}

absl::Status TensorConverterCalculator::CopyMatrixToTensor(const Matrix& matrix,
                                                           float* tensor_ptr) {
  if (row_major_matrix_) {
    auto matrix_map =
        Eigen::Map<RowMajorMatrixXf>(tensor_ptr, matrix.rows(), matrix.cols());
    matrix_map = matrix;
  } else {
    auto matrix_map =
        Eigen::Map<ColMajorMatrixXf>(tensor_ptr, matrix.rows(), matrix.cols());
    matrix_map = matrix;
  }

  return absl::OkStatus();
}

}  // namespace mediapipe
