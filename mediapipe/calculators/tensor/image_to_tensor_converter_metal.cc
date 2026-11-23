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

#include "mediapipe/calculators/tensor/image_to_tensor_converter_metal.h"

#if MEDIAPIPE_METAL_ENABLED

#import <Metal/Metal.h>
#include <TargetConditionals.h>

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
#include "absl/strings/string_view.h"   // from @com_google_absl
#include "mediapipe/calculators/tensor/image_to_tensor_converter.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/gpu/MPPMetalHelper.h"
#include "tflite/delegates/gpu/common/shape.h"  // from @litert
#include "tflite/delegates/gpu/common/types.h"  // from @litert

#if MEDIAPIPE_METAL_ENABLED
#include "mediapipe/framework/formats/tensor_mtl_buffer_view.h"
#endif  // MEDIAPIPE_METAL_ENABLED

namespace mediapipe {

namespace {

// clang-format off
// a square formed by 2 triangles
const float kBasicSquareVertices[] = {
    -1, 1,  0, 1,
    1,  1,  0, 1,
    1,  -1, 0, 1,
    -1, 1,  0, 1,
    1,  -1, 0, 1,
    -1, -1, 0, 1,
};

// maps a texture to kBasicSquareVertices via aspect fill
const float kBasicTextureVertices[] = {
    0, 0, 0, 1,
    1, 0, 0, 1,
    1, 1, 0, 1,
    0, 0, 0, 1,
    1, 1, 0, 1,
    0, 1, 0, 1,
};
// clang-format on

constexpr char kShaderLibHeader[] = R"(
  #include <metal_stdlib>

  using namespace metal;

  struct TextureVertex
  {
    float4 position [[position]];
    float2 uv;
  };
)";

constexpr char kVertexShader[] = R"(
  vertex TextureVertex vertexShader(
      constant float4 *position [[buffer(0)]],
      device float4* tex_coords [[buffer(1)]],
      constant float4x4& transform_matrix [[buffer(2)]],
      uint vid [[vertex_id]]) {
    TextureVertex vert;
    vert.position = position[vid];
    vert.uv = (tex_coords[vid] * transform_matrix).xy;
    return vert;
  }
)";

constexpr char kFragmentShader[] = R"(
  #ifdef OUTPUT_F16C4
  #define Type4 half4
  #define Type half
  #endif  // OUTPUT_F16C4

  #ifdef OUTPUT_F32C4
  #define Type4 float4
  #define Type float
  #endif  // OUTPUT_F32C4

  // 8-bit (uint8 / int8) output via a normalized RGBA8 render target.
  #ifdef OUTPUT_U8C4
  #define Type4 float4
  #define Type float
  #endif  // OUTPUT_U8C4

  fragment Type4 fragmentShader(TextureVertex vertex_output [[stage_in]],
                                  texture2d<Type> texture [[texture(0)]],
                                  constant float* parameters [[buffer(1)]])
  {
    const float alpha = parameters[0];
    const float beta = parameters[1];

    #ifdef CLAMP_TO_ZERO
    constexpr sampler linear_sampler(address::clamp_to_zero, min_filter::linear,
      mag_filter::linear);
    #endif  // CLAMP_TO_ZERO

    #ifdef CLAMP_TO_EDGE
    constexpr sampler linear_sampler(address::clamp_to_edge, min_filter::linear,
      mag_filter::linear);
    #endif  // CLAMP_TO_EDGE

    Type4 texture_pixel = texture.sample(linear_sampler, vertex_output.uv);
    Type4 value = Type4(alpha * texture_pixel.rgb + beta, 0);
    #ifdef OUTPUT_U8C4
    // Round, clamp and store the low byte (two's complement for int8) / 255.
    value = clamp(floor(value + 0.5), Type4(OUTPUT_MIN), Type4(OUTPUT_MAX));
    value = (value - 256.0 * floor(value / 256.0)) / 255.0;
    #endif  // OUTPUT_U8C4
    return value;
  }
)";

// Writes a packed 3-channel (H x W x 3) tensor directly into the output
// buffer, as float32 or (with OUTPUT_BYTES) 8-bit values. Unlike the render
// pipeline above, which writes 4 channels per pixel, this needs no linear
// texture aliasing the buffer, and hence no alignment fallback.
constexpr char kPackedKernel[] = R"(
  #ifdef OUTPUT_BYTES
  #define OutputType uchar
  #else
  #define OutputType float
  #endif  // OUTPUT_BYTES

  kernel void extractSubRectPacked(
      texture2d<float> texture [[texture(0)]],
      device OutputType* output [[buffer(0)]],
      constant float4x4& transform_matrix [[buffer(1)]],
      constant float* parameters [[buffer(2)]],
      uint2 gid [[thread_position_in_grid]]) {
    const float alpha = parameters[0];
    const float beta = parameters[1];
    const int out_width = int(parameters[2]);
    const int out_height = int(parameters[3]);
    if (int(gid.x) >= out_width || int(gid.y) >= out_height) {
      return;
    }

    #ifdef CLAMP_TO_ZERO
    constexpr sampler linear_sampler(address::clamp_to_zero, min_filter::linear,
      mag_filter::linear);
    #endif  // CLAMP_TO_ZERO

    #ifdef CLAMP_TO_EDGE
    constexpr sampler linear_sampler(address::clamp_to_edge, min_filter::linear,
      mag_filter::linear);
    #endif  // CLAMP_TO_EDGE

    // Same mapping as the vertex shader: output pixel center -> ROI -> input.
    const float2 uv = (float2(gid) + 0.5) / float2(out_width, out_height);
    const float2 coord = (float4(uv, 0.0, 1.0) * transform_matrix).xy;
    float3 value = alpha * texture.sample(linear_sampler, coord).rgb + beta;
    #ifdef OUTPUT_BYTES
    // Round and clamp the value to the range of the output type, and truncate
    // it to its low byte (which is the two's complement representation for
    // int8).
    value = clamp(floor(value + 0.5), float3(OUTPUT_MIN), float3(OUTPUT_MAX));
    value = value - 256.0 * floor(value / 256.0);
    #endif  // OUTPUT_BYTES
    const int index = 3 * (int(gid.y) * out_width + int(gid.x));
    output[index] = OutputType(value.r);
    output[index + 1] = OutputType(value.g);
    output[index + 2] = OutputType(value.b);
  }
)";

// kF16C4 / kF32C4 / kU8C4: 4 channels per pixel, rendered into a texture
// aliasing (or blitted into) the output buffer. kF32C3 / kU8C3: 3 channels
// per pixel, written by a compute kernel.
enum class OutputFormat { kF16C4, kF32C4, kU8C4, kF32C3, kU8C3 };

bool IsPacked(OutputFormat output_format) {
  return output_format == OutputFormat::kF32C3 ||
         output_format == OutputFormat::kU8C3;
}

MTLPixelFormat GetPixelFormat(OutputFormat output_format) {
  switch (output_format) {
    case OutputFormat::kF16C4:
      return MTLPixelFormatRGBA16Float;
    case OutputFormat::kF32C4:
      return MTLPixelFormatRGBA32Float;
    case OutputFormat::kU8C4:
      return MTLPixelFormatRGBA8Unorm;
    case OutputFormat::kF32C3:
    case OutputFormat::kU8C3:
      return MTLPixelFormatInvalid;  // Not rendered into a texture.
  }
}
int GetBytesPerRaw(OutputFormat output_format, const tflite::gpu::HW& size) {
  std::size_t type_size;
  switch (output_format) {
    case OutputFormat::kF16C4:
      type_size = sizeof(tflite::gpu::HalfBits);
      break;
    case OutputFormat::kF32C4:
      type_size = sizeof(float);
      break;
    case OutputFormat::kU8C4:
      type_size = sizeof(uint8_t);
      break;
    case OutputFormat::kF32C3:
    case OutputFormat::kU8C3:
      ABSL_LOG(FATAL) << "Packed output formats have no texture rows.";
  }
  constexpr int kNumChannels = 4;
  return size.w * kNumChannels * type_size;
}

class SubRectExtractorMetal {
 public:
  // `output_signed`: For kU8C4 / kU8C3, whether the bytes are int8 instead of
  // uint8.
  static absl::StatusOr<std::unique_ptr<SubRectExtractorMetal>> Make(
      id<MTLDevice> device, OutputFormat output_format, bool output_signed,
      BorderMode border_mode) {
    if (IsPacked(output_format)) {
      id<MTLComputePipelineState> compute_pipeline_state;
      ABSL_RETURN_IF_ERROR(SubRectExtractorMetal::MakeComputePipelineState(
          device, output_format, output_signed, border_mode,
          &compute_pipeline_state));
      return absl::make_unique<SubRectExtractorMetal>(
          device, /*pipeline_state=*/nil, compute_pipeline_state,
          output_format);
    }
    id<MTLRenderPipelineState> pipeline_state;
    ABSL_RETURN_IF_ERROR(SubRectExtractorMetal::MakePipelineState(
        device, output_format, output_signed, border_mode, &pipeline_state));
    return absl::make_unique<SubRectExtractorMetal>(
        device, pipeline_state, /*compute_pipeline_state=*/nil, output_format);
  }

  SubRectExtractorMetal(id<MTLDevice> device,
                        id<MTLRenderPipelineState> pipeline_state,
                        id<MTLComputePipelineState> compute_pipeline_state,
                        OutputFormat output_format)
      : device_(device),
        pipeline_state_(pipeline_state),
        compute_pipeline_state_(compute_pipeline_state),
        output_format_(output_format) {
    if (!IsPacked(output_format_)) {
      positions_buffer_ =
          [device_ newBufferWithBytes:kBasicSquareVertices
                               length:sizeof(kBasicSquareVertices)
                              options:MTLResourceOptionCPUCacheModeDefault];

      tex_coords_buffer_ =
          [device_ newBufferWithBytes:kBasicTextureVertices
                               length:sizeof(kBasicTextureVertices)
                              options:MTLResourceOptionCPUCacheModeDefault];
    }
  }

  absl::Status Execute(id<MTLTexture> input_texture,
                       const RotatedRect& sub_rect, bool flip_horizontally,
                       float alpha, float beta,
                       const tflite::gpu::HW& destination_size,
                       id<MTLCommandBuffer> command_buffer,
                       id<MTLBuffer> destination) {
    if (IsPacked(output_format_)) {
      return InternalExecutePacked(input_texture, sub_rect, flip_horizontally,
                                   alpha, beta, destination_size,
                                   command_buffer, destination);
    }
#if !TARGET_OS_SIMULATOR
    id<MTLTexture> output_texture =
        MTLTextureWithBuffer(destination_size, destination);
    if (output_texture != nil) {
      return InternalExecute(input_texture, sub_rect, flip_horizontally, alpha,
                             beta, destination_size, command_buffer,
                             output_texture);
    }
    const NSUInteger bytes_per_row =
        GetBytesPerRaw(output_format_, destination_size);
    const NSUInteger alignment = [device_
        minimumLinearTextureAlignmentForPixelFormat:GetPixelFormat(
                                                        output_format_)];
    ABSL_LOG_FIRST_N(WARNING, 1)
        << "Cannot render directly into the output buffer: rows of "
        << bytes_per_row << " bytes (" << destination_size.w << "x"
        << destination_size.h << " pixels, "
        << bytes_per_row / destination_size.w
        << " bytes per pixel) with a linear texture alignment of " << alignment
        << " bytes. Falling back to rendering into a separate texture and "
           "blitting it into the output buffer, which is slower.";
#endif  // !TARGET_OS_SIMULATOR
    return InternalExecuteViaBlit(input_texture, sub_rect, flip_horizontally,
                                  alpha, beta, destination_size, command_buffer,
                                  destination);
  }

 private:
#if !TARGET_OS_SIMULATOR
  id<MTLTexture> MTLTextureWithBuffer(const tflite::gpu::HW& size,
                                      id<MTLBuffer> buffer) {
    MTLTextureDescriptor* texture_desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:GetPixelFormat(output_format_)
                                     width:size.w
                                    height:size.h
                                 mipmapped:NO];
    texture_desc.usage = MTLTextureUsageRenderTarget;
    texture_desc.resourceOptions = buffer.resourceOptions;

    NSUInteger output_bytes_per_row = GetBytesPerRaw(output_format_, size);
    NSUInteger alignment = [device_
        minimumLinearTextureAlignmentForPixelFormat:GetPixelFormat(
                                                        output_format_)];
    if (alignment > 0 && output_bytes_per_row % alignment != 0) {
      return nil;
    }

    id<MTLTexture> texture =
        [buffer newTextureWithDescriptor:texture_desc
                                  offset:0
                             bytesPerRow:output_bytes_per_row];
    return texture;
  }
#endif  // !TARGET_OS_SIMULATOR

  absl::Status InternalExecute(id<MTLTexture> input_texture,
                               const RotatedRect& sub_rect,
                               bool flip_horizontally, float alpha, float beta,
                               const tflite::gpu::HW& destination_size,
                               id<MTLCommandBuffer> command_buffer,
                               id<MTLTexture> output_texture) {
    RET_CHECK(command_buffer != nil);
    RET_CHECK(output_texture != nil);

    // Obtain texture mapping coordinates transformation matrix and copy its
    // data to the buffer.
    std::array<float, 16> transform_mat;
    GetRotatedSubRectToRectTransformMatrix(sub_rect, input_texture.width,
                                           input_texture.height,
                                           flip_horizontally, &transform_mat);
    id<MTLBuffer> transform_mat_buffer =
        [device_ newBufferWithBytes:&transform_mat
                             length:sizeof(transform_mat)
                            options:MTLResourceOptionCPUCacheModeDefault];

    // Create parameters wrapper.
    float parameters[] = {alpha, beta};

    // Now everything is ready to go!
    // Setup render pass.
    MTLRenderPassDescriptor* render_pass_desc =
        [MTLRenderPassDescriptor renderPassDescriptor];
    render_pass_desc.colorAttachments[0].texture = output_texture;
    render_pass_desc.colorAttachments[0].storeAction = MTLStoreActionStore;
    render_pass_desc.colorAttachments[0].loadAction = MTLLoadActionClear;

    // Setup render command encoder.
    id<MTLRenderCommandEncoder> command_encoder =
        [command_buffer renderCommandEncoderWithDescriptor:render_pass_desc];
    [command_encoder setRenderPipelineState:pipeline_state_];
    [command_encoder setVertexBuffer:positions_buffer_ offset:0 atIndex:0];
    [command_encoder setVertexBuffer:tex_coords_buffer_ offset:0 atIndex:1];
    [command_encoder setVertexBuffer:transform_mat_buffer offset:0 atIndex:2];
    [command_encoder setFragmentTexture:input_texture atIndex:0];
    [command_encoder setFragmentBytes:&parameters
                               length:sizeof(parameters)
                              atIndex:1];

    [command_encoder drawPrimitives:MTLPrimitiveTypeTriangle
                        vertexStart:0
                        vertexCount:6];
    [command_encoder endEncoding];

    return absl::OkStatus();
  }

  // Fallback for simulator (no buffer-backed linear textures) and unaligned
  // rows.
  absl::Status InternalExecuteViaBlit(id<MTLTexture> input_texture,
                                      const RotatedRect& sub_rect,
                                      bool flip_horizontally, float alpha,
                                      float beta,
                                      const tflite::gpu::HW& destination_size,
                                      id<MTLCommandBuffer> command_buffer,
                                      id<MTLBuffer> destination) {
    MTLTextureDescriptor* texture_desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:GetPixelFormat(output_format_)
                                     width:destination_size.w
                                    height:destination_size.h
                                 mipmapped:NO];
    texture_desc.usage = MTLTextureUsageRenderTarget;
    texture_desc.storageMode = MTLStorageModePrivate;
    id<MTLTexture> output_texture =
        [device_ newTextureWithDescriptor:texture_desc];
    ABSL_RETURN_IF_ERROR(
        InternalExecute(input_texture, sub_rect, flip_horizontally, alpha, beta,
                        destination_size, command_buffer, output_texture));
    id<MTLBlitCommandEncoder> blit_encoder =
        [command_buffer blitCommandEncoder];
    NSUInteger bytes_per_row = GetBytesPerRaw(output_format_, destination_size);
    [blit_encoder copyFromTexture:output_texture
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(destination_size.w,
                                              destination_size.h, 1)
                         toBuffer:destination
                destinationOffset:0
           destinationBytesPerRow:bytes_per_row
         destinationBytesPerImage:bytes_per_row * destination_size.h];
    [blit_encoder endEncoding];
    return absl::OkStatus();
  }

  absl::Status InternalExecutePacked(id<MTLTexture> input_texture,
                                     const RotatedRect& sub_rect,
                                     bool flip_horizontally, float alpha,
                                     float beta,
                                     const tflite::gpu::HW& destination_size,
                                     id<MTLCommandBuffer> command_buffer,
                                     id<MTLBuffer> destination) {
    RET_CHECK(command_buffer != nil);
    RET_CHECK(destination != nil);
    RET_CHECK_GT(destination_size.w, 0);
    RET_CHECK_GT(destination_size.h, 0);

    std::array<float, 16> transform_mat;
    GetRotatedSubRectToRectTransformMatrix(sub_rect, input_texture.width,
                                           input_texture.height,
                                           flip_horizontally, &transform_mat);
    const float parameters[] = {alpha, beta,
                                static_cast<float>(destination_size.w),
                                static_cast<float>(destination_size.h)};

    id<MTLComputeCommandEncoder> command_encoder =
        [command_buffer computeCommandEncoder];
    RET_CHECK(command_encoder != nil);
    [command_encoder setComputePipelineState:compute_pipeline_state_];
    [command_encoder setTexture:input_texture atIndex:0];
    [command_encoder setBuffer:destination offset:0 atIndex:0];
    [command_encoder setBytes:transform_mat.data()
                       length:sizeof(transform_mat)
                      atIndex:1];
    [command_encoder setBytes:parameters length:sizeof(parameters) atIndex:2];

    // The kernel bounds-checks, so round the grid up to whole threadgroups
    // (dispatchThreads would require non-uniform threadgroup support).
    const NSUInteger group_width = std::clamp<NSUInteger>(
        compute_pipeline_state_.threadExecutionWidth, 1, 16);
    const NSUInteger group_height = std::max<NSUInteger>(
        1, compute_pipeline_state_.maxTotalThreadsPerThreadgroup / group_width);
    const MTLSize threads_per_group =
        MTLSizeMake(group_width, std::min<NSUInteger>(group_height, 16), 1);
    const MTLSize groups =
        MTLSizeMake((destination_size.w + threads_per_group.width - 1) /
                        threads_per_group.width,
                    (destination_size.h + threads_per_group.height - 1) /
                        threads_per_group.height,
                    1);
    [command_encoder dispatchThreadgroups:groups
                    threadsPerThreadgroup:threads_per_group];
    [command_encoder endEncoding];
    return absl::OkStatus();
  }

  static absl::string_view GetClampDef(BorderMode border_mode) {
    switch (border_mode) {
      case BorderMode::kReplicate:
        return R"(
          #define CLAMP_TO_EDGE
        )";
      case BorderMode::kZero:
        return R"(
          #define CLAMP_TO_ZERO
        )";
    }
  }

  // Defines OUTPUT_MIN / OUTPUT_MAX, the range of the 8-bit output values.
  static absl::string_view GetOutputRangeDef(bool output_signed) {
    return output_signed ? R"(
          #define OUTPUT_MIN -128.0
          #define OUTPUT_MAX 127.0
        )"
                         : R"(
          #define OUTPUT_MIN 0.0
          #define OUTPUT_MAX 255.0
        )";
  }

  static absl::Status MakeComputePipelineState(
      id<MTLDevice> device, OutputFormat output_format, bool output_signed,
      BorderMode border_mode, id<MTLComputePipelineState>* pipeline_state) {
    RET_CHECK(pipeline_state != nil);
    std::string output_bytes_def;
    if (output_format == OutputFormat::kU8C3) {
      output_bytes_def = absl::StrCat(R"(
        #define OUTPUT_BYTES
      )",
                                      GetOutputRangeDef(output_signed));
    }
    const std::string shader_lib =
        absl::StrCat(kShaderLibHeader, output_bytes_def,
                     GetClampDef(border_mode), kPackedKernel);
    NSString* library_source =
        [[NSString alloc] initWithBytes:shader_lib.data()
                                 length:shader_lib.size()
                               encoding:NSUTF8StringEncoding];
    NSError* error = nil;
    id<MTLLibrary> library =
        [device newLibraryWithSource:library_source options:nil error:&error];
    RET_CHECK(library != nil)
        << "Couldn't create a shader library: "
        << (error ? [[error localizedDescription] UTF8String] : "unknown");
    id<MTLFunction> kernel_function =
        [library newFunctionWithName:@"extractSubRectPacked"];
    RET_CHECK(kernel_function != nil)
        << "Failed creating a new kernel function!";
    error = nil;
    *pipeline_state =
        [device newComputePipelineStateWithFunction:kernel_function
                                              error:&error];
    RET_CHECK(*pipeline_state != nil)
        << "Couldn't create a compute pipeline state: "
        << (error ? [[error localizedDescription] UTF8String] : "unknown");
    return absl::OkStatus();
  }

  static absl::Status MakePipelineState(
      id<MTLDevice> device, OutputFormat output_format, bool output_signed,
      BorderMode border_mode, id<MTLRenderPipelineState>* pipeline_state) {
    RET_CHECK(pipeline_state != nil);

    std::string output_type_def;
    switch (output_format) {
      case OutputFormat::kF16C4:
        output_type_def = R"(
          #define OUTPUT_F16C4
        )";
        break;
      case OutputFormat::kF32C4:
        output_type_def = R"(
          #define OUTPUT_F32C4
        )";
        break;
      case OutputFormat::kU8C4:
        output_type_def = absl::StrCat(R"(
          #define OUTPUT_U8C4
        )",
                                       GetOutputRangeDef(output_signed));
        break;
      case OutputFormat::kF32C3:
      case OutputFormat::kU8C3:
        return absl::InternalError("Packed formats use the compute pipeline.");
    }

    std::string shader_lib =
        absl::StrCat(kShaderLibHeader, output_type_def,
                     GetClampDef(border_mode), kVertexShader, kFragmentShader);
    NSError* error = nil;
    NSString* library_source =
        [NSString stringWithUTF8String:shader_lib.c_str()];

    id<MTLLibrary> library =
        [device newLibraryWithSource:library_source options:nil error:&error];
    RET_CHECK(library != nil) << "Couldn't create a shader library"
                              << [[error localizedDescription] UTF8String];

    id<MTLFunction> vertex_function =
        [library newFunctionWithName:@"vertexShader"];
    RET_CHECK(vertex_function != nil)
        << "Failed creating a new vertex function!";

    id<MTLFunction> fragment_function =
        [library newFunctionWithName:@"fragmentShader"];
    RET_CHECK(fragment_function != nil)
        << "Failed creating a new fragment function!";

    MTLRenderPipelineDescriptor* pipelineDescriptor =
        [MTLRenderPipelineDescriptor new];
    pipelineDescriptor.vertexFunction = vertex_function;
    pipelineDescriptor.fragmentFunction = fragment_function;
    pipelineDescriptor.colorAttachments[0].pixelFormat =
        GetPixelFormat(output_format);

    *pipeline_state =
        [device newRenderPipelineStateWithDescriptor:pipelineDescriptor
                                               error:&error];
    RET_CHECK(error == nil) << "Couldn't create a pipeline state"
                            << [[error localizedDescription] UTF8String];

    return absl::OkStatus();
  }

  id<MTLBuffer> positions_buffer_ = nil;
  id<MTLBuffer> tex_coords_buffer_ = nil;
  id<MTLDevice> device_;
  // Exactly one of these is set, depending on IsPacked(output_format_).
  id<MTLRenderPipelineState> pipeline_state_;
  id<MTLComputePipelineState> compute_pipeline_state_;
  OutputFormat output_format_;
};

class ImageToTensorMetalConverter : public ImageToTensorConverter {
 public:
  absl::Status Init(CalculatorContext* cc, BorderMode border_mode,
                    Tensor::ElementType tensor_type, int num_channels) {
    RET_CHECK(num_channels == 3 || num_channels == 4)
        << "Unsupported number of output channels: " << num_channels;
    bool output_bytes;
    switch (tensor_type) {
      case Tensor::ElementType::kFloat32:
        output_bytes = false;
        break;
      case Tensor::ElementType::kUInt8:
      case Tensor::ElementType::kInt8:
        output_bytes = true;
        break;
      default:
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported output tensor type: ",
                         Tensor::ElementTypeName(tensor_type)));
    }
    const OutputFormat output_format =
        num_channels == 3
            ? (output_bytes ? OutputFormat::kU8C3 : OutputFormat::kF32C3)
            : (output_bytes ? OutputFormat::kU8C4 : OutputFormat::kF32C4);
    tensor_type_ = tensor_type;
    num_channels_ = num_channels;
    @autoreleasepool {
      metal_helper_ = [[MPPMetalHelper alloc] initWithCalculatorContext:cc];
      RET_CHECK(metal_helper_);
      ABSL_ASSIGN_OR_RETURN(
          extractor_,
          SubRectExtractorMetal::Make(
              metal_helper_.mtlDevice, output_format,
              /*output_signed=*/tensor_type == Tensor::ElementType::kInt8,
              border_mode));
    }
    return absl::OkStatus();
  }

  absl::Status Convert(const mediapipe::Image& input, const RotatedRect& roi,
                       float range_min, float range_max,
                       int tensor_buffer_offset,
                       Tensor& output_tensor) override {
    if (input.format() != mediapipe::GpuBufferFormat::kBGRA32 &&
        input.format() != mediapipe::GpuBufferFormat::kRGBAHalf64 &&
        input.format() != mediapipe::GpuBufferFormat::kRGBAFloat128) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Only 4-channel texture input formats are supported, passed format: ",
          static_cast<uint32_t>(input.format())));
    }
    RET_CHECK_EQ(tensor_buffer_offset, 0)
        << "The non-zero tensor_buffer_offset input is not supported yet.";
    ABSL_RETURN_IF_ERROR(ValidateOutputTensorTypeAndRange(
        output_tensor, tensor_type_, range_min, range_max));
    const auto& output_shape = output_tensor.shape();
    ABSL_RETURN_IF_ERROR(ValidateTensorShape(output_shape));

    @autoreleasepool {
      id<MTLTexture> texture =
          [metal_helper_ metalTextureWithGpuBuffer:input.GetGpuBuffer()];

      constexpr float kInputImageRangeMin = 0.0f;
      constexpr float kInputImageRangeMax = 1.0f;
      ABSL_ASSIGN_OR_RETURN(
          auto transform,
          GetValueRangeTransformation(kInputImageRangeMin, kInputImageRangeMax,
                                      range_min, range_max));

      id<MTLCommandBuffer> command_buffer = [metal_helper_ commandBuffer];
      const auto& buffer_view =
          MtlBufferView::GetWriteView(output_tensor, command_buffer);
      ABSL_RETURN_IF_ERROR(extractor_->Execute(
          texture, roi,
          /*flip_horizontally=*/false, transform.scale, transform.offset,
          tflite::gpu::HW(output_shape.dims[1], output_shape.dims[2]),
          command_buffer, buffer_view.buffer()));
      [command_buffer commit];
      return absl::OkStatus();
    }
  }

 private:
  absl::Status ValidateTensorShape(const Tensor::Shape& output_shape) {
    RET_CHECK_EQ(output_shape.dims.size(), 4)
        << "Wrong output dims size: " << output_shape.dims.size();
    RET_CHECK_EQ(output_shape.dims[0], 1)
        << "Handling batch dimension not equal to 1 is not implemented in this "
           "converter.";
    RET_CHECK_EQ(output_shape.dims[3], num_channels_)
        << "Wrong output channel: " << output_shape.dims[3];
    return absl::OkStatus();
  }

  MPPMetalHelper* metal_helper_ = nil;
  Tensor::ElementType tensor_type_ = Tensor::ElementType::kFloat32;
  int num_channels_ = 4;
  std::unique_ptr<SubRectExtractorMetal> extractor_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<ImageToTensorConverter>> CreateMetalConverter(
    CalculatorContext* cc, BorderMode border_mode,
    Tensor::ElementType tensor_type, int num_channels) {
  auto result = absl::make_unique<ImageToTensorMetalConverter>();
  ABSL_RETURN_IF_ERROR(
      result->Init(cc, border_mode, tensor_type, num_channels));

  return result;
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_METAL_ENABLED
