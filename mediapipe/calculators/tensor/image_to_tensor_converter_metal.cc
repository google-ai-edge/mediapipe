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

#include <array>
#include <memory>
#include <vector>

#include "absl/strings/str_cat.h"
#include "mediapipe/calculators/tensor/image_to_tensor_converter.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/gpu/MPPMetalHelper.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

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
    return Type4(alpha * texture_pixel.rgb + beta, 0);
  }
)";

enum class OutputFormat { kF16C4, kF32C4 };

MTLPixelFormat GetPixelFormat(OutputFormat output_format) {
  switch (output_format) {
    case OutputFormat::kF16C4:
      return MTLPixelFormatRGBA16Float;
    case OutputFormat::kF32C4:
      return MTLPixelFormatRGBA32Float;
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
  }
  constexpr int kNumChannels = 4;
  return size.w * kNumChannels * type_size;
}

class SubRectExtractorMetal {
 public:
  static absl::StatusOr<std::unique_ptr<SubRectExtractorMetal>> Make(
      id<MTLDevice> device, OutputFormat output_format,
      BorderMode border_mode) {
    id<MTLRenderPipelineState> pipeline_state;
    MP_RETURN_IF_ERROR(SubRectExtractorMetal::MakePipelineState(
        device, output_format, border_mode, &pipeline_state));

    return absl::make_unique<SubRectExtractorMetal>(device, pipeline_state,
                                                    output_format);
  }

  SubRectExtractorMetal(id<MTLDevice> device,
                        id<MTLRenderPipelineState> pipeline_state,
                        OutputFormat output_format)
      : device_(device),
        pipeline_state_(pipeline_state),
        output_format_(output_format) {
    positions_buffer_ =
        [device_ newBufferWithBytes:kBasicSquareVertices
                             length:sizeof(kBasicSquareVertices)
                            options:MTLResourceOptionCPUCacheModeDefault];

    tex_coords_buffer_ =
        [device_ newBufferWithBytes:kBasicTextureVertices
                             length:sizeof(kBasicTextureVertices)
                            options:MTLResourceOptionCPUCacheModeDefault];
  }

  absl::Status Execute(id<MTLTexture> input_texture,
                       const RotatedRect& sub_rect, bool flip_horizontaly,
                       float alpha, float beta,
                       const tflite::gpu::HW& destination_size,
                       id<MTLCommandBuffer> command_buffer,
                       id<MTLBuffer> destination) {
    auto output_texture = MTLTextureWithBuffer(destination_size, destination);
    return InternalExecute(input_texture, sub_rect, flip_horizontaly, alpha,
                           beta, destination_size, command_buffer,
                           output_texture);
  }

 private:
  id<MTLTexture> MTLTextureWithBuffer(const tflite::gpu::HW& size,
                                      id<MTLBuffer> buffer) {
    MTLTextureDescriptor* texture_desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:GetPixelFormat(output_format_)
                                     width:size.w
                                    height:size.h
                                 mipmapped:NO];
    texture_desc.usage = MTLTextureUsageRenderTarget;

    NSUInteger output_bytes_per_row = GetBytesPerRaw(output_format_, size);

    id<MTLTexture> texture =
        [buffer newTextureWithDescriptor:texture_desc
                                  offset:0
                             bytesPerRow:output_bytes_per_row];
    return texture;
  }

  absl::Status InternalExecute(id<MTLTexture> input_texture,
                               const RotatedRect& sub_rect,
                               bool flip_horizontaly, float alpha, float beta,
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
                                           flip_horizontaly, &transform_mat);
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

  static absl::Status MakePipelineState(
      id<MTLDevice> device, OutputFormat output_format, BorderMode border_mode,
      id<MTLRenderPipelineState>* pipeline_state) {
    RET_CHECK(pipeline_state != nil);

    std::string output_type_def;
    MTLPixelFormat pixel_format;
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
    }

    std::string clamp_def;
    switch (border_mode) {
      case BorderMode::kReplicate: {
        clamp_def = R"(
          #define CLAMP_TO_EDGE
        )";
        break;
      }
      case BorderMode::kZero: {
        clamp_def = R"(
          #define CLAMP_TO_ZERO
        )";
        break;
      }
    }

    std::string shader_lib =
        absl::StrCat(kShaderLibHeader, output_type_def, clamp_def,
                     kVertexShader, kFragmentShader);
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

  id<MTLBuffer> positions_buffer_;
  id<MTLBuffer> tex_coords_buffer_;
  id<MTLDevice> device_;
  id<MTLRenderPipelineState> pipeline_state_;
  OutputFormat output_format_;
};

class MetalProcessor : public ImageToTensorConverter {
 public:
  absl::Status Init(CalculatorContext* cc, BorderMode border_mode) {
    metal_helper_ = [[MPPMetalHelper alloc] initWithCalculatorContext:cc];
    RET_CHECK(metal_helper_);
    ASSIGN_OR_RETURN(extractor_, SubRectExtractorMetal::Make(
                                     metal_helper_.mtlDevice,
                                     OutputFormat::kF32C4, border_mode));
    return absl::OkStatus();
  }

  absl::StatusOr<Tensor> Convert(const mediapipe::Image& input,
                                 const RotatedRect& roi,
                                 const Size& output_dims, float range_min,
                                 float range_max) override {
    if (input.format() != mediapipe::GpuBufferFormat::kBGRA32 &&
        input.format() != mediapipe::GpuBufferFormat::kRGBAHalf64 &&
        input.format() != mediapipe::GpuBufferFormat::kRGBAFloat128) {
      return InvalidArgumentError(absl::StrCat(
          "Only 4-channel texture input formats are supported, passed format: ",
          static_cast<uint32_t>(input.format())));
    }

    @autoreleasepool {
      id<MTLTexture> texture =
          [metal_helper_ metalTextureWithGpuBuffer:input.GetGpuBuffer()];

      constexpr int kNumChannels = 4;
      Tensor tensor(Tensor::ElementType::kFloat32,
                    Tensor::Shape{1, output_dims.height, output_dims.width,
                                  kNumChannels});

      constexpr float kInputImageRangeMin = 0.0f;
      constexpr float kInputImageRangeMax = 1.0f;
      ASSIGN_OR_RETURN(
          auto transform,
          GetValueRangeTransformation(kInputImageRangeMin, kInputImageRangeMax,
                                      range_min, range_max));

      id<MTLCommandBuffer> command_buffer = [metal_helper_ commandBuffer];
      const auto& buffer_view = tensor.GetMtlBufferWriteView(command_buffer);
      MP_RETURN_IF_ERROR(extractor_->Execute(
          texture, roi,
          /*flip_horizontaly=*/false, transform.scale, transform.offset,
          tflite::gpu::HW(output_dims.height, output_dims.width),
          command_buffer, buffer_view.buffer()));
      [command_buffer commit];
      return tensor;
    }
  }

 private:
  MPPMetalHelper* metal_helper_ = nil;
  std::unique_ptr<SubRectExtractorMetal> extractor_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<ImageToTensorConverter>> CreateMetalConverter(
    CalculatorContext* cc, BorderMode border_mode) {
  auto result = absl::make_unique<MetalProcessor>();
  MP_RETURN_IF_ERROR(result->Init(cc, border_mode));

  return result;
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_METAL_ENABLED
