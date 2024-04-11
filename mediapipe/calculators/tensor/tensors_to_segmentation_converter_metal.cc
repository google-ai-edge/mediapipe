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

#if MEDIAPIPE_METAL_ENABLED

#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_segmentation_converter.h"
#include "mediapipe/calculators/tensor/tensors_to_segmentation_utils.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/formats/tensor_mtl_buffer_view.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#import "mediapipe/gpu/MPPMetalHelper.h"
#include "mediapipe/gpu/MPPMetalUtil.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/gpu_origin.pb.h"
#include "mediapipe/gpu/shader_util.h"

namespace mediapipe {
namespace {

using ::mediapipe::tensors_to_segmentation_utils::GetHwcFromDims;
using ::mediapipe::tensors_to_segmentation_utils::GlRender;
using ::mediapipe::tensors_to_segmentation_utils::NumGroups;

constexpr int kWorkgroupSize = 8;  // Block size for GPU shader.
enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };

class TensorsToSegmentationMetalConverter
    : public TensorsToSegmentationConverter {
 public:
  ~TensorsToSegmentationMetalConverter() override;
  absl::Status Init(CalculatorContext* cc,
                    const TensorsToSegmentationCalculatorOptions& options);
  absl::StatusOr<std::unique_ptr<Image>> Convert(
      const std::vector<Tensor>& input_tensors, int output_width,
      int output_height) override;

 private:
  mediapipe::GlCalculatorHelper gpu_helper_;
  // TODO: Refactor upsample program out of the conversion.
  GLuint upsample_program_;
  bool gpu_initialized_ = false;
  MPPMetalHelper* metal_helper_ = nullptr;
  id<MTLComputePipelineState> mask_program_;
};

TensorsToSegmentationMetalConverter::~TensorsToSegmentationMetalConverter() {
  if (gpu_initialized_) {
    gpu_helper_.RunInGlContext([this] {
      if (upsample_program_) glDeleteProgram(upsample_program_);
      upsample_program_ = 0;
      mask_program_ = nil;
    });
  }
}

absl::Status TensorsToSegmentationMetalConverter::Init(
    CalculatorContext* cc,
    const TensorsToSegmentationCalculatorOptions& options) {
  // Initialize metal helper, originally done inside calculator's Open() method.
  metal_helper_ = [[MPPMetalHelper alloc] initWithCalculatorContext:cc];
  RET_CHECK(metal_helper_);

  MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
  MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([this,
                                                 &options]() -> absl::Status {
    // A shader to process a segmentation tensor into an output mask.
    // Currently uses 4 channels for output, and sets R+A channels as mask
    // value.
    const std::string shader_header = R"(
#include <metal_stdlib>
using namespace metal;
)";
    /* Shader defines will be inserted here. */

    const std::string shader_src_main = R"(
kernel void segmentationKernel(
#ifdef TWO_CHANNEL_INPUT
    device float2*     elements        [[ buffer(0) ]],
#else
    device float*      elements        [[ buffer(0) ]],
#endif // TWO_CHANNEL_INPUT
    texture2d<float, access::write>  output_texture  [[ texture(1) ]],
    constant uint*      out_size        [[ buffer(2) ]],
    uint2               gid             [[ thread_position_in_grid ]])
{
  uint out_width = out_size[0];
  uint out_height = out_size[1];

  if (gid.x >= out_width || gid.y >= out_height) { return; }
  uint linear_index = gid.y * out_width + gid.x;

#ifdef TWO_CHANNEL_INPUT
  float2 input_value = elements[linear_index];
#else
  float2 input_value = float2(elements[linear_index], 0.0);
#endif // TWO_CHANNEL_INPUT

// Run activation function.
// One and only one of FN_SOFTMAX,FN_SIGMOID,FN_NONE will be defined.
#ifdef FN_SOFTMAX
  // Only two channel input tensor is supported.
  float2 input_px = input_value.xy;
  float shift = max(input_px.x, input_px.y);
  float softmax_denom = exp(input_px.r - shift) + exp(input_px.g - shift);
  float new_mask_value =
      exp(input_px[OUTPUT_LAYER_INDEX] - shift) / softmax_denom;
#endif // FN_SOFTMAX

#ifdef FN_SIGMOID
  float new_mask_value = 1.0 / (exp(-input_value.x) + 1.0);
#endif // FN_SIGMOID

#ifdef FN_NONE
  float new_mask_value = input_value.x;
#endif // FN_NONE

#ifdef FLIP_Y_COORD
  int y_coord = out_height - gid.y - 1;
#else
  int y_coord = gid.y;
#endif  // defined(FLIP_Y_COORD)
  uint2 output_coordinate = uint2(gid.x, y_coord);

  float4 out_value = float4(new_mask_value, 0.0, 0.0, new_mask_value);
  output_texture.write(out_value, output_coordinate);
}
)";

    // Shader defines.
    using Options = ::mediapipe::TensorsToSegmentationCalculatorOptions;
    const std::string output_layer_index =
        "\n#define OUTPUT_LAYER_INDEX int(" +
        std::to_string(options.output_layer_index()) + ")";
    bool gpu_texture_starts_at_bottom =
        (options.gpu_origin() != mediapipe::GpuOrigin::TOP_LEFT);
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
    id<MTLDevice> device = metal_helper_.mtlDevice;
    NSString* library_source =
        [NSString stringWithUTF8String:shader_src_no_previous.c_str()];
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:library_source
                                                  options:nullptr
                                                    error:&error];
    RET_CHECK(library != nil) << "Couldn't create shader library "
                              << [[error localizedDescription] UTF8String];
    id<MTLFunction> kernel_func = nil;
    kernel_func = [library newFunctionWithName:@"segmentationKernel"];
    RET_CHECK(kernel_func != nil) << "Couldn't create kernel function.";
    mask_program_ =
        [device newComputePipelineStateWithFunction:kernel_func error:&error];
    RET_CHECK(mask_program_ != nil) << "Couldn't create pipeline state " <<
        [[error localizedDescription] UTF8String];

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
TensorsToSegmentationMetalConverter::Convert(
    const std::vector<Tensor>& input_tensors, int output_width,
    int output_height) {
  if (input_tensors.empty()) {
    return absl::InvalidArgumentError("input_tensors vector is empty.");
  }
  std::unique_ptr<Image> output_image_mask;

  MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext(
      [this, &input_tensors, &output_image_mask, output_width,
       output_height]() -> absl::Status {
        // Create initial working mask texture.
        mediapipe::GlTexture small_mask_texture;

        MP_ASSIGN_OR_RETURN(auto hwc,
                            GetHwcFromDims(input_tensors[0].shape().dims));
        auto [tensor_height, tensor_width, tensor_channels] = hwc;

        // Run shader, process mask tensor.
        {
          id<MTLCommandBuffer> command_buffer = [metal_helper_ commandBuffer];
          command_buffer.label = @"SegmentationKernel";
          id<MTLComputeCommandEncoder> command_encoder =
              [command_buffer computeCommandEncoder];
          [command_encoder setComputePipelineState:mask_program_];

          auto read_view =
              MtlBufferView::GetReadView(input_tensors[0], command_buffer);
          [command_encoder setBuffer:read_view.buffer() offset:0 atIndex:0];

          mediapipe::GpuBuffer small_mask_buffer = [metal_helper_
              mediapipeGpuBufferWithWidth:tensor_width
                                   height:tensor_height
                                   format:mediapipe::GpuBufferFormat::kBGRA32];
          id<MTLTexture> small_mask_texture_metal =
              [metal_helper_ metalTextureWithGpuBuffer:small_mask_buffer];
          [command_encoder setTexture:small_mask_texture_metal atIndex:1];

          unsigned int out_size[] = {static_cast<unsigned int>(tensor_width),
                                     static_cast<unsigned int>(tensor_height)};
          [command_encoder setBytes:&out_size
                             length:sizeof(out_size)
                            atIndex:2];

          MTLSize threads_per_group =
              MTLSizeMake(kWorkgroupSize, kWorkgroupSize, 1);
          MTLSize threadgroups =
              MTLSizeMake(NumGroups(tensor_width, kWorkgroupSize),
                          NumGroups(tensor_height, kWorkgroupSize), 1);
          [command_encoder dispatchThreadgroups:threadgroups
                          threadsPerThreadgroup:threads_per_group];
          [command_encoder endEncoding];
          [command_buffer commit];

          small_mask_texture =
              gpu_helper_.CreateSourceTexture(small_mask_buffer);
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
CreateMetalConverter(
    CalculatorContext* cc,
    const mediapipe::TensorsToSegmentationCalculatorOptions& options) {
  auto converter = std::make_unique<TensorsToSegmentationMetalConverter>();
  MP_RETURN_IF_ERROR(converter->Init(cc, options));
  return converter;
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_METAL_ENABLED
