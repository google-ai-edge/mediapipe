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
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "mediapipe/calculators/tensor/tensor_converter_gpu.h"
#include "mediapipe/calculators/tensor/tensor_converter_metal.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/formats/tensor_mtl_buffer_view.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#import "mediapipe/gpu/MPPMetalHelper.h"
#include "mediapipe/gpu/gpu_buffer.h"

namespace mediapipe {

namespace {

constexpr int kWorkgroupSize = 8;  // Block size for GPU shader.

// Commonly used to compute the number of blocks to launch in a kernel.
int NumGroups(const int size, const int group_size) {  // NOLINT
  return (size + group_size - 1) / group_size;
}

class TensorConverteMetalImpl : public TensorConverterGpu {
 public:
  TensorConverteMetalImpl(MPPMetalHelper* gpu_helper,
                          MemoryManager* memory_manager)
      : gpu_helper_(gpu_helper), memory_manager_(memory_manager) {}

  absl::Status Init(std::optional<std::pair<float, float>> output_range,
                    bool include_alpha, bool single_channel,
                    bool flip_vertically, int num_output_channels) {
    num_output_channels_ = num_output_channels;

    id<MTLDevice> device = gpu_helper_.mtlDevice;
    // Shader to convert GL Texture to Metal Buffer,
    // with normalization to either: [0,1] or [-1,1].
    const std::string shader_source = absl::Substitute(
        R"glsl(
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
      )glsl",
        /*$0=*/
        output_range.has_value()
            ? absl::Substitute("pixel = pixel * half($0) + half($1);",
                               (output_range->second - output_range->first),
                               output_range->first)
            : "",
        /*$1=*/num_output_channels_,
        /*$2=*/flip_vertically ? "(in_tex.get_height() - 1 - gid.y)" : "gid.y",
        /*$3=*/
        single_channel ? "" : R"glsl(out_buf[linear_index + 1] = pixel.y;
                                   out_buf[linear_index + 2] = pixel.z;)glsl",
        /*$4=*/include_alpha ? "out_buf[linear_index + 3] = pixel.w;" : "");

    NSString* library_source =
        [NSString stringWithUTF8String:shader_source.c_str()];
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:library_source
                                                  options:nullptr
                                                    error:&error];
    RET_CHECK(library != nil) << "Couldn't create shader library "
                              << [[error localizedDescription] UTF8String];
    id<MTLFunction> kernel_func = nil;
    kernel_func = [library newFunctionWithName:@"convertKernel"];
    RET_CHECK(kernel_func != nil) << "Couldn't create kernel function.";
    to_buffer_program_ =
        [device newComputePipelineStateWithFunction:kernel_func error:&error];
    RET_CHECK(to_buffer_program_ != nil) << "Couldn't create pipeline state " <<
        [[error localizedDescription] UTF8String];
    return absl::OkStatus();
  }

  Tensor Convert(const GpuBuffer& input) override {
    const int width = input.width();
    const int height = input.height();
    const int channels = num_output_channels_;
    Tensor output(Tensor::ElementType::kFloat32,
                  Tensor::Shape{1, height, width, channels}, memory_manager_);
    id<MTLCommandBuffer> command_buffer = [gpu_helper_ commandBuffer];
    command_buffer.label = @"TensorConverterCalculatorConvert";
    id<MTLComputeCommandEncoder> compute_encoder =
        [command_buffer computeCommandEncoder];
    [compute_encoder setComputePipelineState:to_buffer_program_];
    id<MTLTexture> src_texture = [gpu_helper_ metalTextureWithGpuBuffer:input];
    [compute_encoder setTexture:src_texture atIndex:0];
    auto output_view = MtlBufferView::GetWriteView(output, command_buffer);
    [compute_encoder setBuffer:output_view.buffer() offset:0 atIndex:1];
    MTLSize threads_per_group = MTLSizeMake(kWorkgroupSize, kWorkgroupSize, 1);
    MTLSize threadgroups =
        MTLSizeMake(NumGroups(input.width(), kWorkgroupSize),
                    NumGroups(input.height(), kWorkgroupSize), 1);
    [compute_encoder dispatchThreadgroups:threadgroups
                    threadsPerThreadgroup:threads_per_group];
    [compute_encoder endEncoding];
    [command_buffer commit];
    return output;
  }

 private:
  MPPMetalHelper* gpu_helper_ = nullptr;
  MemoryManager* memory_manager_ = nullptr;
  id<MTLComputePipelineState> to_buffer_program_;

  int num_output_channels_ = 0;
};

}  // namespace

absl::StatusOr<std::unique_ptr<TensorConverterGpu>> CreateTensorConverterMetal(
    MPPMetalHelper* gpu_helper, MemoryManager* memory_manager,
    std::optional<std::pair<float, float>> output_range, bool include_alpha,
    bool single_channel, bool flip_vertically, int num_output_channels) {
  auto converter =
      std::make_unique<TensorConverteMetalImpl>(gpu_helper, memory_manager);
  MP_RETURN_IF_ERROR(converter->Init(output_range, include_alpha,
                                     single_channel, flip_vertically,
                                     num_output_channels));
  return converter;
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_METAL_ENABLED
