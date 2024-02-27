// Copyright 2021 The MediaPipe Authors.
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
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/gpu/gpu_origin.pb.h"

#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/shader_util.h"

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
#include "mediapipe/calculators/tensor/tensors_to_segmentation_converter_gl_buffer.h"
#elif !(MEDIAPIPE_METAL_ENABLED)
#include "mediapipe/calculators/tensor/tensors_to_segmentation_converter_gl_texture.h"
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
#endif  // !MEDIAPIPE_DISABLE_GPU

#if !MEDIAPIPE_DISABLE_OPENCV
#include "mediapipe/calculators/tensor/tensors_to_segmentation_converter_opencv.h"
#endif  // !MEDIAPIPE_DISABLE_OPENCV

#if MEDIAPIPE_METAL_ENABLED
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#include "mediapipe/framework/formats/tensor_mtl_buffer_view.h"
#import "mediapipe/gpu/MPPMetalHelper.h"
#include "mediapipe/gpu/MPPMetalUtil.h"
#endif  // MEDIAPIPE_METAL_ENABLED

namespace {
constexpr int kWorkgroupSize = 8;  // Block size for GPU shader.
enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };

constexpr char kTensorsTag[] = "TENSORS";
constexpr char kOutputSizeTag[] = "OUTPUT_SIZE";
constexpr char kMaskTag[] = "MASK";
}  // namespace

namespace mediapipe {

using ::mediapipe::tensors_to_segmentation_utils::CanUseGpu;
using ::mediapipe::tensors_to_segmentation_utils::GetHwcFromDims;
using ::mediapipe::tensors_to_segmentation_utils::GlRender;   // NOLINT
using ::mediapipe::tensors_to_segmentation_utils::NumGroups;  // NOLINT

// Converts Tensors from a tflite segmentation model to an image mask.
//
// Performs optional upscale to OUTPUT_SIZE dimensions if provided,
// otherwise the mask is the same size as input tensor.
//
// If at least one input tensor is already on GPU, processing happens on GPU and
// the output mask is also stored on GPU. Otherwise, processing and the output
// mask are both on CPU.
//
// On GPU, the mask is an RGBA image, in both the R & A channels, scaled 0-1.
// On CPU, the mask is a ImageFormat::VEC32F1 image, with values scaled 0-1.
//
//
// Inputs:
//   One of the following TENSORS tags:
//   TENSORS: Vector of Tensors of type kFloat32. Only the first tensor will be
//            used. The tensor dimensions are specified in this calculator's
//            options.
//   OUTPUT_SIZE(optional): std::pair<int, int>,
//                          If provided, the size to upscale mask to.
//
// Output:
//   MASK: An Image output mask, RGBA(GPU) / VEC32F1(CPU).
//
// Options:
//   See tensors_to_segmentation_calculator.proto
//
// Usage example:
// node {
//   calculator: "TensorsToSegmentationCalculator"
//   input_stream: "TENSORS:tensors"
//   input_stream: "OUTPUT_SIZE:size"
//   output_stream: "MASK:hair_mask"
//   node_options: {
//     [mediapipe.TensorsToSegmentationCalculatorOptions] {
//       output_layer_index: 1
//       # gpu_origin: CONVENTIONAL # or TOP_LEFT
//     }
//   }
// }
//
// TODO Refactor and add support for other backends/platforms.
//
class TensorsToSegmentationCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status LoadOptions(CalculatorContext* cc);
  absl::Status InitGpu(CalculatorContext* cc);
  absl::Status ProcessGpu(CalculatorContext* cc,
                          const std::vector<Tensor>& input_tensors,
                          std::tuple<int, int, int> hwc, int output_width,
                          int output_height);

  bool DoesGpuTextureStartAtBottom() {
    return options_.gpu_origin() != mediapipe::GpuOrigin::TOP_LEFT;
  }
  absl::Status InitConverterIfNecessary(bool use_gpu, CalculatorContext* cc) {
    if (use_gpu) {
#if !MEDIAPIPE_DISABLE_GPU
      if (!gpu_converter_) {
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
        MP_ASSIGN_OR_RETURN(gpu_converter_,
                            CreateGlBufferConverter(cc, options_));
#elif !(MEDIAPIPE_METAL_ENABLED)
        MP_ASSIGN_OR_RETURN(gpu_converter_,
                            CreateGlTextureConverter(cc, options_));
#else
        RET_CHECK_FAIL()
            << "No suitable converter found for current GPU processing.";
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
      }
#endif  // !MEDIAPIPE_DISABLE_GPU
    } else {
#if !MEDIAPIPE_DISABLE_OPENCV
      if (!cpu_converter_) {
        MP_ASSIGN_OR_RETURN(cpu_converter_, CreateOpenCvConverter(options_));
      }
#else
      RET_CHECK_FAIL() << "Cannot initialize OpenCV converter because OpenCV "
                          "processing is disabled.";
#endif  // !MEDIAPIPE_DISABLE_OPENCV
    }
    return absl::OkStatus();
  }

  mediapipe::TensorsToSegmentationCalculatorOptions options_;
  std::unique_ptr<TensorsToSegmentationConverter> cpu_converter_;
  std::unique_ptr<TensorsToSegmentationConverter> gpu_converter_;

#if !MEDIAPIPE_DISABLE_GPU
  mediapipe::GlCalculatorHelper gpu_helper_;
  // TODO: Refactor upsample program out of the conversion.
  GLuint upsample_program_;
  bool gpu_initialized_ = false;
#if MEDIAPIPE_METAL_ENABLED
  MPPMetalHelper* metal_helper_ = nullptr;
  id<MTLComputePipelineState> mask_program_;
#endif  // MEDIAPIPE_METAL_ENABLED
#endif  // !MEDIAPIPE_DISABLE_GPU
};
REGISTER_CALCULATOR(TensorsToSegmentationCalculator);

// static
absl::Status TensorsToSegmentationCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(!cc->Inputs().GetTags().empty());
  RET_CHECK(!cc->Outputs().GetTags().empty());

  // Inputs.
  cc->Inputs().Tag(kTensorsTag).Set<std::vector<Tensor>>();
  if (cc->Inputs().HasTag(kOutputSizeTag)) {
    cc->Inputs().Tag(kOutputSizeTag).Set<std::pair<int, int>>();
  }

  // Outputs.
  cc->Outputs().Tag(kMaskTag).Set<Image>();

  if (CanUseGpu()) {
#if !MEDIAPIPE_DISABLE_GPU
    MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(
        cc, /*request_gpu_as_optional=*/true));
#if MEDIAPIPE_METAL_ENABLED
    MP_RETURN_IF_ERROR([MPPMetalHelper updateContract:cc]);
#endif  // MEDIAPIPE_METAL_ENABLED
#endif  // !MEDIAPIPE_DISABLE_GPU
  }

  return absl::OkStatus();
}

absl::Status TensorsToSegmentationCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  if (CanUseGpu()) {
#if !MEDIAPIPE_DISABLE_GPU
#if MEDIAPIPE_METAL_ENABLED
    metal_helper_ = [[MPPMetalHelper alloc] initWithCalculatorContext:cc];
    RET_CHECK(metal_helper_);
#endif  // MEDIAPIPE_METAL_ENABLED
#endif  // !MEDIAPIPE_DISABLE_GPU
  }

  MP_RETURN_IF_ERROR(LoadOptions(cc));

  return absl::OkStatus();
}

absl::Status TensorsToSegmentationCalculator::Process(CalculatorContext* cc) {
  if (cc->Inputs().Tag(kTensorsTag).IsEmpty()) {
    return absl::OkStatus();
  }

  const auto& input_tensors =
      cc->Inputs().Tag(kTensorsTag).Get<std::vector<Tensor>>();

  bool use_gpu = false;
  if (CanUseGpu()) {
    // Use GPU processing only if at least one input tensor is already on GPU.
    for (const auto& tensor : input_tensors) {
      if (tensor.ready_on_gpu()) {
        use_gpu = true;
        break;
      }
    }
  }

  // Validate tensor channels and activation type.
  {
    RET_CHECK(!input_tensors.empty());
    RET_CHECK(input_tensors[0].element_type() == Tensor::ElementType::kFloat32);
    MP_ASSIGN_OR_RETURN(auto hwc,
                        GetHwcFromDims(input_tensors[0].shape().dims));
    int tensor_channels = std::get<2>(hwc);
    using Options = ::mediapipe::TensorsToSegmentationCalculatorOptions;
    switch (options_.activation()) {
      case Options::NONE:
        RET_CHECK_EQ(tensor_channels, 1);
        break;
      case Options::SIGMOID:
        RET_CHECK_EQ(tensor_channels, 1);
        break;
      case Options::SOFTMAX:
        RET_CHECK_EQ(tensor_channels, 2);
        break;
    }
  }

  // Get dimensions.
  MP_ASSIGN_OR_RETURN(auto hwc, GetHwcFromDims(input_tensors[0].shape().dims));
  auto [tensor_height, tensor_width, tensor_channels] = hwc;
  int output_width = tensor_width, output_height = tensor_height;
  if (cc->Inputs().HasTag(kOutputSizeTag)) {
    const auto& size =
        cc->Inputs().Tag(kOutputSizeTag).Get<std::pair<int, int>>();
    output_width = size.first;
    output_height = size.second;
  }

  if (use_gpu) {
#if !MEDIAPIPE_DISABLE_GPU
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31 || \
    !MEDIAPIPE_METAL_ENABLED
    // Lazily initialize converter
    MP_RETURN_IF_ERROR(InitConverterIfNecessary(use_gpu, cc));
    MP_ASSIGN_OR_RETURN(
        std::unique_ptr<Image> output_mask,
        gpu_converter_->Convert(input_tensors, output_width, output_height));
    cc->Outputs().Tag(kMaskTag).Add(output_mask.release(),
                                    cc->InputTimestamp());
#else
    if (!gpu_initialized_) {
      MP_RETURN_IF_ERROR(InitGpu(cc));
      gpu_initialized_ = true;
    }

    MP_RETURN_IF_ERROR(
        gpu_helper_.RunInGlContext([this, cc, &input_tensors, output_width,
                                    output_height, hwc]() -> absl::Status {
          MP_RETURN_IF_ERROR(
              ProcessGpu(cc, input_tensors, hwc, output_width, output_height));
          return absl::OkStatus();
        }));
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31 ||
        // !MEDIAPIPE_METAL_ENABLED
#else
    RET_CHECK_FAIL() << "GPU processing disabled.";
#endif  // !MEDIAPIPE_DISABLE_GPU
  } else {
#if !MEDIAPIPE_DISABLE_OPENCV
    // Lazily initialize converter.
    MP_RETURN_IF_ERROR(InitConverterIfNecessary(use_gpu, cc));
    MP_ASSIGN_OR_RETURN(
        std::unique_ptr<Image> output_mask,
        cpu_converter_->Convert(input_tensors, output_width, output_height));
    cc->Outputs().Tag(kMaskTag).Add(output_mask.release(),
                                    cc->InputTimestamp());
#else
    RET_CHECK_FAIL() << "OpenCV processing disabled.";
#endif  // !MEDIAPIPE_DISABLE_OPENCV
  }

  return absl::OkStatus();
}

absl::Status TensorsToSegmentationCalculator::Close(CalculatorContext* cc) {
#if !MEDIAPIPE_DISABLE_GPU
#if !(MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31) && \
    MEDIAPIPE_METAL_ENABLED
  if (!gpu_initialized_) {
    return absl::OkStatus();
  }

  gpu_helper_.RunInGlContext([this] {
    if (upsample_program_) glDeleteProgram(upsample_program_);
    upsample_program_ = 0;
    mask_program_ = nil;
  });
#endif  // !(MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31) &&
        // MEDIAPIPE_METAL_ENABLED
#endif  // !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
}

// Steps:
// 1. receive tensor
// 2. process segmentation tensor into small mask
// 3. upsample small mask into output mask to be same size as input image
absl::Status TensorsToSegmentationCalculator::ProcessGpu(
    CalculatorContext* cc, const std::vector<Tensor>& input_tensors,
    std::tuple<int, int, int> hwc, int output_width, int output_height) {
#if !MEDIAPIPE_DISABLE_GPU
#if !(MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31) && \
    MEDIAPIPE_METAL_ENABLED

  // Create initial working mask texture.
  mediapipe::GlTexture small_mask_texture;

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
    [command_encoder setBytes:&out_size length:sizeof(out_size) atIndex:2];

    MTLSize threads_per_group = MTLSizeMake(kWorkgroupSize, kWorkgroupSize, 1);
    MTLSize threadgroups =
        MTLSizeMake(NumGroups(tensor_width, kWorkgroupSize),
                    NumGroups(tensor_height, kWorkgroupSize), 1);
    [command_encoder dispatchThreadgroups:threadgroups
                    threadsPerThreadgroup:threads_per_group];
    [command_encoder endEncoding];
    [command_buffer commit];

    small_mask_texture = gpu_helper_.CreateSourceTexture(small_mask_buffer);
  }

  // Upsample small mask into output.
  mediapipe::GlTexture output_texture = gpu_helper_.CreateDestinationTexture(
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
  auto output_image = output_texture.GetFrame<Image>();
  cc->Outputs().Tag(kMaskTag).Add(output_image.release(), cc->InputTimestamp());

  // Cleanup
  output_texture.Release();
#else
  RET_CHECK_FAIL() << "OpenGL implementations should not go this path.";
#endif  // !(MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31) &&
        // MEDIAPIPE_METAL_ENABLED
#endif  // !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
}

absl::Status TensorsToSegmentationCalculator::LoadOptions(
    CalculatorContext* cc) {
  // Get calculator options specified in the graph.
  options_ = cc->Options<mediapipe::TensorsToSegmentationCalculatorOptions>();

  return absl::OkStatus();
}

absl::Status TensorsToSegmentationCalculator::InitGpu(CalculatorContext* cc) {
#if !MEDIAPIPE_DISABLE_GPU
#if !(MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31) && \
    MEDIAPIPE_METAL_ENABLED
  // METAL
  MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
  MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([this]() -> absl::Status {
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
        std::to_string(options_.output_layer_index()) + ")";
    const std::string flip_y_coord =
        DoesGpuTextureStartAtBottom() ? "\n#define FLIP_Y_COORD" : "";
    const std::string fn_none =
        options_.activation() == Options::NONE ? "\n#define FN_NONE" : "";
    const std::string fn_sigmoid =
        options_.activation() == Options::SIGMOID ? "\n#define FN_SIGMOID" : "";
    const std::string fn_softmax =
        options_.activation() == Options::SOFTMAX ? "\n#define FN_SOFTMAX" : "";
    const std::string two_channel = options_.activation() == Options::SOFTMAX
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
#else
  RET_CHECK_FAIL() << "OpenGL implementations should not go this path.";
#endif  // !(MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31) &&
        // MEDIAPIPE_METAL_ENABLED
#endif  // !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
}

}  // namespace mediapipe
