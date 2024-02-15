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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "absl/types/optional.h"
#include "mediapipe/calculators/tensor/tensor_converter_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensor_converter_cpu.h"
#include "mediapipe/calculators/tensor/tensor_converter_gpu.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/gpu/gpu_origin.pb.h"

#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_buffer_format.h"
#include "mediapipe/gpu/gpu_origin.pb.h"

#if MEDIAPIPE_METAL_ENABLED
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#include "mediapipe/framework/formats/tensor_mtl_buffer_view.h"
#import "mediapipe/gpu/MPPMetalHelper.h"
#elif MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
#include "mediapipe/gpu/gl_calculator_helper.h"

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
#include "mediapipe/calculators/tensor/tensor_converter_gl31.h"
#else
#include "mediapipe/calculators/tensor/tensor_converter_gl30.h"
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31

#endif  // MEDIAPIPE_METAL_ENABLED
#endif  // !MEDIAPIPE_DISABLE_GPU

namespace {

constexpr int kWorkgroupSize = 8;  // Block size for GPU shader.
// Commonly used to compute the number of blocks to launch in a kernel.
int NumGroups(const int size, const int group_size) {  // NOLINT
  return (size + group_size - 1) / group_size;
}

absl::StatusOr<bool> ShouldFlipVertically(
    const mediapipe::TensorConverterCalculatorOptions& options, bool use_gpu) {
  if (options.has_flip_vertically() && options.has_gpu_origin()) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "Cannot specify both flip_vertically and gpu_origin options"));
  }

  if (!options.has_gpu_origin()) {
    // Fall back to flip_vertically.
    return options.flip_vertically();
  }

  // Warn if gpu_origin is specified with a CPU input image.
  // Those are always TOP_LEFT, so no flipping is necessary.
  if (!use_gpu) {
    ABSL_LOG(WARNING)
        << "Ignoring gpu_origin option since IMAGE_GPU input is not specified";
    return false;
  }

  switch (options.gpu_origin()) {
    case mediapipe::GpuOrigin::TOP_LEFT:
      return false;
    case mediapipe::GpuOrigin::DEFAULT:
    case mediapipe::GpuOrigin::CONVENTIONAL:
      // TOP_LEFT on Metal, BOTTOM_LEFT on OpenGL.
#ifdef __APPLE__
      return false;
#else
      return true;
#endif
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Unhandled GPU origin %i", options.gpu_origin()));
  }
}

constexpr char kImageFrameTag[] = "IMAGE";
constexpr char kGpuBufferTag[] = "IMAGE_GPU";
constexpr char kTensorsTag[] = "TENSORS";
constexpr char kMatrixTag[] = "MATRIX";

constexpr std::pair<float, float> kDefaultOutputRange = {0.0f, 1.0f};

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
  absl::Status LoadOptions(CalculatorContext* cc, bool use_gpu);
  absl::Status ProcessCPU(CalculatorContext* cc);
  absl::Status ProcessGPU(CalculatorContext* cc);

#if MEDIAPIPE_METAL_ENABLED
  MPPMetalHelper* gpu_helper_ = nullptr;
  id<MTLComputePipelineState> to_buffer_program_;
#elif MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
  GlCalculatorHelper gpu_helper_;
#endif  // MEDIAPIPE_METAL_ENABLED
  bool initialized_ = false;
  bool use_gpu_ = false;
  std::optional<std::pair<float, float>> output_range_;
  bool flip_vertically_ = false;
  bool row_major_matrix_ = false;
  int max_num_channels_ = 3;

  std::unique_ptr<TensorConverterGpu> tensor_converter_gpu_;
};
REGISTER_CALCULATOR(TensorConverterCalculator);

absl::Status TensorConverterCalculator::GetContract(CalculatorContract* cc) {
  // Confirm only one of the input streams is present.
  RET_CHECK(static_cast<int>(cc->Inputs().HasTag(kImageFrameTag)) +
                static_cast<int>(cc->Inputs().HasTag(kGpuBufferTag)) +
                static_cast<int>(cc->Inputs().HasTag(kMatrixTag)) ==
            1)
      << "Only one input tag of {IMAGE, IMAGE_GPU, MATRIX} may be specified";

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
    MP_RETURN_IF_ERROR(GlCalculatorHelper::UpdateContract(cc));
#endif  // MEDIAPIPE_METAL_ENABLED
  }
#endif  // !MEDIAPIPE_DISABLE_GPU

  RET_CHECK(cc->Outputs().HasTag(kTensorsTag));
  cc->Outputs().Tag(kTensorsTag).Set<std::vector<Tensor>>();
  return absl::OkStatus();
}

absl::Status TensorConverterCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

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

  MP_RETURN_IF_ERROR(LoadOptions(cc, use_gpu_));

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
    gpu_helper_.RunInGlContext([this] { tensor_converter_gpu_.reset(); });
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
    MP_ASSIGN_OR_RETURN(Tensor output,
                        ConvertImageFrameToTensorOnCpu(
                            image_frame,
                            output_range_.has_value() ? output_range_.value()
                                                      : kDefaultOutputRange,
                            flip_vertically_, max_num_channels_));
    output_tensors->emplace_back(std::move(output));
  } else if (cc->Inputs().HasTag(kMatrixTag)) {
    if (cc->Inputs().Tag(kMatrixTag).IsEmpty()) {
      return absl::OkStatus();
    }
    const auto& matrix = cc->Inputs().Tag(kMatrixTag).Get<Matrix>();
    MP_ASSIGN_OR_RETURN(Tensor output,
                        ConvertMatrixToTensorOnCpu(matrix, row_major_matrix_));
    output_tensors->emplace_back(std::move(output));
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
  auto output_tensors = std::make_unique<std::vector<Tensor>>();
#if MEDIAPIPE_METAL_ENABLED
  const int width = input.width();
  const int height = input.height();
  const int channels = max_num_channels_;
  output_tensors->emplace_back(Tensor::ElementType::kFloat32,
                               Tensor::Shape{1, height, width, channels});
  id<MTLCommandBuffer> command_buffer = [gpu_helper_ commandBuffer];
  command_buffer.label = @"TensorConverterCalculatorConvert";
  id<MTLComputeCommandEncoder> compute_encoder =
      [command_buffer computeCommandEncoder];
  [compute_encoder setComputePipelineState:to_buffer_program_];
  id<MTLTexture> src_texture = [gpu_helper_ metalTextureWithGpuBuffer:input];
  [compute_encoder setTexture:src_texture atIndex:0];
  auto output_view =
      MtlBufferView::GetWriteView(output_tensors->at(0), command_buffer);
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
        Tensor output = tensor_converter_gpu_->Convert(input);
        output_tensors->emplace_back(std::move(output));
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
  mediapipe::GpuBufferFormat format = input.format();
  const bool include_alpha = (max_num_channels_ == 4);
  const bool single_channel = (max_num_channels_ == 1);

  RET_CHECK(format == mediapipe::GpuBufferFormat::kBGRA32 ||
            format == mediapipe::GpuBufferFormat::kRGB24 ||
            format == mediapipe::GpuBufferFormat::kRGBA32 ||
            format == mediapipe::GpuBufferFormat::kRGBAFloat128 ||
            format == mediapipe::GpuBufferFormat::kRGBAHalf64 ||
            format == mediapipe::GpuBufferFormat::kGrayFloat32 ||
            format == mediapipe::GpuBufferFormat::kGrayHalf16 ||
            format == mediapipe::GpuBufferFormat::kOneComponent8)
      << "Unsupported GPU input format: " << static_cast<uint32_t>(format);
  if (include_alpha) {
    RET_CHECK(format == mediapipe::GpuBufferFormat::kBGRA32 ||
              format == mediapipe::GpuBufferFormat::kRGBA32 ||
              format == mediapipe::GpuBufferFormat::kRGBAFloat128 ||
              format == mediapipe::GpuBufferFormat::kRGBAHalf64)
        << "Num input channels is less than desired output, input format: "
        << static_cast<uint32_t>(format);
  }

#if MEDIAPIPE_METAL_ENABLED
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
      output_range_.has_value()
          ? absl::Substitute("pixel = pixel * half($0) + half($1);",
                             (output_range_->second - output_range_->first),
                             output_range_->first)
          : "",
      /*$1=*/max_num_channels_,
      /*$2=*/flip_vertically_ ? "(in_tex.get_height() - 1 - gid.y)" : "gid.y",
      /*$3=*/
      single_channel ? "" : R"glsl(out_buf[linear_index + 1] = pixel.y;
                                   out_buf[linear_index + 2] = pixel.z;)glsl",
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
  MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext(
      [this, &input, &include_alpha, &single_channel]() -> absl::Status {
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
        tensor_converter_gpu_ = CreateTensorConverterGl31(gpu_helper_);
#else
        tensor_converter_gpu_ = CreateTensorConverterGl30(gpu_helper_);
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
        return tensor_converter_gpu_->Init(
            input.width(), input.height(), output_range_, include_alpha,
            single_channel, flip_vertically_, max_num_channels_);
      }));
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
#endif  // !MEDIAPIPE_DISABLE_GPU
  return absl::OkStatus();
}

absl::Status TensorConverterCalculator::LoadOptions(CalculatorContext* cc,
                                                    bool use_gpu) {
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
    ABSL_CHECK_GT(output_range_->second, output_range_->first);
  }

  // Custom div and sub values.
  if (options.use_custom_normalization()) {
    output_range_.emplace(std::pair<float, float>(
        -options.custom_sub(),
        -options.custom_sub() + 255.0 / options.custom_div()));
  }

  // Get y-flip mode.
  MP_ASSIGN_OR_RETURN(flip_vertically_, ShouldFlipVertically(options, use_gpu));

  // Get row_major_matrix mode.
  row_major_matrix_ = options.row_major_matrix();

  // Get desired way to handle input channels.
  max_num_channels_ = options.max_num_channels();
  ABSL_CHECK_GE(max_num_channels_, 1);
  ABSL_CHECK_LE(max_num_channels_, 4);
  ABSL_CHECK_NE(max_num_channels_, 2);
  return absl::OkStatus();
}

}  // namespace mediapipe
