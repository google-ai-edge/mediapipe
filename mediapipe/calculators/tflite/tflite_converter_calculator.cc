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

#include "mediapipe/calculators/tflite/tflite_converter_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/resource_util.h"
#include "mediapipe/util/tflite/config.h"
#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/interpreter.h"

#ifndef MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gpu_buffer.h"
#endif  // MEDIAPIPE_DISABLE_GPU

#if MEDIAPIPE_TFLITE_GL_INFERENCE
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_program.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_shader.h"
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE

#if MEDIAPIPE_TFLITE_METAL_INFERENCE
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#import "mediapipe/gpu/MPPMetalHelper.h"
#include "mediapipe/gpu/MPPMetalUtil.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "tensorflow/lite/delegates/gpu/metal_delegate.h"
#endif  // MEDIAPIPE_TFLITE_METAL_INFERENCE

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
constexpr char kTensorsGpuTag[] = "TENSORS_GPU";
constexpr char kMatrixTag[] = "MATRIX";
}  // namespace

namespace mediapipe {

namespace {
#if MEDIAPIPE_TFLITE_GL_INFERENCE
using ::tflite::gpu::gl::CreateReadWriteShaderStorageBuffer;
using ::tflite::gpu::gl::GlProgram;
using ::tflite::gpu::gl::GlShader;
struct GPUData {
  int elements = 1;
  GpuTensor buffer;
  GlShader shader;
  GlProgram program;
};
#elif MEDIAPIPE_TFLITE_METAL_INFERENCE
struct GPUData {
  int elements = 1;
  GpuTensor buffer;
  id<MTLComputePipelineState> pipeline_state;
};
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE

}  // namespace

// Calculator for normalizing and converting an ImageFrame or Matrix
// into a TfLiteTensor (float 32) or a GpuBuffer to a tflite::gpu::GlBuffer
// or MTLBuffer.
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
//  TENSORS - Vector of TfLiteTensor of type kTfLiteFloat32, or kTfLiteUint8.
//  TENSORS_GPU - vector of GlBuffer or MTLBuffer.
//
// Example use:
// node {
//   calculator: "TfLiteConverterCalculator"
//   input_stream: "IMAGE:input_image"
//   output_stream: "TENSORS:image_tensor"
//   options: {
//     [mediapipe.TfLiteConverterCalculatorOptions.ext] {
//       zero_center: true
//     }
//   }
// }
//
// IMPORTANT Notes:
//  No conversion between CPU/GPU is done.
//  Inputs/outputs must match type: CPU->CPU or GPU->GPU.
//  GPU tensors are currently only supported on mobile platforms.
//  This calculator uses FixedSizeInputStreamHandler by default.
//
// Note: Input defines output, so only these type sets are supported:
// IMAGE -> TENSORS | IMAGE_GPU -> TENSORS_GPU | MATRIX -> TENSORS
//
class TfLiteConverterCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;
  ::mediapipe::Status Close(CalculatorContext* cc) override;

 private:
  ::mediapipe::Status InitGpu(CalculatorContext* cc);
  ::mediapipe::Status LoadOptions(CalculatorContext* cc);
  template <class T>
  ::mediapipe::Status NormalizeImage(const ImageFrame& image_frame,
                                     bool flip_vertically, float* tensor_ptr);
  ::mediapipe::Status CopyMatrixToTensor(const Matrix& matrix,
                                         float* tensor_ptr);
  ::mediapipe::Status ProcessCPU(CalculatorContext* cc);
  ::mediapipe::Status ProcessGPU(CalculatorContext* cc);

  std::unique_ptr<tflite::Interpreter> interpreter_ = nullptr;

#if MEDIAPIPE_TFLITE_GL_INFERENCE
  mediapipe::GlCalculatorHelper gpu_helper_;
  std::unique_ptr<GPUData> gpu_data_out_;
#elif MEDIAPIPE_TFLITE_METAL_INFERENCE
  MPPMetalHelper* gpu_helper_ = nullptr;
  std::unique_ptr<GPUData> gpu_data_out_;
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE

  bool initialized_ = false;
  bool use_gpu_ = false;
  absl::optional<std::pair<float, float>> output_range_;
  bool flip_vertically_ = false;
  bool row_major_matrix_ = false;
  bool use_quantized_tensors_ = false;
  int max_num_channels_ = 3;
};
REGISTER_CALCULATOR(TfLiteConverterCalculator);

namespace {
template <class CC>
bool ShouldUseGpu(CC* cc) {
#if MEDIAPIPE_TFLITE_GPU_SUPPORTED
  return cc->Inputs().HasTag(kGpuBufferTag) ||
         cc->Outputs().HasTag(kTensorsGpuTag);
#else
  return false;
#endif  // MEDIAPIPE_TFLITE_GPU_SUPPORTED
}
}  // namespace

::mediapipe::Status TfLiteConverterCalculator::GetContract(
    CalculatorContract* cc) {
  // Confirm only one of the input streams is present.
  RET_CHECK(cc->Inputs().HasTag(kImageFrameTag) ^
            cc->Inputs().HasTag(kGpuBufferTag) ^
            cc->Inputs().HasTag(kMatrixTag));

  // Confirm only one of the output streams is present.
  RET_CHECK(cc->Outputs().HasTag(kTensorsTag) ^
            cc->Outputs().HasTag(kTensorsGpuTag));

  if (cc->Inputs().HasTag(kImageFrameTag)) {
    cc->Inputs().Tag(kImageFrameTag).Set<ImageFrame>();
  }
  if (cc->Inputs().HasTag(kMatrixTag)) {
    cc->Inputs().Tag(kMatrixTag).Set<Matrix>();
  }
#ifndef MEDIAPIPE_DISABLE_GPU
  if (cc->Inputs().HasTag(kGpuBufferTag)) {
    cc->Inputs().Tag(kGpuBufferTag).Set<mediapipe::GpuBuffer>();
  }
#endif  // MEDIAPIPE_DISABLE_GPU

  if (cc->Outputs().HasTag(kTensorsTag)) {
    cc->Outputs().Tag(kTensorsTag).Set<std::vector<TfLiteTensor>>();
  }
  if (cc->Outputs().HasTag(kTensorsGpuTag)) {
    cc->Outputs().Tag(kTensorsGpuTag).Set<std::vector<GpuTensor>>();
  }

  if (ShouldUseGpu(cc)) {
#if MEDIAPIPE_TFLITE_GL_INFERENCE
    MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
#elif MEDIAPIPE_TFLITE_METAL_INFERENCE
    MP_RETURN_IF_ERROR([MPPMetalHelper updateContract:cc]);
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE
  }

  // Assign this calculator's default InputStreamHandler.
  cc->SetInputStreamHandler("FixedSizeInputStreamHandler");

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteConverterCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  MP_RETURN_IF_ERROR(LoadOptions(cc));

  use_gpu_ = ShouldUseGpu(cc);

  if (use_gpu_) {
    // Cannot mix CPU/GPU streams.
    RET_CHECK(cc->Inputs().HasTag(kGpuBufferTag) &&
              cc->Outputs().HasTag(kTensorsGpuTag));
    // Cannot use quantization.
    use_quantized_tensors_ = false;
#if MEDIAPIPE_TFLITE_GL_INFERENCE
    MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
#elif MEDIAPIPE_TFLITE_METAL_INFERENCE
    gpu_helper_ = [[MPPMetalHelper alloc] initWithCalculatorContext:cc];
    RET_CHECK(gpu_helper_);
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE
  } else {
    interpreter_ = absl::make_unique<tflite::Interpreter>();
    interpreter_->AddTensors(1);
    interpreter_->SetInputs({0});
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteConverterCalculator::Process(CalculatorContext* cc) {
  if (use_gpu_) {
    if (cc->Inputs().Tag(kGpuBufferTag).IsEmpty()) {
      return ::mediapipe::OkStatus();
    }
    if (!initialized_) {
      MP_RETURN_IF_ERROR(InitGpu(cc));
      initialized_ = true;
    }
    // Convert to GPU tensors type.
    MP_RETURN_IF_ERROR(ProcessGPU(cc));
  } else {
    // Convert to CPU tensors or Matrix type.
    MP_RETURN_IF_ERROR(ProcessCPU(cc));
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteConverterCalculator::Close(CalculatorContext* cc) {
  interpreter_.reset();
#if MEDIAPIPE_TFLITE_GL_INFERENCE
  gpu_helper_.RunInGlContext([this] { gpu_data_out_.reset(); });
#elif MEDIAPIPE_TFLITE_METAL_INFERENCE
  gpu_data_out_.reset();
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE
  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteConverterCalculator::ProcessCPU(
    CalculatorContext* cc) {
  if (cc->Inputs().HasTag(kImageFrameTag)) {
    if (cc->Inputs().Tag(kImageFrameTag).IsEmpty()) {
      return ::mediapipe::OkStatus();
    }
    // CPU ImageFrame to TfLiteTensor conversion.

    const auto& image_frame =
        cc->Inputs().Tag(kImageFrameTag).Get<ImageFrame>();
    const int height = image_frame.Height();
    const int width = image_frame.Width();
    const int channels = image_frame.NumberOfChannels();
    const int channels_preserved = std::min(channels, max_num_channels_);
    const mediapipe::ImageFormat::Format format = image_frame.Format();

    if (!initialized_) {
      if (!(format == mediapipe::ImageFormat::SRGBA ||
            format == mediapipe::ImageFormat::SRGB ||
            format == mediapipe::ImageFormat::GRAY8 ||
            format == mediapipe::ImageFormat::VEC32F1))
        RET_CHECK_FAIL() << "Unsupported CPU input format.";
      TfLiteQuantization quant;
      if (use_quantized_tensors_) {
        RET_CHECK(format != mediapipe::ImageFormat::VEC32F1)
            << "Only 8-bit input images are supported for quantization.";
        quant.type = kTfLiteAffineQuantization;
        auto quant_params = static_cast<TfLiteAffineQuantization*>(
            malloc(sizeof(TfLiteAffineQuantization)));
        quant_params->scale = TfLiteFloatArrayCreate(1);
        quant_params->scale->data[0] = 1.0;
        quant_params->zero_point = TfLiteIntArrayCreate(1);
        quant_params->zero_point->data[0] = 0;
        quant_params->quantized_dimension = 0;
        quant.params = quant_params;
        interpreter_->SetTensorParametersReadWrite(0, kTfLiteUInt8, "",
                                                   {channels_preserved}, quant);
      } else {
        // Initialize structure for no quantization.
        quant.type = kTfLiteNoQuantization;
        quant.params = nullptr;
        interpreter_->SetTensorParametersReadWrite(0, kTfLiteFloat32, "",
                                                   {channels_preserved}, quant);
      }
      initialized_ = true;
    }

    const int tensor_idx = interpreter_->inputs()[0];
    TfLiteTensor* tensor = interpreter_->tensor(tensor_idx);
    interpreter_->ResizeInputTensor(tensor_idx,
                                    {height, width, channels_preserved});
    interpreter_->AllocateTensors();

    // Copy image data into tensor.
    if (use_quantized_tensors_) {
      const int width_padding =
          image_frame.WidthStep() / image_frame.ByteDepth() - width * channels;
      const uint8* image_buffer =
          reinterpret_cast<const uint8*>(image_frame.PixelData());
      uint8* tensor_buffer = tensor->data.uint8;
      RET_CHECK(tensor_buffer);
      for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
          for (int channel = 0; channel < channels_preserved; ++channel) {
            *tensor_buffer++ = image_buffer[channel];
          }
          image_buffer += channels;
        }
        image_buffer += width_padding;
      }
    } else {
      float* tensor_buffer = tensor->data.f;
      RET_CHECK(tensor_buffer);
      if (image_frame.ByteDepth() == 1) {
        MP_RETURN_IF_ERROR(NormalizeImage<uint8>(image_frame, flip_vertically_,
                                                 tensor_buffer));
      } else if (image_frame.ByteDepth() == 4) {
        MP_RETURN_IF_ERROR(NormalizeImage<float>(image_frame, flip_vertically_,
                                                 tensor_buffer));
      } else {
        return ::mediapipe::InternalError(
            "Only byte-based (8 bit) and float (32 bit) images supported.");
      }
    }

    auto output_tensors = absl::make_unique<std::vector<TfLiteTensor>>();
    output_tensors->emplace_back(*tensor);
    cc->Outputs()
        .Tag(kTensorsTag)
        .Add(output_tensors.release(), cc->InputTimestamp());
  } else if (cc->Inputs().HasTag(kMatrixTag)) {
    if (cc->Inputs().Tag(kMatrixTag).IsEmpty()) {
      return ::mediapipe::OkStatus();
    }
    // CPU Matrix to TfLiteTensor conversion.
    const auto& matrix = cc->Inputs().Tag(kMatrixTag).Get<Matrix>();
    const int height = matrix.rows();
    const int width = matrix.cols();
    const int channels = 1;

    if (!initialized_) {
      interpreter_->SetTensorParametersReadWrite(
          /*tensor_index=*/0, /*type=*/kTfLiteFloat32, /*name=*/"",
          /*dims=*/{channels}, /*quantization=*/TfLiteQuantization());
      initialized_ = true;
    }

    const int tensor_idx = interpreter_->inputs()[0];
    TfLiteTensor* tensor = interpreter_->tensor(tensor_idx);
    interpreter_->ResizeInputTensor(tensor_idx, {height, width, channels});
    interpreter_->AllocateTensors();

    float* tensor_ptr = tensor->data.f;
    RET_CHECK(tensor_ptr);

    MP_RETURN_IF_ERROR(CopyMatrixToTensor(matrix, tensor_ptr));

    auto output_tensors = absl::make_unique<std::vector<TfLiteTensor>>();
    output_tensors->emplace_back(*tensor);
    cc->Outputs()
        .Tag(kTensorsTag)
        .Add(output_tensors.release(), cc->InputTimestamp());
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteConverterCalculator::ProcessGPU(
    CalculatorContext* cc) {
#if MEDIAPIPE_TFLITE_GL_INFERENCE
  // GpuBuffer to tflite::gpu::GlBuffer conversion.
  const auto& input =
      cc->Inputs().Tag(kGpuBufferTag).Get<mediapipe::GpuBuffer>();
  MP_RETURN_IF_ERROR(
      gpu_helper_.RunInGlContext([this, &input]() -> ::mediapipe::Status {
        // Convert GL texture into TfLite GlBuffer (SSBO).
        auto src = gpu_helper_.CreateSourceTexture(input);
        glActiveTexture(GL_TEXTURE0 + 0);
        glBindTexture(GL_TEXTURE_2D, src.name());
        MP_RETURN_IF_ERROR(gpu_data_out_->buffer.BindToIndex(1));
        const tflite::gpu::uint3 workgroups = {
            NumGroups(input.width(), kWorkgroupSize),
            NumGroups(input.height(), kWorkgroupSize), 1};
        MP_RETURN_IF_ERROR(gpu_data_out_->program.Dispatch(workgroups));
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
        src.Release();
        return ::mediapipe::OkStatus();
      }));

  // Copy into outputs.
  auto output_tensors = absl::make_unique<std::vector<GpuTensor>>();
  MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext(
      [this, &output_tensors]() -> ::mediapipe::Status {
        output_tensors->resize(1);
        {
          GpuTensor& tensor = output_tensors->at(0);
          MP_RETURN_IF_ERROR(CreateReadWriteShaderStorageBuffer<float>(
              gpu_data_out_->elements, &tensor));
          MP_RETURN_IF_ERROR(CopyBuffer(gpu_data_out_->buffer, tensor));
        }
        return ::mediapipe::OkStatus();
      }));
  cc->Outputs()
      .Tag(kTensorsGpuTag)
      .Add(output_tensors.release(), cc->InputTimestamp());
#elif MEDIAPIPE_TFLITE_METAL_INFERENCE
  // GpuBuffer to id<MTLBuffer> conversion.
  const auto& input =
      cc->Inputs().Tag(kGpuBufferTag).Get<mediapipe::GpuBuffer>();
  id<MTLCommandBuffer> command_buffer = [gpu_helper_ commandBuffer];

  id<MTLTexture> src_texture = [gpu_helper_ metalTextureWithGpuBuffer:input];
  command_buffer.label = @"TfLiteConverterCalculatorConvertAndBlit";
  id<MTLComputeCommandEncoder> compute_encoder =
      [command_buffer computeCommandEncoder];
  [compute_encoder setComputePipelineState:gpu_data_out_->pipeline_state];
  [compute_encoder setTexture:src_texture atIndex:0];
  [compute_encoder setBuffer:gpu_data_out_->buffer offset:0 atIndex:1];
  MTLSize threads_per_group = MTLSizeMake(kWorkgroupSize, kWorkgroupSize, 1);
  MTLSize threadgroups =
      MTLSizeMake(NumGroups(input.width(), kWorkgroupSize),
                  NumGroups(input.height(), kWorkgroupSize), 1);
  [compute_encoder dispatchThreadgroups:threadgroups
                  threadsPerThreadgroup:threads_per_group];
  [compute_encoder endEncoding];

  // Copy into outputs.
  // TODO Avoid this copy.
  auto output_tensors = absl::make_unique<std::vector<GpuTensor>>();
  output_tensors->resize(1);
  id<MTLDevice> device = gpu_helper_.mtlDevice;
  output_tensors->at(0) =
      [device newBufferWithLength:gpu_data_out_->elements * sizeof(float)
                          options:MTLResourceStorageModeShared];
  [MPPMetalUtil blitMetalBufferTo:output_tensors->at(0)
                             from:gpu_data_out_->buffer
                         blocking:false
                    commandBuffer:command_buffer];

  cc->Outputs()
      .Tag(kTensorsGpuTag)
      .Add(output_tensors.release(), cc->InputTimestamp());
#else
  RET_CHECK_FAIL() << "GPU processing is not enabled.";
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteConverterCalculator::InitGpu(CalculatorContext* cc) {
#if MEDIAPIPE_TFLITE_GPU_SUPPORTED
  // Get input image sizes.
  const auto& input =
      cc->Inputs().Tag(kGpuBufferTag).Get<mediapipe::GpuBuffer>();
  mediapipe::ImageFormat::Format format =
      mediapipe::ImageFormatForGpuBufferFormat(input.format());
  gpu_data_out_ = absl::make_unique<GPUData>();
  gpu_data_out_->elements = input.height() * input.width() * max_num_channels_;
  const bool include_alpha = (max_num_channels_ == 4);
  const bool single_channel = (max_num_channels_ == 1);
  if (!(format == mediapipe::ImageFormat::GRAY8 ||
        format == mediapipe::ImageFormat::SRGB ||
        format == mediapipe::ImageFormat::SRGBA))
    RET_CHECK_FAIL() << "Unsupported GPU input format.";
  if (include_alpha && (format != mediapipe::ImageFormat::SRGBA))
    RET_CHECK_FAIL() << "Num input channels is less than desired output.";
#endif  // MEDIAPIPE_TFLITE_GPU_SUPPORTED

#if MEDIAPIPE_TFLITE_GL_INFERENCE
  MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext(
      [this, &include_alpha, &input, &single_channel]() -> ::mediapipe::Status {
        // Device memory.
        MP_RETURN_IF_ERROR(
            ::tflite::gpu::gl::CreateReadWriteShaderStorageBuffer<float>(
                gpu_data_out_->elements, &gpu_data_out_->buffer));

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
                ? absl::Substitute(
                      "pixel = pixel * float($0) + float($1);",
                      (output_range_->second - output_range_->first),
                      output_range_->first)
                : "",
            /*$4=*/flip_vertically_ ? "(width_height.y - 1 - gid.y)" : "gid.y",
            /*$5=*/
            single_channel
                ? ""
                : R"(output_data.elements[linear_index + 1] = pixel.y;
                            output_data.elements[linear_index + 2] = pixel.z;)",
            /*$6=*/
            include_alpha ? "output_data.elements[linear_index + 3] = pixel.w;"
                          : "",
            /*$7=*/max_num_channels_);
        MP_RETURN_IF_ERROR(GlShader::CompileShader(
            GL_COMPUTE_SHADER, shader_source, &gpu_data_out_->shader));
        MP_RETURN_IF_ERROR(GlProgram::CreateWithShader(
            gpu_data_out_->shader, &gpu_data_out_->program));
        return ::mediapipe::OkStatus();
      }));

#elif MEDIAPIPE_TFLITE_METAL_INFERENCE

  RET_CHECK(include_alpha)
      << "iOS GPU inference currently accepts only RGBA input.";

  // Device memory.
  id<MTLDevice> device = gpu_helper_.mtlDevice;
  gpu_data_out_->buffer =
      [device newBufferWithLength:gpu_data_out_->elements * sizeof(float)
                          options:MTLResourceStorageModeShared];

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
    $0 pixel = $0(in_tex.sample(texture_sampler, coord).$1);
    $2   // normalize [-1,1]
    const int linear_index = $4 * ($3 * in_tex.get_width() + gid.x);
    out_buf[linear_index + 0] = pixel.x;
    out_buf[linear_index + 1] = pixel.y;
    out_buf[linear_index + 2] = pixel.z;
    $5  // alpha channel
  }
      )",
      /*$0=*/include_alpha ? "float4" : "float3",
      /*$1=*/include_alpha ? "rgba" : "rgb",
      /*$2=*/
      output_range_.has_value()
          ? absl::Substitute("pixel = pixel * float($0) + float($1);",
                             (output_range_->second - output_range_->first),
                             output_range_->first)
          : "",
      /*$3=*/flip_vertically_ ? "(in_tex.get_height() - 1 - gid.y)" : "gid.y",
      /*$4=*/include_alpha ? 4 : 3,
      /*$5=*/include_alpha ? "out_buf[linear_index + 3] = pixel.w;" : "");

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
  gpu_data_out_->pipeline_state =
      [device newComputePipelineStateWithFunction:kernel_func error:&error];
  RET_CHECK(gpu_data_out_->pipeline_state != nil)
      << "Couldn't create pipeline state "
      << [[error localizedDescription] UTF8String];
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteConverterCalculator::LoadOptions(
    CalculatorContext* cc) {
  // Get calculator options specified in the graph.
  const auto& options =
      cc->Options<::mediapipe::TfLiteConverterCalculatorOptions>();

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
#if defined(MEDIAPIPE_IOS)
  if (cc->Inputs().HasTag(kGpuBufferTag))
    // Currently on iOS, tflite gpu input tensor must be 4 channels,
    // so input image must be 4 channels also (checked in InitGpu).
    max_num_channels_ = 4;
#endif

  // Get tensor type, float or quantized.
  use_quantized_tensors_ = options.use_quantized_tensors();

  return ::mediapipe::OkStatus();
}

template <class T>
::mediapipe::Status TfLiteConverterCalculator::NormalizeImage(
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

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteConverterCalculator::CopyMatrixToTensor(
    const Matrix& matrix, float* tensor_ptr) {
  if (row_major_matrix_) {
    auto matrix_map =
        Eigen::Map<RowMajorMatrixXf>(tensor_ptr, matrix.rows(), matrix.cols());
    matrix_map = matrix;
  } else {
    auto matrix_map =
        Eigen::Map<ColMajorMatrixXf>(tensor_ptr, matrix.rows(), matrix.cols());
    matrix_map = matrix;
  }

  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
