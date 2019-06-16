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

#include "mediapipe/calculators/tflite/tflite_converter_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/resource_util.h"
#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/interpreter.h"

#if defined(__ANDROID__)
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_program.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_shader.h"
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#endif  // ANDROID

namespace {
constexpr int kWorkgroupSize = 8;  // Block size for GPU shader.
// Commonly used to compute the number of blocks to launch in a kernel.
int RoundUp(const int size, const int multiple) {
  return (size + multiple - 1) / multiple;
}
}  // namespace

namespace mediapipe {

#if defined(__ANDROID__)
using ::tflite::gpu::gl::GlBuffer;
using ::tflite::gpu::gl::GlProgram;
using ::tflite::gpu::gl::GlShader;
struct GPUData {
  int width;
  int height;
  int channels;
  GlBuffer ssbo;
  GlShader shader;
  GlProgram program;
};
#endif  // ANDROID

// Calculator for normalizing and converting an ImageFrame or GpuBuffer
// into a TfLiteTensor (float 32) or tflite::gpu::GlBuffer, respetively.
//
// This calculator is designed to be used with the TfLiteInferenceCalcualtor,
// as a pre-processing step for calculator inputs.
//
// Input data is normalized to [-1,1] (default) or [0,1], specified by options.
//
// Input:
//  IMAGE - ImageFrame (assumed to be 8-bit or 32-bit data).
//  IMAGE_GPU - GpuBuffer (assumed to be RGBA or RGB GL texture)
//
// Output:
//  TENSORS - Vector of TfLiteTensor of type kTfLiteFloat32
//  TENSORS_GPU - vector of GlBuffer
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
//  GPU tensors are currently only supported on Android.
//  This calculator uses FixedSizeInputStreamHandler by default.
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
                                     bool zero_center, bool flip_vertically,
                                     float* tensor_buffer);

  std::unique_ptr<tflite::Interpreter> interpreter_ = nullptr;

#if defined(__ANDROID__)
  mediapipe::GlCalculatorHelper gpu_helper_;
  std::unique_ptr<GPUData> gpu_data_out_;
#endif

  bool initialized_ = false;
  bool use_gpu_ = false;
  bool zero_center_ = true;  // normalize range to [-1,1] | otherwise [0,1]
  bool flip_vertically_ = false;
  int max_num_channels_ = 3;
};
REGISTER_CALCULATOR(TfLiteConverterCalculator);

::mediapipe::Status TfLiteConverterCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag("IMAGE") || cc->Inputs().HasTag("IMAGE_GPU"));
  RET_CHECK(cc->Outputs().HasTag("TENSORS") ||
            cc->Outputs().HasTag("TENSORS_GPU"));

  if (cc->Inputs().HasTag("IMAGE")) cc->Inputs().Tag("IMAGE").Set<ImageFrame>();
#if defined(__ANDROID__)
  if (cc->Inputs().HasTag("IMAGE_GPU"))
    cc->Inputs().Tag("IMAGE_GPU").Set<mediapipe::GpuBuffer>();
#endif

  if (cc->Outputs().HasTag("TENSORS"))
    cc->Outputs().Tag("TENSORS").Set<std::vector<TfLiteTensor>>();
#if defined(__ANDROID__)
  if (cc->Outputs().HasTag("TENSORS_GPU"))
    cc->Outputs().Tag("TENSORS_GPU").Set<std::vector<GlBuffer>>();
#endif

#if defined(__ANDROID__)
  RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
#endif

  // Assign this calculator's default InputStreamHandler.
  cc->SetInputStreamHandler("FixedSizeInputStreamHandler");

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteConverterCalculator::Open(CalculatorContext* cc) {
  RETURN_IF_ERROR(LoadOptions(cc));

  if (cc->Inputs().HasTag("IMAGE_GPU") ||
      cc->Outputs().HasTag("IMAGE_OUT_GPU")) {
#if defined(__ANDROID__)
    use_gpu_ = true;
#else
    RET_CHECK_FAIL() << "GPU processing on non-Android not supported yet.";
#endif
  }

  if (use_gpu_) {
    // Cannot mix CPU/GPU streams.
    RET_CHECK(cc->Inputs().HasTag("IMAGE_GPU") &&
              cc->Outputs().HasTag("TENSORS_GPU"));
#if defined(__ANDROID__)
    RETURN_IF_ERROR(gpu_helper_.Open(cc));
#endif
  } else {
    interpreter_ = absl::make_unique<tflite::Interpreter>();
    interpreter_->AddTensors(1);
    interpreter_->SetInputs({0});
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteConverterCalculator::Process(CalculatorContext* cc) {
  if (use_gpu_) {
    // GpuBuffer to tflite::gpu::GlBuffer conversion.
#if defined(__ANDROID__)
    if (!initialized_) {
      RETURN_IF_ERROR(InitGpu(cc));
      initialized_ = true;
    }

    const auto& input =
        cc->Inputs().Tag("IMAGE_GPU").Get<mediapipe::GpuBuffer>();
    RETURN_IF_ERROR(
        gpu_helper_.RunInGlContext([this, &input]() -> ::mediapipe::Status {
          // Convert GL texture into TfLite GlBuffer (SSBO).
          auto src = gpu_helper_.CreateSourceTexture(input);
          glActiveTexture(GL_TEXTURE0 + 0);
          glBindTexture(GL_TEXTURE_2D, src.name());
          auto status = gpu_data_out_->ssbo.BindToIndex(1);
          if (!status.ok()) {
            return ::mediapipe::InternalError(status.error_message());
          }
          const tflite::gpu::uint3 workgroups = {
              RoundUp(gpu_data_out_->width, kWorkgroupSize),
              RoundUp(gpu_data_out_->height, kWorkgroupSize), 1};
          status = gpu_data_out_->program.Dispatch(workgroups);
          if (!status.ok()) {
            return ::mediapipe::InternalError(status.error_message());
          }
          glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
          glBindTexture(GL_TEXTURE_2D, 0);
          src.Release();
          return ::mediapipe::OkStatus();
        }));

    auto output_tensors = absl::make_unique<std::vector<GlBuffer>>();
    output_tensors->resize(1);
    for (int i = 0; i < 1; ++i) {
      GlBuffer& tensor = output_tensors->at(i);
      using ::tflite::gpu::gl::CreateReadWriteShaderStorageBuffer;
      auto status = CreateReadWriteShaderStorageBuffer<float>(
          gpu_data_out_->width * gpu_data_out_->height *
              gpu_data_out_->channels,
          &tensor);
      if (!status.ok()) {
        return ::mediapipe::InternalError(status.error_message());
      }
      tflite::gpu::gl::CopyBuffer(gpu_data_out_->ssbo, tensor);
    }
    cc->Outputs()
        .Tag("TENSORS_GPU")
        .Add(output_tensors.release(), cc->InputTimestamp());
#else
    RET_CHECK_FAIL()
        << "GPU input on non-Android devices is not supported yet.";
#endif
  } else {
    // CPU ImageFrame to TfLiteTensor conversion.

    const auto& image_frame = cc->Inputs().Tag("IMAGE").Get<ImageFrame>();
    const int height = image_frame.Height();
    const int width = image_frame.Width();
    const int channels_preserved =
        std::min(image_frame.NumberOfChannels(), max_num_channels_);

    if (!(image_frame.Format() == mediapipe::ImageFormat::SRGBA ||
          image_frame.Format() == mediapipe::ImageFormat::SRGB ||
          image_frame.Format() == mediapipe::ImageFormat::GRAY8 ||
          image_frame.Format() == mediapipe::ImageFormat::VEC32F1))
      RET_CHECK_FAIL() << "Unsupported CPU input format.";

    if (!initialized_) {
      interpreter_->SetTensorParametersReadWrite(
          0, kTfLiteFloat32, "", {channels_preserved}, TfLiteQuantization());
      initialized_ = true;
    }

    const int tensor_idx = interpreter_->inputs()[0];
    TfLiteTensor* tensor = interpreter_->tensor(tensor_idx);
    interpreter_->ResizeInputTensor(tensor_idx,
                                    {height, width, channels_preserved});
    interpreter_->AllocateTensors();

    float* tensor_buffer = tensor->data.f;
    RET_CHECK(tensor_buffer);

    if (image_frame.ByteDepth() == 1) {
      RETURN_IF_ERROR(NormalizeImage<uint8>(image_frame, zero_center_,
                                            flip_vertically_, tensor_buffer));
    } else if (image_frame.ByteDepth() == 4) {
      RETURN_IF_ERROR(NormalizeImage<float>(image_frame, zero_center_,
                                            flip_vertically_, tensor_buffer));
    } else {
      return ::mediapipe::InternalError(
          "Only byte-based (8 bit) and float (32 bit) images supported.");
    }

    auto output_tensors = absl::make_unique<std::vector<TfLiteTensor>>();
    output_tensors->emplace_back(*tensor);
    cc->Outputs().Tag("TENSORS").Add(output_tensors.release(),
                                     cc->InputTimestamp());
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteConverterCalculator::Close(CalculatorContext* cc) {
#if defined(__ANDROID__)
  gpu_helper_.RunInGlContext([this] { gpu_data_out_.reset(); });
#endif  // __ANDROID__
  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteConverterCalculator::InitGpu(CalculatorContext* cc) {
#if defined(__ANDROID__)
  // Get input image sizes.
  const auto& input = cc->Inputs().Tag("IMAGE_GPU").Get<mediapipe::GpuBuffer>();

  mediapipe::ImageFormat::Format format =
      mediapipe::ImageFormatForGpuBufferFormat(input.format());

  gpu_data_out_ = absl::make_unique<GPUData>();
  gpu_data_out_->height = input.height();
  gpu_data_out_->width = input.width();
  gpu_data_out_->channels = max_num_channels_;  // desired output channels

  const bool include_alpha = (max_num_channels_ == 4);

  if (!(format == mediapipe::ImageFormat::SRGB ||
        format == mediapipe::ImageFormat::SRGBA))
    RET_CHECK_FAIL() << "Unsupported GPU input format.";

  if (include_alpha && (format != mediapipe::ImageFormat::SRGBA))
    RET_CHECK_FAIL() << "Num input channels is less than desired output.";

  // Shader to convert GL Texture to Shader Storage Buffer Object (SSBO),
  // with normalization to either: [0,1] or [-1,1].
  auto status = ::tflite::gpu::gl::CreateReadWriteShaderStorageBuffer<float>(
      gpu_data_out_->width * gpu_data_out_->height * gpu_data_out_->channels,
      &gpu_data_out_->ssbo);
  if (!status.ok()) {
    return ::mediapipe::InternalError(status.error_message());
  }
  const std::string shader_source = absl::Substitute(
      R"( #version 310 es
          layout(local_size_x = $0, local_size_y = $0) in;
          layout(binding = 0) uniform sampler2D input_texture;
          layout(std430, binding = 1) buffer Output {float elements[];} output_data;
          ivec2 width_height = ivec2($1, $2);
          void main() {
            ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
            if (gid.x >= width_height.x || gid.y >= width_height.y) return;
            $5  // pixel fetch
            $3  // normalize [-1,1]
            int linear_index = $7 * ($4 * width_height.x + gid.x);
            output_data.elements[linear_index + 0] = pixel.x;
            output_data.elements[linear_index + 1] = pixel.y;
            output_data.elements[linear_index + 2] = pixel.z;
            $6  // alpha channel
          })",
      /*$0=*/kWorkgroupSize, /*$1=*/gpu_data_out_->width,
      /*$2=*/gpu_data_out_->height,
      /*$3=*/zero_center_ ? "pixel = (pixel - 0.5) * 2.0;" : "",
      /*$4=*/flip_vertically_ ? "(width_height.y - 1 - gid.y)" : "gid.y",
      /*$5=*/
      include_alpha ? "vec4 pixel = texelFetch(input_texture, gid, 0);"
                    : "vec3 pixel = texelFetch(input_texture, gid, 0).xyz;",
      /*$6=*/
      include_alpha ? "output_data.elements[linear_index + 3] = pixel.w;" : "",
      /*$7=*/include_alpha ? 4 : 3);
  status = GlShader::CompileShader(GL_COMPUTE_SHADER, shader_source,
                                   &gpu_data_out_->shader);
  if (!status.ok()) {
    return ::mediapipe::InternalError(status.error_message());
  }
  status = GlProgram::CreateWithShader(gpu_data_out_->shader,
                                       &gpu_data_out_->program);
  if (!status.ok()) {
    return ::mediapipe::InternalError(status.error_message());
  }
#endif  // ANDROID

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteConverterCalculator::LoadOptions(
    CalculatorContext* cc) {
  // Get calculator options specified in the graph.
  const auto& options =
      cc->Options<::mediapipe::TfLiteConverterCalculatorOptions>();

  // Get data normalization mode.
  zero_center_ = options.zero_center();

  // Get y-flip mode.
  flip_vertically_ = options.flip_vertically();

  // Get desired way to handle input channels.
  max_num_channels_ = options.max_num_channels();
  // Currently only alpha channel toggling is suppored.
  CHECK_GE(max_num_channels_, 3);
  CHECK_LE(max_num_channels_, 4);

  return ::mediapipe::OkStatus();
}

template <class T>
::mediapipe::Status TfLiteConverterCalculator::NormalizeImage(
    const ImageFrame& image_frame, bool zero_center, bool flip_vertically,
    float* tensor_buffer) {
  const int height = image_frame.Height();
  const int width = image_frame.Width();
  const int channels = image_frame.NumberOfChannels();
  const int channels_preserved = std::min(channels, max_num_channels_);
  const int channels_ignored = channels - channels_preserved;

  float div, sub;
  if (zero_center) {
    // [-1,1]
    div = 127.5f;
    sub = 1.0f;
  } else {
    // [0,1]
    div = 255.0f;
    sub = 0.0f;
  }

  for (int i = 0; i < height; ++i) {
    const T* image_ptr = reinterpret_cast<const T*>(
        image_frame.PixelData() +
        (flip_vertically ? height - 1 - i : i) * image_frame.WidthStep());
    for (int j = 0; j < width; ++j) {
      for (int c = 0; c < channels_preserved; ++c) {
        *tensor_buffer++ = *image_ptr++ / div - sub;
      }
      image_ptr += channels_ignored;
    }
  }

  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
