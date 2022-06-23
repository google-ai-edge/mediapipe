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

#include <vector>

#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "mediapipe/calculators/tensor/tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/gpu/gpu_origin.pb.h"
#include "mediapipe/util/resource_util.h"
#include "tensorflow/lite/interpreter.h"

#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/shader_util.h"
#endif  // !MEDIAPIPE_DISABLE_GPU

#if !MEDIAPIPE_DISABLE_OPENCV
#include "mediapipe/framework/formats/image_opencv.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#endif  // !MEDIAPIPE_DISABLE_OPENCV

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
#include "tensorflow/lite/delegates/gpu/gl/converters/util.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_program.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_shader.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_texture.h"
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31

#if MEDIAPIPE_METAL_ENABLED
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#import "mediapipe/gpu/MPPMetalHelper.h"
#include "mediapipe/gpu/MPPMetalUtil.h"
#endif  // MEDIAPIPE_METAL_ENABLED

namespace {
constexpr int kWorkgroupSize = 8;  // Block size for GPU shader.
enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };

// Commonly used to compute the number of blocks to launch in a kernel.
int NumGroups(const int size, const int group_size) {  // NOLINT
  return (size + group_size - 1) / group_size;
}

bool CanUseGpu() {
#if !MEDIAPIPE_DISABLE_GPU || MEDIAPIPE_METAL_ENABLED
  // TODO: Configure GPU usage policy in individual calculators.
  constexpr bool kAllowGpuProcessing = true;
  return kAllowGpuProcessing;
#else
  return false;
#endif  // !MEDIAPIPE_DISABLE_GPU || MEDIAPIPE_METAL_ENABLED
}

constexpr char kTensorsTag[] = "TENSORS";
constexpr char kOutputSizeTag[] = "OUTPUT_SIZE";
constexpr char kMaskTag[] = "MASK";

absl::StatusOr<std::tuple<int, int, int>> GetHwcFromDims(
    const std::vector<int>& dims) {
  if (dims.size() == 3) {
    return std::make_tuple(dims[0], dims[1], dims[2]);
  } else if (dims.size() == 4) {
    // BHWC format check B == 1
    RET_CHECK_EQ(1, dims[0]) << "Expected batch to be 1 for BHWC heatmap";
    return std::make_tuple(dims[1], dims[2], dims[3]);
  } else {
    RET_CHECK(false) << "Invalid shape for segmentation tensor " << dims.size();
  }
}
}  // namespace

namespace mediapipe {

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
using ::tflite::gpu::gl::GlProgram;
using ::tflite::gpu::gl::GlShader;
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31

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
//   TENSORS: Vector of Tensor,
//            The tensor dimensions are specified in this calculator's options.
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
  absl::Status ProcessGpu(CalculatorContext* cc);
  absl::Status ProcessCpu(CalculatorContext* cc);
  void GlRender();

  bool DoesGpuTextureStartAtBottom() {
    return options_.gpu_origin() != mediapipe::GpuOrigin_Mode_TOP_LEFT;
  }

#if !MEDIAPIPE_DISABLE_OPENCV
  template <class T>
  absl::Status ApplyActivation(cv::Mat& tensor_mat, cv::Mat* small_mask_mat);
#endif  // !MEDIAPIPE_DISABLE_OPENCV
  ::mediapipe::TensorsToSegmentationCalculatorOptions options_;

#if !MEDIAPIPE_DISABLE_GPU
  mediapipe::GlCalculatorHelper gpu_helper_;
  GLuint upsample_program_;
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
  std::unique_ptr<GlProgram> mask_program_31_;
#else
  GLuint mask_program_20_;
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
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
    MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
#if MEDIAPIPE_METAL_ENABLED
    MP_RETURN_IF_ERROR([MPPMetalHelper updateContract:cc]);
#endif  // MEDIAPIPE_METAL_ENABLED
#endif  // !MEDIAPIPE_DISABLE_GPU
  }

  return absl::OkStatus();
}

absl::Status TensorsToSegmentationCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));
  bool use_gpu = false;

  if (CanUseGpu()) {
#if !MEDIAPIPE_DISABLE_GPU
    use_gpu = true;
    MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
#if MEDIAPIPE_METAL_ENABLED
    metal_helper_ = [[MPPMetalHelper alloc] initWithCalculatorContext:cc];
    RET_CHECK(metal_helper_);
#endif  // MEDIAPIPE_METAL_ENABLED
#endif  // !MEDIAPIPE_DISABLE_GPU
  }

  MP_RETURN_IF_ERROR(LoadOptions(cc));

  if (use_gpu) {
#if !MEDIAPIPE_DISABLE_GPU
    MP_RETURN_IF_ERROR(InitGpu(cc));
#else
    RET_CHECK_FAIL() << "GPU processing disabled.";
#endif  // !MEDIAPIPE_DISABLE_GPU
  }

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
    ASSIGN_OR_RETURN(auto hwc, GetHwcFromDims(input_tensors[0].shape().dims));
    int tensor_channels = std::get<2>(hwc);
    typedef mediapipe::TensorsToSegmentationCalculatorOptions Options;
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

  if (use_gpu) {
#if !MEDIAPIPE_DISABLE_GPU
    MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([this, cc]() -> absl::Status {
      MP_RETURN_IF_ERROR(ProcessGpu(cc));
      return absl::OkStatus();
    }));
#else
    RET_CHECK_FAIL() << "GPU processing disabled.";
#endif  // !MEDIAPIPE_DISABLE_GPU
  } else {
#if !MEDIAPIPE_DISABLE_OPENCV
    MP_RETURN_IF_ERROR(ProcessCpu(cc));
#else
    RET_CHECK_FAIL() << "OpenCV processing disabled.";
#endif  // !MEDIAPIPE_DISABLE_OPENCV
  }

  return absl::OkStatus();
}

absl::Status TensorsToSegmentationCalculator::Close(CalculatorContext* cc) {
#if !MEDIAPIPE_DISABLE_GPU
  gpu_helper_.RunInGlContext([this] {
    if (upsample_program_) glDeleteProgram(upsample_program_);
    upsample_program_ = 0;
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
    mask_program_31_.reset();
#else
    if (mask_program_20_) glDeleteProgram(mask_program_20_);
    mask_program_20_ = 0;
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
#if MEDIAPIPE_METAL_ENABLED
    mask_program_ = nil;
#endif  // MEDIAPIPE_METAL_ENABLED
  });
#endif  // !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
}

absl::Status TensorsToSegmentationCalculator::ProcessCpu(
    CalculatorContext* cc) {
#if !MEDIAPIPE_DISABLE_OPENCV
  // Get input streams, and dimensions.
  const auto& input_tensors =
      cc->Inputs().Tag(kTensorsTag).Get<std::vector<Tensor>>();
  ASSIGN_OR_RETURN(auto hwc, GetHwcFromDims(input_tensors[0].shape().dims));
  auto [tensor_height, tensor_width, tensor_channels] = hwc;
  int output_width = tensor_width, output_height = tensor_height;
  if (cc->Inputs().HasTag(kOutputSizeTag)) {
    const auto& size =
        cc->Inputs().Tag(kOutputSizeTag).Get<std::pair<int, int>>();
    output_width = size.first;
    output_height = size.second;
  }

  // Create initial working mask.
  cv::Mat small_mask_mat(cv::Size(tensor_width, tensor_height), CV_32FC1);

  // Wrap input tensor.
  auto raw_input_tensor = &input_tensors[0];
  auto raw_input_view = raw_input_tensor->GetCpuReadView();
  const float* raw_input_data = raw_input_view.buffer<float>();
  cv::Mat tensor_mat(cv::Size(tensor_width, tensor_height),
                     CV_MAKETYPE(CV_32F, tensor_channels),
                     const_cast<float*>(raw_input_data));

  // Process mask tensor and apply activation function.
  if (tensor_channels == 2) {
    MP_RETURN_IF_ERROR(ApplyActivation<cv::Vec2f>(tensor_mat, &small_mask_mat));
  } else if (tensor_channels == 1) {
    RET_CHECK(mediapipe::TensorsToSegmentationCalculatorOptions::SOFTMAX !=
              options_.activation());  // Requires 2 channels.
    if (mediapipe::TensorsToSegmentationCalculatorOptions::NONE ==
        options_.activation())  // Pass-through optimization.
      tensor_mat.copyTo(small_mask_mat);
    else
      MP_RETURN_IF_ERROR(ApplyActivation<float>(tensor_mat, &small_mask_mat));
  } else {
    RET_CHECK_FAIL() << "Unsupported number of tensor channels "
                     << tensor_channels;
  }

  // Send out image as CPU packet.
  std::shared_ptr<ImageFrame> mask_frame = std::make_shared<ImageFrame>(
      ImageFormat::VEC32F1, output_width, output_height);
  std::unique_ptr<Image> output_mask = absl::make_unique<Image>(mask_frame);
  auto output_mat = formats::MatView(output_mask.get());
  // Upsample small mask into output.
  cv::resize(small_mask_mat, *output_mat,
             cv::Size(output_width, output_height));
  cc->Outputs().Tag(kMaskTag).Add(output_mask.release(), cc->InputTimestamp());
#endif  // !MEDIAPIPE_DISABLE_OPENCV

  return absl::OkStatus();
}

#if !MEDIAPIPE_DISABLE_OPENCV
template <class T>
absl::Status TensorsToSegmentationCalculator::ApplyActivation(
    cv::Mat& tensor_mat, cv::Mat* small_mask_mat) {
  // Configure activation function.
  const int output_layer_index = options_.output_layer_index();
  typedef mediapipe::TensorsToSegmentationCalculatorOptions Options;
  const auto activation_fn = [&](const cv::Vec2f& mask_value) {
    float new_mask_value = 0;
    // TODO consider moving switch out of the loop,
    // and also avoid float/Vec2f casting.
    switch (options_.activation()) {
      case Options::NONE: {
        new_mask_value = mask_value[0];
        break;
      }
      case Options::SIGMOID: {
        const float pixel0 = mask_value[0];
        new_mask_value = 1.0 / (std::exp(-pixel0) + 1.0);
        break;
      }
      case Options::SOFTMAX: {
        const float pixel0 = mask_value[0];
        const float pixel1 = mask_value[1];
        const float max_pixel = std::max(pixel0, pixel1);
        const float min_pixel = std::min(pixel0, pixel1);
        const float softmax_denom =
            /*exp(max_pixel - max_pixel)=*/1.0f +
            std::exp(min_pixel - max_pixel);
        new_mask_value = std::exp(mask_value[output_layer_index] - max_pixel) /
                         softmax_denom;
        break;
      }
    }
    return new_mask_value;
  };

  // Process mask tensor.
  for (int i = 0; i < tensor_mat.rows; ++i) {
    for (int j = 0; j < tensor_mat.cols; ++j) {
      const T& input_pix = tensor_mat.at<T>(i, j);
      const float mask_value = activation_fn(input_pix);
      small_mask_mat->at<float>(i, j) = mask_value;
    }
  }

  return absl::OkStatus();
}
#endif  // !MEDIAPIPE_DISABLE_OPENCV

// Steps:
// 1. receive tensor
// 2. process segmentation tensor into small mask
// 3. upsample small mask into output mask to be same size as input image
absl::Status TensorsToSegmentationCalculator::ProcessGpu(
    CalculatorContext* cc) {
#if !MEDIAPIPE_DISABLE_GPU
  // Get input streams, and dimensions.
  const auto& input_tensors =
      cc->Inputs().Tag(kTensorsTag).Get<std::vector<Tensor>>();
  ASSIGN_OR_RETURN(auto hwc, GetHwcFromDims(input_tensors[0].shape().dims));
  auto [tensor_height, tensor_width, tensor_channels] = hwc;
  int output_width = tensor_width, output_height = tensor_height;
  if (cc->Inputs().HasTag(kOutputSizeTag)) {
    const auto& size =
        cc->Inputs().Tag(kOutputSizeTag).Get<std::pair<int, int>>();
    output_width = size.first;
    output_height = size.second;
  }

  // Create initial working mask texture.
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
  tflite::gpu::gl::GlTexture small_mask_texture;
#else
  mediapipe::GlTexture small_mask_texture;
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31

  // Run shader, process mask tensor.
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
  {
    MP_RETURN_IF_ERROR(CreateReadWriteRgbaImageTexture(
        tflite::gpu::DataType::UINT8,  // GL_RGBA8
        {tensor_width, tensor_height}, &small_mask_texture));

    const int output_index = 0;
    glBindImageTexture(output_index, small_mask_texture.id(), 0, GL_FALSE, 0,
                       GL_WRITE_ONLY, GL_RGBA8);

    auto read_view = input_tensors[0].GetOpenGlBufferReadView();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, read_view.name());

    const tflite::gpu::uint3 workgroups = {
        NumGroups(tensor_width, kWorkgroupSize),
        NumGroups(tensor_height, kWorkgroupSize), 1};

    glUseProgram(mask_program_31_->id());
    glUniform2i(glGetUniformLocation(mask_program_31_->id(), "out_size"),
                tensor_width, tensor_height);

    MP_RETURN_IF_ERROR(mask_program_31_->Dispatch(workgroups));
  }
#elif MEDIAPIPE_METAL_ENABLED
  {
    id<MTLCommandBuffer> command_buffer = [metal_helper_ commandBuffer];
    command_buffer.label = @"SegmentationKernel";
    id<MTLComputeCommandEncoder> command_encoder =
        [command_buffer computeCommandEncoder];
    [command_encoder setComputePipelineState:mask_program_];

    auto read_view = input_tensors[0].GetMtlBufferReadView(command_buffer);
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
#else
  {
    small_mask_texture = gpu_helper_.CreateDestinationTexture(
        tensor_width, tensor_height,
        mediapipe::GpuBufferFormat::kBGRA32);  // actually GL_RGBA8

    // Go through CPU if not already texture 2D (no direct conversion yet).
    // Tensor::GetOpenGlTexture2dReadView() doesn't automatically convert types.
    if (!input_tensors[0].ready_as_opengl_texture_2d()) {
      (void)input_tensors[0].GetCpuReadView();
    }

    auto read_view = input_tensors[0].GetOpenGlTexture2dReadView();

    gpu_helper_.BindFramebuffer(small_mask_texture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, read_view.name());
    glUseProgram(mask_program_20_);
    GlRender();
    glBindTexture(GL_TEXTURE_2D, 0);
    glFlush();
  }
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31

  // Upsample small mask into output.
  mediapipe::GlTexture output_texture = gpu_helper_.CreateDestinationTexture(
      output_width, output_height,
      mediapipe::GpuBufferFormat::kBGRA32);  // actually GL_RGBA8

  // Run shader, upsample result.
  {
    gpu_helper_.BindFramebuffer(output_texture);
    glActiveTexture(GL_TEXTURE1);
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
    glBindTexture(GL_TEXTURE_2D, small_mask_texture.id());
#else
    glBindTexture(GL_TEXTURE_2D, small_mask_texture.name());
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
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
#endif  // !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
}

void TensorsToSegmentationCalculator::GlRender() {
#if !MEDIAPIPE_DISABLE_GPU
  static const GLfloat square_vertices[] = {
      -1.0f, -1.0f,  // bottom left
      1.0f,  -1.0f,  // bottom right
      -1.0f, 1.0f,   // top left
      1.0f,  1.0f,   // top right
  };
  static const GLfloat texture_vertices[] = {
      0.0f, 0.0f,  // bottom left
      1.0f, 0.0f,  // bottom right
      0.0f, 1.0f,  // top left
      1.0f, 1.0f,  // top right
  };

  // vertex storage
  GLuint vbo[2];
  glGenBuffers(2, vbo);
  GLuint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  // vbo 0
  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(GLfloat), square_vertices,
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(ATTRIB_VERTEX);
  glVertexAttribPointer(ATTRIB_VERTEX, 2, GL_FLOAT, 0, 0, nullptr);

  // vbo 1
  glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
  glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(GLfloat), texture_vertices,
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(ATTRIB_TEXTURE_POSITION);
  glVertexAttribPointer(ATTRIB_TEXTURE_POSITION, 2, GL_FLOAT, 0, 0, nullptr);

  // draw
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  // cleanup
  glDisableVertexAttribArray(ATTRIB_VERTEX);
  glDisableVertexAttribArray(ATTRIB_TEXTURE_POSITION);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
  glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(2, vbo);
#endif  // !MEDIAPIPE_DISABLE_GPU
}

absl::Status TensorsToSegmentationCalculator::LoadOptions(
    CalculatorContext* cc) {
  // Get calculator options specified in the graph.
  options_ = cc->Options<::mediapipe::TensorsToSegmentationCalculatorOptions>();

  return absl::OkStatus();
}

absl::Status TensorsToSegmentationCalculator::InitGpu(CalculatorContext* cc) {
#if !MEDIAPIPE_DISABLE_GPU
  MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([this]() -> absl::Status {
  // A shader to process a segmentation tensor into an output mask.
  // Currently uses 4 channels for output, and sets R+A channels as mask value.
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
    // GLES 3.1
    const tflite::gpu::uint3 workgroup_size = {kWorkgroupSize, kWorkgroupSize,
                                               1};
    const std::string shader_header =
        absl::StrCat(tflite::gpu::gl::GetShaderHeader(workgroup_size), R"(
precision highp float;

layout(rgba8, binding = 0) writeonly uniform highp image2D output_texture;

uniform ivec2 out_size;
)");
    /* Shader defines will be inserted here. */

    const std::string shader_src_main = R"(
layout(std430, binding = 2) readonly buffer B0 {
#ifdef TWO_CHANNEL_INPUT
  vec2 elements[];
#else
  float elements[];
#endif // TWO_CHANNEL_INPUT
} input_data;   // data tensor

void main() {
  int out_width = out_size.x;
  int out_height = out_size.y;

  ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
  if (gid.x >= out_width || gid.y >= out_height) { return; }
  int linear_index = gid.y * out_width + gid.x;

#ifdef TWO_CHANNEL_INPUT
  vec2 input_value = input_data.elements[linear_index];
#else
  vec2 input_value = vec2(input_data.elements[linear_index], 0.0);
#endif // TWO_CHANNEL_INPUT

// Run activation function.
// One and only one of FN_SOFTMAX,FN_SIGMOID,FN_NONE will be defined.
#ifdef FN_SOFTMAX
  // Only two channel input tensor is supported.
  vec2 input_px = input_value.rg;
  float shift = max(input_px.r, input_px.g);
  float softmax_denom = exp(input_px.r - shift) + exp(input_px.g - shift);
  float new_mask_value =
      exp(input_px[OUTPUT_LAYER_INDEX] - shift) / softmax_denom;
#endif // FN_SOFTMAX

#ifdef FN_SIGMOID
  float new_mask_value = 1.0 / (exp(-input_value.r) + 1.0);
#endif // FN_SIGMOID

#ifdef FN_NONE
  float new_mask_value = input_value.r;
#endif // FN_NONE

#ifdef FLIP_Y_COORD
  int y_coord = out_height - gid.y - 1;
#else
  int y_coord = gid.y;
#endif  // defined(FLIP_Y_COORD)
  ivec2 output_coordinate = ivec2(gid.x, y_coord);

  vec4 out_value = vec4(new_mask_value, 0.0, 0.0, new_mask_value);
  imageStore(output_texture, output_coordinate, out_value);
})";

#elif MEDIAPIPE_METAL_ENABLED
    // METAL
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

#else
    // GLES 2.0
    const std::string shader_header = absl::StrCat(
        std::string(mediapipe::kMediaPipeFragmentShaderPreamble), R"(
DEFAULT_PRECISION(mediump, float)
)");
    /* Shader defines will be inserted here. */

    const std::string shader_src_main = R"(
in vec2 sample_coordinate;

uniform sampler2D input_texture;

#ifdef GL_ES
#define fragColor gl_FragColor
#else
out vec4 fragColor;
#endif  // defined(GL_ES);

void main() {
#ifdef FLIP_Y_COORD
  float y_coord = 1.0 - sample_coordinate.y;
#else
  float y_coord = sample_coordinate.y;
#endif  // defined(FLIP_Y_COORD)
  vec2 adjusted_coordinate = vec2(sample_coordinate.x, y_coord);
  vec4 input_value = texture2D(input_texture, adjusted_coordinate);

  // Run activation function.
  // One and only one of FN_SOFTMAX,FN_SIGMOID,FN_NONE will be defined.

#ifdef FN_SOFTMAX
  // Only two channel input tensor is supported.
  vec2 input_px = input_value.rg;
  float shift = max(input_px.r, input_px.g);
  float softmax_denom = exp(input_px.r - shift) + exp(input_px.g - shift);
  float new_mask_value =
      exp(mix(input_px.r, input_px.g, float(OUTPUT_LAYER_INDEX)) - shift) / softmax_denom;
#endif // FN_SOFTMAX

#ifdef FN_SIGMOID
  float new_mask_value = 1.0 / (exp(-input_value.r) + 1.0);
#endif // FN_SIGMOID

#ifdef FN_NONE
  float new_mask_value = input_value.r;
#endif // FN_NONE

  vec4 out_value = vec4(new_mask_value, 0.0, 0.0, new_mask_value);
  fragColor = out_value;
})";
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31

    // Shader defines.
    typedef mediapipe::TensorsToSegmentationCalculatorOptions Options;
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
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
    GlShader shader_without_previous;
    MP_RETURN_IF_ERROR(GlShader::CompileShader(
        GL_COMPUTE_SHADER, shader_src_no_previous, &shader_without_previous));
    mask_program_31_ = absl::make_unique<GlProgram>();
    MP_RETURN_IF_ERROR(GlProgram::CreateWithShader(shader_without_previous,
                                                   mask_program_31_.get()));
#elif MEDIAPIPE_METAL_ENABLED
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
#else
    mediapipe::GlhCreateProgram(
        mediapipe::kBasicVertexShader, shader_src_no_previous.c_str(),
        NUM_ATTRIBUTES, &attr_name[0], attr_location, &mask_program_20_);
    RET_CHECK(mask_program_20_) << "Problem initializing the program.";
    glUseProgram(mask_program_20_);
    glUniform1i(glGetUniformLocation(mask_program_20_, "input_texture"), 1);
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31

    // Simple pass-through program, used for hardware upsampling.
    mediapipe::GlhCreateProgram(
        mediapipe::kBasicVertexShader, mediapipe::kBasicTexturedFragmentShader,
        NUM_ATTRIBUTES, &attr_name[0], attr_location, &upsample_program_);
    RET_CHECK(upsample_program_) << "Problem initializing the program.";
    glUseProgram(upsample_program_);
    glUniform1i(glGetUniformLocation(upsample_program_, "video_frame"), 1);

    return absl::OkStatus();
  }));
#endif  // !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
}

}  // namespace mediapipe
