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

#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "mediapipe/calculators/tflite/tflite_tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/resource_util.h"
#include "tensorflow/lite/interpreter.h"

#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/shader_util.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_program.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_shader.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_texture.h"
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#endif  // !MEDIAPIPE_DISABLE_GPU

namespace {
constexpr int kWorkgroupSize = 8;  // Block size for GPU shader.
enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };
// Commonly used to compute the number of blocks to launch in a kernel.
int NumGroups(const int size, const int group_size) {  // NOLINT
  return (size + group_size - 1) / group_size;
}
float Clamp(float val, float min, float max) {
  return std::min(std::max(val, min), max);
}

constexpr char kTensorsTag[] = "TENSORS";
constexpr char kTensorsGpuTag[] = "TENSORS_GPU";
constexpr char kSizeImageTag[] = "REFERENCE_IMAGE";
constexpr char kSizeImageGpuTag[] = "REFERENCE_IMAGE_GPU";
constexpr char kMaskTag[] = "MASK";
constexpr char kMaskGpuTag[] = "MASK_GPU";
constexpr char kPrevMaskTag[] = "PREV_MASK";
constexpr char kPrevMaskGpuTag[] = "PREV_MASK_GPU";

}  // namespace

namespace mediapipe {

#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
using ::tflite::gpu::gl::CopyBuffer;
using ::tflite::gpu::gl::CreateReadWriteRgbaImageTexture;
using ::tflite::gpu::gl::CreateReadWriteShaderStorageBuffer;
using ::tflite::gpu::gl::GlBuffer;
using ::tflite::gpu::gl::GlProgram;
using ::tflite::gpu::gl::GlShader;
#endif  // !MEDIAPIPE_DISABLE_GPU

// Converts TFLite tensors from a tflite segmentation model to an image mask.
//
// Performs optional upscale to REFERENCE_IMAGE dimensions if provided,
// otherwise the mask is the same size as input tensor.
//
// Produces result as an RGBA image, with the mask in both R & A channels. The
// value of each pixel is the probability of the specified class after softmax,
// scaled to 255 on CPU. The class can be specified through the
// |output_layer_index| option.
//
// Inputs:
//   One of the following TENSORS tags:
//   TENSORS: Vector of TfLiteTensor of type kTfLiteFloat32.
//            The tensor dimensions are specified in this calculator's options.
//   TENSORS_GPU: Vector of GlBuffer.
//   One of the following REFERENCE_IMAGE tags:
//   REFERENCE_IMAGE (optional): An ImageFrame input image,
//                               used only for output dimensions.
//   REFERENCE_IMAGE_GPU (optional): A GpuBuffer input image,
//                                   used only for output dimensions.
//   One of the following PREV_MASK tags:
//   PREV_MASK (optional): An ImageFrame input mask, Gray, RGB or RGBA, [0-255].
//   PREV_MASK_GPU (optional): A GpuBuffer input mask, RGBA, [0-1].
// Output:
//   One of the following MASK tags:
//   MASK: An ImageFrame output mask, RGBA.
//   MASK_GPU: A GpuBuffer output mask, RGBA.
//
// Options:
//   See tflite_segmentation_calculator.proto
//
// Usage example:
// node {
//   calculator: "TfLiteTensorsToSegmentationCalculator"
//   input_stream: "TENSORS_GPU:tensors"
//   input_stream: "IMAGE_GPU:input_video"
//   output_stream: "MASK_GPU:hair_mask"
//   node_options: {
//     [mediapipe.TfLiteTensorsToSegmentationCalculatorOptions] {
//       tensor_in_width: 512
//       tensor_in_height: 512
//       tensor_in_channels: 2
//       combine_with_previous_ratio: 1.0
//       output_layer_index: 1
//     }
//   }
// }
//
class TfLiteTensorsToSegmentationCalculator : public CalculatorBase {
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

  ::mediapipe::TfLiteTensorsToSegmentationCalculatorOptions options_;

  int tensor_width_ = 0;
  int tensor_height_ = 0;
  int tensor_channels_ = 0;

  bool use_gpu_ = false;
#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
  mediapipe::GlCalculatorHelper gpu_helper_;
  std::unique_ptr<GlProgram> mask_program_with_prev_;
  std::unique_ptr<GlProgram> mask_program_no_prev_;
  std::unique_ptr<GlBuffer> tensor_buffer_;
  GLuint upsample_program_;
#endif  // !MEDIAPIPE_DISABLE_GPU
};
REGISTER_CALCULATOR(TfLiteTensorsToSegmentationCalculator);

// static
absl::Status TfLiteTensorsToSegmentationCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(!cc->Inputs().GetTags().empty());
  RET_CHECK(!cc->Outputs().GetTags().empty());

  bool use_gpu = false;

  // Inputs CPU.
  if (cc->Inputs().HasTag(kTensorsTag)) {
    cc->Inputs().Tag(kTensorsTag).Set<std::vector<TfLiteTensor>>();
  }
  if (cc->Inputs().HasTag(kPrevMaskTag)) {
    cc->Inputs().Tag(kPrevMaskTag).Set<ImageFrame>();
  }
  if (cc->Inputs().HasTag(kSizeImageTag)) {
    cc->Inputs().Tag(kSizeImageTag).Set<ImageFrame>();
  }

  // Inputs GPU.
#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
  if (cc->Inputs().HasTag(kTensorsGpuTag)) {
    cc->Inputs().Tag(kTensorsGpuTag).Set<std::vector<GlBuffer>>();
    use_gpu |= true;
  }
  if (cc->Inputs().HasTag(kPrevMaskGpuTag)) {
    cc->Inputs().Tag(kPrevMaskGpuTag).Set<mediapipe::GpuBuffer>();
    use_gpu |= true;
  }
  if (cc->Inputs().HasTag(kSizeImageGpuTag)) {
    cc->Inputs().Tag(kSizeImageGpuTag).Set<mediapipe::GpuBuffer>();
    use_gpu |= true;
  }
#endif  // !MEDIAPIPE_DISABLE_GPU

  // Outputs.
  if (cc->Outputs().HasTag(kMaskTag)) {
    cc->Outputs().Tag(kMaskTag).Set<ImageFrame>();
  }
#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
  if (cc->Outputs().HasTag(kMaskGpuTag)) {
    cc->Outputs().Tag(kMaskGpuTag).Set<mediapipe::GpuBuffer>();
    use_gpu |= true;
  }
#endif  // !MEDIAPIPE_DISABLE_GPU

  if (use_gpu) {
#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
    MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
#endif  // !MEDIAPIPE_DISABLE_GPU
  }
  return absl::OkStatus();
}

absl::Status TfLiteTensorsToSegmentationCalculator::Open(
    CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  if (cc->Inputs().HasTag(kTensorsGpuTag)) {
    use_gpu_ = true;
#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
    MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
#endif  // !MEDIAPIPE_DISABLE_GPU
  }

  MP_RETURN_IF_ERROR(LoadOptions(cc));

  if (use_gpu_) {
#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
    MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([this, cc]() -> absl::Status {
      MP_RETURN_IF_ERROR(InitGpu(cc));
      return absl::OkStatus();
    }));
#else
    RET_CHECK_FAIL() << "GPU processing not enabled.";
#endif  // !MEDIAPIPE_DISABLE_GPU
  }

  return absl::OkStatus();
}

absl::Status TfLiteTensorsToSegmentationCalculator::Process(
    CalculatorContext* cc) {
  if (use_gpu_) {
#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
    MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([this, cc]() -> absl::Status {
      MP_RETURN_IF_ERROR(ProcessGpu(cc));
      return absl::OkStatus();
    }));
#endif  // !MEDIAPIPE_DISABLE_GPU
  } else {
    MP_RETURN_IF_ERROR(ProcessCpu(cc));
  }

  return absl::OkStatus();
}

absl::Status TfLiteTensorsToSegmentationCalculator::Close(
    CalculatorContext* cc) {
#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
  gpu_helper_.RunInGlContext([this] {
    if (upsample_program_) glDeleteProgram(upsample_program_);
    upsample_program_ = 0;
    mask_program_with_prev_.reset();
    mask_program_no_prev_.reset();
    tensor_buffer_.reset();
  });
#endif  // !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
}

absl::Status TfLiteTensorsToSegmentationCalculator::ProcessCpu(
    CalculatorContext* cc) {
  if (cc->Inputs().Tag(kTensorsTag).IsEmpty()) {
    return absl::OkStatus();
  }

  // Get input streams.
  const auto& input_tensors =
      cc->Inputs().Tag(kTensorsTag).Get<std::vector<TfLiteTensor>>();
  const bool has_prev_mask = cc->Inputs().HasTag(kPrevMaskTag) &&
                             !cc->Inputs().Tag(kPrevMaskTag).IsEmpty();
  const ImageFrame placeholder;
  const auto& input_mask =
      has_prev_mask ? cc->Inputs().Tag(kPrevMaskTag).Get<ImageFrame>()
                    : placeholder;
  int output_width = tensor_width_, output_height = tensor_height_;
  if (cc->Inputs().HasTag(kSizeImageTag)) {
    const auto& input_image = cc->Inputs().Tag(kSizeImageTag).Get<ImageFrame>();
    output_width = input_image.Width();
    output_height = input_image.Height();
  }
  RET_CHECK_EQ(input_tensors.size(), 1);

  // Create initial working mask.
  cv::Mat small_mask_mat(cv::Size(tensor_width_, tensor_height_), CV_8UC4);

  // Get input previous mask.
  cv::Mat input_mask_mat;
  if (has_prev_mask) {
    cv::Mat temp_mask_mat = formats::MatView(&input_mask);
    if (temp_mask_mat.channels() != 4) {
      cv::Mat converted_mat;
      cv::cvtColor(temp_mask_mat, converted_mat,
                   temp_mask_mat.channels() == 1 ? cv::COLOR_GRAY2RGBA
                                                 : cv::COLOR_RGB2RGBA);
      temp_mask_mat = converted_mat.clone();
    }
    cv::resize(temp_mask_mat, input_mask_mat, small_mask_mat.size());
  }

  // Copy input tensor.
  const TfLiteTensor* raw_input_tensor = &input_tensors[0];
  const float* raw_input_data = raw_input_tensor->data.f;
  cv::Mat tensor_mat(cv::Size(tensor_width_, tensor_height_),
                     CV_MAKETYPE(CV_32F, tensor_channels_));
  float* tensor_mat_ptr = tensor_mat.ptr<float>();
  memcpy(tensor_mat_ptr, raw_input_data, raw_input_tensor->bytes);

  // Process mask tensor.
  // Run softmax over tensor output and blend with previous mask.
  const int output_layer_index = options_.output_layer_index();
  const float combine_with_prev_ratio = options_.combine_with_previous_ratio();
  for (int i = 0; i < tensor_height_; ++i) {
    for (int j = 0; j < tensor_width_; ++j) {
      // Only two channel input tensor is supported.
      const cv::Vec2f input_pix = tensor_mat.at<cv::Vec2f>(i, j);
      const float shift = std::max(input_pix[0], input_pix[1]);
      const float softmax_denom =
          std::exp(input_pix[0] - shift) + std::exp(input_pix[1] - shift);
      float new_mask_value =
          std::exp(input_pix[output_layer_index] - shift) / softmax_denom;
      // Combine previous value with current using uncertainty^2 as mixing coeff
      if (has_prev_mask) {
        const float prev_mask_value =
            input_mask_mat.at<cv::Vec4b>(i, j)[0] / 255.0f;
        const float eps = 0.001;
        float uncertainty_alpha =
            1.0 +
            (new_mask_value * std::log(new_mask_value + eps) +
             (1.0 - new_mask_value) * std::log(1.0 - new_mask_value + eps)) /
                std::log(2.0f);
        uncertainty_alpha = Clamp(uncertainty_alpha, 0.0f, 1.0f);
        // Equivalent to: a = 1 - (1 - a) * (1 - a);  (squaring the uncertainty)
        uncertainty_alpha *= 2.0 - uncertainty_alpha;
        const float mixed_mask_value =
            new_mask_value * uncertainty_alpha +
            prev_mask_value * (1.0f - uncertainty_alpha);
        new_mask_value = mixed_mask_value * combine_with_prev_ratio +
                         (1.0f - combine_with_prev_ratio) * new_mask_value;
      }
      const uchar mask_value = static_cast<uchar>(new_mask_value * 255);
      // Set both R and A channels for convenience.
      const cv::Vec4b out_value = {mask_value, 0, 0, mask_value};
      small_mask_mat.at<cv::Vec4b>(i, j) = out_value;
    }
  }

  if (options_.flip_vertically()) cv::flip(small_mask_mat, small_mask_mat, 0);

  // Upsample small mask into output.
  cv::Mat large_mask_mat;
  cv::resize(small_mask_mat, large_mask_mat,
             cv::Size(output_width, output_height));

  // Send out image as CPU packet.
  std::unique_ptr<ImageFrame> output_mask = absl::make_unique<ImageFrame>(
      ImageFormat::SRGBA, output_width, output_height);
  cv::Mat output_mat = formats::MatView(output_mask.get());
  large_mask_mat.copyTo(output_mat);
  cc->Outputs().Tag(kMaskTag).Add(output_mask.release(), cc->InputTimestamp());

  return absl::OkStatus();
}

// Steps:
// 1. receive tensor and optional previous mask
// 2. process segmentation tensor into small mask
// 3. upsample small mask into output mask to be same size as input image
absl::Status TfLiteTensorsToSegmentationCalculator::ProcessGpu(
    CalculatorContext* cc) {
  if (cc->Inputs().Tag(kTensorsGpuTag).IsEmpty()) {
    return absl::OkStatus();
  }
#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
  // Get input streams.
  const auto& input_tensors =
      cc->Inputs().Tag(kTensorsGpuTag).Get<std::vector<GlBuffer>>();
  const bool has_prev_mask = cc->Inputs().HasTag(kPrevMaskGpuTag) &&
                             !cc->Inputs().Tag(kPrevMaskGpuTag).IsEmpty();
  const auto& input_mask =
      has_prev_mask
          ? cc->Inputs().Tag(kPrevMaskGpuTag).Get<mediapipe::GpuBuffer>()
          : mediapipe::GpuBuffer();
  int output_width = tensor_width_, output_height = tensor_height_;
  if (cc->Inputs().HasTag(kSizeImageGpuTag)) {
    const auto& input_image =
        cc->Inputs().Tag(kSizeImageGpuTag).Get<mediapipe::GpuBuffer>();
    output_width = input_image.width();
    output_height = input_image.height();
  }
  RET_CHECK_EQ(input_tensors.size(), 1);

  // Create initial working mask texture.
  ::tflite::gpu::gl::GlTexture small_mask_texture;
  MP_RETURN_IF_ERROR(CreateReadWriteRgbaImageTexture(
      tflite::gpu::DataType::UINT8,  // GL_RGBA8
      {tensor_width_, tensor_height_}, &small_mask_texture));

  // Get input previous mask.
  auto input_mask_texture = has_prev_mask
                                ? gpu_helper_.CreateSourceTexture(input_mask)
                                : mediapipe::GlTexture();

  // Copy input tensor.
  MP_RETURN_IF_ERROR(CopyBuffer(input_tensors[0], *tensor_buffer_));

  // Run shader, process mask tensor.
  // Run softmax over tensor output and blend with previous mask.
  {
    const int output_index = 0;
    glBindImageTexture(output_index, small_mask_texture.id(), 0, GL_FALSE, 0,
                       GL_WRITE_ONLY, GL_RGBA8);
    MP_RETURN_IF_ERROR(tensor_buffer_->BindToIndex(2));

    const tflite::gpu::uint3 workgroups = {
        NumGroups(tensor_width_, kWorkgroupSize),
        NumGroups(tensor_height_, kWorkgroupSize), 1};

    if (!has_prev_mask) {
      MP_RETURN_IF_ERROR(mask_program_no_prev_->Dispatch(workgroups));
    } else {
      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, input_mask_texture.name());
      MP_RETURN_IF_ERROR(mask_program_with_prev_->Dispatch(workgroups));
      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, 0);
    }
  }

  // Upsample small mask into output.
  mediapipe::GlTexture output_texture = gpu_helper_.CreateDestinationTexture(
      output_width, output_height,
      mediapipe::GpuBufferFormat::kBGRA32);  // actually GL_RGBA8

  // Run shader, upsample result.
  {
    gpu_helper_.BindFramebuffer(output_texture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, small_mask_texture.id());
    GlRender();
    glBindTexture(GL_TEXTURE_2D, 0);
    glFlush();
  }

  // Send out image as GPU packet.
  auto output_image = output_texture.GetFrame<mediapipe::GpuBuffer>();
  cc->Outputs()
      .Tag(kMaskGpuTag)
      .Add(output_image.release(), cc->InputTimestamp());

  // Cleanup
  input_mask_texture.Release();
  output_texture.Release();
#endif  // !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
}

void TfLiteTensorsToSegmentationCalculator::GlRender() {
#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
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

  // program
  glUseProgram(upsample_program_);

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

absl::Status TfLiteTensorsToSegmentationCalculator::LoadOptions(
    CalculatorContext* cc) {
  // Get calculator options specified in the graph.
  options_ =
      cc->Options<::mediapipe::TfLiteTensorsToSegmentationCalculatorOptions>();

  if (!options_.has_tensor_width() || !options_.has_tensor_height() ||
      !options_.has_tensor_channels())
    RET_CHECK_FAIL() << "Missing tensor dimensions in options.";

  tensor_width_ = options_.tensor_width();
  tensor_height_ = options_.tensor_height();
  tensor_channels_ = options_.tensor_channels();
  RET_CHECK_EQ(tensor_channels_, 2)
      << "Only 2 channel segmentation tensor currently supported";

  return absl::OkStatus();
}

absl::Status TfLiteTensorsToSegmentationCalculator::InitGpu(
    CalculatorContext* cc) {
#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
  MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([this]() -> absl::Status {
    // A shader to process a segmentation tensor into an output mask,
    // and use an optional previous mask as input.
    // Currently uses 4 channels for output,
    // and sets both R and A channels as mask value.
    const std::string shader_src_template =
        R"( #version 310 es

layout(local_size_x = $0, local_size_y = $0, local_size_z = 1) in;

precision highp float;

layout(std430, binding = 2) readonly buffer B0 {
  vec2 elements[];
} input_data;   // data tensor
layout(binding = 1) uniform sampler2D input_texture;   // previous mask
layout(rgba8, binding = 0) writeonly uniform highp image2D output_texture;

uniform ivec2 out_size;

const int output_layer_index = int($1);
const float combine_with_previous_ratio = float($2);

// Will be replaced with either '#define READ_PREVIOUS' or empty string
$3 //DEFINE_READ_PREVIOUS

void main() {
  int out_width = out_size.x;
  int out_height = out_size.y;

  ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
  if (gid.x >= out_width || gid.y >= out_height) { return; }

  int linear_index = gid.y * out_width + gid.x;
  vec2 input_value = input_data.elements[linear_index];

  // Only two channel input tensor is supported.
  vec2 input_px = input_value.rg;
  float shift = max(input_px.r, input_px.g);
  float softmax_denom = exp(input_px.r - shift) + exp(input_px.g - shift);
  float new_mask_value =
      exp(input_px[output_layer_index] - shift) / softmax_denom;

  // Combine previous value with current using uncertainty^2 as mixing parameter
#ifdef READ_PREVIOUS
  vec2 normalized_gid = vec2(gid) / vec2(out_width - 1, out_height - 1);
  float prev_mask_value = texture(input_texture, normalized_gid).r;

  float eps = 0.001;
  float uncertainty_alpha =
      1.0 + (new_mask_value * log(new_mask_value + eps) +
             (1.0 - new_mask_value) * log(1.0 - new_mask_value + eps)) /
                log(2.0f);
  uncertainty_alpha = clamp(uncertainty_alpha, 0.0, 1.0);
  // equivalent to a = 1 - (1 - a) * (1 - a);  (squaring the uncertainty)
  uncertainty_alpha *= 2.0 - uncertainty_alpha;

  float mixed_mask_value = new_mask_value * uncertainty_alpha +
                           prev_mask_value * (1.0f - uncertainty_alpha);

  // Use user provided value to mix raw value & a value mixed with previous mask
  new_mask_value = mixed_mask_value * combine_with_previous_ratio +
                 (1.0f - combine_with_previous_ratio) * new_mask_value;
#endif  // READ_PREVIOUS

  int y_coord = int($4);
  ivec2 output_coordinate = ivec2(gid.x, y_coord);
  // Set both R and A channels for convenience.
  vec4 out_value = vec4(new_mask_value, 0.0, 0.0, new_mask_value);
  imageStore(output_texture, output_coordinate, out_value);
})";

    const std::string shader_src_no_previous = absl::Substitute(
        shader_src_template, kWorkgroupSize, options_.output_layer_index(),
        options_.combine_with_previous_ratio(), "",
        options_.flip_vertically() ? "out_height - gid.y - 1" : "gid.y");
    const std::string shader_src_with_previous = absl::Substitute(
        shader_src_template, kWorkgroupSize, options_.output_layer_index(),
        options_.combine_with_previous_ratio(), "#define READ_PREVIOUS",
        options_.flip_vertically() ? "out_height - gid.y - 1" : "gid.y");

    // Shader programs.
    GlShader shader_without_previous;
    MP_RETURN_IF_ERROR(GlShader::CompileShader(
        GL_COMPUTE_SHADER, shader_src_no_previous, &shader_without_previous));
    mask_program_no_prev_ = absl::make_unique<GlProgram>();
    MP_RETURN_IF_ERROR(GlProgram::CreateWithShader(
        shader_without_previous, mask_program_no_prev_.get()));
    GlShader shader_with_previous;
    MP_RETURN_IF_ERROR(GlShader::CompileShader(
        GL_COMPUTE_SHADER, shader_src_with_previous, &shader_with_previous));
    mask_program_with_prev_ = absl::make_unique<GlProgram>();
    MP_RETURN_IF_ERROR(GlProgram::CreateWithShader(
        shader_with_previous, mask_program_with_prev_.get()));

    // Buffer storage for input tensor.
    size_t tensor_length = tensor_width_ * tensor_height_ * tensor_channels_;
    tensor_buffer_ = absl::make_unique<GlBuffer>();
    MP_RETURN_IF_ERROR(CreateReadWriteShaderStorageBuffer<float>(
        tensor_length, tensor_buffer_.get()));

    // Parameters.
    glUseProgram(mask_program_with_prev_->id());
    glUniform2i(glGetUniformLocation(mask_program_with_prev_->id(), "out_size"),
                tensor_width_, tensor_height_);
    glUniform1i(
        glGetUniformLocation(mask_program_with_prev_->id(), "input_texture"),
        1);
    glUseProgram(mask_program_no_prev_->id());
    glUniform2i(glGetUniformLocation(mask_program_no_prev_->id(), "out_size"),
                tensor_width_, tensor_height_);
    glUniform1i(
        glGetUniformLocation(mask_program_no_prev_->id(), "input_texture"), 1);

    // Vertex shader attributes.
    const GLint attr_location[NUM_ATTRIBUTES] = {
        ATTRIB_VERTEX,
        ATTRIB_TEXTURE_POSITION,
    };
    const GLchar* attr_name[NUM_ATTRIBUTES] = {
        "position",
        "texture_coordinate",
    };

    // Simple pass-through shader, used for hardware upsampling.
    std::string upsample_shader_base = R"(
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
    out vec4 fragColor;
  #endif  // defined(GL_ES)

  in vec2 sample_coordinate;
  uniform sampler2D input_data;

  void main() {
    vec4 pix = texture2D(input_data, sample_coordinate);
    fragColor = pix;
  }
)";

    // Program
    mediapipe::GlhCreateProgram(
        mediapipe::kBasicVertexShader, upsample_shader_base.c_str(),
        NUM_ATTRIBUTES, &attr_name[0], attr_location, &upsample_program_);
    RET_CHECK(upsample_program_) << "Problem initializing the program.";

    // Parameters
    glUseProgram(upsample_program_);
    glUniform1i(glGetUniformLocation(upsample_program_, "input_data"), 1);

    return absl::OkStatus();
  }));
#endif  // !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
}

}  // namespace mediapipe
