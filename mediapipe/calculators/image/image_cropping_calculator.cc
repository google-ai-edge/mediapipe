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

#include "mediapipe/calculators/image/image_cropping_calculator.h"

#include <cmath>

#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/shader_util.h"
#endif  // !MEDIAPIPE_DISABLE_GPU

namespace {
enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };
}  // namespace

namespace mediapipe {

namespace {

#if !MEDIAPIPE_DISABLE_GPU

#endif  // !MEDIAPIPE_DISABLE_GPU

constexpr char kRectTag[] = "RECT";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kHeightTag[] = "HEIGHT";
constexpr char kImageTag[] = "IMAGE";
constexpr char kImageGpuTag[] = "IMAGE_GPU";
constexpr char kWidthTag[] = "WIDTH";

}  // namespace

REGISTER_CALCULATOR(ImageCroppingCalculator);

absl::Status ImageCroppingCalculator::GetContract(CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag(kImageTag) ^ cc->Inputs().HasTag(kImageGpuTag));
  RET_CHECK(cc->Outputs().HasTag(kImageTag) ^
            cc->Outputs().HasTag(kImageGpuTag));

  bool use_gpu = false;

  if (cc->Inputs().HasTag(kImageTag)) {
    RET_CHECK(cc->Outputs().HasTag(kImageTag));
    cc->Inputs().Tag(kImageTag).Set<ImageFrame>();
    cc->Outputs().Tag(kImageTag).Set<ImageFrame>();
  }
#if !MEDIAPIPE_DISABLE_GPU
  if (cc->Inputs().HasTag(kImageGpuTag)) {
    RET_CHECK(cc->Outputs().HasTag(kImageGpuTag));
    cc->Inputs().Tag(kImageGpuTag).Set<GpuBuffer>();
    cc->Outputs().Tag(kImageGpuTag).Set<GpuBuffer>();
    use_gpu |= true;
  }
#endif  // !MEDIAPIPE_DISABLE_GPU

  int flags = 0;
  if (cc->Inputs().HasTag(kRectTag)) {
    ++flags;
  }
  if (cc->Inputs().HasTag(kWidthTag) && cc->Inputs().HasTag(kHeightTag)) {
    ++flags;
  }
  if (cc->Inputs().HasTag(kNormRectTag)) {
    ++flags;
  }
  if (cc->Options<mediapipe::ImageCroppingCalculatorOptions>()
          .has_norm_width() &&
      cc->Options<mediapipe::ImageCroppingCalculatorOptions>()
          .has_norm_height()) {
    ++flags;
  }
  if (cc->Options<mediapipe::ImageCroppingCalculatorOptions>().has_width() &&
      cc->Options<mediapipe::ImageCroppingCalculatorOptions>().has_height()) {
    ++flags;
  }
  RET_CHECK(flags == 1) << "Illegal combination of input streams/options.";

  if (cc->Inputs().HasTag(kRectTag)) {
    cc->Inputs().Tag(kRectTag).Set<Rect>();
  }
  if (cc->Inputs().HasTag(kNormRectTag)) {
    cc->Inputs().Tag(kNormRectTag).Set<NormalizedRect>();
  }
  if (cc->Inputs().HasTag(kWidthTag)) {
    cc->Inputs().Tag(kWidthTag).Set<int>();
  }
  if (cc->Inputs().HasTag(kHeightTag)) {
    cc->Inputs().Tag(kHeightTag).Set<int>();
  }

  if (use_gpu) {
#if !MEDIAPIPE_DISABLE_GPU
    MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
#endif  // !MEDIAPIPE_DISABLE_GPU
  }

  return absl::OkStatus();
}

absl::Status ImageCroppingCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  if (cc->Inputs().HasTag(kImageGpuTag)) {
    use_gpu_ = true;
  }

  options_ = cc->Options<mediapipe::ImageCroppingCalculatorOptions>();
  output_max_width_ =
      options_.has_output_max_width() ? options_.output_max_width() : FLT_MAX;
  output_max_height_ =
      options_.has_output_max_height() ? options_.output_max_height() : FLT_MAX;

  if (use_gpu_) {
#if !MEDIAPIPE_DISABLE_GPU
    MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
#else
    RET_CHECK_FAIL() << "GPU processing is for Android and iOS only.";
#endif  // !MEDIAPIPE_DISABLE_GPU
  }

  // Validate border mode.
  if (use_gpu_) {
    MP_RETURN_IF_ERROR(ValidateBorderModeForGPU(cc));
  } else {
    MP_RETURN_IF_ERROR(ValidateBorderModeForCPU(cc));
  }

  return absl::OkStatus();
}

absl::Status ImageCroppingCalculator::Process(CalculatorContext* cc) {
  if (cc->Inputs().HasTag(kRectTag) && cc->Inputs().Tag(kRectTag).IsEmpty()) {
    VLOG(1) << "RECT is empty for timestamp: " << cc->InputTimestamp();
    return absl::OkStatus();
  }
  if (cc->Inputs().HasTag(kNormRectTag) &&
      cc->Inputs().Tag(kNormRectTag).IsEmpty()) {
    VLOG(1) << "NORM_RECT is empty for timestamp: " << cc->InputTimestamp();
    return absl::OkStatus();
  }
  if (use_gpu_) {
#if !MEDIAPIPE_DISABLE_GPU
    MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([this, cc]() -> absl::Status {
      if (!gpu_initialized_) {
        MP_RETURN_IF_ERROR(InitGpu(cc));
        gpu_initialized_ = true;
      }
      MP_RETURN_IF_ERROR(RenderGpu(cc));
      return absl::OkStatus();
    }));
#endif  // !MEDIAPIPE_DISABLE_GPU
  } else {
    MP_RETURN_IF_ERROR(RenderCpu(cc));
  }
  return absl::OkStatus();
}

absl::Status ImageCroppingCalculator::Close(CalculatorContext* cc) {
#if !MEDIAPIPE_DISABLE_GPU
  gpu_helper_.RunInGlContext([this] {
    if (program_) glDeleteProgram(program_);
    program_ = 0;
  });
  gpu_initialized_ = false;
#endif  // !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
}

absl::Status ImageCroppingCalculator::ValidateBorderModeForCPU(
    CalculatorContext* cc) {
  int border_mode;
  return GetBorderModeForOpenCV(cc, &border_mode);
}

absl::Status ImageCroppingCalculator::ValidateBorderModeForGPU(
    CalculatorContext* cc) {
  mediapipe::ImageCroppingCalculatorOptions options =
      cc->Options<mediapipe::ImageCroppingCalculatorOptions>();

  switch (options.border_mode()) {
    case mediapipe::ImageCroppingCalculatorOptions::BORDER_ZERO:
      LOG(WARNING) << "BORDER_ZERO mode is not supported by GPU "
                   << "implementation and will fall back into BORDER_REPLICATE";
      break;
    case mediapipe::ImageCroppingCalculatorOptions::BORDER_REPLICATE:
      break;
    default:
      RET_CHECK_FAIL() << "Unsupported border mode for GPU: "
                       << options.border_mode();
  }

  return absl::OkStatus();
}

absl::Status ImageCroppingCalculator::RenderCpu(CalculatorContext* cc) {
  if (cc->Inputs().Tag(kImageTag).IsEmpty()) {
    return absl::OkStatus();
  }
  const auto& input_img = cc->Inputs().Tag(kImageTag).Get<ImageFrame>();
  cv::Mat input_mat = formats::MatView(&input_img);

  RectSpec specs = GetCropSpecs(cc, input_img.Width(), input_img.Height());
  int target_width = specs.width, target_height = specs.height,
      rect_center_x = specs.center_x, rect_center_y = specs.center_y;
  float rotation = specs.rotation;

  // Get border mode and value for OpenCV.
  int border_mode;
  MP_RETURN_IF_ERROR(GetBorderModeForOpenCV(cc, &border_mode));

  const cv::RotatedRect min_rect(cv::Point2f(rect_center_x, rect_center_y),
                                 cv::Size2f(target_width, target_height),
                                 rotation * 180.f / M_PI);
  cv::Mat src_points;
  cv::boxPoints(min_rect, src_points);

  float output_width = min_rect.size.width;
  float output_height = min_rect.size.height;
  float scale = std::min({1.0f, output_max_width_ / output_width,
                          output_max_height_ / output_height});
  output_width *= scale;
  output_height *= scale;

  float dst_corners[8] = {0,
                          output_height - 1,
                          0,
                          0,
                          output_width - 1,
                          0,
                          output_width - 1,
                          output_height - 1};
  cv::Mat dst_points = cv::Mat(4, 2, CV_32F, dst_corners);
  cv::Mat projection_matrix =
      cv::getPerspectiveTransform(src_points, dst_points);
  cv::Mat cropped_image;
  cv::warpPerspective(input_mat, cropped_image, projection_matrix,
                      cv::Size(output_width, output_height),
                      /* flags = */ 0,
                      /* borderMode = */ border_mode);

  std::unique_ptr<ImageFrame> output_frame(new ImageFrame(
      input_img.Format(), cropped_image.cols, cropped_image.rows));
  cv::Mat output_mat = formats::MatView(output_frame.get());
  cropped_image.copyTo(output_mat);
  cc->Outputs().Tag(kImageTag).Add(output_frame.release(),
                                   cc->InputTimestamp());
  return absl::OkStatus();
}

absl::Status ImageCroppingCalculator::RenderGpu(CalculatorContext* cc) {
  if (cc->Inputs().Tag(kImageGpuTag).IsEmpty()) {
    return absl::OkStatus();
  }
#if !MEDIAPIPE_DISABLE_GPU
  const Packet& input_packet = cc->Inputs().Tag(kImageGpuTag).Value();
  const auto& input_buffer = input_packet.Get<mediapipe::GpuBuffer>();
  auto src_tex = gpu_helper_.CreateSourceTexture(input_buffer);

  int out_width, out_height;
  GetOutputDimensions(cc, src_tex.width(), src_tex.height(), &out_width,
                      &out_height);
  auto dst_tex = gpu_helper_.CreateDestinationTexture(out_width, out_height);

  // Run cropping shader on GPU.
  {
    gpu_helper_.BindFramebuffer(dst_tex);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(src_tex.target(), src_tex.name());

    GlRender();

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, 0);
    glFlush();
  }

  // Send result image in GPU packet.
  auto output = dst_tex.GetFrame<mediapipe::GpuBuffer>();
  cc->Outputs().Tag(kImageGpuTag).Add(output.release(), cc->InputTimestamp());

  // Cleanup
  src_tex.Release();
  dst_tex.Release();
#endif  // !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
}

void ImageCroppingCalculator::GlRender() {
#if !MEDIAPIPE_DISABLE_GPU
  static const GLfloat square_vertices[] = {
      -1.0f, -1.0f,  // bottom left
      1.0f,  -1.0f,  // bottom right
      -1.0f, 1.0f,   // top left
      1.0f,  1.0f,   // top right
  };
  const GLfloat* texture_vertices = &transformed_points_[0];

  // program
  glUseProgram(program_);

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

absl::Status ImageCroppingCalculator::InitGpu(CalculatorContext* cc) {
#if !MEDIAPIPE_DISABLE_GPU
  const GLint attr_location[NUM_ATTRIBUTES] = {
      ATTRIB_VERTEX,
      ATTRIB_TEXTURE_POSITION,
  };
  const GLchar* attr_name[NUM_ATTRIBUTES] = {
      "position",
      "texture_coordinate",
  };

  // Simple pass-through shader.
  const GLchar* frag_src = GLES_VERSION_COMPAT
      R"(
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
    uniform sampler2D input_frame;

    void main() {
      vec4 pix = texture2D(input_frame, sample_coordinate);
      fragColor = pix;
    }
  )";

  // Program
  mediapipe::GlhCreateProgram(mediapipe::kBasicVertexShader, frag_src,
                              NUM_ATTRIBUTES, &attr_name[0], attr_location,
                              &program_);
  RET_CHECK(program_) << "Problem initializing the program.";

  // Parameters
  glUseProgram(program_);
  glUniform1i(glGetUniformLocation(program_, "input_frame"), 1);
#endif  // !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
}

// For GPU only.
void ImageCroppingCalculator::GetOutputDimensions(CalculatorContext* cc,
                                                  int src_width, int src_height,
                                                  int* dst_width,
                                                  int* dst_height) {
  RectSpec specs = GetCropSpecs(cc, src_width, src_height);
  int crop_width = specs.width, crop_height = specs.height,
      x_center = specs.center_x, y_center = specs.center_y;
  float rotation = specs.rotation;

  const float half_width = crop_width / 2.0f;
  const float half_height = crop_height / 2.0f;
  const float corners[] = {-half_width, -half_height, half_width, -half_height,
                           -half_width, half_height,  half_width, half_height};

  for (int i = 0; i < 4; ++i) {
    const float rotated_x = std::cos(rotation) * corners[i * 2] -
                            std::sin(rotation) * corners[i * 2 + 1];
    const float rotated_y = std::sin(rotation) * corners[i * 2] +
                            std::cos(rotation) * corners[i * 2 + 1];

    transformed_points_[i * 2] = ((rotated_x + x_center) / src_width);
    transformed_points_[i * 2 + 1] = ((rotated_y + y_center) / src_height);
  }

  // Find the boundaries of the transformed rectangle.
  float col_min = transformed_points_[0];
  float col_max = transformed_points_[0];
  float row_min = transformed_points_[1];
  float row_max = transformed_points_[1];
  for (int i = 1; i < 4; ++i) {
    col_min = std::min(col_min, transformed_points_[i * 2]);
    col_max = std::max(col_max, transformed_points_[i * 2]);
    row_min = std::min(row_min, transformed_points_[i * 2 + 1]);
    row_max = std::max(row_max, transformed_points_[i * 2 + 1]);
  }

  int width = static_cast<int>(std::round((col_max - col_min) * src_width));
  int height = static_cast<int>(std::round((row_max - row_min) * src_height));

  float scale =
      std::min({1.0f, output_max_width_ / width, output_max_height_ / height});
  width *= scale;
  height *= scale;

  // Minimum output dimension 1x1 prevents creation of textures with 0x0.
  *dst_width = std::max(1, width);
  *dst_height = std::max(1, height);
}

RectSpec ImageCroppingCalculator::GetCropSpecs(const CalculatorContext* cc,
                                               int src_width, int src_height) {
  // Get the size of the cropping box.
  int crop_width = src_width;
  int crop_height = src_height;
  // Get the center of cropping box. Default is the at the center.
  int x_center = src_width / 2;
  int y_center = src_height / 2;
  // Get the rotation of the cropping box.
  float rotation = 0.0f;
  // Get the normalized width and height if specified by the inputs or options.
  float normalized_width = 0.0f;
  float normalized_height = 0.0f;

  mediapipe::ImageCroppingCalculatorOptions options =
      cc->Options<mediapipe::ImageCroppingCalculatorOptions>();

  // width/height, norm_width/norm_height from input streams take precednece.
  if (cc->Inputs().HasTag(kRectTag)) {
    const auto& rect = cc->Inputs().Tag(kRectTag).Get<Rect>();
    // Only use the rect if it is valid.
    if (rect.width() > 0 && rect.height() > 0) {
      x_center = rect.x_center();
      y_center = rect.y_center();
      crop_width = rect.width();
      crop_height = rect.height();
      rotation = rect.rotation();
    }
  } else if (cc->Inputs().HasTag(kNormRectTag)) {
    const auto& norm_rect =
        cc->Inputs().Tag(kNormRectTag).Get<NormalizedRect>();
    if (norm_rect.width() > 0.0 && norm_rect.height() > 0.0) {
      normalized_width = norm_rect.width();
      normalized_height = norm_rect.height();
      x_center = std::round(norm_rect.x_center() * src_width);
      y_center = std::round(norm_rect.y_center() * src_height);
      rotation = norm_rect.rotation();
    }
  } else if (cc->Inputs().HasTag(kWidthTag) &&
             cc->Inputs().HasTag(kHeightTag)) {
    crop_width = cc->Inputs().Tag(kWidthTag).Get<int>();
    crop_height = cc->Inputs().Tag(kHeightTag).Get<int>();
  } else if (options.has_width() && options.has_height()) {
    crop_width = options.width();
    crop_height = options.height();
  } else if (options.has_norm_width() && options.has_norm_height()) {
    normalized_width = options.norm_width();
    normalized_height = options.norm_height();
  }

  // Get the crop width and height from the normalized width and height.
  if (normalized_width > 0 && normalized_height > 0) {
    crop_width = std::round(normalized_width * src_width);
    crop_height = std::round(normalized_height * src_height);
  }

  // Rotation and center values from input streams take precedence, so only
  // look at those values in the options if kRectTag and kNormRectTag are not
  // present from the inputs.
  if (!cc->Inputs().HasTag(kRectTag) && !cc->Inputs().HasTag(kNormRectTag)) {
    if (options.has_norm_center_x() && options.has_norm_center_y()) {
      x_center = std::round(options.norm_center_x() * src_width);
      y_center = std::round(options.norm_center_y() * src_height);
    }
    if (options.has_rotation()) {
      rotation = options.rotation();
    }
  }

  return {crop_width, crop_height, x_center, y_center, rotation};
}

absl::Status ImageCroppingCalculator::GetBorderModeForOpenCV(
    CalculatorContext* cc, int* border_mode) {
  mediapipe::ImageCroppingCalculatorOptions options =
      cc->Options<mediapipe::ImageCroppingCalculatorOptions>();

  switch (options.border_mode()) {
    case mediapipe::ImageCroppingCalculatorOptions::BORDER_ZERO:
      *border_mode = cv::BORDER_CONSTANT;
      break;
    case mediapipe::ImageCroppingCalculatorOptions::BORDER_REPLICATE:
      *border_mode = cv::BORDER_REPLICATE;
      break;
    default:
      RET_CHECK_FAIL() << "Unsupported border mode for CPU: "
                       << options.border_mode();
  }

  return absl::OkStatus();
}

}  // namespace mediapipe
