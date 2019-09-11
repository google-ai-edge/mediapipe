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

#include <cmath>

#include "mediapipe/calculators/image/image_cropping_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/shader_util.h"
#endif  // __ANDROID__ or iOS

namespace {
enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };
}  // namespace

namespace mediapipe {

#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)

#endif  // __ANDROID__ or iOS

// Crops the input texture to the given rectangle region. The rectangle can
// be at arbitrary location on the image with rotation. If there's rotation, the
// output texture will have the size of the input rectangle. The rotation should
// be in radian, see rect.proto for detail.
//
// Input:
//   One of the following two tags:
//   IMAGE - ImageFrame representing the input image.
//   IMAGE_GPU - GpuBuffer representing the input image.
//   One of the following two tags (optional if WIDTH/HEIGHT is specified):
//   RECT - A Rect proto specifying the width/height and location of the
//          cropping rectangle.
//   NORM_RECT - A NormalizedRect proto specifying the width/height and location
//               of the cropping rectangle in normalized coordinates.
//   Alternative tags to RECT (optional if RECT/NORM_RECT is specified):
//   WIDTH - The desired width of the output cropped image,
//           based on image center
//   HEIGHT - The desired height of the output cropped image,
//            based on image center
//
// Output:
//   One of the following two tags:
//   IMAGE - Cropped ImageFrame
//   IMAGE_GPU - Cropped GpuBuffer.
//
// Note: input_stream values take precedence over options defined in the graph.
//
class ImageCroppingCalculator : public CalculatorBase {
 public:
  ImageCroppingCalculator() = default;
  ~ImageCroppingCalculator() override = default;

  static ::mediapipe::Status GetContract(CalculatorContract* cc);
  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;
  ::mediapipe::Status Close(CalculatorContext* cc) override;

 private:
  ::mediapipe::Status RenderCpu(CalculatorContext* cc);
  ::mediapipe::Status RenderGpu(CalculatorContext* cc);
  ::mediapipe::Status InitGpu(CalculatorContext* cc);
  void GlRender();
  void GetOutputDimensions(CalculatorContext* cc, int src_width, int src_height,
                           int* dst_width, int* dst_height);

  mediapipe::ImageCroppingCalculatorOptions options_;

  bool use_gpu_ = false;
  // Output texture corners (4) after transoformation in normalized coordinates.
  float transformed_points_[8];
#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
  bool gpu_initialized_ = false;
  mediapipe::GlCalculatorHelper gpu_helper_;
  GLuint program_ = 0;
#endif  // __ANDROID__ or iOS
};
REGISTER_CALCULATOR(ImageCroppingCalculator);

::mediapipe::Status ImageCroppingCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag("IMAGE") ^ cc->Inputs().HasTag("IMAGE_GPU"));
  RET_CHECK(cc->Outputs().HasTag("IMAGE") ^ cc->Outputs().HasTag("IMAGE_GPU"));

  if (cc->Inputs().HasTag("IMAGE")) {
    RET_CHECK(cc->Outputs().HasTag("IMAGE"));
    cc->Inputs().Tag("IMAGE").Set<ImageFrame>();
    cc->Outputs().Tag("IMAGE").Set<ImageFrame>();
  }
#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
  if (cc->Inputs().HasTag("IMAGE_GPU")) {
    RET_CHECK(cc->Outputs().HasTag("IMAGE_GPU"));
    cc->Inputs().Tag("IMAGE_GPU").Set<GpuBuffer>();
    cc->Outputs().Tag("IMAGE_GPU").Set<GpuBuffer>();
  }
#endif  // __ANDROID__ or iOS

  if (cc->Inputs().HasTag("RECT")) {
    cc->Inputs().Tag("RECT").Set<Rect>();
  }
  if (cc->Inputs().HasTag("NORM_RECT")) {
    cc->Inputs().Tag("NORM_RECT").Set<NormalizedRect>();
  }
  if (cc->Inputs().HasTag("WIDTH")) {
    cc->Inputs().Tag("WIDTH").Set<int>();
  }
  if (cc->Inputs().HasTag("HEIGHT")) {
    cc->Inputs().Tag("HEIGHT").Set<int>();
  }

#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
  MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
#endif  // __ANDROID__ or iOS

  return ::mediapipe::OkStatus();
}

::mediapipe::Status ImageCroppingCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  if (cc->Inputs().HasTag("IMAGE_GPU")) {
    use_gpu_ = true;
  }

  options_ = cc->Options<mediapipe::ImageCroppingCalculatorOptions>();

  if (use_gpu_) {
#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
    MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
#else
    RET_CHECK_FAIL() << "GPU processing is for Android and iOS only.";
#endif  // __ANDROID__ or iOS
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status ImageCroppingCalculator::Process(CalculatorContext* cc) {
  if (use_gpu_) {
#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
    MP_RETURN_IF_ERROR(
        gpu_helper_.RunInGlContext([this, cc]() -> ::mediapipe::Status {
          if (!gpu_initialized_) {
            MP_RETURN_IF_ERROR(InitGpu(cc));
            gpu_initialized_ = true;
          }
          MP_RETURN_IF_ERROR(RenderGpu(cc));
          return ::mediapipe::OkStatus();
        }));
#endif  // __ANDROID__ or iOS
  } else {
    MP_RETURN_IF_ERROR(RenderCpu(cc));
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status ImageCroppingCalculator::Close(CalculatorContext* cc) {
#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
  gpu_helper_.RunInGlContext([this] {
    if (program_) glDeleteProgram(program_);
    program_ = 0;
  });
  gpu_initialized_ = false;
#endif  // __ANDROID__ or iOS

  return ::mediapipe::OkStatus();
}

::mediapipe::Status ImageCroppingCalculator::RenderCpu(CalculatorContext* cc) {
  const auto& input_img = cc->Inputs().Tag("IMAGE").Get<ImageFrame>();
  cv::Mat input_mat = formats::MatView(&input_img);

  float rect_center_x = input_img.Width() / 2.0f;
  float rect_center_y = input_img.Height() / 2.0f;
  float rotation = 0.0f;
  int target_width = input_img.Width();
  int target_height = input_img.Height();
  if (cc->Inputs().HasTag("RECT")) {
    const auto& rect = cc->Inputs().Tag("RECT").Get<Rect>();
    if (rect.width() > 0 && rect.height() > 0 && rect.x_center() >= 0 &&
        rect.y_center() >= 0) {
      rect_center_x = rect.x_center();
      rect_center_y = rect.y_center();
      target_width = rect.width();
      target_height = rect.height();
      rotation = rect.rotation();
    }
  } else if (cc->Inputs().HasTag("NORM_RECT")) {
    const auto& rect = cc->Inputs().Tag("NORM_RECT").Get<NormalizedRect>();
    if (rect.width() > 0.0 && rect.height() > 0.0 && rect.x_center() >= 0.0 &&
        rect.y_center() >= 0.0) {
      rect_center_x = std::round(rect.x_center() * input_img.Width());
      rect_center_y = std::round(rect.y_center() * input_img.Height());
      target_width = std::round(rect.width() * input_img.Width());
      target_height = std::round(rect.height() * input_img.Height());
      rotation = rect.rotation();
    }
  } else {
    if (cc->Inputs().HasTag("WIDTH") && cc->Inputs().HasTag("HEIGHT")) {
      target_width = cc->Inputs().Tag("WIDTH").Get<int>();
      target_height = cc->Inputs().Tag("HEIGHT").Get<int>();
    } else if (options_.has_width() && options_.has_height()) {
      target_width = options_.width();
      target_height = options_.height();
    }
    rotation = options_.rotation();
  }

  const cv::RotatedRect min_rect(cv::Point2f(rect_center_x, rect_center_y),
                                 cv::Size2f(target_width, target_height),
                                 rotation * 180.f / M_PI);
  cv::Mat src_points;
  cv::boxPoints(min_rect, src_points);

  float dst_corners[8] = {0,
                          min_rect.size.height - 1,
                          0,
                          0,
                          min_rect.size.width - 1,
                          0,
                          min_rect.size.width - 1,
                          min_rect.size.height - 1};
  cv::Mat dst_points = cv::Mat(4, 2, CV_32F, dst_corners);
  cv::Mat projection_matrix =
      cv::getPerspectiveTransform(src_points, dst_points);
  cv::Mat cropped_image;
  cv::warpPerspective(input_mat, cropped_image, projection_matrix,
                      cv::Size(min_rect.size.width, min_rect.size.height));

  std::unique_ptr<ImageFrame> output_frame(new ImageFrame(
      input_img.Format(), cropped_image.cols, cropped_image.rows));
  cv::Mat output_mat = formats::MatView(output_frame.get());
  cropped_image.copyTo(output_mat);
  cc->Outputs().Tag("IMAGE").Add(output_frame.release(), cc->InputTimestamp());
  return ::mediapipe::OkStatus();
}

::mediapipe::Status ImageCroppingCalculator::RenderGpu(CalculatorContext* cc) {
  if (cc->Inputs().Tag("IMAGE_GPU").IsEmpty()) {
    return ::mediapipe::OkStatus();
  }
#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
  const Packet& input_packet = cc->Inputs().Tag("IMAGE_GPU").Value();
  const auto& input_buffer = input_packet.Get<mediapipe::GpuBuffer>();
  auto src_tex = gpu_helper_.CreateSourceTexture(input_buffer);

  int out_width, out_height;
  GetOutputDimensions(cc, src_tex.width(), src_tex.height(), &out_width,
                      &out_height);
  auto dst_tex = gpu_helper_.CreateDestinationTexture(out_width, out_height);

  // Run cropping shader on GPU.
  {
    gpu_helper_.BindFramebuffer(dst_tex);  // GL_TEXTURE0

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(src_tex.target(), src_tex.name());

    GlRender();

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, 0);
    glFlush();
  }

  // Send result image in GPU packet.
  auto output = dst_tex.GetFrame<mediapipe::GpuBuffer>();
  cc->Outputs().Tag("IMAGE_GPU").Add(output.release(), cc->InputTimestamp());

  // Cleanup
  src_tex.Release();
  dst_tex.Release();
#endif  // __ANDROID__ or iOS

  return ::mediapipe::OkStatus();
}

void ImageCroppingCalculator::GlRender() {
#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
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

#endif  // __ANDROID__ or iOS
}

::mediapipe::Status ImageCroppingCalculator::InitGpu(CalculatorContext* cc) {
#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
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
#endif  // __ANDROID__ or iOS

  return ::mediapipe::OkStatus();
}

// For GPU only.
void ImageCroppingCalculator::GetOutputDimensions(CalculatorContext* cc,
                                                  int src_width, int src_height,
                                                  int* dst_width,
                                                  int* dst_height) {
  // Get the size of the cropping box.
  int crop_width = src_width;
  int crop_height = src_height;
  // Get the center of cropping box. Default is the at the center.
  int x_center = src_width / 2;
  int y_center = src_height / 2;
  // Get the rotation of the cropping box.
  float rotation = 0.0f;
  if (cc->Inputs().HasTag("RECT")) {
    const auto& rect = cc->Inputs().Tag("RECT").Get<Rect>();
    // Only use the rect if it is valid.
    if (rect.width() > 0 && rect.height() > 0 && rect.x_center() >= 0 &&
        rect.y_center() >= 0) {
      x_center = rect.x_center();
      y_center = rect.y_center();
      crop_width = rect.width();
      crop_height = rect.height();
      rotation = rect.rotation();
    }
  } else if (cc->Inputs().HasTag("NORM_RECT")) {
    const auto& rect = cc->Inputs().Tag("NORM_RECT").Get<NormalizedRect>();
    // Only use the rect if it is valid.
    if (rect.width() > 0.0 && rect.height() > 0.0 && rect.x_center() >= 0.0 &&
        rect.y_center() >= 0.0) {
      x_center = std::round(rect.x_center() * src_width);
      y_center = std::round(rect.y_center() * src_height);
      crop_width = std::round(rect.width() * src_width);
      crop_height = std::round(rect.height() * src_height);
      rotation = rect.rotation();
    }
  } else {
    if (cc->Inputs().HasTag("WIDTH") && cc->Inputs().HasTag("HEIGHT")) {
      crop_width = cc->Inputs().Tag("WIDTH").Get<int>();
      crop_height = cc->Inputs().Tag("HEIGHT").Get<int>();
    } else if (options_.has_width() && options_.has_height()) {
      crop_width = options_.width();
      crop_height = options_.height();
    }
    rotation = options_.rotation();
  }

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

  *dst_width = std::round((col_max - col_min) * src_width);
  *dst_height = std::round((row_max - row_min) * src_height);
}

}  // namespace mediapipe
