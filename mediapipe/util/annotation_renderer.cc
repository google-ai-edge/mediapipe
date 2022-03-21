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

#include "mediapipe/util/annotation_renderer.h"

#include <math.h>

#include <algorithm>
#include <cmath>

#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/vector.h"
#include "mediapipe/util/color.pb.h"

namespace mediapipe {
namespace {

using Arrow = RenderAnnotation::Arrow;
using FilledOval = RenderAnnotation::FilledOval;
using FilledRectangle = RenderAnnotation::FilledRectangle;
using FilledRoundedRectangle = RenderAnnotation::FilledRoundedRectangle;
using Point = RenderAnnotation::Point;
using Line = RenderAnnotation::Line;
using GradientLine = RenderAnnotation::GradientLine;
using Oval = RenderAnnotation::Oval;
using Rectangle = RenderAnnotation::Rectangle;
using RoundedRectangle = RenderAnnotation::RoundedRectangle;
using Text = RenderAnnotation::Text;

int ClampThickness(int thickness) {
  constexpr int kMaxThickness = 32767;  // OpenCV MAX_THICKNESS
  return std::clamp(thickness, 1, kMaxThickness);
}

bool NormalizedtoPixelCoordinates(double normalized_x, double normalized_y,
                                  int image_width, int image_height, int* x_px,
                                  int* y_px) {
  CHECK(x_px != nullptr);
  CHECK(y_px != nullptr);
  CHECK_GT(image_width, 0);
  CHECK_GT(image_height, 0);

  if (normalized_x < 0 || normalized_x > 1.0 || normalized_y < 0 ||
      normalized_y > 1.0) {
    VLOG(1) << "Normalized coordinates must be between 0.0 and 1.0";
  }

  *x_px = static_cast<int32>(round(normalized_x * image_width));
  *y_px = static_cast<int32>(round(normalized_y * image_height));

  return true;
}

cv::Scalar MediapipeColorToOpenCVColor(const Color& color) {
  return cv::Scalar(color.r(), color.g(), color.b());
}

cv::RotatedRect RectangleToOpenCVRotatedRect(int left, int top, int right,
                                             int bottom, double rotation) {
  return cv::RotatedRect(
      cv::Point2f((left + right) / 2.f, (top + bottom) / 2.f),
      cv::Size2f(right - left, bottom - top), rotation / M_PI * 180.f);
}

void cv_line2(cv::Mat& img, const cv::Point& start, const cv::Point& end,
              const cv::Scalar& color1, const cv::Scalar& color2,
              int thickness) {
  cv::LineIterator iter(img, start, end, /*cv::LINE_4=*/4);
  for (int i = 0; i < iter.count; i++, iter++) {
    const double alpha = static_cast<double>(i) / iter.count;
    const cv::Scalar new_color(color1 * (1.0 - alpha) + color2 * alpha);
    const cv::Rect rect(iter.pos(), cv::Size(thickness, thickness));
    cv::rectangle(img, rect, new_color, /*cv::FILLED=*/-1, /*cv::LINE_4=*/4);
  }
}

}  // namespace

void AnnotationRenderer::RenderDataOnImage(const RenderData& render_data) {
  for (const auto& annotation : render_data.render_annotations()) {
    if (annotation.data_case() == RenderAnnotation::kRectangle) {
      DrawRectangle(annotation);
    } else if (annotation.data_case() == RenderAnnotation::kRoundedRectangle) {
      DrawRoundedRectangle(annotation);
    } else if (annotation.data_case() == RenderAnnotation::kFilledRectangle) {
      DrawFilledRectangle(annotation);
    } else if (annotation.data_case() ==
               RenderAnnotation::kFilledRoundedRectangle) {
      DrawFilledRoundedRectangle(annotation);
    } else if (annotation.data_case() == RenderAnnotation::kOval) {
      DrawOval(annotation);
    } else if (annotation.data_case() == RenderAnnotation::kFilledOval) {
      DrawFilledOval(annotation);
    } else if (annotation.data_case() == RenderAnnotation::kText) {
      DrawText(annotation);
    } else if (annotation.data_case() == RenderAnnotation::kPoint) {
      DrawPoint(annotation);
    } else if (annotation.data_case() == RenderAnnotation::kLine) {
      DrawLine(annotation);
    } else if (annotation.data_case() == RenderAnnotation::kGradientLine) {
      DrawGradientLine(annotation);
    } else if (annotation.data_case() == RenderAnnotation::kArrow) {
      DrawArrow(annotation);
    } else {
      LOG(FATAL) << "Unknown annotation type: " << annotation.data_case();
    }
  }
}

void AnnotationRenderer::AdoptImage(cv::Mat* input_image) {
  image_width_ = input_image->cols;
  image_height_ = input_image->rows;

  // No pixel data copy here, only headers are copied.
  mat_image_ = *input_image;
}

int AnnotationRenderer::GetImageWidth() const { return mat_image_.cols; }
int AnnotationRenderer::GetImageHeight() const { return mat_image_.rows; }

void AnnotationRenderer::SetFlipTextVertically(bool flip) {
  flip_text_vertically_ = flip;
}

void AnnotationRenderer::SetScaleFactor(float scale_factor) {
  if (scale_factor > 0.0f) scale_factor_ = std::min(scale_factor, 1.0f);
}

void AnnotationRenderer::DrawRectangle(const RenderAnnotation& annotation) {
  int left = -1;
  int top = -1;
  int right = -1;
  int bottom = -1;
  const auto& rectangle = annotation.rectangle();
  if (rectangle.normalized()) {
    CHECK(NormalizedtoPixelCoordinates(rectangle.left(), rectangle.top(),
                                       image_width_, image_height_, &left,
                                       &top));
    CHECK(NormalizedtoPixelCoordinates(rectangle.right(), rectangle.bottom(),
                                       image_width_, image_height_, &right,
                                       &bottom));
  } else {
    left = static_cast<int>(rectangle.left() * scale_factor_);
    top = static_cast<int>(rectangle.top() * scale_factor_);
    right = static_cast<int>(rectangle.right() * scale_factor_);
    bottom = static_cast<int>(rectangle.bottom() * scale_factor_);
  }

  const cv::Scalar color = MediapipeColorToOpenCVColor(annotation.color());
  const int thickness =
      ClampThickness(round(annotation.thickness() * scale_factor_));
  if (rectangle.rotation() != 0.0) {
    const auto& rect = RectangleToOpenCVRotatedRect(left, top, right, bottom,
                                                    rectangle.rotation());
    const int kNumVertices = 4;
    cv::Point2f vertices[kNumVertices];
    rect.points(vertices);
    for (int i = 0; i < kNumVertices; i++) {
      cv::line(mat_image_, vertices[i], vertices[(i + 1) % kNumVertices], color,
               thickness);
    }
  } else {
    cv::Rect rect(left, top, right - left, bottom - top);
    cv::rectangle(mat_image_, rect, color, thickness);
  }
  if (rectangle.has_top_left_thickness()) {
    const auto& rect = RectangleToOpenCVRotatedRect(left, top, right, bottom,
                                                    rectangle.rotation());
    const int kNumVertices = 4;
    cv::Point2f vertices[kNumVertices];
    rect.points(vertices);
    const int top_left_thickness =
        ClampThickness(round(rectangle.top_left_thickness() * scale_factor_));
    cv::ellipse(mat_image_, vertices[1],
                cv::Size(top_left_thickness, top_left_thickness), 0.0, 0, 360,
                color, -1);
  }
}

void AnnotationRenderer::DrawFilledRectangle(
    const RenderAnnotation& annotation) {
  int left = -1;
  int top = -1;
  int right = -1;
  int bottom = -1;
  const auto& rectangle = annotation.filled_rectangle().rectangle();
  if (rectangle.normalized()) {
    CHECK(NormalizedtoPixelCoordinates(rectangle.left(), rectangle.top(),
                                       image_width_, image_height_, &left,
                                       &top));
    CHECK(NormalizedtoPixelCoordinates(rectangle.right(), rectangle.bottom(),
                                       image_width_, image_height_, &right,
                                       &bottom));
  } else {
    left = static_cast<int>(rectangle.left() * scale_factor_);
    top = static_cast<int>(rectangle.top() * scale_factor_);
    right = static_cast<int>(rectangle.right() * scale_factor_);
    bottom = static_cast<int>(rectangle.bottom() * scale_factor_);
  }

  const cv::Scalar color = MediapipeColorToOpenCVColor(annotation.color());
  if (rectangle.rotation() != 0.0) {
    const auto& rect = RectangleToOpenCVRotatedRect(left, top, right, bottom,
                                                    rectangle.rotation());
    const int kNumVertices = 4;
    cv::Point2f vertices2f[kNumVertices];
    rect.points(vertices2f);
    // Convert cv::Point2f[] to cv::Point[].
    cv::Point vertices[kNumVertices];
    for (int i = 0; i < kNumVertices; ++i) {
      vertices[i] = vertices2f[i];
    }
    cv::fillConvexPoly(mat_image_, vertices, kNumVertices, color);
  } else {
    cv::Rect rect(left, top, right - left, bottom - top);
    cv::rectangle(mat_image_, rect, color, -1);
  }
}

void AnnotationRenderer::DrawRoundedRectangle(
    const RenderAnnotation& annotation) {
  int left = -1;
  int top = -1;
  int right = -1;
  int bottom = -1;
  const auto& rectangle = annotation.rounded_rectangle().rectangle();
  if (rectangle.normalized()) {
    CHECK(NormalizedtoPixelCoordinates(rectangle.left(), rectangle.top(),
                                       image_width_, image_height_, &left,
                                       &top));
    CHECK(NormalizedtoPixelCoordinates(rectangle.right(), rectangle.bottom(),
                                       image_width_, image_height_, &right,
                                       &bottom));
  } else {
    left = static_cast<int>(rectangle.left() * scale_factor_);
    top = static_cast<int>(rectangle.top() * scale_factor_);
    right = static_cast<int>(rectangle.right() * scale_factor_);
    bottom = static_cast<int>(rectangle.bottom() * scale_factor_);
  }

  const cv::Scalar color = MediapipeColorToOpenCVColor(annotation.color());
  const int thickness =
      ClampThickness(round(annotation.thickness() * scale_factor_));
  const int corner_radius =
      round(annotation.rounded_rectangle().corner_radius() * scale_factor_);
  const int line_type = annotation.rounded_rectangle().line_type();
  DrawRoundedRectangle(mat_image_, cv::Point(left, top),
                       cv::Point(right, bottom), color, thickness, line_type,
                       corner_radius);
}

void AnnotationRenderer::DrawFilledRoundedRectangle(
    const RenderAnnotation& annotation) {
  int left = -1;
  int top = -1;
  int right = -1;
  int bottom = -1;
  const auto& rectangle =
      annotation.filled_rounded_rectangle().rounded_rectangle().rectangle();
  if (rectangle.normalized()) {
    CHECK(NormalizedtoPixelCoordinates(rectangle.left(), rectangle.top(),
                                       image_width_, image_height_, &left,
                                       &top));
    CHECK(NormalizedtoPixelCoordinates(rectangle.right(), rectangle.bottom(),
                                       image_width_, image_height_, &right,
                                       &bottom));
  } else {
    left = static_cast<int>(rectangle.left() * scale_factor_);
    top = static_cast<int>(rectangle.top() * scale_factor_);
    right = static_cast<int>(rectangle.right() * scale_factor_);
    bottom = static_cast<int>(rectangle.bottom() * scale_factor_);
  }

  const cv::Scalar color = MediapipeColorToOpenCVColor(annotation.color());
  const int corner_radius =
      annotation.rounded_rectangle().corner_radius() * scale_factor_;
  const int line_type = annotation.rounded_rectangle().line_type();
  DrawRoundedRectangle(mat_image_, cv::Point(left, top),
                       cv::Point(right, bottom), color, -1, line_type,
                       corner_radius);
}

void AnnotationRenderer::DrawRoundedRectangle(cv::Mat src, cv::Point top_left,
                                              cv::Point bottom_right,
                                              const cv::Scalar& line_color,
                                              int thickness, int line_type,
                                              int corner_radius) {
  // Corners:
  // p1 - p2
  // |     |
  // p4 - p3
  cv::Point p1 = top_left;
  cv::Point p2 = cv::Point(bottom_right.x, top_left.y);
  cv::Point p3 = bottom_right;
  cv::Point p4 = cv::Point(top_left.x, bottom_right.y);

  // Draw edges of the rectangle
  cv::line(src, cv::Point(p1.x + corner_radius, p1.y),
           cv::Point(p2.x - corner_radius, p2.y), line_color, thickness,
           line_type);
  cv::line(src, cv::Point(p2.x, p2.y + corner_radius),
           cv::Point(p3.x, p3.y - corner_radius), line_color, thickness,
           line_type);
  cv::line(src, cv::Point(p4.x + corner_radius, p4.y),
           cv::Point(p3.x - corner_radius, p3.y), line_color, thickness,
           line_type);
  cv::line(src, cv::Point(p1.x, p1.y + corner_radius),
           cv::Point(p4.x, p4.y - corner_radius), line_color, thickness,
           line_type);

  // Draw arcs at corners.
  cv::ellipse(src, p1 + cv::Point(corner_radius, corner_radius),
              cv::Size(corner_radius, corner_radius), 180.0, 0, 90, line_color,
              thickness, line_type);
  cv::ellipse(src, p2 + cv::Point(-corner_radius, corner_radius),
              cv::Size(corner_radius, corner_radius), 270.0, 0, 90, line_color,
              thickness, line_type);
  cv::ellipse(src, p3 + cv::Point(-corner_radius, -corner_radius),
              cv::Size(corner_radius, corner_radius), 0.0, 0, 90, line_color,
              thickness, line_type);
  cv::ellipse(src, p4 + cv::Point(corner_radius, -corner_radius),
              cv::Size(corner_radius, corner_radius), 90.0, 0, 90, line_color,
              thickness, line_type);
}

void AnnotationRenderer::DrawOval(const RenderAnnotation& annotation) {
  int left = -1;
  int top = -1;
  int right = -1;
  int bottom = -1;
  const auto& enclosing_rectangle = annotation.oval().rectangle();
  if (enclosing_rectangle.normalized()) {
    CHECK(NormalizedtoPixelCoordinates(enclosing_rectangle.left(),
                                       enclosing_rectangle.top(), image_width_,
                                       image_height_, &left, &top));
    CHECK(NormalizedtoPixelCoordinates(
        enclosing_rectangle.right(), enclosing_rectangle.bottom(), image_width_,
        image_height_, &right, &bottom));
  } else {
    left = static_cast<int>(enclosing_rectangle.left() * scale_factor_);
    top = static_cast<int>(enclosing_rectangle.top() * scale_factor_);
    right = static_cast<int>(enclosing_rectangle.right() * scale_factor_);
    bottom = static_cast<int>(enclosing_rectangle.bottom() * scale_factor_);
  }

  cv::Point center((left + right) / 2, (top + bottom) / 2);
  cv::Size size((right - left) / 2, (bottom - top) / 2);
  const double rotation = enclosing_rectangle.rotation() / M_PI * 180.f;
  const cv::Scalar color = MediapipeColorToOpenCVColor(annotation.color());
  const int thickness =
      ClampThickness(round(annotation.thickness() * scale_factor_));
  cv::ellipse(mat_image_, center, size, rotation, 0, 360, color, thickness);
}

void AnnotationRenderer::DrawFilledOval(const RenderAnnotation& annotation) {
  int left = -1;
  int top = -1;
  int right = -1;
  int bottom = -1;
  const auto& enclosing_rectangle = annotation.filled_oval().oval().rectangle();
  if (enclosing_rectangle.normalized()) {
    CHECK(NormalizedtoPixelCoordinates(enclosing_rectangle.left(),
                                       enclosing_rectangle.top(), image_width_,
                                       image_height_, &left, &top));
    CHECK(NormalizedtoPixelCoordinates(
        enclosing_rectangle.right(), enclosing_rectangle.bottom(), image_width_,
        image_height_, &right, &bottom));
  } else {
    left = static_cast<int>(enclosing_rectangle.left() * scale_factor_);
    top = static_cast<int>(enclosing_rectangle.top() * scale_factor_);
    right = static_cast<int>(enclosing_rectangle.right() * scale_factor_);
    bottom = static_cast<int>(enclosing_rectangle.bottom() * scale_factor_);
  }

  cv::Point center((left + right) / 2, (top + bottom) / 2);
  cv::Size size(std::max(0, (right - left) / 2),
                std::max(0, (bottom - top) / 2));
  const double rotation = enclosing_rectangle.rotation() / M_PI * 180.f;
  const cv::Scalar color = MediapipeColorToOpenCVColor(annotation.color());
  cv::ellipse(mat_image_, center, size, rotation, 0, 360, color, -1);
}

void AnnotationRenderer::DrawArrow(const RenderAnnotation& annotation) {
  int x_start = -1;
  int y_start = -1;
  int x_end = -1;
  int y_end = -1;

  const auto& arrow = annotation.arrow();
  if (arrow.normalized()) {
    CHECK(NormalizedtoPixelCoordinates(arrow.x_start(), arrow.y_start(),
                                       image_width_, image_height_, &x_start,
                                       &y_start));
    CHECK(NormalizedtoPixelCoordinates(arrow.x_end(), arrow.y_end(),
                                       image_width_, image_height_, &x_end,
                                       &y_end));
  } else {
    x_start = static_cast<int>(arrow.x_start() * scale_factor_);
    y_start = static_cast<int>(arrow.y_start() * scale_factor_);
    x_end = static_cast<int>(arrow.x_end() * scale_factor_);
    y_end = static_cast<int>(arrow.y_end() * scale_factor_);
  }

  cv::Point arrow_start(x_start, y_start);
  cv::Point arrow_end(x_end, y_end);
  const cv::Scalar color = MediapipeColorToOpenCVColor(annotation.color());
  const int thickness =
      ClampThickness(round(annotation.thickness() * scale_factor_));

  // Draw the main arrow line.
  cv::line(mat_image_, arrow_start, arrow_end, color, thickness);

  // Compute the arrowtip left and right vectors.
  Vector2_d L_start(static_cast<double>(x_start), static_cast<double>(y_start));
  Vector2_d L_end(static_cast<double>(x_end), static_cast<double>(y_end));
  Vector2_d U = (L_end - L_start).Normalize();
  Vector2_d V = U.Ortho();
  double line_length = (L_end - L_start).Norm();
  constexpr double kArrowTipLengthProportion = 0.2;
  double arrowtip_length = kArrowTipLengthProportion * line_length;
  Vector2_d arrowtip_left = L_end - arrowtip_length * U + arrowtip_length * V;
  Vector2_d arrowtip_right = L_end - arrowtip_length * U - arrowtip_length * V;

  // Draw the arrowtip left and right lines.
  cv::Point arrowtip_left_start(static_cast<int>(round(arrowtip_left[0])),
                                static_cast<int>(round(arrowtip_left[1])));
  cv::Point arrowtip_right_start(static_cast<int>(round(arrowtip_right[0])),
                                 static_cast<int>(round(arrowtip_right[1])));
  cv::line(mat_image_, arrowtip_left_start, arrow_end, color, thickness);
  cv::line(mat_image_, arrowtip_right_start, arrow_end, color, thickness);
}

void AnnotationRenderer::DrawPoint(const RenderAnnotation& annotation) {
  const auto& point = annotation.point();
  int x = -1;
  int y = -1;
  if (point.normalized()) {
    CHECK(NormalizedtoPixelCoordinates(point.x(), point.y(), image_width_,
                                       image_height_, &x, &y));
  } else {
    x = static_cast<int>(point.x() * scale_factor_);
    y = static_cast<int>(point.y() * scale_factor_);
  }

  cv::Point point_to_draw(x, y);
  const cv::Scalar color = MediapipeColorToOpenCVColor(annotation.color());
  const int thickness =
      ClampThickness(round(annotation.thickness() * scale_factor_));
  cv::circle(mat_image_, point_to_draw, thickness, color, -1);
}

void AnnotationRenderer::DrawLine(const RenderAnnotation& annotation) {
  int x_start = -1;
  int y_start = -1;
  int x_end = -1;
  int y_end = -1;

  const auto& line = annotation.line();
  if (line.normalized()) {
    CHECK(NormalizedtoPixelCoordinates(line.x_start(), line.y_start(),
                                       image_width_, image_height_, &x_start,
                                       &y_start));
    CHECK(NormalizedtoPixelCoordinates(line.x_end(), line.y_end(), image_width_,
                                       image_height_, &x_end, &y_end));
  } else {
    x_start = static_cast<int>(line.x_start() * scale_factor_);
    y_start = static_cast<int>(line.y_start() * scale_factor_);
    x_end = static_cast<int>(line.x_end() * scale_factor_);
    y_end = static_cast<int>(line.y_end() * scale_factor_);
  }

  cv::Point start(x_start, y_start);
  cv::Point end(x_end, y_end);
  const cv::Scalar color = MediapipeColorToOpenCVColor(annotation.color());
  const int thickness =
      ClampThickness(round(annotation.thickness() * scale_factor_));
  cv::line(mat_image_, start, end, color, thickness);
}

void AnnotationRenderer::DrawGradientLine(const RenderAnnotation& annotation) {
  int x_start = -1;
  int y_start = -1;
  int x_end = -1;
  int y_end = -1;

  const auto& line = annotation.gradient_line();
  if (line.normalized()) {
    CHECK(NormalizedtoPixelCoordinates(line.x_start(), line.y_start(),
                                       image_width_, image_height_, &x_start,
                                       &y_start));
    CHECK(NormalizedtoPixelCoordinates(line.x_end(), line.y_end(), image_width_,
                                       image_height_, &x_end, &y_end));
  } else {
    x_start = static_cast<int>(line.x_start() * scale_factor_);
    y_start = static_cast<int>(line.y_start() * scale_factor_);
    x_end = static_cast<int>(line.x_end() * scale_factor_);
    y_end = static_cast<int>(line.y_end() * scale_factor_);
  }

  const cv::Point start(x_start, y_start);
  const cv::Point end(x_end, y_end);
  const int thickness =
      ClampThickness(round(annotation.thickness() * scale_factor_));
  const cv::Scalar color1 = MediapipeColorToOpenCVColor(line.color1());
  const cv::Scalar color2 = MediapipeColorToOpenCVColor(line.color2());
  cv_line2(mat_image_, start, end, color1, color2, thickness);
}

void AnnotationRenderer::DrawText(const RenderAnnotation& annotation) {
  int left = -1;
  int baseline = -1;
  int font_size = -1;

  const auto& text = annotation.text();
  if (text.normalized()) {
    CHECK(NormalizedtoPixelCoordinates(text.left(), text.baseline(),
                                       image_width_, image_height_, &left,
                                       &baseline));
    font_size = static_cast<int>(round(text.font_height() * image_height_));
  } else {
    left = static_cast<int>(text.left() * scale_factor_);
    baseline = static_cast<int>(text.baseline() * scale_factor_);
    font_size = static_cast<int>(text.font_height() * scale_factor_);
  }

  cv::Point origin(left, baseline);
  const cv::Scalar color = MediapipeColorToOpenCVColor(annotation.color());
  const int thickness =
      ClampThickness(round(annotation.thickness() * scale_factor_));
  const int font_face = text.font_face();

  const double font_scale = ComputeFontScale(font_face, font_size, thickness);
  int text_baseline = 0;
  cv::Size text_size = cv::getTextSize(text.display_text(), font_face,
                                       font_scale, thickness, &text_baseline);

  if (text.center_horizontally()) {
    origin.x -= text_size.width / 2;
  }
  if (text.center_vertically()) {
    origin.y += text_size.height / 2;
  }

  cv::putText(mat_image_, text.display_text(), origin, font_face, font_scale,
              color, thickness, /*lineType=*/8,
              /*bottomLeftOrigin=*/flip_text_vertically_);
}

double AnnotationRenderer::ComputeFontScale(int font_face, int font_size,
                                            int thickness) {
  double base_line;
  double cap_line;

  // The details below of how to compute the font scale from font face,
  // thickness, and size were inferred from the OpenCV implementation.
  switch (font_face) {
    case cv::FONT_HERSHEY_SIMPLEX:
    case cv::FONT_HERSHEY_DUPLEX:
    case cv::FONT_HERSHEY_COMPLEX:
    case cv::FONT_HERSHEY_TRIPLEX:
    case cv::FONT_HERSHEY_SCRIPT_SIMPLEX:
    case cv::FONT_HERSHEY_SCRIPT_COMPLEX:
      base_line = 9;
      cap_line = 12;
      break;
    case cv::FONT_HERSHEY_PLAIN:
      base_line = 5;
      cap_line = 4;
      break;
    case cv::FONT_HERSHEY_COMPLEX_SMALL:
      base_line = 6;
      cap_line = 7;
      break;
    default:
      return -1;
  }

  const double thick = static_cast<double>(thickness + 1);
  return (static_cast<double>(font_size) - (thick / 2.0F)) /
         (cap_line + base_line);
}

}  // namespace mediapipe
