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
//#include <android/log.h>

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

static const std::vector<int> UPPER_LIP = {61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78};
static const std::vector<int> LOWER_LIP = {61, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146};
static const std::vector<int> FACE_OVAL = {10, 338, 338, 297, 297, 332, 332, 284, 284, 251, 251, 389, 389, 356, 356,
                                           454, 454, 323, 323, 361, 361, 288, 288, 397, 397, 365, 365, 379, 379, 378,
                                           378, 400, 400, 377, 377, 152, 152, 148, 148, 176, 176, 149, 149, 150, 150,
                                           136, 136, 172, 172, 58, 58, 132, 132, 93, 93, 234, 234, 127, 127, 162, 162,
                                           21, 21, 54, 54, 103, 103, 67, 67, 109, 109, 10};
static const std::vector<int> MOUTH_INSIDE = {78, 191, 80, 81, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95};
static const std::vector<int> PART_FOREHEAD_B = {21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 301, 293, 334, 296, 336, 9, 107, 66, 105, 63, 71};
static const std::vector<int> LEFT_EYE = {130, 33, 246, 161, 160, 159, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7};
static const std::vector<int> RIGHT_EYE = {362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382};
static const std::vector<int> LIPS = {61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146};
static const std::vector<int> LEFT_BROW = {70, 63, 105, 66, 107, 55, 65, 52, 53, 46};
static const std::vector<int> RIGHT_BROW = {336, 296, 334, 293, 301, 300, 283, 282, 295, 285};

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

void AnnotationRenderer::RenderDataOnImage(const RenderData &render_data)
{
  if (render_data.render_annotations().size()){
    DrawLipstick(render_data);
    WhitenTeeth(render_data);
    //SmoothFace(render_data);
  }
  else
  {
    LOG(FATAL) << "Unknown annotation type: ";
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

cv::Mat AnnotationRenderer::FormFacePartMask(std::vector<int> orderList, const RenderData &render_data)
{
  int c = 0;
  std::vector<cv::Point> points_array;
  cv::Mat mask = cv::Mat::zeros(mat_image_.size(), CV_32F);
  for (auto order : orderList)
  {
    c = 0;
    for (auto &annotation : render_data.render_annotations())
    {
      if (annotation.data_case() == RenderAnnotation::kPoint)
      {
        if (order == c)
        {
          const auto &point = annotation.point();
          int x = -1;
          int y = -1;
          if (point.normalized())
          {
            CHECK(NormalizedtoPixelCoordinates(point.x(), point.y(), image_width_,
                                               image_height_, &x, &y));
          }
          else
          {
            x = static_cast<int>(point.x() * scale_factor_);
            y = static_cast<int>(point.y() * scale_factor_);
          }
          points_array.push_back(cv::Point(x, y));
        }
        c += 1;
      }
    }
  }

  if (points_array.size() != orderList.size()){
    mask.convertTo(mask, CV_8U);
    return mask;
  }

  std::vector<std::vector<cv::Point>> points_array_wrapper;
  points_array_wrapper.push_back(points_array);

  cv::fillPoly(mask, points_array_wrapper, cv::Scalar::all(255), cv::LINE_AA);
  mask.convertTo(mask, CV_8U);

  return mask;
}


std::tuple<double, double, double, double> AnnotationRenderer::GetFaceBox(const RenderData &render_data)
{
  std::vector<int> x_s, y_s;
  double box_min_y, box_max_y, box_max_x, box_min_x;

  for (auto &annotation : render_data.render_annotations())
  {
    if (annotation.data_case() == RenderAnnotation::kPoint)
    {
      const auto &point = annotation.point();
      int x = -1;
      int y = -1;
      if (point.normalized())
      {
        CHECK(NormalizedtoPixelCoordinates(point.x(), point.y(), image_width_,
                                           image_height_, &x, &y));
      }
      else
      {
        x = static_cast<int32>(point.x() * scale_factor_);
        y = static_cast<int32>(point.y() * scale_factor_);
      }
      x_s.push_back(point.x());
      x_s.push_back(point.y());
    }
  }
  cv::minMaxLoc(y_s, &box_min_y, &box_max_y);
  cv::minMaxLoc(x_s, &box_min_x, &box_max_x);
  box_min_y = box_min_y * 0.9;

  return std::make_tuple(box_min_x, box_min_y, box_max_x, box_max_y);
}

cv::Mat AnnotationRenderer::predict_forehead_mask(const RenderData &render_data, double face_box_min_y)
{

  cv::Mat part_forehead_mask = AnnotationRenderer::FormFacePartMask(PART_FOREHEAD_B, render_data);
  part_forehead_mask.convertTo(part_forehead_mask, CV_32F, 1.0 / 255);
  part_forehead_mask.convertTo(part_forehead_mask, CV_8U);

  cv::Mat image_sm, image_sm_hsv, skinMask;

  cv::resize(mat_image_, image_sm, cv::Size(mat_image_.size().width, mat_image_.size().height));
  cv::cvtColor(image_sm, image_sm_hsv, cv::COLOR_BGR2HSV);

  std::vector<int> x, y;
  std::vector<cv::Point> location;
  // std::cout << "R (numpy)   = " << std::endl << cv::format(part_forehead_mask, cv::Formatter::FMT_NUMPY ) << std::endl << std::endl;

  cv::Vec3d hsv_min, hsv_max;

  std::vector<cv::Mat> channels(3);
  cv::split(image_sm_hsv, channels);
  std::vector<std::vector<double>> minx(3), maxx(3);
  int c = 0;
  for (auto ch : channels)
  {
    cv::Mat row, mask_row;
    double min, max;
    for (int i = 0; i < ch.rows; i++)
    {
      row = ch.row(i);
      mask_row = part_forehead_mask.row(i);
      cv::minMaxLoc(row, &min, &max, 0, 0, mask_row);
      minx[c].push_back(min);
      maxx[c].push_back(max);
    }
    c++;
  }
  for (int i = 0; i < 3; i++)
  {
    hsv_min[i] = *std::min_element(minx[i].begin(), minx[i].end());
  }
  for (int i = 0; i < 3; i++)
  {
    hsv_max[i] = *std::max_element(maxx[i].begin(), maxx[i].end());
  }

  cv::Mat _forehead_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(1, 1));
  cv::inRange(image_sm_hsv, hsv_min, hsv_max, skinMask);
  cv::erode(skinMask, skinMask, _forehead_kernel, cv::Point(-1, -1), 2);
  cv::dilate(skinMask, skinMask, _forehead_kernel, cv::Point(-1, -1), 2);
  skinMask.convertTo(skinMask, CV_8U, 1.0 / 255);

  cv::findNonZero(skinMask, location);

  double max_part_f, x_min_part, x_max_part;

  for (auto &i : location)
  {
    x.push_back(i.x);
    y.push_back(i.y);
  }

  cv::minMaxLoc(y, NULL, &max_part_f);
  cv::minMaxLoc(x, &x_min_part, &x_max_part);

  cv::Mat new_skin_mask = cv::Mat::zeros(skinMask.size(), CV_8U);

  new_skin_mask(cv::Range(face_box_min_y, max_part_f), cv::Range(x_min_part, x_max_part)) =
      skinMask(cv::Range(face_box_min_y, max_part_f), cv::Range(x_min_part, x_max_part));

  return new_skin_mask;
}

void AnnotationRenderer::SmoothFace(const RenderData &render_data)
{
  cv::Mat not_full_face = cv::Mat(FormFacePartMask(FACE_OVAL, render_data)) +
                          cv::Mat(predict_forehead_mask(render_data, std::get<1>(GetFaceBox(render_data)))) -
                          cv::Mat(FormFacePartMask(LEFT_EYE, render_data)) -
                          cv::Mat(FormFacePartMask(RIGHT_EYE, render_data)) -
                          cv::Mat(FormFacePartMask(LEFT_BROW, render_data)) -
                          cv::Mat(FormFacePartMask(RIGHT_BROW, render_data)) -
                          cv::Mat(FormFacePartMask(LIPS, render_data));

  cv::resize(not_full_face,
             not_full_face,
             mat_image_.size(), 0, 0,
             cv::INTER_LINEAR);

  std::vector<int> x, y;
  std::vector<cv::Point> location;

  cv::findNonZero(not_full_face, location);

  double min_y, min_x, max_x, max_y;

  for (auto &i : location)
  {
    x.push_back(i.x);
    y.push_back(i.y);
  }

  cv::minMaxLoc(x, &min_x, &max_x);
  cv::minMaxLoc(y, &min_y, &max_y);

  cv::Mat patch_face = mat_image_(cv::Range(min_y, max_y), cv::Range(min_x, max_x));
  cv::Mat patch_nff = not_full_face(cv::Range(min_y, max_y), cv::Range(min_x, max_x));
  cv::Mat patch_new, patch_wow;
  cv::cvtColor(patch_face, patch_wow, cv::COLOR_RGBA2RGB);
  if (patch_wow.data != patch_new.data) {
    cv::bilateralFilter(patch_wow, patch_new, 12, 50, 50);
  }

  cv::Mat patch_new_nff, patch_new_mask, patch, patch_face_nff;

  patch_new.copyTo(patch_new_nff, patch_nff);

  patch_face.copyTo(patch_face_nff, patch_nff);
  cv::cvtColor(patch_face_nff, patch_face_nff, cv::COLOR_RGBA2RGB);

  patch_new_mask = 0.85 * patch_new_nff + 0.15 * patch_face_nff;
  
  patch = cv::min(255, patch_new_mask);
  patch.copyTo(patch_face, patch_nff);
}

cv::Mat matmul32F(cv::Mat& bgr, cv::Mat& mask)
{
    assert(bgr.type() == CV_32FC3 && mask.type() == CV_32FC1 && bgr.size() == mask.size());
    int H = bgr.rows;
    int W = bgr.cols;
    cv::Mat dst(bgr.size(), bgr.type());

    if (bgr.isContinuous() && mask.isContinuous())
    {
        W *= H;
        H = 1;
    }

    for( int i = 0; i < H; ++i)
    {
        float* pdst = ((float*)dst.data)+i*W*3;
        float* pbgr = ((float*)bgr.data)+i*W*3;
        float* pmask = ((float*)mask.data) + i*W;
        for ( int j = 0; j < W; ++j)
        {
            (*pdst++) = (*pbgr++) *(*pmask);
            (*pdst++) = (*pbgr++) *(*pmask);
            (*pdst++) = (*pbgr++) *(*pmask);
            pmask+=1;
        }
    }
    return dst;
}

void AnnotationRenderer::DrawLipstick(const RenderData &render_data)
{
  cv::Mat spec_lips_mask, upper_lips_mask, lower_lips_mask;
  spec_lips_mask = cv::Mat::zeros(mat_image_.size(), CV_32F);
  upper_lips_mask = cv::Mat::zeros(mat_image_.size(), CV_32F);
  lower_lips_mask = cv::Mat::zeros(mat_image_.size(), CV_32F);

  upper_lips_mask = AnnotationRenderer::FormFacePartMask(UPPER_LIP, render_data);
  lower_lips_mask = AnnotationRenderer::FormFacePartMask(LOWER_LIP, render_data);

  spec_lips_mask = upper_lips_mask + lower_lips_mask;
//
  spec_lips_mask.convertTo(spec_lips_mask, CV_8U);
//
  cv::resize(spec_lips_mask, spec_lips_mask, mat_image_.size(), cv::INTER_LINEAR);
//
  std::vector<int> x, y;
  std::vector<cv::Point> location;

  cv::findNonZero(spec_lips_mask, location);

  for (auto &i : location)
  {
    x.push_back(i.x);
    y.push_back(i.y);
  }

  if (!(x.empty()) && !(y.empty()))
  {
    double min_y, max_y, max_x, min_x;
    cv::minMaxLoc(y, &min_y, &max_y);
    cv::minMaxLoc(x, &min_x, &max_x);

    cv::Mat lips_crop_mask = spec_lips_mask(cv::Range(min_y, max_y), cv::Range(min_x, max_x));
    lips_crop_mask.convertTo(lips_crop_mask, CV_32F, 1.0 / 255);

    cv::Mat lips_crop = cv::Mat(mat_image_(cv::Range(min_y, max_y), cv::Range(min_x, max_x)));

    cv::Mat lips_blend = cv::Mat(lips_crop.size().height, lips_crop.size().width, CV_32FC4, cv::Scalar(255.0, 0, 0, 0));

    std::vector<cv::Mat> channels(4);

    cv::split(lips_blend, channels);
    channels[3] = lips_crop_mask * 20;

    cv::merge(channels, lips_blend);

    cv::Mat tmp_lip_mask;

    channels[3].convertTo(tmp_lip_mask, CV_32FC1, 1.0 / 255);

    cv::split(lips_blend, channels);
    for (auto &ch : channels)
    {
      cv::multiply(ch, tmp_lip_mask, ch, 1.0, CV_32F);
    }
    cv::merge(channels, lips_blend);

    cv::subtract(1.0, tmp_lip_mask, tmp_lip_mask, cv::noArray(), CV_32F);

    cv::split(lips_crop, channels);
    for (auto &ch : channels)
    {
      cv::multiply(ch, tmp_lip_mask, ch, 1.0, CV_8U);
    }
    cv::merge(channels, lips_crop);

    cv::add(lips_blend, lips_crop, lips_crop, cv::noArray(), CV_8U);

    lips_crop = cv::abs(lips_crop);

    cvtColor(lips_crop, lips_crop, cv::COLOR_RGBA2RGB);

    cv::Mat slice = mat_image_(cv::Range(min_y, max_y), cv::Range(min_x, max_x));
    lips_crop_mask.convertTo(lips_crop_mask, slice.type());
    slice.copyTo(slice, lips_crop_mask);

    cv::Mat masked_lips_crop, slice_gray;
    lips_crop.copyTo(masked_lips_crop, lips_crop_mask);

    cv::cvtColor(masked_lips_crop, slice_gray, cv::COLOR_RGB2GRAY);

    masked_lips_crop.copyTo(slice, slice_gray);
  }
}

void AnnotationRenderer::WhitenTeeth(const RenderData &render_data)
{
  cv::Mat mouth_mask, mouth;

  mouth_mask = cv::Mat::zeros(mat_image_.size(), CV_32F);
  mouth_mask = AnnotationRenderer::FormFacePartMask(MOUTH_INSIDE, render_data);

  cv::resize(mouth_mask, mouth, mat_image_.size(), cv::INTER_LINEAR);
  mouth.convertTo(mouth, CV_8U);

  std::vector<int> x, y;
  std::vector<cv::Point> location;

  cv::findNonZero(mouth, location);

  for (auto &i : location)
  {
    x.push_back(i.x);
    y.push_back(i.y);
  }

  if (!(x.empty()) && !(y.empty()))
  {
    double mouth_min_y, mouth_max_y, mouth_max_x, mouth_min_x;
    cv::minMaxLoc(y, &mouth_min_y, &mouth_max_y);
    cv::minMaxLoc(x, &mouth_min_x, &mouth_max_x);
    double mh = mouth_max_y - mouth_min_y;
    double mw = mouth_max_x - mouth_min_x;
    cv::Mat mouth_crop_mask;
    mouth.convertTo(mouth, CV_32F, 1.0 / 255);
    mouth.convertTo(mouth, CV_32F, 1.0 / 255);
    if (mh / mw > 0.17)
    {
      mouth_min_y = static_cast<int>(std::max(mouth_min_y - mh * 0.1, 0.0));
      mouth_max_y = static_cast<int>(std::min(mouth_max_y + mh * 0.1, (double)image_height_));
      mouth_min_x = static_cast<int>(std::max(mouth_min_x - mw * 0.1, 0.0));
      mouth_max_x = static_cast<int>(std::min(mouth_max_x + mw * 0.1, (double)image_width_));
      mouth_crop_mask = mouth(cv::Range(mouth_min_y, mouth_max_y), cv::Range(mouth_min_x, mouth_max_x));
      cv::Mat img_hsv, tmp_mask, img_hls;
      cv::cvtColor(mat_image_(cv::Range(mouth_min_y, mouth_max_y), cv::Range(mouth_min_x, mouth_max_x)), img_hsv,
                   cv::COLOR_RGBA2RGB);
      cv::cvtColor(img_hsv, img_hsv,
                   cv::COLOR_RGB2HSV);

      cv::Mat _mouth_erode_kernel = cv::getStructuringElement(
          cv::MORPH_ELLIPSE, cv::Size(7, 7));

      cv::erode(mouth_crop_mask * 255, tmp_mask, _mouth_erode_kernel, cv::Point(-1, -1), 3);
      cv::GaussianBlur(tmp_mask, tmp_mask, cv::Size(51, 51), 0);

      img_hsv.convertTo(img_hsv, CV_8U);

      std::vector<cv::Mat> channels(3);
      cv::split(img_hsv, channels);

      cv::Mat tmp;
      cv::multiply(channels[1], tmp_mask, tmp, 0.3, CV_8U);
      cv::subtract(channels[1], tmp, channels[1], cv::noArray(), CV_8U);
      channels[1] = cv::min(255, channels[1]);
      cv::merge(channels, img_hsv);

      cv::cvtColor(img_hsv, img_hsv, cv::COLOR_HSV2RGB);
      cv::cvtColor(img_hsv, img_hls, cv::COLOR_RGB2HLS);

      cv::split(img_hls, channels);
      cv::multiply(channels[1], tmp_mask, tmp, 0.3, CV_8U);
      cv::add(channels[1], tmp, channels[1], cv::noArray(), CV_8U);
      channels[1] = cv::min(255, channels[1]);
      cv::merge(channels, img_hls);

      cv::cvtColor(img_hls, img_hls, cv::COLOR_HLS2RGB);
      cv::cvtColor(img_hls, img_hls, cv::COLOR_RGB2RGBA);
      // std::cout << "R (numpy)   = " << std::endl << cv::format(img_hls, cv::Formatter::FMT_NUMPY ) << std::endl << std::endl;

      cv::Mat slice = mat_image_(cv::Range(mouth_min_y, mouth_max_y), cv::Range(mouth_min_x, mouth_max_x));
      img_hls.copyTo(slice);
    }
  }
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
