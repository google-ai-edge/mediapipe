// Copyright 2020 The MediaPipe Authors.
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

#include "mediapipe/modules/objectron/calculators/box_util.h"

#include <math.h>

#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/util/tracking/box_tracker.pb.h"

namespace mediapipe {
void ComputeBoundingRect(const std::vector<cv::Point2f>& points,
                         mediapipe::TimedBoxProto* box) {
  CHECK(box != nullptr);
  float top = 1.0f;
  float bottom = 0.0f;
  float left = 1.0f;
  float right = 0.0f;
  for (const auto& point : points) {
    top = std::min(top, point.y);
    bottom = std::max(bottom, point.y);
    left = std::min(left, point.x);
    right = std::max(right, point.x);
  }
  box->set_top(top);
  box->set_bottom(bottom);
  box->set_left(left);
  box->set_right(right);
  // We are currently only doing axis aligned bounding box. If we need to
  // compute rotated bounding box, then we need the original image aspect ratio,
  // map back to original image space, compute cv::convexHull, then for each
  // edge of the hull, rotate according to edge orientation, find the box.
  box->set_rotation(0.0f);
}

float ComputeBoxIoU(const TimedBoxProto& box1, const TimedBoxProto& box2) {
  cv::Point2f box1_center((box1.left() + box1.right()) * 0.5f,
                          (box1.top() + box1.bottom()) * 0.5f);
  cv::Size2f box1_size(box1.right() - box1.left(), box1.bottom() - box1.top());
  cv::RotatedRect rect1(box1_center, box1_size,
                        -box1.rotation() * 180.0f / M_PI);
  cv::Point2f box2_center((box2.left() + box2.right()) * 0.5f,
                          (box2.top() + box2.bottom()) * 0.5f);
  cv::Size2f box2_size(box2.right() - box2.left(), box2.bottom() - box2.top());
  cv::RotatedRect rect2(box2_center, box2_size,
                        -box2.rotation() * 180.0f / M_PI);
  std::vector<cv::Point2f> intersections_unsorted;
  std::vector<cv::Point2f> intersections;
  cv::rotatedRectangleIntersection(rect1, rect2, intersections_unsorted);
  if (intersections_unsorted.size() < 3) {
    return 0.0f;
  }
  cv::convexHull(intersections_unsorted, intersections);

  // We use Shoelace formula to compute area of polygons.
  float intersection_area = 0.0f;
  for (int i = 0; i < intersections.size(); ++i) {
    const auto& curr_pt = intersections[i];
    const int i_next = (i + 1) == intersections.size() ? 0 : (i + 1);
    const auto& next_pt = intersections[i_next];
    intersection_area += (curr_pt.x * next_pt.y - next_pt.x * curr_pt.y);
  }
  intersection_area = std::abs(intersection_area) * 0.5f;

  // Compute union area
  const float union_area =
      rect1.size.area() + rect2.size.area() - intersection_area + 1e-5f;

  const float iou = intersection_area / union_area;
  return iou;
}

std::vector<cv::Point2f> ComputeBoxCorners(const TimedBoxProto& box,
                                           float width, float height) {
  // Rotate 4 corner w.r.t. center.
  const cv::Point2f center(0.5f * (box.left() + box.right()) * width,
                           0.5f * (box.top() + box.bottom()) * height);
  const std::vector<cv::Point2f> corners{
      cv::Point2f(box.left() * width, box.top() * height),
      cv::Point2f(box.left() * width, box.bottom() * height),
      cv::Point2f(box.right() * width, box.bottom() * height),
      cv::Point2f(box.right() * width, box.top() * height)};

  const float cos_a = std::cos(box.rotation());
  const float sin_a = std::sin(box.rotation());
  std::vector<cv::Point2f> transformed_corners(4);
  for (int k = 0; k < 4; ++k) {
    // Scale and rotate w.r.t. center.
    const cv::Point2f rad = corners[k] - center;
    const cv::Point2f rot_rad(cos_a * rad.x - sin_a * rad.y,
                              sin_a * rad.x + cos_a * rad.y);
    transformed_corners[k] = center + rot_rad;
    transformed_corners[k].x /= width;
    transformed_corners[k].y /= height;
  }
  return transformed_corners;
}

cv::Mat PerspectiveTransformBetweenBoxes(const TimedBoxProto& src_box,
                                         const TimedBoxProto& dst_box,
                                         const float aspect_ratio) {
  std::vector<cv::Point2f> box1_corners =
      ComputeBoxCorners(src_box, /*width*/ aspect_ratio, /*height*/ 1.0f);
  std::vector<cv::Point2f> box2_corners =
      ComputeBoxCorners(dst_box, /*width*/ aspect_ratio, /*height*/ 1.0f);
  cv::Mat affine_transform = cv::getPerspectiveTransform(
      /*src*/ box1_corners, /*dst*/ box2_corners);
  cv::Mat output_affine;
  affine_transform.convertTo(output_affine, CV_32FC1);
  return output_affine;
}

cv::Point2f MapPoint(const TimedBoxProto& src_box, const TimedBoxProto& dst_box,
                     const cv::Point2f& src_point, float width, float height) {
  const cv::Point2f src_center(
      0.5f * (src_box.left() + src_box.right()) * width,
      0.5f * (src_box.top() + src_box.bottom()) * height);
  const cv::Point2f dst_center(
      0.5f * (dst_box.left() + dst_box.right()) * width,
      0.5f * (dst_box.top() + dst_box.bottom()) * height);
  const float scale_x =
      (dst_box.right() - dst_box.left()) / (src_box.right() - src_box.left());
  const float scale_y =
      (dst_box.bottom() - dst_box.top()) / (src_box.bottom() - src_box.top());
  const float rotation = dst_box.rotation() - src_box.rotation();
  const cv::Point2f rad =
      cv::Point2f(src_point.x * width, src_point.y * height) - src_center;
  const float rad_x = rad.x * scale_x;
  const float rad_y = rad.y * scale_y;
  const float cos_a = std::cos(rotation);
  const float sin_a = std::sin(rotation);
  const cv::Point2f rot_rad(cos_a * rad_x - sin_a * rad_y,
                            sin_a * rad_x + cos_a * rad_y);
  const cv::Point2f dst_point_image = dst_center + rot_rad;
  const cv::Point2f dst_point(dst_point_image.x / width,
                              dst_point_image.y / height);
  return dst_point;
}

}  // namespace mediapipe
