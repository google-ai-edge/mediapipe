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

#include "mediapipe/graphs/object_detection_3d/calculators/box_util.h"

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/util/tracking/box_tracker.pb.h"

namespace mediapipe {
namespace {

TEST(BoxUtilTest, TestComputeBoundingRect) {
  std::vector<cv::Point2f> points{
      cv::Point2f(0.35f, 0.25f), cv::Point2f(0.3f, 0.3f),
      cv::Point2f(0.2f, 0.4f),   cv::Point2f(0.3f, 0.1f),
      cv::Point2f(0.2f, 0.2f),   cv::Point2f(0.5f, 0.3f),
      cv::Point2f(0.4f, 0.4f),   cv::Point2f(0.5f, 0.1f),
      cv::Point2f(0.4f, 0.2f)};
  TimedBoxProto box;
  ComputeBoundingRect(points, &box);
  EXPECT_FLOAT_EQ(0.1f, box.top());
  EXPECT_FLOAT_EQ(0.4f, box.bottom());
  EXPECT_FLOAT_EQ(0.2f, box.left());
  EXPECT_FLOAT_EQ(0.5f, box.right());
}

TEST(BoxUtilTest, TestComputeBoxIoU) {
  TimedBoxProto box1;
  box1.set_top(0.2f);
  box1.set_bottom(0.6f);
  box1.set_left(0.1f);
  box1.set_right(0.3f);
  box1.set_rotation(0.0f);
  TimedBoxProto box2 = box1;
  box2.set_rotation(/*pi/2*/ 1.570796f);
  const float box_area =
      (box1.bottom() - box1.top()) * (box1.right() - box1.left());
  const float box_intersection =
      (box1.right() - box1.left()) * (box1.right() - box1.left());
  const float expected_iou =
      box_intersection / (box_area * 2 - box_intersection);
  EXPECT_NEAR(expected_iou, ComputeBoxIoU(box1, box2), 3e-5f);

  TimedBoxProto box3;
  box3.set_top(0.2f);
  box3.set_bottom(0.6f);
  box3.set_left(0.5f);
  box3.set_right(0.7f);
  EXPECT_NEAR(0.0f, ComputeBoxIoU(box1, box3), 3e-5f);
}

TEST(BoxUtilTest, TestPerspectiveTransformBetweenBoxes) {
  TimedBoxProto box1;
  const float height = 4.0f;
  const float width = 3.0f;
  box1.set_top(1.0f / height);
  box1.set_bottom(2.0f / height);
  box1.set_left(1.0f / width);
  box1.set_right(2.0f / width);
  TimedBoxProto box2;
  box2.set_top(1.0f / height);
  box2.set_bottom(2.0f / height);
  box2.set_left(1.0f / width);
  box2.set_right(2.0f / width);
  box2.set_rotation(/*pi/4*/ -0.785398f);
  cv::Mat transform =
      PerspectiveTransformBetweenBoxes(box1, box2, width / height);
  const float kTolerence = 1e-5f;
  const cv::Vec3f original_position(1.5f / width, 1.0f / height, 1.0f);
  const cv::Mat transformed_position = transform * cv::Mat(original_position);
  EXPECT_NEAR(
      (1.5f - 0.5f * std::sqrt(2) / 2.0f) / width,
      transformed_position.at<float>(0) / transformed_position.at<float>(2),
      kTolerence);
  EXPECT_NEAR(
      (1.5f - 0.5f * std::sqrt(2) / 2.0f) / height,
      transformed_position.at<float>(1) / transformed_position.at<float>(2),
      kTolerence);
}

TEST(BoxUtilTest, TestMapPoint) {
  const float height = 4.0f;
  const float width = 3.0f;
  TimedBoxProto box1;
  box1.set_top(1.0f / height);
  box1.set_bottom(2.0f / height);
  box1.set_left(1.0f / width);
  box1.set_right(2.0f / width);
  TimedBoxProto box2;
  box2.set_top(1.0f / height);
  box2.set_bottom(2.0f / height);
  box2.set_left(1.0f / width);
  box2.set_right(2.0f / width);
  box2.set_rotation(/*pi/4*/ -0.785398f);

  cv::Point2f src_point1(1.2f / width, 1.4f / height);
  cv::Point2f src_point2(1.3f / width, 1.8f / height);
  const float distance1 = std::sqrt(0.1 * 0.1 + 0.4 * 0.4);
  cv::Point2f dst_point1 = MapPoint(box1, box2, src_point1, width, height);
  cv::Point2f dst_point2 = MapPoint(box1, box2, src_point2, width, height);
  const float distance2 =
      std::sqrt((dst_point1.x * width - dst_point2.x * width) *
                    (dst_point1.x * width - dst_point2.x * width) +
                (dst_point1.y * height - dst_point2.y * height) *
                    (dst_point1.y * height - dst_point2.y * height));
  EXPECT_NEAR(distance1, distance2, 1e-5f);
}

}  // namespace
}  // namespace mediapipe
