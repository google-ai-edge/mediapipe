// Copyright 2022 The MediaPipe Authors.
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

#include "mediapipe/framework/formats/location_opencv.h"

#include "mediapipe/framework/formats/annotation/rasterization.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/rectangle.h"

namespace mediapipe {

// 7x3x1 test mask pattern containing the following region types: bordering left
// and right edges, multiple and single pixel lengths, multiple and single
// segments per row.
static const int kWidth = 7;
static const int kHeight = 3;
const std::vector<uint8> kTestPatternVector = {0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0,
                                               0, 0, 0, 1, 0, 1, 0, 1, 0, 0};

// Interval {y, x_start, x_end} representation of kTestPatternVector.
const std::vector<std::vector<int>> kTestPatternIntervals = {
    {0, 5, 6}, {1, 1, 2}, {2, 0, 0}, {2, 2, 2}, {2, 4, 4}};

static const float kEps = 0.0001f;

Location TestPatternIntervalsToMaskLocation() {
  LocationData data;
  data.set_format(LocationData::MASK);
  data.mutable_mask()->set_width(kWidth);
  data.mutable_mask()->set_height(kHeight);
  for (const auto& test_interval : kTestPatternIntervals) {
    auto interval =
        data.mutable_mask()->mutable_rasterization()->add_interval();
    interval->set_y(test_interval[0]);
    interval->set_left_x(test_interval[1]);
    interval->set_right_x(test_interval[2]);
  }
  return Location(data);
}

TEST(LocationOpencvTest, CreateBBoxLocation) {
  const int x_start = 1;
  const int y_start = 2;
  const int width = 3;
  const int height = 4;

  const cv::Rect cv_rect(x_start, y_start, width, height);
  Location location = CreateBBoxLocation(cv_rect);
  auto rect = location.GetBBox<Rectangle_i>();

  const std::vector<int> cv_rect_dims(
      {cv_rect.x, cv_rect.y, cv_rect.width, cv_rect.height});
  const std::vector<int> rect_dims(
      {rect.xmin(), rect.ymin(), rect.Width(), rect.Height()});
  EXPECT_EQ(cv_rect_dims, rect_dims);
}

TEST(LocationOpencvTest, CreateCvMaskLocation) {
  cv::Mat_<uint8> test_mask(kHeight, kWidth,
                            const_cast<uint8*>(kTestPatternVector.data()));
  Location location = CreateCvMaskLocation(test_mask);
  auto intervals = location.ConvertToProto().mask().rasterization().interval();
  EXPECT_EQ(intervals.size(), kTestPatternIntervals.size());
  for (int i = 0; i < intervals.size(); ++i) {
    const std::vector<int> vec = {intervals[i].y(), intervals[i].left_x(),
                                  intervals[i].right_x()};
    EXPECT_EQ(vec, kTestPatternIntervals[i]);
  }
}

TEST(LocationOpenCvTest, EnlargeLocationMaskGrow) {
  const float grow_factor = 1.3;
  auto test_location = TestPatternIntervalsToMaskLocation();
  const float sum = cv::sum(*GetCvMask(test_location))[0];
  EnlargeLocation(test_location, grow_factor);
  const float grown_sum = cv::sum(*GetCvMask(test_location))[0];
  EXPECT_GT(grown_sum, sum);
}

TEST(LocationOpenCvTest, EnlargeMaskShrink) {
  const float shrink_factor = 0.7;
  auto test_location = TestPatternIntervalsToMaskLocation();
  const float sum = cv::sum(*GetCvMask(test_location))[0];
  EnlargeLocation(test_location, shrink_factor);
  const float shrunk_sum = cv::sum(*GetCvMask(test_location))[0];
  EXPECT_GT(sum, shrunk_sum);
}

TEST(LocationOpenCvTest, EnlargeBBox) {
  const float test_factor = 1.2f;
  auto relative_bbox =
      Location::CreateRelativeBBoxLocation(0.5f, 0.3f, 0.2f, 0.6f);
  EnlargeLocation(relative_bbox, test_factor);
  auto enlarged_relative_bbox_rect = relative_bbox.GetRelativeBBox();

  EXPECT_NEAR(enlarged_relative_bbox_rect.xmin(), 0.48f, kEps);
  EXPECT_NEAR(enlarged_relative_bbox_rect.ymin(), 0.24f, kEps);
  EXPECT_NEAR(enlarged_relative_bbox_rect.Width(), 0.24f, kEps);
  EXPECT_NEAR(enlarged_relative_bbox_rect.Height(), 0.72f, kEps);

  auto bbox = Location::CreateBBoxLocation(50, 30, 20, 60);
  EnlargeLocation(bbox, test_factor);
  auto enlarged_bbox_rect = bbox.GetBBox<Rectangle_i>();

  EXPECT_EQ(enlarged_bbox_rect.xmin(), 48);
  EXPECT_EQ(enlarged_bbox_rect.ymin(), 24);
  EXPECT_EQ(enlarged_bbox_rect.Width(), 24);
  EXPECT_EQ(enlarged_bbox_rect.Height(), 72);
}

TEST(LocationOpenCvTest, ConvertRelativeBBoxToCvMask) {
  const float rel_x_min = 0.1;
  const float rel_y_min = 0.2;
  const float rel_width = 0.3;
  const float rel_height = 0.6;
  const int width = 10;
  const int height = 20;
  cv::Size expected_size(width, height);

  LocationData data;
  data.set_format(LocationData::RELATIVE_BOUNDING_BOX);
  data.mutable_relative_bounding_box()->set_xmin(rel_x_min);
  data.mutable_relative_bounding_box()->set_ymin(rel_y_min);
  data.mutable_relative_bounding_box()->set_width(rel_width);
  data.mutable_relative_bounding_box()->set_height(rel_height);
  Location test_location(data);

  const int x_start = rel_x_min * width;
  const int x_end = x_start + rel_width * width;
  const int y_start = rel_y_min * height;
  const int y_end = y_start + rel_height * height;

  const auto cv_mask = *ConvertToCvMask(test_location, width, height);
  EXPECT_EQ(cv_mask.size(), expected_size);
  for (int y = 0; y < cv_mask.rows; ++y) {
    for (int x = 0; x < cv_mask.cols; ++x) {
      bool in_mask = (x >= x_start && x < x_end && y >= y_start && y < y_end);
      float expected_value = in_mask ? 1 : 0;
      ASSERT_EQ(cv_mask.at<float>(y, x), expected_value);
    }
  }
}

TEST(LocationOpenCvTest, GetCvMask) {
  auto test_location = TestPatternIntervalsToMaskLocation();
  auto cv_mask = *GetCvMask(test_location);
  EXPECT_EQ(cv_mask.cols * cv_mask.rows, kTestPatternVector.size());
  int flat_idx = 0;
  for (auto it = cv_mask.begin<uint8>(); it != cv_mask.end<uint8>(); ++it) {
    const uint8 expected_value = kTestPatternVector[flat_idx] == 0 ? 0 : 255;
    EXPECT_EQ(*it, expected_value);
    flat_idx++;
  }
}

}  // namespace mediapipe
