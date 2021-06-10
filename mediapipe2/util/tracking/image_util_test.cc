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

#include "mediapipe/util/tracking/image_util.h"

#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {

template <int border>
void TestCopyBorder(cv::Mat* full_size) {
  full_size->setTo(0);

  const int width = full_size->cols - 2 * border;
  // Fill center with per pixel value [1, 2, 3].
  for (int y = border; y < full_size->rows - border; ++y) {
    float* row_ptr = full_size->ptr<float>(y);
    for (int x = border; x < full_size->cols - border; ++x) {
      int value = x - border;
      row_ptr[3 * x] = value + 0.1f;
      row_ptr[3 * x + 1] = value + 0.2f;
      row_ptr[3 * x + 2] = value + 0.3f;
    }
  }

  CopyMatBorder<float, border, 3>(full_size);

  // Main part should not be modified.
  for (int y = border; y < full_size->rows - border; ++y) {
    float* row_ptr = full_size->ptr<float>(y);
    for (int x = border; x < full_size->cols - border; ++x) {
      int value = x - border;
      EXPECT_EQ(row_ptr[3 * x], value + 0.1f);
      EXPECT_EQ(row_ptr[3 * x + 1], value + 0.2f);
      EXPECT_EQ(row_ptr[3 * x + 2], value + 0.3f);
    }
  }

  for (int y = 0; y < full_size->rows; ++y) {
    const float* row_ptr = full_size->ptr<float>(y);
    for (int x = 0; x < full_size->cols; ++x) {
      // Memory copy, expect exact floating point equality.
      int value = x - border;
      if (x < border) {
        // Cap to valid area.
        value = std::min(width - 1, (border - 1) - x);
      } else if (x >= full_size->cols - border) {
        // Last column minus distance from frame boundary, capped
        // to valid area.
        value = std::max(0, width - 1 - (x - full_size->cols + border));
      }
      EXPECT_EQ(row_ptr[3 * x], value + 0.1f);
      EXPECT_EQ(row_ptr[3 * x + 1], value + 0.2f);
      EXPECT_EQ(row_ptr[3 * x + 2], value + 0.3f);
    }
  }
}

TEST(PushPullFilteringTest, CopyBorder) {
  cv::Mat full_size(100, 50, CV_32FC3);
  TestCopyBorder<1>(&full_size);
  TestCopyBorder<2>(&full_size);
  TestCopyBorder<3>(&full_size);
  TestCopyBorder<4>(&full_size);
  TestCopyBorder<5>(&full_size);
}

TEST(PushPullFilteringTest, CopyBorderSmallFrame) {
  cv::Mat full_size(3, 3, CV_32FC3);
  TestCopyBorder<1>(&full_size);

  full_size.create(5, 5, CV_32FC3);
  TestCopyBorder<2>(&full_size);

  full_size.create(7, 7, CV_32FC3);
  TestCopyBorder<3>(&full_size);

  full_size.create(9, 9, CV_32FC3);
  TestCopyBorder<4>(&full_size);
}

}  // namespace mediapipe
