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

#include "mediapipe/framework/formats/motion/optical_flow_field.h"

#include <algorithm>
#include <memory>
#include <string>

#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/logging.h"
#include "tensorflow/core/framework/tensor.h"

namespace mediapipe {

namespace {
TEST(OpticalFlowField, ConstructsAndSerializes) {
  cv::Mat_<cv::Point2f> original_flow(31, 15);
  for (int r = 0; r < original_flow.rows; ++r) {
    for (int c = 0; c < original_flow.cols; ++c) {
      original_flow(r, c) = cv::Point2f(r * c / 2.0, r * c / 3.0);
    }
  }
  std::unique_ptr<OpticalFlowField> flow(new OpticalFlowField(original_flow));
  EXPECT_EQ(original_flow.cols, flow->width());
  EXPECT_EQ(original_flow.rows, flow->height());

  OpticalFlowFieldData proto;
  flow->ConvertToProto(&proto);

  EXPECT_EQ(flow->width(), proto.width());
  EXPECT_EQ(flow->height(), proto.height());
  EXPECT_EQ(original_flow.rows * original_flow.cols, proto.dx_size());
  EXPECT_EQ(original_flow.rows * original_flow.cols, proto.dy_size());

  OpticalFlowField from_proto;
  from_proto.SetFromProto(proto);

  ASSERT_EQ(original_flow.cols, from_proto.width());
  ASSERT_EQ(original_flow.rows, from_proto.height());

  for (int r = 0; r < original_flow.rows; ++r) {
    for (int c = 0; c < original_flow.cols; ++c) {
      const cv::Point2f& flow_at_r_c =
          from_proto.flow_data().at<cv::Point2f>(r, c);
      EXPECT_FLOAT_EQ(original_flow(r, c).x, flow_at_r_c.x);
      EXPECT_FLOAT_EQ(original_flow(r, c).y, flow_at_r_c.y);
    }
  }
}

TEST(OpticalFlowField, ConvertsFromTensorflow) {
  const int height = 3;
  const int width = 5;
  // Construct a flow field with (dx, dy) = (r, c).
  tensorflow::Tensor flow_tensor(tensorflow::DT_FLOAT, {height, width, 2});
  auto eigen_flow = flow_tensor.tensor<float, 3>();
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      eigen_flow(r, c, 0) = r;
      eigen_flow(r, c, 1) = c;
    }
  }
  OpticalFlowField flow_field;
  flow_field.CopyFromTensor(flow_tensor);
  ASSERT_EQ(height, flow_field.height());
  ASSERT_EQ(width, flow_field.width());
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      const auto& dx_dy = flow_field.flow_data().at<cv::Point2f>(r, c);
      EXPECT_EQ(r, dx_dy.x);
      EXPECT_EQ(c, dx_dy.y);
    }
  }
}

TEST(OpticalFlowField, ZeroFlowCanVisualize) {
  cv::Mat_<cv::Point2f> flow(15, 15, cv::Point2f(0.0f, 0.0f));
  std::unique_ptr<OpticalFlowField> flow_field(new OpticalFlowField(flow));
  cv::Mat viz = flow_field->GetVisualization();
  cv::Mat saturated_viz = flow_field->GetVisualizationSaturatedAt(1.0);
  EXPECT_EQ(flow.rows, viz.rows);
  EXPECT_EQ(flow.cols, viz.cols);
  EXPECT_EQ(flow.rows, saturated_viz.rows);
  EXPECT_EQ(flow.cols, saturated_viz.cols);
  // All zero flow fields should be white.
  for (int r = 0; r < viz.rows; ++r) {
    for (int c = 0; c < viz.cols; ++c) {
      for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(255, viz.at<cv::Vec3b>(r, c)[i]);
        EXPECT_EQ(255, saturated_viz.at<cv::Vec3b>(r, c)[i]);
      }
    }
  }
}

TEST(OpticalFlowField, FollowFlowWorks) {
  const int width = 17;
  const int height = 15;
  cv::Mat_<cv::Point2f> flow(height, width);
  // dx = 3x + y;
  // dy = 2x - y;
  for (int y = 0; y < flow.rows; ++y) {
    for (int x = 0; x < flow.cols; ++x) {
      flow(y, x) = cv::Point2f(3 * x + y, 2 * x - y);
    }
  }
  OpticalFlowField flow_field(flow);
  float new_x;
  float new_y;
  // Out of bounds tests.
  EXPECT_FALSE(flow_field.FollowFlow(-1, 0, &new_x, &new_y));
  EXPECT_FALSE(flow_field.FollowFlow(0, -0.5, &new_x, &new_y));
  EXPECT_FALSE(flow_field.FollowFlow(width, 0.5 * height, &new_x, &new_y));
  EXPECT_FALSE(flow_field.FollowFlow(0.5 * width, height, &new_x, &new_y));

  // Test corners.
  float x = 0.0f;
  float y = 0.0f;
  EXPECT_TRUE(flow_field.FollowFlow(x, y, &new_x, &new_y));
  EXPECT_FLOAT_EQ(new_x, x + 3 * x + y);
  EXPECT_FLOAT_EQ(new_y, y + 2 * x - y);

  x = width - 1;
  EXPECT_TRUE(flow_field.FollowFlow(x, y, &new_x, &new_y));
  EXPECT_FLOAT_EQ(new_x, x + 3 * x + y);
  EXPECT_FLOAT_EQ(new_y, y + 2 * x - y);

  y = height - 1;
  EXPECT_TRUE(flow_field.FollowFlow(x, y, &new_x, &new_y));
  EXPECT_FLOAT_EQ(new_x, x + 3 * x + y);
  EXPECT_FLOAT_EQ(new_y, y + 2 * x - y);

  x = 0.0f;
  EXPECT_TRUE(flow_field.FollowFlow(x, y, &new_x, &new_y));
  EXPECT_FLOAT_EQ(new_x, x + 3 * x + y);
  EXPECT_FLOAT_EQ(new_y, y + 2 * x - y);

  // Integer y coordinate (still linear).
  x = 0.25 * width;
  y = 1;
  EXPECT_TRUE(flow_field.FollowFlow(x, y, &new_x, &new_y));
  EXPECT_FLOAT_EQ(new_x, x + 3 * x + y);
  EXPECT_FLOAT_EQ(new_y, y + 2 * x - y);
  // Integer x coordinate (still linear).
  x = 1;
  y = 0.7 * height;
  EXPECT_TRUE(flow_field.FollowFlow(x, y, &new_x, &new_y));
  EXPECT_FLOAT_EQ(new_x, x + 3 * x + y);
  EXPECT_FLOAT_EQ(new_y, y + 2 * x - y);

  // Somewhere in the middle of a pixel, interpolation not linear.
  x = 4.9f;
  y = 3.2f;
  float dx = x - static_cast<int>(x);
  float dy = y - static_cast<int>(y);
  EXPECT_TRUE(flow_field.FollowFlow(x, y, &new_x, &new_y));
  float p00 = 3 * 4 + 3;  // 3 * floor(x) + floor(y)
  float p01 = 3 * 4 + 4;  // 3 * floor(x) + ceil(y)
  float p10 = 3 * 5 + 3;  // 3 * ceil(x) + floor(y)
  float p11 = 3 * 5 + 4;  // 3 * ceil(x) + ceil(y)
  float interpolated = p00 + (p10 - p00) * dx + (p01 - p00) * dy +
                       (p00 - p01 - p10 + p11) * dx * dy;
  EXPECT_FLOAT_EQ(new_x, x + interpolated);
  p00 = 2 * 4 - 3;  // 2 * floor(x) - floor(y)
  p01 = 2 * 4 - 4;  // 2 * floor(x) - ceil(y)
  p10 = 2 * 5 - 3;  // 2 * ceil(x) - floor(y)
  p11 = 2 * 5 - 4;  // 2 * ceil(x) - ceil(y)
  interpolated = p00 + (p10 - p00) * dx + (p01 - p00) * dy +
                 (p00 - p01 - p10 + p11) * dx * dy;
  EXPECT_FLOAT_EQ(new_y, y + interpolated);
}

TEST(OpticalFlowField, ConvertsToCorrespondences) {
  const int width = 17;
  const int height = 15;
  cv::Mat_<cv::Point2f> flow(height, width);
  // dx = 3x + y;
  // dy = 2x - y;
  for (int y = 0; y < flow.rows; ++y) {
    for (int x = 0; x < flow.cols; ++x) {
      flow(y, x) = cv::Point2f(3 * x + y, 2 * x - y);
    }
  }
  OpticalFlowField flow_field(flow);
  // Correspondences should be:
  // x' = x + dx = 4x + y
  // y' = y + dy = 2x.
  cv::Mat_<cv::Point2f> correspondences = flow_field.ConvertToCorrespondences();
  EXPECT_EQ(flow_field.height(), correspondences.rows);
  EXPECT_EQ(flow_field.width(), correspondences.cols);
  for (int y = 0; y < correspondences.rows; ++y) {
    for (int x = 0; x < correspondences.cols; ++x) {
      cv::Point2f matched_pixel = correspondences(y, x);
      EXPECT_FLOAT_EQ(4 * x + y, matched_pixel.x);
      EXPECT_FLOAT_EQ(2 * x, matched_pixel.y);
    }
  }
}

// Tests that AllWithinMargin can be used to determine exact equality.
TEST(OpticalFlowField, ExactEquality) {
  cv::Mat_<cv::Point2f> flow_data(3, 5);
  for (int y = 0; y < flow_data.rows; ++y) {
    for (int x = 0; x < flow_data.cols; ++x) {
      flow_data(y, x) = cv::Point2f((y - 1.5) / 2.0 + 2 * x, y + 3 - x);
    }
  }
  // Flows with matching flow fields are exactly equal.
  OpticalFlowField flow1(flow_data);
  OpticalFlowField flow2(flow_data);
  EXPECT_TRUE(flow1.AllWithinMargin(flow2, 0.0));
  EXPECT_TRUE(flow2.AllWithinMargin(flow1, 0.0));

  // A flow field with different dimensions is not equal.
  OpticalFlowField wrong_size(flow_data.rowRange(0, 2));
  EXPECT_FALSE(wrong_size.AllWithinMargin(flow1, 0.0));
  EXPECT_FALSE(flow1.AllWithinMargin(wrong_size, 0.0));

  // A flow that differs at one pixel is not equal.
  flow_data(0, 0) += cv::Point2f(0.1, 0);
  OpticalFlowField noisy_flow(flow_data);
  EXPECT_FALSE(flow1.AllWithinMargin(noisy_flow, 0.0));
  EXPECT_FALSE(noisy_flow.AllWithinMargin(flow1, 0.0));
}

// Tests that AllWithinMargin computes approximate equality.
TEST(OpticalFlowField, AllWithinMargin) {
  cv::Mat_<cv::Point2f> flow_data(3, 5);
  for (int y = 0; y < flow_data.rows; ++y) {
    for (int x = 0; x < flow_data.cols; ++x) {
      flow_data(y, x) = cv::Point2f((y - 1.5) / 2.0 + 2 * x, y + 3 - x);
    }
  }
  // Exactly equal flow fields are equal given a non-zero margin.
  OpticalFlowField flow1(flow_data);
  OpticalFlowField flow2(flow_data);
  EXPECT_TRUE(flow1.AllWithinMargin(flow2, 0.5));
  EXPECT_TRUE(flow2.AllWithinMargin(flow1, 0.5));

  // Flows with different dimensions are still not equal given a margin.
  OpticalFlowField wrong_size(flow_data.rowRange(0, 2));
  EXPECT_FALSE(wrong_size.AllWithinMargin(flow1, 0.5));
  EXPECT_FALSE(flow1.AllWithinMargin(wrong_size, 0.5));

  // Adding a smaller value than the margin to one of the pixels is ok.
  flow_data(0, 0) += cv::Point2f(0.1, 0);
  OpticalFlowField noisy_flow(flow_data);
  EXPECT_TRUE(flow1.AllWithinMargin(noisy_flow, 0.5));
  EXPECT_TRUE(noisy_flow.AllWithinMargin(flow1, 0.5));

  // Adding a larger offset than the margin breaks equality.
  noisy_flow.mutable_flow_data().at<cv::Point2f>(1, 1) += cv::Point2f(0, -0.6);
  EXPECT_FALSE(flow1.AllWithinMargin(noisy_flow, 0.5));
  EXPECT_FALSE(noisy_flow.AllWithinMargin(flow1, 0.5));
}

TEST(OpticalFlowField, Occlusions) {
  cv::Mat_<cv::Point2f> forward(3, 4);
  cv::Mat_<cv::Point2f> backward(3, 4);
  // Background moves down by one pixel.
  for (int y = 0; y < forward.rows; ++y) {
    for (int x = 0; x < forward.cols; ++x) {
      forward(y, x) = cv::Point2f(0.0, 1.0);
      backward(y, x) = cv::Point2f(0.0, -1.0);
    }
  }
  // Occluder at (x, y) = (1, 1) stays stationary.
  forward(1, 1) = cv::Point2f(0.0, 0.0);
  backward(1, 1) = cv::Point2f(0.0, 0.0);

  Location occlusion_mask;
  Location disocclusion_mask;
  OpticalFlowField::EstimateMotionConsistencyOcclusions(
      OpticalFlowField(forward), OpticalFlowField(backward), 0.5,
      &occlusion_mask, &disocclusion_mask);
  std::unique_ptr<cv::Mat> occlusion_mat = occlusion_mask.GetCvMask();
  std::unique_ptr<cv::Mat> disocclusion_mat = disocclusion_mask.GetCvMask();
  EXPECT_EQ(3, occlusion_mat->rows);
  EXPECT_EQ(3, disocclusion_mat->rows);
  EXPECT_EQ(4, occlusion_mat->cols);
  EXPECT_EQ(4, disocclusion_mat->cols);
  for (int x = 0; x < occlusion_mat->cols; ++x) {
    for (int y = 0; y < occlusion_mat->rows; ++y) {
      // Bottom row and pixel at (x, y) = (1, 0) are occluded.
      if (y == occlusion_mat->rows - 1 || (x == 1 && y == 0)) {
        EXPECT_GT(occlusion_mat->at<uint8>(y, x), 0);
      } else {
        EXPECT_EQ(0, occlusion_mat->at<uint8>(y, x));
      }
      // Top row and pixel at (x, y) = (1, 2) are disoccluded.
      if (y == 0 || (x == 1 && y == 2)) {
        EXPECT_GT(disocclusion_mat->at<uint8>(y, x), 0);
      } else {
        EXPECT_EQ(0, disocclusion_mat->at<uint8>(y, x));
      }
    }
  }
}

TEST(OpticalFlowField, ResizeRoundTrip) {
  // This constructor takes rows, cols, not width, height!
  cv::Mat_<cv::Point2f> original_flow(31, 15);
  for (int r = 0; r < original_flow.rows; ++r) {
    for (int c = 0; c < original_flow.cols; ++c) {
      original_flow(r, c) = cv::Point2f(r * c / 2.0, r * c / 3.0);
    }
  }
  std::unique_ptr<OpticalFlowField> flow(new OpticalFlowField(original_flow));
  ASSERT_EQ(flow->width(), 15);
  ASSERT_EQ(flow->height(), 31);
  flow->Resize(30, 62);
  EXPECT_EQ(flow->width(), 30);
  EXPECT_EQ(flow->height(), 62);
  flow->Resize(15, 31);
  EXPECT_EQ(original_flow.cols, flow->width());
  EXPECT_EQ(original_flow.rows, flow->height());

  // Check for round trip equality ignoring boundary conditions.
  for (int r = 1; r < original_flow.rows - 1; ++r) {
    for (int c = 1; c < original_flow.cols - 1; ++c) {
      EXPECT_FLOAT_EQ(flow->flow_data().at<cv::Point2f>(r, c).x,
                      original_flow.at<cv::Point2f>(r, c).x);
      EXPECT_FLOAT_EQ(flow->flow_data().at<cv::Point2f>(r, c).y,
                      original_flow.at<cv::Point2f>(r, c).y);
    }
  }
}

TEST(OpticalFlowField, ResizeOneWay) {
  // Initial flow field is (dx, dy) = (1.0, -1.0) in a 1 x 1 image.
  OpticalFlowField flow;
  flow.Allocate(1, 1);
  flow.mutable_flow_data().at<cv::Point2f>(0, 0) = cv::Point2f(1.0, -1.0);
  // Resize to a 3 x 2 image. Flow should now be (3.0, -2.0).
  flow.Resize(3, 2);
  EXPECT_EQ(flow.width(), 3);
  EXPECT_EQ(flow.height(), 2);
  for (int r = 0; r < flow.height(); ++r) {
    for (int c = 0; c < flow.width(); ++c) {
      EXPECT_FLOAT_EQ(flow.flow_data().at<cv::Point2f>(r, c).x, 3.0);
      EXPECT_FLOAT_EQ(flow.flow_data().at<cv::Point2f>(r, c).y, -2.0);
    }
  }
}

}  // namespace

}  // namespace mediapipe
