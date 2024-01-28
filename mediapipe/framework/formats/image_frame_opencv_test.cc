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

#include "mediapipe/framework/formats/image_frame_opencv.h"

#include <cstdint>

#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/logging.h"

namespace mediapipe {
namespace {

// Set image_frame to a constant per-channel pix_value.
template <class T>
void SetToColor(const T* pix_value, ImageFrame* image_frame) {
  const int cols = image_frame->Width();
  const int rows = image_frame->Height();
  const int channels = image_frame->NumberOfChannels();
  const int width_padding =
      image_frame->WidthStep() / (sizeof(T)) - cols * channels;
  T* pixel = reinterpret_cast<T*>(image_frame->MutablePixelData());
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      for (int channel = 0; channel < channels; ++channel) {
        pixel[channel] = pix_value[channel];
      }
      pixel += channels;
    }
    pixel += width_padding;
  }
}

TEST(ImageFrameOpencvTest, ConvertToMat) {
  const int i_width = 123, i_height = 45;
  ImageFrame frame1(ImageFormat::GRAY8, i_width, i_height);
  ImageFrame frame2(ImageFormat::GRAY8, i_width, i_height);

  // Check adding constant images.
  const uint8_t frame1_val = 12;
  const uint8_t frame2_val = 34;
  SetToColor<uint8_t>(&frame1_val, &frame1);
  SetToColor<uint8_t>(&frame2_val, &frame2);
  // Get Mat wrapper around ImageFrame memory (zero copy).
  cv::Mat frame1_mat = formats::MatView(&frame1);
  cv::Mat frame2_mat = formats::MatView(&frame2);
  // Use OpenCV functions directly on ImageFrame data.
  cv::Mat frame_sum = frame1_mat + frame2_mat;
  const auto frame_avg = static_cast<int>(cv::mean(frame_sum)[0]);
  EXPECT_EQ(frame_avg, frame1_val + frame2_val);

  // Check setting min/max pixels.
  uint8_t* frame1_ptr = frame1.MutablePixelData();
  frame1_ptr[(i_width - 5) + (i_height - 5) * frame1.WidthStep()] = 1;
  frame1_ptr[(i_width - 6) + (i_height - 6) * frame1.WidthStep()] = 100;
  double min, max;
  cv::Point min_loc, max_loc;
  cv::minMaxLoc(frame1_mat, &min, &max, &min_loc, &max_loc);
  EXPECT_EQ(min, 1);
  EXPECT_EQ(min_loc.x, i_width - 5);
  EXPECT_EQ(min_loc.y, i_height - 5);
  EXPECT_EQ(max, 100);
  EXPECT_EQ(max_loc.x, i_width - 6);
  EXPECT_EQ(max_loc.y, i_height - 6);
}

TEST(ImageFrameOpencvTest, ConvertToIpl) {
  const int i_width = 123, i_height = 45;
  ImageFrame frame1(ImageFormat::GRAY8, i_width, i_height);
  ImageFrame frame2(ImageFormat::GRAY8, i_width, i_height);

  // Check adding constant images.
  const uint8_t frame1_val = 12;
  const uint8_t frame2_val = 34;
  SetToColor<uint8_t>(&frame1_val, &frame1);
  SetToColor<uint8_t>(&frame2_val, &frame2);
  const cv::Mat frame1_mat = formats::MatView(&frame1);
  const cv::Mat frame2_mat = formats::MatView(&frame2);
  const cv::Mat frame_sum = frame1_mat + frame2_mat;
  const auto frame_avg = static_cast<int>(cv::mean(frame_sum).val[0]);
  EXPECT_EQ(frame_avg, frame1_val + frame2_val);

  // Check setting min/max pixels.
  uint8_t* frame1_ptr = frame1.MutablePixelData();
  frame1_ptr[(i_width - 5) + (i_height - 5) * frame1.WidthStep()] = 1;
  frame1_ptr[(i_width - 6) + (i_height - 6) * frame1.WidthStep()] = 100;
  double min, max;
  cv::Point min_loc, max_loc;
  cv::minMaxLoc(frame1_mat, &min, &max, &min_loc, &max_loc);
  EXPECT_EQ(min, 1);
  EXPECT_EQ(min_loc.x, i_width - 5);
  EXPECT_EQ(min_loc.y, i_height - 5);
  EXPECT_EQ(max, 100);
  EXPECT_EQ(max_loc.x, i_width - 6);
  EXPECT_EQ(max_loc.y, i_height - 6);
}

TEST(ImageFrameOpencvTest, ImageFormats) {
  const int i_width = 123, i_height = 45;
  ImageFrame frame_g8(ImageFormat::GRAY8, i_width, i_height);
  ImageFrame frame_g16(ImageFormat::GRAY16, i_width, i_height);
  ImageFrame frame_v32f1(ImageFormat::VEC32F1, i_width, i_height);
  ImageFrame frame_v32f2(ImageFormat::VEC32F2, i_width, i_height);
  ImageFrame frame_v32f4(ImageFormat::VEC32F4, i_width, i_height);
  ImageFrame frame_c3(ImageFormat::SRGB, i_width, i_height);
  ImageFrame frame_c4(ImageFormat::SRGBA, i_width, i_height);

  cv::Mat mat_g8 = formats::MatView(&frame_g8);
  cv::Mat mat_g16 = formats::MatView(&frame_g16);
  cv::Mat mat_v32f1 = formats::MatView(&frame_v32f1);
  cv::Mat mat_v32f2 = formats::MatView(&frame_v32f2);
  cv::Mat mat_v32f4 = formats::MatView(&frame_v32f4);
  cv::Mat mat_c3 = formats::MatView(&frame_c3);
  cv::Mat mat_c4 = formats::MatView(&frame_c4);

  EXPECT_EQ(mat_g8.type(), CV_8UC1);
  EXPECT_EQ(mat_g16.type(), CV_16UC1);
  EXPECT_EQ(mat_v32f1.type(), CV_32FC1);
  EXPECT_EQ(mat_v32f2.type(), CV_32FC2);
  EXPECT_EQ(mat_v32f4.type(), CV_32FC4);
  EXPECT_EQ(mat_c3.type(), CV_8UC3);
  EXPECT_EQ(mat_c4.type(), CV_8UC4);
}

}  // namespace
}  // namespace mediapipe
