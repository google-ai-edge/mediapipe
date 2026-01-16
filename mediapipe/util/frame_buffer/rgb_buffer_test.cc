// Copyright 2023 The MediaPipe Authors.
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

#include "mediapipe/util/frame_buffer/rgb_buffer.h"

#include <cstdlib>
#include <utility>

#include "absl/log/absl_log.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/util/frame_buffer/float_buffer.h"
#include "mediapipe/util/frame_buffer/gray_buffer.h"
#include "mediapipe/util/frame_buffer/yuv_buffer.h"

// The default implementation of halide_error calls abort(), which we don't
// want. Instead, log the error and let the filter invocation fail.
extern "C" void halide_error(void*, const char* message) {
  ABSL_LOG(ERROR) << "Halide Error: " << message;
}

namespace mediapipe {
namespace frame_buffer {
namespace {

// Fill a halide_buffer_t channel with the given value.
void Fill(halide_buffer_t* buffer, int channel, int value) {
  for (int y = 0; y < buffer->dim[1].extent; ++y) {
    for (int x = 0; x < buffer->dim[0].extent; ++x) {
      buffer->host[buffer->dim[1].stride * y + buffer->dim[0].stride * x +
                   buffer->dim[2].stride * channel] = value;
    }
  }
}

// Fill an RgbBuffer with (0, 0, 0). Fills the alpha channel if present.
void Fill(RgbBuffer* buffer) {
  for (int c = 0; c < buffer->channels(); ++c) {
    Fill(buffer->buffer(), c, 0);
  }
}

// Returns a padded RGB buffer. The metadata are defined as width: 4, height: 2,
// row_stride: 18, channels: 3.
RgbBuffer GetPaddedRgbBuffer() {
  static uint8_t rgb_buffer_with_padding[] = {
      10, 20, 30, 20, 30, 40, 30, 40, 50,  40, 50,  60,  0, 0, 0, 0, 0, 0,
      20, 40, 60, 40, 60, 80, 60, 80, 100, 80, 100, 120, 0, 0, 0, 0, 0, 0};
  return RgbBuffer(rgb_buffer_with_padding,
                   /*width=*/4, /*height=*/2,
                   /*row_stride=*/18, /*alpha=*/false);
}

// Returns a padded RGB buffer. The metadata are defined as width: 4, height: 2,
// row_stride: 24, channels: 4.
RgbBuffer GetPaddedRgbaBuffer() {
  static uint8_t rgb_buffer_with_padding[] = {
      10, 20, 30,  255, 20, 30,  40,  255, 30, 40, 50, 255, 40, 50, 60, 255,
      0,  0,  0,   0,   0,  0,   0,   0,   20, 40, 60, 255, 40, 60, 80, 255,
      60, 80, 100, 255, 80, 100, 120, 255, 0,  0,  0,  0,   0,  0,  0,  0};
  return RgbBuffer(rgb_buffer_with_padding,
                   /*width=*/4, /*height=*/2,
                   /*row_stride=*/24, /*alpha=*/true);
}

// TODO: Consider move these helper methods into a util class.
// Returns true if the data in the two arrays are the same. Otherwise, return
// false.
bool CompareArray(const uint8_t* lhs_ptr, const uint8_t* rhs_ptr, int width,
                  int height) {
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      if (lhs_ptr[i * width + j] != rhs_ptr[i * width + j]) {
        return false;
      }
    }
  }
  return true;
}

// Returns true if the data in the two arrays are the same. Otherwise, return
// false.
bool CompareArray(const float* lhs_ptr, const float* rhs_ptr, int width,
                  int height) {
  constexpr float kTolerancy = 1e-6;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      if (std::abs(lhs_ptr[i * width + j] - rhs_ptr[i * width + j]) >
          kTolerancy) {
        return false;
      }
    }
  }
  return true;
}

// Returns true if the halide buffers of two input GrayBuffer are identical.
// Otherwise, returns false;
bool CompareBuffer(const GrayBuffer& lhs, const GrayBuffer& rhs) {
  if (lhs.width() != rhs.width() || lhs.height() != rhs.height()) {
    return false;
  }
  const uint8_t* reference_ptr = const_cast<GrayBuffer&>(lhs).buffer()->host;
  const uint8_t* converted_ptr = const_cast<GrayBuffer&>(rhs).buffer()->host;
  return CompareArray(reference_ptr, converted_ptr, lhs.width(), lhs.height());
}

// Returns true if the halide buffers of two input RgbBuffer are identical.
// Otherwise, returns false;
bool CompareBuffer(const RgbBuffer& lhs, const RgbBuffer& rhs) {
  if (lhs.width() != rhs.width() || lhs.height() != rhs.height() ||
      lhs.row_stride() != rhs.row_stride() ||
      lhs.channels() != rhs.channels()) {
    return false;
  }
  const uint8_t* reference_ptr = const_cast<RgbBuffer&>(lhs).buffer()->host;
  const uint8_t* converted_ptr = const_cast<RgbBuffer&>(rhs).buffer()->host;
  return CompareArray(reference_ptr, converted_ptr, lhs.row_stride(),
                      lhs.height());
}

// Returns true if the halide buffers of two input YuvBuffer are identical.
// Otherwise, returns false;
bool CompareBuffer(const YuvBuffer& lhs, const YuvBuffer& rhs) {
  if (lhs.width() != rhs.width() || lhs.height() != rhs.height()) {
    return false;
  }
  const uint8_t* reference_ptr = const_cast<YuvBuffer&>(lhs).y_buffer()->host;
  const uint8_t* converted_ptr = const_cast<YuvBuffer&>(rhs).y_buffer()->host;
  if (!CompareArray(reference_ptr, converted_ptr, lhs.width(), lhs.height())) {
    return false;
  }
  reference_ptr = const_cast<YuvBuffer&>(lhs).uv_buffer()->host;
  converted_ptr = const_cast<YuvBuffer&>(rhs).uv_buffer()->host;
  return CompareArray(reference_ptr, converted_ptr, lhs.width(),
                      lhs.height() / 2);
}

// Returns true if the halide buffers of two input FloatBuffer are identical.
// Otherwise, returns false;
bool CompareBuffer(const FloatBuffer& lhs, const FloatBuffer& rhs) {
  if (lhs.width() != rhs.width() || lhs.height() != rhs.height() ||
      lhs.channels() != rhs.channels()) {
    return false;
  }
  const float* reference_ptr = reinterpret_cast<const float*>(
      const_cast<FloatBuffer&>(lhs).buffer()->host);
  const float* converted_ptr = reinterpret_cast<const float*>(
      const_cast<FloatBuffer&>(rhs).buffer()->host);
  return CompareArray(reference_ptr, converted_ptr, lhs.width(), lhs.height());
}

TEST(RgbBufferTest, Properties) {
  RgbBuffer rgb(2, 8, false), rgba(2, 8, true);
  EXPECT_EQ(2, rgb.width());
  EXPECT_EQ(8, rgb.height());
  EXPECT_EQ(3, rgb.channels());

  EXPECT_EQ(2, rgba.width());
  EXPECT_EQ(8, rgba.height());
  EXPECT_EQ(4, rgba.channels());
}

TEST(RgbBufferTest, PropertiesOfPaddedRgb) {
  RgbBuffer rgb_buffer = GetPaddedRgbBuffer();
  EXPECT_EQ(rgb_buffer.width(), 4);
  EXPECT_EQ(rgb_buffer.height(), 2);
  EXPECT_EQ(rgb_buffer.row_stride(), 18);
  EXPECT_EQ(rgb_buffer.channels(), 3);
}

TEST(RgbBufferTest, PropertiesOfPaddedRgba) {
  RgbBuffer rgb_buffer = GetPaddedRgbaBuffer();
  EXPECT_EQ(rgb_buffer.width(), 4);
  EXPECT_EQ(rgb_buffer.height(), 2);
  EXPECT_EQ(rgb_buffer.row_stride(), 24);
  EXPECT_EQ(rgb_buffer.channels(), 4);
}

TEST(RgbBufferTest, Release) {
  RgbBuffer source(8, 8, true);
  delete[] source.Release();
}

TEST(RgbBufferTest, Assign) {
  RgbBuffer source(8, 8, false);
  RgbBuffer sink(nullptr, 0, 0, false);
  sink = source;
  EXPECT_EQ(8, sink.width());
  EXPECT_EQ(8, sink.height());
  EXPECT_EQ(3, sink.channels());

  sink = RgbBuffer(16, 16, true);
  EXPECT_EQ(16, sink.width());
  EXPECT_EQ(16, sink.height());
  EXPECT_EQ(4, sink.channels());
}

TEST(RgbBufferTest, MoveAssign) {
  RgbBuffer source(8, 8, false);
  RgbBuffer sink(nullptr, 0, 0, true);
  sink = std::move(source);
  EXPECT_EQ(nullptr, source.Release());
  EXPECT_EQ(8, sink.width());
  EXPECT_EQ(8, sink.height());
}

TEST(RgbBufferTest, MoveConstructor) {
  RgbBuffer source(8, 8, false);
  RgbBuffer sink(std::move(source));
  EXPECT_EQ(nullptr, source.Release());
  EXPECT_EQ(8, sink.width());
  EXPECT_EQ(8, sink.height());
}

TEST(RgbBufferTest, RgbCrop) {
  RgbBuffer source(8, 8, false);
  EXPECT_TRUE(source.Crop(2, 2, 6, 6));
}

TEST(RgbBufferTest, RgbaCrop) {
  RgbBuffer source(8, 8, true);
  EXPECT_TRUE(source.Crop(2, 2, 6, 6));
}

// Some operations expect images with a platform-dependent minimum width
// because their implementations are vectorized.

TEST(RgbBufferTest, RgbResize) {
  RgbBuffer source(128, 8, false);
  RgbBuffer result(32, 4, false);
  Fill(&source);
  EXPECT_TRUE(source.Resize(&result));

  // Test odd result sizes too.
  source = RgbBuffer(64, 16, false);
  result = RgbBuffer(32, 7, false);
  Fill(&source);
  EXPECT_TRUE(source.Resize(&result));
}

TEST(RgbBufferTest, RgbaResize) {
  RgbBuffer source(128, 8, true);
  RgbBuffer result(32, 4, true);
  Fill(&source);
  EXPECT_TRUE(source.Resize(&result));

  // Test odd result sizes too.
  source = RgbBuffer(64, 16, true);
  result = RgbBuffer(32, 7, true);
  Fill(&source);
  EXPECT_TRUE(source.Resize(&result));
}

// Note: RGB-to-RGBA conversion currently doesn't work.
TEST(RgbBufferTest, RgbResizeDifferentFormat) {
  RgbBuffer source(128, 8, false);
  RgbBuffer result(16, 4, true);
  Fill(&source);
  EXPECT_FALSE(source.Resize(&result));
}

TEST(RgbBufferTest, RgbaResizeDifferentFormat) {
  RgbBuffer source(128, 8, true);
  RgbBuffer result(16, 4, false);
  Fill(&source);
  EXPECT_TRUE(source.Resize(&result));
}

TEST(RgbBufferTest, PaddedRgbResize) {
  const int target_width = 2;
  const int target_height = 1;
  RgbBuffer source = GetPaddedRgbBuffer();
  RgbBuffer result(target_width, target_height, /*alpha=*/false);

  ASSERT_TRUE(source.Resize(&result));
  EXPECT_EQ(result.width(), target_width);
  EXPECT_EQ(result.height(), target_height);
  EXPECT_EQ(result.channels(), 3);
  EXPECT_EQ(result.row_stride(), target_width * /*pixel_stride=*/3);

  uint8_t rgb_data[] = {10, 20, 30, 30, 40, 50};
  RgbBuffer rgb_buffer =
      RgbBuffer(rgb_data, target_width, target_height, /*alpha=*/false);
  EXPECT_TRUE(CompareBuffer(rgb_buffer, result));
}

TEST(RgbBufferTest, PaddedRgbaResize) {
  const int target_width = 2;
  const int target_height = 1;
  RgbBuffer source = GetPaddedRgbaBuffer();
  RgbBuffer result(target_width, target_height, /*alpha=*/true);

  ASSERT_TRUE(source.Resize(&result));
  EXPECT_EQ(result.width(), target_width);
  EXPECT_EQ(result.height(), target_height);
  EXPECT_EQ(result.channels(), 4);
  EXPECT_EQ(result.row_stride(), target_width * /*pixel_stride=*/4);

  uint8_t rgb_data[] = {10, 20, 30, 255, 30, 40, 50, 255};
  RgbBuffer rgb_buffer =
      RgbBuffer(rgb_data, target_width, target_height, /*alpha=*/true);
  EXPECT_TRUE(CompareBuffer(rgb_buffer, result));
}

TEST(RgbBufferTest, RgbRotateCheckSize) {
  RgbBuffer source(4, 8, false);
  RgbBuffer result(8, 4, false);
  Fill(&source);
  EXPECT_TRUE(source.Rotate(90, &result));
}

TEST(RgbBufferTest, RgbRotateCheckData) {
  uint8_t* data = new uint8_t[12];
  data[0] = data[1] = data[2] = 1;    // Pixel 1
  data[3] = data[4] = data[5] = 2;    // Pixel 2
  data[6] = data[7] = data[8] = 3;    // Pixel 3
  data[9] = data[10] = data[11] = 4;  // Pixel 4
  RgbBuffer source(data, 2, 2, false);
  RgbBuffer result(2, 2, false);
  source.Rotate(90, &result);
  EXPECT_EQ(2, result.buffer()->host[0]);
  EXPECT_EQ(4, result.buffer()->host[3]);
  EXPECT_EQ(1, result.buffer()->host[6]);
  EXPECT_EQ(3, result.buffer()->host[9]);
  delete[] data;
}

TEST(RgbBufferTest, RgbRotateDifferentFormat) {
  RgbBuffer source(4, 8, true);
  RgbBuffer result(8, 4, false);
  Fill(&source);
  EXPECT_TRUE(source.Rotate(90, &result));
}

// Note: RGB-to-RGBA conversion currently doesn't work.
TEST(RgbBufferTest, RgbRotateDifferentFormatFail) {
  RgbBuffer source(4, 8, false);
  RgbBuffer result(8, 4, true);
  Fill(&source);
  EXPECT_FALSE(source.Rotate(90, &result));
}

TEST(RgbBufferTest, RgbaRotate) {
  RgbBuffer source(4, 8, true);
  RgbBuffer result(8, 4, true);
  Fill(&source);
  EXPECT_TRUE(source.Rotate(90, &result));
}

TEST(RgbBufferTest, RgbaRotateDifferentFormat) {
  RgbBuffer source(4, 8, true);
  RgbBuffer result(8, 4, false);
  Fill(&source);
  EXPECT_TRUE(source.Rotate(90, &result));
}

// Note: RGB-to-RGBA conversion currently doesn't work.
TEST(RgbBufferTest, RgbaRotateDifferentFormatFail) {
  RgbBuffer source(4, 8, false);
  RgbBuffer result(8, 4, true);
  Fill(&source);
  EXPECT_FALSE(source.Rotate(90, &result));
}

TEST(RgbBufferTest, PaddedRgbRotateCheckData) {
  const int target_width = 2;
  const int target_height = 4;
  RgbBuffer source = GetPaddedRgbBuffer();
  RgbBuffer result(target_width, target_height, /*alpha=*/false);

  ASSERT_TRUE(source.Rotate(/*angle=*/90, &result));
  EXPECT_EQ(result.width(), target_width);
  EXPECT_EQ(result.height(), target_height);
  EXPECT_EQ(result.channels(), 3);
  EXPECT_EQ(result.row_stride(), target_width * /*pixel_stride=*/3);

  uint8_t rgb_data[] = {40, 50, 60, 80, 100, 120, 30, 40, 50, 60, 80, 100,
                        20, 30, 40, 40, 60,  80,  10, 20, 30, 20, 40, 60};
  RgbBuffer rgb_buffer =
      RgbBuffer(rgb_data, target_width, target_height, /*alpha=*/false);
  EXPECT_TRUE(CompareBuffer(rgb_buffer, result));
}

TEST(RgbBufferTest, PaddedRgbaRotateCheckData) {
  const int target_width = 2;
  const int target_height = 4;
  RgbBuffer result(target_width, target_height, /*alpha=*/true);
  RgbBuffer source = GetPaddedRgbaBuffer();

  ASSERT_TRUE(source.Rotate(/*angle=*/90, &result));
  EXPECT_EQ(result.width(), target_width);
  EXPECT_EQ(result.height(), target_height);
  EXPECT_EQ(result.channels(), 4);
  EXPECT_EQ(result.row_stride(), target_width * /*pixel_stride=*/4);

  uint8_t rgb_data[] = {40,  50,  60, 255, 80,  100, 120, 255, 30,  40, 50,
                        255, 60,  80, 100, 255, 20,  30,  40,  255, 40, 60,
                        80,  255, 10, 20,  30,  255, 20,  40,  60,  255};
  RgbBuffer rgb_buffer =
      RgbBuffer(rgb_data, target_width, target_height, /*alpha=*/true);
  EXPECT_TRUE(CompareBuffer(rgb_buffer, result));
}

TEST(RgbBufferTest, RgbaFlip) {
  RgbBuffer source(16, 16, true);
  RgbBuffer result(16, 16, true);
  Fill(&source);
  EXPECT_TRUE(source.FlipHorizontally(&result));
  EXPECT_TRUE(source.FlipVertically(&result));
}

// Note: Neither RGBA-to-RGB nor RGB-to-RGBA conversion currently works.
TEST(RgbBufferTest, RgbaFlipDifferentFormatFail) {
  RgbBuffer source(16, 16, false);
  RgbBuffer result(16, 16, true);
  Fill(&source);
  Fill(&result);
  EXPECT_FALSE(source.FlipHorizontally(&result));
  EXPECT_FALSE(result.FlipHorizontally(&source));
  EXPECT_FALSE(source.FlipVertically(&result));
  EXPECT_FALSE(result.FlipVertically(&source));
}

TEST(RgbBufferTest, PaddedRgbFlipHorizontally) {
  const int target_width = 4;
  const int target_height = 2;
  RgbBuffer result(target_width, target_height, /*alpha=*/false);
  RgbBuffer source = GetPaddedRgbBuffer();

  ASSERT_TRUE(source.FlipHorizontally(&result));
  EXPECT_EQ(result.width(), target_width);
  EXPECT_EQ(result.height(), target_height);
  EXPECT_EQ(result.channels(), 3);
  EXPECT_EQ(result.row_stride(), target_width * /*pixel_stride=*/3);

  uint8_t rgb_data[] = {40, 50,  60,  30, 40, 50,  20, 30, 40, 10, 20, 30,
                        80, 100, 120, 60, 80, 100, 40, 60, 80, 20, 40, 60};
  RgbBuffer rgb_buffer =
      RgbBuffer(rgb_data, target_width, target_height, /*alpha=*/false);
  EXPECT_TRUE(CompareBuffer(rgb_buffer, result));
}

TEST(RgbBufferTest, PaddedRgbaFlipHorizontally) {
  const int target_width = 4;
  const int target_height = 2;
  RgbBuffer result(target_width, target_height, /*alpha=*/true);
  RgbBuffer source = GetPaddedRgbaBuffer();

  ASSERT_TRUE(source.FlipHorizontally(&result));
  EXPECT_EQ(result.width(), target_width);
  EXPECT_EQ(result.height(), target_height);
  EXPECT_EQ(result.channels(), 4);
  EXPECT_EQ(result.row_stride(), target_width * /*pixel_stride=*/4);

  uint8_t rgb_data[] = {40,  50,  60, 255, 30,  40,  50,  255, 20,  30, 40,
                        255, 10,  20, 30,  255, 80,  100, 120, 255, 60, 80,
                        100, 255, 40, 60,  80,  255, 20,  40,  60,  255};
  RgbBuffer rgb_buffer =
      RgbBuffer(rgb_data, target_width, target_height, /*alpha=*/true);
  EXPECT_TRUE(CompareBuffer(rgb_buffer, result));
}

TEST(RgbBufferTest, RgbConvertNv21) {
  RgbBuffer source(32, 8, false);
  YuvBuffer result(32, 8, YuvBuffer::NV21);
  Fill(&source);
  EXPECT_TRUE(source.Convert(&result));
}

TEST(RgbBufferTest, RgbaConvertNv21) {
  RgbBuffer source(32, 8, true);
  YuvBuffer result(32, 8, YuvBuffer::NV21);
  Fill(&source);
  EXPECT_TRUE(source.Convert(&result));
}

TEST(RgbBufferTest, PaddedRgbConvertNv21) {
  const int target_width = 4;
  const int target_height = 2;
  YuvBuffer result(target_width, target_height, YuvBuffer::NV21);
  RgbBuffer source = GetPaddedRgbBuffer();

  ASSERT_TRUE(source.Convert(&result));
  EXPECT_EQ(result.width(), target_width);
  EXPECT_EQ(result.height(), target_height);

  uint8_t yuv_data[] = {18, 28, 38, 48, 36, 56, 76, 96, 122, 135, 122, 135};
  YuvBuffer yuv_buffer =
      YuvBuffer(yuv_data, target_width, target_height, YuvBuffer::NV21);
  EXPECT_TRUE(CompareBuffer(yuv_buffer, result));
}

TEST(RgbBufferTest, PaddedRgbaConvertNv21) {
  const int target_width = 4;
  const int target_height = 2;
  YuvBuffer result(target_width, target_height, YuvBuffer::NV21);
  RgbBuffer source = GetPaddedRgbaBuffer();

  ASSERT_TRUE(source.Convert(&result));
  EXPECT_EQ(result.width(), target_width);
  EXPECT_EQ(result.height(), target_height);

  uint8_t yuv_data[] = {18, 28, 38, 48, 36, 56, 76, 96, 122, 135, 122, 135};
  YuvBuffer yuv_buffer =
      YuvBuffer(yuv_data, target_width, target_height, YuvBuffer::NV21);
  EXPECT_TRUE(CompareBuffer(yuv_buffer, result));
}

TEST(RgbBufferTest, RgbConvertGray) {
  uint8_t* data = new uint8_t[6];
  data[0] = 200;
  data[1] = 100;
  data[2] = 0;
  data[3] = 0;
  data[4] = 200;
  data[5] = 100;
  RgbBuffer source(data, 2, 1, false);
  GrayBuffer result(2, 1);
  EXPECT_TRUE(source.Convert(&result));
  EXPECT_EQ(118, result.buffer()->host[0]);
  EXPECT_EQ(129, result.buffer()->host[1]);
  delete[] data;
}

TEST(RgbBufferTest, RgbaConvertGray) {
  uint8_t* data = new uint8_t[8];
  data[0] = 200;
  data[1] = 100;
  data[2] = 0;
  data[3] = 1;
  data[4] = 0;
  data[5] = 200;
  data[6] = 100;
  data[7] = 50;
  RgbBuffer source(data, 2, 1, true);
  GrayBuffer result(2, 1);
  EXPECT_TRUE(source.Convert(&result));
  EXPECT_EQ(118, result.buffer()->host[0]);
  EXPECT_EQ(129, result.buffer()->host[1]);
  delete[] data;
}

TEST(RgbBufferTest, PaddedRgbConvertGray) {
  const int target_width = 4;
  const int target_height = 2;
  GrayBuffer result(target_width, target_height);
  RgbBuffer source = GetPaddedRgbBuffer();

  ASSERT_TRUE(source.Convert(&result));
  EXPECT_EQ(result.width(), target_width);
  EXPECT_EQ(result.height(), target_height);

  uint8_t gray_data[] = {18, 28, 38, 48, 36, 56, 76, 96};
  GrayBuffer gray_buffer = GrayBuffer(gray_data, target_width, target_height);
  EXPECT_TRUE(CompareBuffer(gray_buffer, result));
}

TEST(RgbBufferTest, PaddedRgbaConvertGray) {
  const int target_width = 4;
  const int target_height = 2;
  GrayBuffer result(target_width, target_height);
  RgbBuffer source = GetPaddedRgbaBuffer();

  ASSERT_TRUE(source.Convert(&result));
  EXPECT_EQ(result.width(), target_width);
  EXPECT_EQ(result.height(), target_height);

  uint8_t gray_data[] = {18, 28, 38, 48, 36, 56, 76, 96};
  GrayBuffer gray_buffer = GrayBuffer(gray_data, target_width, target_height);
  EXPECT_TRUE(CompareBuffer(gray_buffer, result));
}

TEST(RgbBufferTest, RgbConvertRgba) {
  constexpr int kWidth = 2, kHeight = 1;
  uint8_t rgb_data[] = {200, 100, 50, 100, 50, 20};
  RgbBuffer source(rgb_data, kWidth, kHeight, false);
  RgbBuffer result(kWidth, kHeight, true);

  ASSERT_TRUE(source.Convert(&result));

  uint8_t rgba_data[] = {200, 100, 50, 255, 100, 50, 20, 255};
  RgbBuffer rgba_buffer = RgbBuffer(rgba_data, kWidth, kHeight, true);
  EXPECT_TRUE(CompareBuffer(rgba_buffer, result));
}

TEST(RgbBufferTest, PaddedRgbConvertRgba) {
  constexpr int kWidth = 4, kHeight = 2;
  RgbBuffer source = GetPaddedRgbBuffer();
  RgbBuffer result(kWidth, kHeight, true);
  ASSERT_TRUE(source.Convert(&result));

  uint8_t rgba_data[]{10,  20,  30, 255, 20,  30,  40, 255, 30,  40, 50,
                      255, 40,  50, 60,  255, 20,  40, 60,  255, 40, 60,
                      80,  255, 60, 80,  100, 255, 80, 100, 120, 255};
  RgbBuffer rgba_buffer = RgbBuffer(rgba_data, kWidth, kHeight, true);
  EXPECT_TRUE(CompareBuffer(rgba_buffer, result));
}

TEST(RgbBufferTest, RgbaConvertRgb) {
  constexpr int kWidth = 2, kHeight = 1;
  uint8_t rgba_data[] = {200, 100, 50, 30, 100, 50, 20, 70};
  RgbBuffer source(rgba_data, kWidth, kHeight, true);
  RgbBuffer result(kWidth, kHeight, false);

  ASSERT_TRUE(source.Convert(&result));

  uint8_t rgb_data[] = {200, 100, 50, 100, 50, 20};
  RgbBuffer rgb_buffer = RgbBuffer(rgb_data, kWidth, kHeight, false);
  EXPECT_TRUE(CompareBuffer(rgb_buffer, result));
}

TEST(RgbBufferTest, PaddedRgbaConvertRgb) {
  constexpr int kWidth = 4, kHeight = 2;
  RgbBuffer source = GetPaddedRgbaBuffer();
  RgbBuffer result(kWidth, kHeight, false);

  ASSERT_TRUE(source.Convert(&result));

  uint8_t rgb_data[] = {10, 20, 30, 20, 30, 40, 30, 40, 50,  40, 50,  60,
                        20, 40, 60, 40, 60, 80, 60, 80, 100, 80, 100, 120};
  RgbBuffer rgb_buffer = RgbBuffer(rgb_data, kWidth, kHeight, false);
  EXPECT_TRUE(CompareBuffer(rgb_buffer, result));
}

TEST(RgbBufferTest, RgbToFloat) {
  constexpr int kWidth = 2, kHeight = 1, kChannels = 3;
  constexpr float kScale = 0.01f, kOffset = 0.5f;
  uint8_t rgb_data[] = {200, 100, 50, 100, 50, 20};
  RgbBuffer source(rgb_data, kWidth, kHeight, false);
  FloatBuffer result(kWidth, kHeight, kChannels);

  ASSERT_TRUE(source.ToFloat(kScale, kOffset, &result));

  float float_data[] = {2.5f, 1.5f, 1.0f, 1.5f, 1.0f, 0.7f};
  FloatBuffer float_buffer =
      FloatBuffer(float_data, kWidth, kHeight, kChannels);
  EXPECT_TRUE(CompareBuffer(float_buffer, result));
}

TEST(RgbBufferTest, PaddedRgbToFloat) {
  constexpr int kWidth = 4, kHeight = 2, kChannels = 3;
  constexpr float kScale = 0.01f, kOffset = 0.0f;
  RgbBuffer source = GetPaddedRgbBuffer();
  FloatBuffer result(kWidth, kHeight, kChannels);

  ASSERT_TRUE(source.ToFloat(kScale, kOffset, &result));

  float float_data[] = {0.1f, 0.2f, 0.3f, 0.2f, 0.3f, 0.4f, 0.3f, 0.4f,
                        0.5f, 0.4f, 0.5f, 0.6f, 0.2f, 0.4f, 0.6f, 0.4f,
                        0.6f, 0.8f, 0.6f, 0.8f, 1.0f, 0.8f, 1.0f, 1.2f};
  FloatBuffer float_buffer =
      FloatBuffer(float_data, kWidth, kHeight, kChannels);
  EXPECT_TRUE(CompareBuffer(float_buffer, result));
}

TEST(RgbBufferTest, RgbaToFloat) {
  constexpr int kWidth = 2, kHeight = 1, kChannels = 4;
  constexpr float kScale = 0.01f, kOffset = 0.5f;
  uint8_t rgba_data[] = {200, 100, 50, 30, 100, 50, 20, 70};
  RgbBuffer source(rgba_data, kWidth, kHeight, true);
  FloatBuffer result(kWidth, kHeight, kChannels);

  ASSERT_TRUE(source.ToFloat(kScale, kOffset, &result));

  float float_data[] = {2.5f, 1.5f, 1.0f, 0.8f, 1.5f, 1.0f, 0.7f, 1.2f};
  FloatBuffer float_buffer =
      FloatBuffer(float_data, kWidth, kHeight, kChannels);
  EXPECT_TRUE(CompareBuffer(float_buffer, result));
}

TEST(RgbBufferTest, PaddedRgbaToFloat) {
  constexpr int kWidth = 4, kHeight = 2, kChannels = 4;
  constexpr float kScale = 0.01f, kOffset = 0.0f;
  RgbBuffer source = GetPaddedRgbaBuffer();
  FloatBuffer result(kWidth, kHeight, kChannels);

  ASSERT_TRUE(source.ToFloat(kScale, kOffset, &result));

  float float_data[] = {0.1f, 0.2f, 0.3f, 2.55f, 0.2f, 0.3f, 0.4f, 2.55f,
                        0.3f, 0.4f, 0.5f, 2.55f, 0.4f, 0.5f, 0.6f, 2.55f,
                        0.2f, 0.4f, 0.6f, 2.55f, 0.4f, 0.6f, 0.8f, 2.55f,
                        0.6f, 0.8f, 1.0f, 2.55f, 0.8f, 1.0f, 1.2f, 2.55f};
  FloatBuffer float_buffer =
      FloatBuffer(float_data, kWidth, kHeight, kChannels);
  EXPECT_TRUE(CompareBuffer(float_buffer, result));
}

}  // namespace
}  // namespace frame_buffer
}  // namespace mediapipe
