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

#include "mediapipe/util/frame_buffer/yuv_buffer.h"

#include <utility>

#include "absl/log/absl_log.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/util/frame_buffer/rgb_buffer.h"

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

// Fill a YuvBuffer with the given YUV color.
void Fill(YuvBuffer* buffer, uint8_t y, uint8_t u, uint8_t v) {
  Fill(buffer->y_buffer(), 0, y);
  Fill(buffer->uv_buffer(), 1, u);
  Fill(buffer->uv_buffer(), 0, v);
}

TEST(YuvBufferTest, Properties) {
  YuvBuffer yuv(2, 8, YuvBuffer::NV21);
  EXPECT_EQ(2, yuv.width());
  EXPECT_EQ(8, yuv.height());
}

TEST(YuvBufferTest, Release) {
  YuvBuffer source(8, 8, YuvBuffer::NV21);
  delete[] source.Release();
}

TEST(YuvBufferTest, Assign) {
  YuvBuffer source(8, 8, YuvBuffer::NV21);
  YuvBuffer sink(nullptr, 0, 0, YuvBuffer::NV21);
  sink = source;
  EXPECT_EQ(8, sink.width());
  EXPECT_EQ(8, sink.height());

  sink = YuvBuffer(16, 16, YuvBuffer::NV21);
  EXPECT_EQ(16, sink.width());
  EXPECT_EQ(16, sink.height());
}

TEST(YuvBufferTest, MoveAssign) {
  YuvBuffer source(8, 8, YuvBuffer::NV21);
  YuvBuffer sink(nullptr, 0, 0, YuvBuffer::NV21);
  sink = std::move(source);
  EXPECT_EQ(nullptr, source.Release());
  EXPECT_EQ(8, sink.width());
  EXPECT_EQ(8, sink.height());
}

TEST(YuvBufferTest, MoveConstructor) {
  YuvBuffer source(8, 8, YuvBuffer::NV21);
  YuvBuffer sink(std::move(source));
  EXPECT_EQ(nullptr, source.Release());
  EXPECT_EQ(8, sink.width());
  EXPECT_EQ(8, sink.height());
}

TEST(YuvBufferTest, GenericSemiplanarLayout) {
  uint8_t y_plane[16], uv_plane[8];
  YuvBuffer buffer(y_plane, uv_plane, uv_plane + 1, 4, 4, 4, 4, 2);
  Fill(&buffer, 16, 32, 64);

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(y_plane[i], 16) << i;
  }
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(uv_plane[2 * i], 32);
    EXPECT_EQ(uv_plane[2 * i + 1], 64);
  }
}

TEST(YuvBufferTest, GenericPlanarLayout) {
  uint8_t y_plane[16], u_plane[4], v_plane[4];
  YuvBuffer buffer(y_plane, u_plane, v_plane, 4, 4, 4, 2, 1);
  Fill(&buffer, 16, 32, 64);

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(y_plane[i], 16) << i;
  }
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(u_plane[i], 32);
    EXPECT_EQ(v_plane[i], 64);
  }
}

TEST(YuvBufferTest, Nv21Crop) {
  YuvBuffer source(8, 8, YuvBuffer::NV21);
  EXPECT_TRUE(source.Crop(2, 2, 6, 6));
}

TEST(YuvBufferTest, Nv21Resize) {
  YuvBuffer source(8, 8, YuvBuffer::NV21);
  YuvBuffer result(4, 4, YuvBuffer::NV21);
  Fill(&source, 16, 32, 64);
  EXPECT_TRUE(source.Resize(&result));

  // Test odd result sizes too.
  source = YuvBuffer(500, 362, YuvBuffer::NV21);
  result = YuvBuffer(320, 231, YuvBuffer::NV21);
  Fill(&source, 16, 32, 64);
  EXPECT_TRUE(source.Resize(&result));
}

TEST(YuvBufferTest, Nv21ResizeDifferentFormat) {
  YuvBuffer source(8, 8, YuvBuffer::NV21);
  YuvBuffer result(4, 4, YuvBuffer::YV12);
  Fill(&source, 16, 32, 64);
  EXPECT_TRUE(source.Resize(&result));
}

TEST(YuvBufferTest, Nv21Rotate) {
  YuvBuffer source(4, 8, YuvBuffer::NV21);
  YuvBuffer result(8, 4, YuvBuffer::NV21);
  Fill(&source, 16, 32, 64);
  EXPECT_TRUE(source.Rotate(90, &result));
}

TEST(YuvBufferTest, Nv21RotateDifferentFormat) {
  YuvBuffer source(8, 8, YuvBuffer::NV21);
  YuvBuffer result(8, 8, YuvBuffer::YV12);
  Fill(&source, 16, 32, 64);
  EXPECT_TRUE(source.Rotate(90, &result));
}

TEST(YuvBufferTest, Nv21RotateFailBounds) {
  // Expect failure if the destination doesn't have the correct bounds.
  YuvBuffer source(4, 8, YuvBuffer::NV21);
  YuvBuffer result(4, 8, YuvBuffer::NV21);
  Fill(&source, 16, 32, 64);
  EXPECT_FALSE(source.Rotate(90, &result));
}

TEST(YuvBufferTest, Nv21Flip) {
  YuvBuffer source(16, 16, YuvBuffer::NV21);
  YuvBuffer result(16, 16, YuvBuffer::NV21);
  Fill(&source, 16, 32, 64);
  EXPECT_TRUE(source.FlipHorizontally(&result));
  EXPECT_TRUE(source.FlipVertically(&result));
}

TEST(YuvBufferTest, Nv21FlipDifferentFormat) {
  YuvBuffer source(16, 16, YuvBuffer::NV21);
  YuvBuffer result(16, 16, YuvBuffer::YV12);
  Fill(&source, 16, 32, 64);
  EXPECT_TRUE(source.FlipHorizontally(&result));
  EXPECT_TRUE(source.FlipVertically(&result));
}

TEST(YuvBufferTest, Nv21ConvertRgb) {
  // Note that RGB conversion expects at least images of width >= 32 because
  // the implementation is vectorized.
  YuvBuffer source(32, 8, YuvBuffer::NV21);
  Fill(&source, 52, 170, 90);

  RgbBuffer result_rgb(32, 8, false);
  EXPECT_TRUE(source.Convert(false, &result_rgb));

  RgbBuffer result_rgba(32, 8, true);
  EXPECT_TRUE(source.Convert(false, &result_rgba));

  uint8_t* pixels = result_rgba.buffer()->host;
  ASSERT_TRUE(pixels);
  EXPECT_EQ(pixels[0], 0);
  EXPECT_EQ(pixels[1], 65);
  EXPECT_EQ(pixels[2], 126);
  EXPECT_EQ(pixels[3], 255);
}

TEST(YuvBufferTest, Nv21ConvertRgbCropped) {
  // Note that RGB conversion expects at least images of width >= 32 because
  // the implementation is vectorized.
  YuvBuffer source(1024, 768, YuvBuffer::NV21);
  Fill(&source, 52, 170, 90);

  // YUV images must be left-and top-aligned to even X/Y coordinates,
  // regardless of whether the target image has even or odd width/height.
  EXPECT_FALSE(source.Crop(1, 1, 512, 384));
  EXPECT_FALSE(source.Crop(1, 1, 511, 383));

  YuvBuffer source1(source);
  EXPECT_TRUE(source1.Crop(64, 64, 512, 384));
  RgbBuffer result_rgb(source1.width(), source1.height(), false);
  EXPECT_TRUE(source1.Convert(false, &result_rgb));

  YuvBuffer source2(source);
  EXPECT_TRUE(source2.Crop(64, 64, 511, 383));
  RgbBuffer result_rgba(source2.width(), source2.height(), true);
  EXPECT_TRUE(source2.Convert(false, &result_rgba));

  uint8_t* pixels = result_rgba.buffer()->host;
  ASSERT_TRUE(pixels);
  EXPECT_EQ(pixels[0], 0);
  EXPECT_EQ(pixels[1], 65);
  EXPECT_EQ(pixels[2], 126);
  EXPECT_EQ(pixels[3], 255);
}

TEST(YuvBufferTest, Nv21ConvertRgbHalve) {
  YuvBuffer source(64, 8, YuvBuffer::NV21);
  Fill(&source, 52, 170, 90);

  RgbBuffer result_rgb(32, 4, false);
  EXPECT_TRUE(source.Convert(true, &result_rgb));

  RgbBuffer result_rgba(32, 4, true);
  EXPECT_TRUE(source.Convert(true, &result_rgba));

  uint8_t* pixels = result_rgba.buffer()->host;
  ASSERT_TRUE(pixels);
  EXPECT_EQ(pixels[0], 0);
  EXPECT_EQ(pixels[1], 65);
  EXPECT_EQ(pixels[2], 126);
  EXPECT_EQ(pixels[3], 255);
}

}  // namespace
}  // namespace frame_buffer
}  // namespace mediapipe
