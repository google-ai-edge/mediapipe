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

#include "mediapipe/util/frame_buffer/gray_buffer.h"

#include <utility>

#include "absl/log/absl_log.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

// The default implementation of halide_error calls abort(), which we don't
// want. Instead, log the error and let the filter invocation fail.
extern "C" void halide_error(void*, const char* message) {
  ABSL_LOG(ERROR) << "Halide Error: " << message;
}

namespace mediapipe {
namespace frame_buffer {
namespace {

// Fill a GrayBuffer with zeroes.
void Fill(GrayBuffer* gray_buffer) {
  halide_buffer_t* buffer = gray_buffer->buffer();
  for (int y = 0; y < buffer->dim[1].extent; ++y) {
    for (int x = 0; x < buffer->dim[0].extent; ++x) {
      buffer->host[buffer->dim[1].stride * y + buffer->dim[0].stride * x] = 0;
    }
  }
}

TEST(GrayBufferTest, Properties) {
  GrayBuffer buffer(5, 4);
  EXPECT_EQ(5, buffer.width());
  EXPECT_EQ(4, buffer.height());
}

TEST(GrayBufferTest, Release) {
  GrayBuffer buffer(4, 4);
  delete[] buffer.Release();
}

TEST(GrayBufferTest, Assign) {
  GrayBuffer buffer(3, 2);
  GrayBuffer sink(nullptr, 0, 0);
  Fill(&buffer);
  sink = buffer;
  EXPECT_EQ(3, sink.width());
  EXPECT_EQ(2, sink.height());

  sink = GrayBuffer(5, 4);
  EXPECT_EQ(5, sink.width());
  EXPECT_EQ(4, sink.height());
}

TEST(GrayBufferTest, MoveAssign) {
  GrayBuffer buffer(3, 2);
  GrayBuffer sink(nullptr, 0, 0);
  Fill(&buffer);
  sink = std::move(buffer);
  EXPECT_EQ(nullptr, buffer.Release());
  EXPECT_EQ(3, sink.width());
  EXPECT_EQ(2, sink.height());
}

TEST(GrayBufferTest, MoveConstructor) {
  GrayBuffer buffer(5, 4);
  GrayBuffer sink(std::move(buffer));
  Fill(&buffer);
  EXPECT_EQ(nullptr, buffer.Release());
  EXPECT_EQ(5, sink.width());
  EXPECT_EQ(4, sink.height());
}

TEST(GrayBufferTest, Crop) {
  GrayBuffer source(8, 8);
  EXPECT_TRUE(source.Crop(2, 2, 6, 6));
}

TEST(GrayBufferTest, Resize_Even) {
  uint8_t* data = new uint8_t[16];
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      data[x + y * 4] = x + y * 4;
    }
  }
  GrayBuffer source(data, 4, 4);
  GrayBuffer result(2, 2);
  EXPECT_TRUE(source.Resize(&result));
  EXPECT_EQ(0, result.buffer()->host[0]);
  EXPECT_EQ(2, result.buffer()->host[1]);
  EXPECT_EQ(8, result.buffer()->host[2]);
  EXPECT_EQ(10, result.buffer()->host[3]);
  delete[] data;
}

TEST(GrayBufferTest, Resize_Odd) {
  uint8_t* data = new uint8_t[16];
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      data[x + y * 4] = x + y * 4;
    }
  }
  GrayBuffer source(data, 4, 4);
  GrayBuffer result(1, 3);
  EXPECT_TRUE(source.Resize(&result));
  EXPECT_EQ(0, result.buffer()->host[0]);
  EXPECT_EQ(5, result.buffer()->host[1]);
  EXPECT_EQ(11, result.buffer()->host[2]);
  delete[] data;
}

TEST(GrayBufferTest, Rotate) {
  GrayBuffer buffer(5, 4);
  GrayBuffer result(4, 5);
  Fill(&buffer);
  EXPECT_TRUE(buffer.Rotate(90, &result));
}

TEST(GrayBufferTest, Rotate_90) {
  uint8_t* data = new uint8_t[4];
  data[0] = 1;
  data[1] = 2;
  data[2] = 3;
  data[3] = 4;
  GrayBuffer buffer(data, 2, 2);
  GrayBuffer result(2, 2);
  EXPECT_TRUE(buffer.Rotate(90, &result));

  EXPECT_EQ(2, result.buffer()->host[0]);
  EXPECT_EQ(4, result.buffer()->host[1]);
  EXPECT_EQ(1, result.buffer()->host[2]);
  EXPECT_EQ(3, result.buffer()->host[3]);

  delete[] data;
}

TEST(GrayBufferTest, Rotate_180) {
  uint8_t* data = new uint8_t[4];
  data[0] = 1;
  data[1] = 2;
  data[2] = 3;
  data[3] = 4;
  GrayBuffer buffer(data, 2, 2);
  GrayBuffer result(2, 2);
  EXPECT_TRUE(buffer.Rotate(180, &result));
  EXPECT_EQ(4, result.buffer()->host[0]);
  EXPECT_EQ(3, result.buffer()->host[1]);
  EXPECT_EQ(2, result.buffer()->host[2]);
  EXPECT_EQ(1, result.buffer()->host[3]);
  delete[] data;
}

TEST(GrayBufferTest, Rotate_270) {
  uint8_t* data = new uint8_t[4];
  data[0] = 1;
  data[1] = 2;
  data[2] = 3;
  data[3] = 4;
  GrayBuffer buffer(data, 2, 2);
  GrayBuffer result(2, 2);
  EXPECT_TRUE(buffer.Rotate(270, &result));
  EXPECT_EQ(3, result.buffer()->host[0]);
  EXPECT_EQ(1, result.buffer()->host[1]);
  EXPECT_EQ(4, result.buffer()->host[2]);
  EXPECT_EQ(2, result.buffer()->host[3]);
  delete[] data;
}

TEST(GrayBufferTest, Flip) {
  GrayBuffer buffer(5, 4);
  GrayBuffer result(5, 4);
  Fill(&buffer);
  EXPECT_TRUE(buffer.FlipHorizontally(&result));
  EXPECT_TRUE(buffer.FlipVertically(&result));
}

TEST(GrayBufferTest, Flip_Horizontally) {
  uint8_t* data = new uint8_t[4];
  data[0] = 1;
  data[1] = 2;
  data[2] = 3;
  data[3] = 4;
  GrayBuffer buffer(data, 2, 2);
  GrayBuffer result(2, 2);
  EXPECT_TRUE(buffer.FlipHorizontally(&result));
  EXPECT_EQ(2, result.buffer()->host[0]);
  EXPECT_EQ(1, result.buffer()->host[1]);
  EXPECT_EQ(4, result.buffer()->host[2]);
  EXPECT_EQ(3, result.buffer()->host[3]);
  delete[] data;
}

TEST(GrayBufferTest, Flip_Vertically) {
  uint8_t* data = new uint8_t[4];
  data[0] = 1;
  data[1] = 2;
  data[2] = 3;
  data[3] = 4;
  GrayBuffer buffer(data, 2, 2);
  GrayBuffer result(2, 2);
  EXPECT_TRUE(buffer.FlipVertically(&result));
  EXPECT_EQ(3, result.buffer()->host[0]);
  EXPECT_EQ(4, result.buffer()->host[1]);
  EXPECT_EQ(1, result.buffer()->host[2]);
  EXPECT_EQ(2, result.buffer()->host[3]);
  delete[] data;
}

}  // namespace
}  // namespace frame_buffer
}  // namespace mediapipe
