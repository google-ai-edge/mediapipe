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

#include "mediapipe/gpu/gpu_buffer.h"

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/gpu/gpu_test_base.h"

namespace mediapipe {
namespace {

class GpuBufferTest : public GpuTestBase {};

TEST_F(GpuBufferTest, BasicTest) {
  RunInGlContext([this] {
    GpuBuffer buffer = gpu_shared_.gpu_buffer_pool.GetBuffer(300, 200);
    EXPECT_EQ(buffer.width(), 300);
    EXPECT_EQ(buffer.height(), 200);
    EXPECT_TRUE(buffer);
    EXPECT_FALSE(buffer == nullptr);

    GpuBuffer no_buffer;
    EXPECT_FALSE(no_buffer);
    EXPECT_TRUE(no_buffer == nullptr);

    GpuBuffer buffer2 = buffer;
    EXPECT_EQ(buffer, buffer);
    EXPECT_EQ(buffer, buffer2);
    EXPECT_NE(buffer, no_buffer);

    buffer = nullptr;
    EXPECT_TRUE(buffer == nullptr);
    EXPECT_TRUE(buffer == no_buffer);
  });
}

}  // anonymous namespace
}  // namespace mediapipe
