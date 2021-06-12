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

#include "mediapipe/framework/formats/image_frame_pool.h"

#include "absl/memory/memory.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace {

using Pair = std::pair<int, int>;

constexpr int kWidth = 300;
constexpr int kHeight = 200;
constexpr ImageFormat::Format kFormat = ImageFormat::SRGBA;
constexpr int kKeepCount = 2;

class ImageFramePoolTest : public ::testing::Test {
 protected:
  ImageFramePoolTest() {
    pool_ = ImageFramePool::Create(kWidth, kHeight, kFormat, kKeepCount);
  }

  void SetUp() override {}

  std::shared_ptr<ImageFramePool> pool_;
};

TEST_F(ImageFramePoolTest, GetBuffer) {
  EXPECT_EQ(Pair(0, 0), pool_->GetInUseAndAvailableCounts());
  auto buffer = pool_->GetBuffer();
  EXPECT_EQ(Pair(1, 0), pool_->GetInUseAndAvailableCounts());
  buffer = nullptr;
  EXPECT_EQ(Pair(0, 1), pool_->GetInUseAndAvailableCounts());
  buffer = pool_->GetBuffer();
  EXPECT_EQ(Pair(1, 0), pool_->GetInUseAndAvailableCounts());
}

TEST_F(ImageFramePoolTest, GetMoreBuffers) {
  EXPECT_EQ(Pair(0, 0), pool_->GetInUseAndAvailableCounts());
  std::vector<ImageFrameSharedPtr> buffers;

  // Create kKeepCount + 1 buffers
  for (int i = 0; i <= kKeepCount; i++) {
    buffers.emplace_back(pool_->GetBuffer());
  }
  EXPECT_EQ(Pair(kKeepCount + 1, 0), pool_->GetInUseAndAvailableCounts());

  // Delete one
  buffers.resize(kKeepCount);
  EXPECT_EQ(Pair(kKeepCount, 0), pool_->GetInUseAndAvailableCounts());

  // Delete all
  buffers.resize(0);
  EXPECT_EQ(Pair(0, kKeepCount), pool_->GetInUseAndAvailableCounts());

  // Create one more
  buffers.emplace_back(pool_->GetBuffer());
  EXPECT_EQ(Pair(1, kKeepCount - 1), pool_->GetInUseAndAvailableCounts());
}

TEST_F(ImageFramePoolTest, DeleteNotLast) {
  EXPECT_EQ(Pair(0, 0), pool_->GetInUseAndAvailableCounts());
  std::vector<ImageFrameSharedPtr> buffers;

  // Create kKeepCount + 1 buffers
  for (int i = 0; i <= kKeepCount; i++) {
    buffers.emplace_back(pool_->GetBuffer());
  }
  EXPECT_EQ(Pair(kKeepCount + 1, 0), pool_->GetInUseAndAvailableCounts());

  // Delete second
  buffers.erase(buffers.begin() + 1);
  EXPECT_EQ(Pair(kKeepCount, 0), pool_->GetInUseAndAvailableCounts());

  // Delete first
  buffers.erase(buffers.begin());
  EXPECT_EQ(Pair(kKeepCount - 1, 1), pool_->GetInUseAndAvailableCounts());
}

TEST(ImageFrameBufferPoolStaticTest, BufferCanOutlivePool) {
  auto pool = ImageFramePool::Create(kWidth, kHeight, kFormat, kKeepCount);
  auto buffer = pool->GetBuffer();
  pool = nullptr;
  buffer = nullptr;
}

}  // anonymous namespace
}  // namespace mediapipe
