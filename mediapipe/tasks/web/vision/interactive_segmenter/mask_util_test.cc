// Copyright 2026 The MediaPipe Authors.
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

#include "mediapipe/tasks/web/vision/interactive_segmenter/mask_util.h"

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"

namespace mediapipe::tasks::web::vision {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;

TEST(MaskUtilTest, CopiesOneChannelMaskSuccessfully) {
  // 2x3 image with 1 channel
  // Source layout (top-to-bottom):
  // Row 0: [1, 2]
  // Row 1: [3, 4]
  // Row 2: [5, 6]
  std::vector<uint8_t> src = {1, 2, 3, 4, 5, 6};
  std::vector<uint8_t> dest(6, 0);

  absl::Status status = CopyMask(src.data(), /*width=*/2, /*height=*/3,
                                 /*channels=*/1, /*channel_size=*/1,
                                 /*width_step=*/2, dest.data());
  EXPECT_TRUE(status.ok());

  // Expected destination layout (sequential copy):
  // Row 0: [1, 2]
  // Row 1: [3, 4]
  // Row 2: [5, 6]
  std::vector<uint8_t> expected = {1, 2, 3, 4, 5, 6};
  EXPECT_THAT(dest, Eq(expected));
}

TEST(MaskUtilTest, CopiesFourChannelMaskExtractingFirstChannelByte) {
  // 2x2 image with 4 channels (e.g. RGBA, and channel_size = 1 byte)
  // Only the first byte of each pixel is expected in the destination.
  // Source layout:
  // Row 0: Pixel 0: [10, 0, 0, 0], Pixel 1: [20, 0, 0, 0]
  // Row 1: Pixel 2: [30, 0, 0, 0], Pixel 3: [40, 0, 0, 0]
  std::vector<uint8_t> src = {10, 0, 0, 0, 20, 0, 0, 0,
                              30, 0, 0, 0, 40, 0, 0, 0};
  std::vector<uint8_t> dest(4, 0);

  absl::Status status = CopyMask(src.data(), /*width=*/2, /*height=*/2,
                                 /*channels=*/4, /*channel_size=*/1,
                                 /*width_step=*/8, dest.data());
  EXPECT_TRUE(status.ok());

  // Expected destination layout (sequential copy, extracted first channel):
  // Row 0: [10, 20]
  // Row 1: [30, 40]
  std::vector<uint8_t> expected = {10, 20, 30, 40};
  EXPECT_THAT(dest, Eq(expected));
}

TEST(MaskUtilTest, CopiesMaskWithRowStridePaddingSuccessfully) {
  // 2x2 image with 1 channel, but with width_step (row stride) = 4 due to
  // padding. Source layout (each row is padded to 4 bytes): Row 0: [1, 2,
  // (padded 0, 0)] Row 1: [3, 4, (padded 0, 0)]
  std::vector<uint8_t> src = {1, 2, 0, 0, 3, 4, 0, 0};
  std::vector<uint8_t> dest(4, 0);

  // Dest does not have padding, it must be packed tightly (width * height *
  // channel_size = 4 bytes)
  absl::Status status = CopyMask(src.data(), /*width=*/2, /*height=*/2,
                                 /*channels=*/1, /*channel_size=*/1,
                                 /*width_step=*/4, dest.data());
  EXPECT_TRUE(status.ok());

  // Expected destination layout (sequential copy):
  // Row 0: [1, 2]
  // Row 1: [3, 4]
  std::vector<uint8_t> expected = {1, 2, 3, 4};
  EXPECT_THAT(dest, Eq(expected));
}

TEST(MaskUtilTest, ReturnsErrorForNullBuffers) {
  std::vector<uint8_t> buf = {1, 2, 3, 4};

  absl::Status status = CopyMask(nullptr, 2, 2, 1, 1, 2, buf.data());
  EXPECT_THAT(status.code(), Eq(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(), HasSubstr("buffers must not be null"));

  status = CopyMask(buf.data(), 2, 2, 1, 1, 2, nullptr);
  EXPECT_THAT(status.code(), Eq(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(), HasSubstr("buffers must not be null"));
}

TEST(MaskUtilTest, ReturnsErrorForInvalidDimensions) {
  std::vector<uint8_t> src = {1, 2, 3, 4};
  std::vector<uint8_t> dest(4, 0);

  absl::Status status =
      CopyMask(src.data(), /*width=*/0, /*height=*/2, 1, 1, 2, dest.data());
  EXPECT_THAT(status.code(), Eq(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(), HasSubstr("strictly positive"));

  status =
      CopyMask(src.data(), /*width=*/2, /*height=*/-1, 1, 1, 2, dest.data());
  EXPECT_THAT(status.code(), Eq(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(), HasSubstr("strictly positive"));
}

TEST(MaskUtilTest, ReturnsErrorForUnsupportedChannels) {
  std::vector<uint8_t> src = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint8_t> dest(4, 0);

  // 2 channels is unsupported
  absl::Status status =
      CopyMask(src.data(), 2, 2, /*channels=*/2, 1, 4, dest.data());
  EXPECT_THAT(status.code(), Eq(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(), HasSubstr("Unsupported number of channels"));

  // 3 channels is unsupported
  status = CopyMask(src.data(), 2, 2, /*channels=*/3, 1, 6, dest.data());
  EXPECT_THAT(status.code(), Eq(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(), HasSubstr("Unsupported number of channels"));
}

TEST(MaskUtilTest, ReturnsErrorForInvalidWidthStep) {
  std::vector<uint8_t> src = {1, 2, 3, 4};
  std::vector<uint8_t> dest(4, 0);

  // width_step (1) is smaller than row size (width * channels * channel_size =
  // 2)
  absl::Status status =
      CopyMask(src.data(), 2, 2, 1, 1, /*width_step=*/1, dest.data());
  EXPECT_THAT(status.code(), Eq(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(), HasSubstr("Width step is smaller"));
}

}  // namespace
}  // namespace mediapipe::tasks::web::vision
