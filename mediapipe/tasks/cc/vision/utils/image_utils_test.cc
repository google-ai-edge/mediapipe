/* Copyright 2022 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe::tasks::vision::utils {
namespace {

using ::mediapipe::Tensor;

TEST(ImageUtilsTest, FailedImageFromBuffer) {
  constexpr int width = 1;
  constexpr int height = 1;
  constexpr int max_channels = 1;
  const std::vector<uint8_t> buffer(width * height * max_channels, 0);

  const ImageFormat::Format format = ImageFormat::UNKNOWN;
  const absl::StatusOr<Image> image =
      CreateImageFromBuffer(format, &buffer[0], width, height);
  EXPECT_EQ(image.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(image.status().message(),
            "Expected image of SRGB, SRGBA or SBGRA format, but found 0.");
}

TEST(ImageUtilsTest, FailedGetImageLikeTensorShape) {
  Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape{9});

  auto shape = GetImageLikeTensorShape(tensor);
  EXPECT_EQ(shape.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(shape.status().message(),
            "Tensor should have 2, 3, or 4 dims, received: 1");
}

class ImageUtilsParamTest
    : public ::testing::TestWithParam<ImageFormat::Format> {};

TEST_P(ImageUtilsParamTest, SuccessfulImageFromBuffer) {
  constexpr int width = 4;
  constexpr int height = 4;
  constexpr int max_channels = 4;
  const std::vector<uint8_t> buffer(width * height * max_channels, 0);

  const ImageFormat::Format format = GetParam();
  const absl::StatusOr<Image> image =
      CreateImageFromBuffer(format, &buffer[0], width, height);
  EXPECT_TRUE(image.status().ok());
  EXPECT_EQ(image->GetImageFrameSharedPtr()->Format(), format);
  EXPECT_EQ(image->GetImageFrameSharedPtr()->Width(), width);
  EXPECT_EQ(image->GetImageFrameSharedPtr()->Height(), height);
}

INSTANTIATE_TEST_SUITE_P(ImageUtilsTests, ImageUtilsParamTest,
                         testing::Values(ImageFormat::SRGB, ImageFormat::SRGBA,
                                         ImageFormat::SBGRA));
}  // namespace
}  // namespace mediapipe::tasks::vision::utils
