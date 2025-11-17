/* Copyright 2024 The MediaPipe Authors.

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

#include "mediapipe/tasks/c/vision/interactive_segmenter/interactive_segmenter.h"

#include <string>

#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/keypoint.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/test/test_utils.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/core/image_test_util.h"
#include "mediapipe/tasks/c/vision/image_segmenter/image_segmenter_result.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace {

using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::c::test::CreateCategoryMaskFromImage;
using ::mediapipe::tasks::c::test::SimilarToUint8Mask;
using ::mediapipe::tasks::vision::DecodeImageFromFile;
using ::mediapipe::tasks::vision::core::CreateEmptyGpuMpImage;
using ::mediapipe::tasks::vision::core::GetImage;
using ::mediapipe::tasks::vision::core::ScopedMpImage;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kModelName[] = "ptm_512_hdt_ptm_woid.tflite";
constexpr char kImageFile[] = "penguins_large.jpg";
constexpr char kMaskImageFile[] = "penguins_large_mask.png";

// Magnification factor used when creating the golden category masks to make
// them more human-friendly. Since interactive segmenter has only 2 categories,
// the golden mask uses 0 or 255 for each pixel.
constexpr int kGoldenMaskMagnificationFactor = 255;

std::string GetFullPath(absl::string_view file_name) {
  return JoinPath("./", kTestDataDirectory, file_name);
}

TEST(InteractiveSegmenterTest,
     ImageModeTestSucceedsWithCategoryMaskAndKeypoint) {
  const ScopedMpImage image = GetImage(GetFullPath(kImageFile));
  ASSERT_NE(image, nullptr);

  const std::string model_path = GetFullPath(kModelName);
  InteractiveSegmenterOptions options = {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = model_path.c_str()},
      .output_confidence_masks = false,
      .output_category_mask = true,
  };

  MpInteractiveSegmenterPtr segmenter;
  ASSERT_EQ(MpInteractiveSegmenterCreate(&options, &segmenter), kMpOk);

  ImageSegmenterResult result;

  // Initialize the keypoint
  NormalizedKeypoint keypoint = {.x = 0.329f, .y = 0.545f};

  // Initialize the ROI using brace initialization
  RegionOfInterest roi = {.format = RegionOfInterest::kKeypoint,
                          .keypoint = &keypoint,
                          .scribble = nullptr,
                          .scribble_count = 0};

  ASSERT_EQ(MpInteractiveSegmenterSegmentImage(
                segmenter, image.get(), &roi,
                /* image_processing_options= */ nullptr, &result),
            kMpOk);

  auto expected_mask_image = DecodeImageFromFile(GetFullPath(kMaskImageFile));
  const MpMask expected_mask = CreateCategoryMaskFromImage(expected_mask_image);
  const MpImagePtr actual_mask = result.category_mask;
  EXPECT_GT(SimilarToUint8Mask(actual_mask, &expected_mask,
                               kGoldenMaskMagnificationFactor),
            0.9f);
  MpInteractiveSegmenterCloseResult(&result);
  ASSERT_EQ(MpInteractiveSegmenterClose(segmenter), kMpOk);

  delete[] expected_mask.image_frame.image_buffer;
}

// Test here fails since the model metadata has no Activation type.
TEST(InteractiveSegmenterTest,
     ImageModeTestSucceedsWithCategoryMaskAndScribble) {
  const ScopedMpImage image = GetImage(GetFullPath(kImageFile));
  ASSERT_NE(image, nullptr);

  const std::string model_path = GetFullPath(kModelName);
  InteractiveSegmenterOptions options = {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = model_path.c_str()},
      .output_confidence_masks = false,
      .output_category_mask = true,
  };

  MpInteractiveSegmenterPtr segmenter;
  ASSERT_EQ(MpInteractiveSegmenterCreate(&options, &segmenter), kMpOk);

  ImageSegmenterResult result;

  NormalizedKeypoint keypoints[3] = {{.x = 0.44f, .y = 0.70f},
                                     {.x = 0.44f, .y = 0.71f},
                                     {.x = 0.44f, .y = 0.72f}};

  // Initialize RegionOfInterestC
  RegionOfInterest roi = {.format = RegionOfInterest::kScribble,
                          .keypoint = nullptr,
                          .scribble = keypoints,
                          .scribble_count = 3};

  ASSERT_EQ(MpInteractiveSegmenterSegmentImage(
                segmenter, image.get(), &roi,
                /* image_processing_options= */ nullptr, &result),
            kMpOk);

  auto expected_mask_image = DecodeImageFromFile(GetFullPath(kMaskImageFile));
  const MpMask expected_mask = CreateCategoryMaskFromImage(expected_mask_image);
  const MpImagePtr actual_mask = result.category_mask;
  EXPECT_GT(SimilarToUint8Mask(actual_mask, &expected_mask,
                               kGoldenMaskMagnificationFactor),
            0.84f);
  MpInteractiveSegmenterCloseResult(&result);
  ASSERT_EQ(MpInteractiveSegmenterClose(segmenter), kMpOk);

  delete[] expected_mask.image_frame.image_buffer;
}

TEST(InteractiveSegmenterTest, ImageModeTestWithRotation) {
  const ScopedMpImage image = GetImage(GetFullPath(kImageFile));
  ASSERT_NE(image, nullptr);

  const std::string model_path = GetFullPath(kModelName);
  InteractiveSegmenterOptions options = {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = model_path.c_str()},
      .output_confidence_masks = false,
      .output_category_mask = true,
  };

  MpInteractiveSegmenterPtr segmenter;
  ASSERT_EQ(MpInteractiveSegmenterCreate(&options, &segmenter), kMpOk);

  ImageSegmenterResult result;

  // Initialize the keypoint
  NormalizedKeypoint keypoint = {.x = 0.329f, .y = 0.545f};

  // Initialize the ROI using brace initialization
  RegionOfInterest roi = {.format = RegionOfInterest::kKeypoint,
                          .keypoint = &keypoint,
                          .scribble = nullptr,
                          .scribble_count = 0};

  ImageProcessingOptions image_processing_options = {
      .has_region_of_interest = false, .rotation_degrees = -90};

  ASSERT_EQ(
      MpInteractiveSegmenterSegmentImage(segmenter, image.get(), &roi,
                                         &image_processing_options, &result),
      kMpOk);

  auto expected_mask_image = DecodeImageFromFile(GetFullPath(kMaskImageFile));
  const MpMask expected_mask = CreateCategoryMaskFromImage(expected_mask_image);
  const MpImagePtr actual_mask = result.category_mask;
  EXPECT_GT(SimilarToUint8Mask(actual_mask, &expected_mask,
                               kGoldenMaskMagnificationFactor),
            0.9f);
  MpInteractiveSegmenterCloseResult(&result);
  ASSERT_EQ(MpInteractiveSegmenterClose(segmenter), kMpOk);

  delete[] expected_mask.image_frame.image_buffer;
}

TEST(InteractiveSegmenterTest, InvalidArgumentHandling) {
  // It is an error to set neither the asset buffer nor the path.
  InteractiveSegmenterOptions options = {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = nullptr},
      .output_confidence_masks = false,
      .output_category_mask = true,
  };

  MpInteractiveSegmenterPtr segmenter = nullptr;
  MpStatus status = MpInteractiveSegmenterCreate(&options, &segmenter);
  EXPECT_EQ(segmenter, nullptr);

  EXPECT_EQ(status, kMpInvalidArgument);
}

TEST(InteractiveSegmenterTest, FailedRecognitionHandling) {
  const std::string model_path = GetFullPath(kModelName);
  InteractiveSegmenterOptions options = {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = model_path.c_str()},
      .output_confidence_masks = false,
      .output_category_mask = true,
  };

  MpInteractiveSegmenterPtr segmenter;
  ASSERT_EQ(MpInteractiveSegmenterCreate(&options, &segmenter), kMpOk);

  const ScopedMpImage mp_image = CreateEmptyGpuMpImage();
  ImageSegmenterResult result;

  NormalizedKeypoint keypoint = {.x = 0.0f, .y = 0.0f};

  RegionOfInterest roi = {.format = RegionOfInterest::kKeypoint,
                          .keypoint = &keypoint,
                          .scribble = nullptr,
                          .scribble_count = 0};

  MpStatus status = MpInteractiveSegmenterSegmentImage(
      segmenter, mp_image.get(), &roi,
      /* image_processing_options= */ nullptr, &result);
  EXPECT_EQ(status, kMpInvalidArgument);
  ASSERT_EQ(MpInteractiveSegmenterClose(segmenter), kMpOk);
}

}  // namespace
