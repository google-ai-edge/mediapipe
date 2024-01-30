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

#include <cstdint>
#include <cstdlib>
#include <string>

#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/c/vision/image_segmenter/image_segmenter_result.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace {

using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::vision::DecodeImageFromFile;
using testing::HasSubstr;

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

MpMask CreateCategoryMaskFromFile(const std::string& file_path) {
  auto cpp_expected_mask_image = DecodeImageFromFile(GetFullPath(file_path));
  const auto& cpp_expected_mask_image_frame =
      cpp_expected_mask_image->GetImageFrameSharedPtr();

  const int pixel_data_size =
      cpp_expected_mask_image_frame->PixelDataSizeStoredContiguously();
  uint8_t* pixel_data = new uint8_t[pixel_data_size];
  cpp_expected_mask_image_frame->CopyToBuffer(pixel_data, pixel_data_size);

  MpMask mask = {
      .type = MpMask::IMAGE_FRAME,
      .image_frame = {.mask_format = MaskFormat::UINT8,
                      .image_buffer = pixel_data,
                      .width = cpp_expected_mask_image_frame->Width(),
                      .height = cpp_expected_mask_image_frame->Height()}};

  return mask;
}

float SimilarToUint8Mask(const MpMask* actual_mask, const MpMask* expected_mask,
                         int magnification_factor) {
  //   Validate that both images are of the same size and type
  if (actual_mask->image_frame.width != expected_mask->image_frame.width ||
      actual_mask->image_frame.height != expected_mask->image_frame.height ||
      actual_mask->image_frame.mask_format != MaskFormat::UINT8 ||
      expected_mask->image_frame.mask_format != MaskFormat::UINT8) {
    return 0;  // Not similar
  }

  int consistent_pixels = 0;
  int total_pixels =
      actual_mask->image_frame.width * actual_mask->image_frame.height;

  const uint8_t* buffer_actual = actual_mask->image_frame.image_buffer;
  const uint8_t* buffer_expected = expected_mask->image_frame.image_buffer;

  for (int i = 0; i < total_pixels; ++i) {
    // Apply magnification factor and compare
    if (buffer_actual[i] * magnification_factor == buffer_expected[i]) {
      consistent_pixels++;
    }
  }

  float similarity = (float)consistent_pixels / total_pixels;
  return similarity;
}

TEST(InteractiveSegmenterTest,
     ImageModeTestSucceedsWithCategoryMaskAndKeypoint) {
  const auto image = DecodeImageFromFile(GetFullPath(kImageFile));
  ASSERT_TRUE(image.ok());

  const std::string model_path = GetFullPath(kModelName);
  InteractiveSegmenterOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* output_confidence_masks= */ false,
      /* output_category_mask= */ true,
  };

  void* segmenter =
      interactive_segmenter_create(&options, /* error_msg */ nullptr);
  EXPECT_NE(segmenter, nullptr);

  const auto& image_frame = image->GetImageFrameSharedPtr();
  const MpImage mp_image = {
      .type = MpImage::IMAGE_FRAME,
      .image_frame = {.format = static_cast<ImageFormat>(image_frame->Format()),
                      .image_buffer = image_frame->PixelData(),
                      .width = image_frame->Width(),
                      .height = image_frame->Height()}};

  ImageSegmenterResult result;

  // Initialize the keypoint
  NormalizedKeypoint keypoint = {0.329f, 0.545f, nullptr, 0.0f, false};

  // Initialize the ROI using brace initialization
  RegionOfInterest roi = {.format = RegionOfInterest::kKeypoint,
                          .keypoint = &keypoint,
                          .scribble = nullptr,
                          .scribble_count = 0};

  const int error =
      interactive_segmenter_segment_image(segmenter, mp_image, roi, &result,
                                          /* error_msg */ nullptr);
  EXPECT_EQ(error, 0);

  const MpMask expected_mask = CreateCategoryMaskFromFile(kMaskImageFile);
  const MpMask actual_mask = result.category_mask;
  EXPECT_GT(SimilarToUint8Mask(&actual_mask, &expected_mask,
                               kGoldenMaskMagnificationFactor),
            0.9f);
  interactive_segmenter_close_result(&result);
  interactive_segmenter_close(segmenter, /* error_msg */ nullptr);

  delete[] expected_mask.image_frame.image_buffer;
}

TEST(InteractiveSegmenterTest,
     ImageModeTestSucceedsWithCategoryMaskAndScribble) {
  const auto image = DecodeImageFromFile(GetFullPath(kImageFile));
  ASSERT_TRUE(image.ok());

  const std::string model_path = GetFullPath(kModelName);
  InteractiveSegmenterOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* output_confidence_masks= */ false,
      /* output_category_mask= */ true,
  };

  void* segmenter =
      interactive_segmenter_create(&options, /* error_msg */ nullptr);
  EXPECT_NE(segmenter, nullptr);

  const auto& image_frame = image->GetImageFrameSharedPtr();
  const MpImage mp_image = {
      .type = MpImage::IMAGE_FRAME,
      .image_frame = {.format = static_cast<ImageFormat>(image_frame->Format()),
                      .image_buffer = image_frame->PixelData(),
                      .width = image_frame->Width(),
                      .height = image_frame->Height()}};

  ImageSegmenterResult result;

  NormalizedKeypoint keypoints[3] = {{0.44f, 0.70f, nullptr, 0.0f, false},
                                     {0.44f, 0.71f, nullptr, 0.0f, false},
                                     {0.44f, 0.72f, nullptr, 0.0f, false}};

  // Initialize RegionOfInterestC
  RegionOfInterest roi = {.format = RegionOfInterest::kScribble,
                          .keypoint = nullptr,
                          .scribble = keypoints,
                          .scribble_count = 3};

  const int error =
      interactive_segmenter_segment_image(segmenter, mp_image, roi, &result,
                                          /* error_msg */ nullptr);
  EXPECT_EQ(error, 0);

  const MpMask expected_mask = CreateCategoryMaskFromFile(kMaskImageFile);
  const MpMask actual_mask = result.category_mask;
  EXPECT_GT(SimilarToUint8Mask(&actual_mask, &expected_mask,
                               kGoldenMaskMagnificationFactor),
            0.84f);
  interactive_segmenter_close_result(&result);
  interactive_segmenter_close(segmenter, /* error_msg */ nullptr);

  free(keypoints);
  delete[] expected_mask.image_frame.image_buffer;
}

TEST(InteractiveSegmenterTest, InvalidArgumentHandling) {
  // It is an error to set neither the asset buffer nor the path.
  InteractiveSegmenterOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ nullptr},
      /* output_confidence_masks= */ false,
      /* output_category_mask= */ true,
  };

  char* error_msg;
  void* segmenter = interactive_segmenter_create(&options, &error_msg);
  EXPECT_EQ(segmenter, nullptr);

  EXPECT_THAT(error_msg, HasSubstr("ExternalFile must specify"));

  free(error_msg);
}

TEST(InteractiveSegmenterTest, FailedRecognitionHandling) {
  const std::string model_path = GetFullPath(kModelName);
  InteractiveSegmenterOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* output_confidence_masks= */ false,
      /* output_category_mask= */ true,
  };

  void* segmenter = interactive_segmenter_create(&options, /* error_msg */
                                                 nullptr);
  EXPECT_NE(segmenter, nullptr);

  const MpImage mp_image = {.type = MpImage::GPU_BUFFER, .gpu_buffer = {}};
  ImageSegmenterResult result;
  char* error_msg;

  NormalizedKeypoint keypoint = {0.0f, 0.0f, nullptr, 0.0f, false};

  RegionOfInterest roi = {.format = RegionOfInterest::kKeypoint,
                          .keypoint = &keypoint,
                          .scribble = nullptr,
                          .scribble_count = 0};

  interactive_segmenter_segment_image(segmenter, mp_image, roi, &result,
                                      &error_msg);
  EXPECT_THAT(error_msg, HasSubstr("GPU Buffer not supported yet"));
  free(error_msg);
  interactive_segmenter_close(segmenter, /* error_msg */ nullptr);
}

}  // namespace
