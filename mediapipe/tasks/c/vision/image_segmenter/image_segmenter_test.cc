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

#include "mediapipe/tasks/c/vision/image_segmenter/image_segmenter.h"

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
constexpr char kModelName[] = "deeplabv3.tflite";
constexpr char kImageFile[] = "cat.jpg";
constexpr int kIterations = 5;
constexpr float kGoldenMaskSimilarity = 0.98;

std::string GetFullPath(absl::string_view file_name) {
  return JoinPath("./", kTestDataDirectory, file_name);
}

double CalculateSoftIOU(const MpMask m1, const MpMask m2) {
  double intersectionSum = 0.0;
  double unionSum = 0.0;

  if (m1.type == MpImage::IMAGE_FRAME &&
      m1.image_frame.mask_format == MaskFormat::MASK_FORMAT_FLOAT &&
      m2.type == MpImage::IMAGE_FRAME &&
      m2.image_frame.mask_format == MaskFormat::MASK_FORMAT_FLOAT) {
    int totalPixels = m1.image_frame.width * m1.image_frame.height;
    for (int i = 0; i < totalPixels; ++i) {
      float val1 = ((float*)m1.image_frame.buffer_float)[i];
      float val2 = ((float*)m2.image_frame.buffer_float)[i];
      intersectionSum += val1 * val2;
      unionSum += val1 * val1 + val2 * val2 - val1 * val2;
    }
  }

  return unionSum > 0.0 ? intersectionSum / unionSum : 0.0;
}

int SimilarToFloatMask(const MpMask actual_mask, const MpMask expected_mask,
                       float similarity_threshold) {
  if (actual_mask.image_frame.width != expected_mask.image_frame.width ||
      actual_mask.image_frame.height != expected_mask.image_frame.height) {
    return 0;  // Not similar
  }

  double iou = CalculateSoftIOU(actual_mask, expected_mask);
  return iou > similarity_threshold;
}

int SimilarToUint8Mask(const MpMask* actual_mask, const MpMask* expected_mask,
                       float similarity_threshold, int magnification_factor) {
  // Validate that both images are of the same size and type
  if (actual_mask->image_frame.width != expected_mask->image_frame.width ||
      actual_mask->image_frame.height != expected_mask->image_frame.height ||
      actual_mask->image_frame.mask_format != MaskFormat::MASK_FORMAT_UINT8 ||
      expected_mask->image_frame.mask_format != MaskFormat::MASK_FORMAT_UINT8) {
    return 0;  // Not similar
  }

  int consistent_pixels = 0;
  int total_pixels =
      actual_mask->image_frame.width * actual_mask->image_frame.height;

  for (int i = 0; i < total_pixels; ++i) {
    // Apply magnification factor and compare
    if (actual_mask->image_frame.buffer_uint8[i] * magnification_factor ==
        expected_mask->image_frame.buffer_uint8[i]) {
      consistent_pixels++;
    }
  }

  float similarity = (float)consistent_pixels / total_pixels;
  return similarity >= similarity_threshold;
}

TEST(ImageSegmenterTest, ImageModeTestSucceedsWithConfidenceMask) {
  const auto image = DecodeImageFromFile(GetFullPath(kImageFile));
  ASSERT_TRUE(image.ok());

  const std::string model_path = GetFullPath(kModelName);
  ImageSegmenterOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::IMAGE,
      /* display_names_locale= */ "en",
      /* output_confidence_masks= */ true,
      /* output_category_mask= */ false,
  };

  void* segmenter = image_segmenter_create(&options, /* error_msg */ nullptr);
  EXPECT_NE(segmenter, nullptr);

  const auto& image_frame = image->GetImageFrameSharedPtr();
  const MpImage mp_image = {
      .type = MpImage::IMAGE_FRAME,
      .image_frame = {.format = static_cast<ImageFormat>(image_frame->Format()),
                      .image_buffer = image_frame->PixelData(),
                      .width = image_frame->Width(),
                      .height = image_frame->Height()}};

  ImageSegmenterResult result;
  image_segmenter_segment_image(segmenter, mp_image, &result,
                                /* error_msg */ nullptr);
  EXPECT_NE(result.confidence_masks, nullptr);
  EXPECT_EQ(result.confidence_masks_count, 21);

  const auto cpp_expected_mask_image =
      DecodeImageFromFile(GetFullPath("cat_mask.jpg"));
  const auto& cpp_expected_mask_image_frame =
      cpp_expected_mask_image->GetImageFrameSharedPtr();
  MpMask expected_mask = {
      .type = MpMask::IMAGE_FRAME,
      .image_frame = {.mask_format = MaskFormat::MASK_FORMAT_FLOAT,
                      .buffer_float = cpp_expected_mask_image_frame->PixelData(),
                      .width = cpp_expected_mask_image_frame->Width(),
                      .height = cpp_expected_mask_image_frame->Height()}};

  MpMask actual_mask = result.confidence_masks[8];

  EXPECT_TRUE(
      SimilarToFloatMask(actual_mask, expected_mask, kGoldenMaskSimilarity));
  image_segmenter_close_result(&result);
  image_segmenter_close(segmenter, /* error_msg */ nullptr);
}

TEST(ImageSegmenterTest, VideoModeTest) {
  const auto image = DecodeImageFromFile(GetFullPath(kImageFile));
  ASSERT_TRUE(image.ok());

  const std::string model_path = GetFullPath(kModelName);
  ImageSegmenterOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::VIDEO,
      /* display_names_locale= */ "en",
      /* output_confidence_masks= */ true,
      /* output_category_mask= */ false,
  };

  void* segmenter = image_segmenter_create(&options, /* error_msg */ nullptr);
  EXPECT_NE(segmenter, nullptr);

  const auto& image_frame = image->GetImageFrameSharedPtr();
  const MpImage mp_image = {
      .type = MpImage::IMAGE_FRAME,
      .image_frame = {.format = static_cast<ImageFormat>(image_frame->Format()),
                      .image_buffer = image_frame->PixelData(),
                      .width = image_frame->Width(),
                      .height = image_frame->Height()}};

  for (int i = 0; i < kIterations; ++i) {
    ImageSegmenterResult result;
    image_segmenter_segment_for_video(segmenter, mp_image, i, &result,
                                      /* error_msg */ nullptr);

    image_segmenter_close_result(&result);
  }
  image_segmenter_close(segmenter, /* error_msg */ nullptr);
}

// A structure to support LiveStreamModeTest below. This structure holds a
// static method `Fn` for a callback function of C API. A `static` qualifier
// allows to take an address of the method to follow API style. Another static
// struct member is `last_timestamp` that is used to verify that current
// timestamp is greater than the previous one.
struct LiveStreamModeCallback {
  static int64_t last_timestamp;
  static void Fn(const ImageSegmenterResult* segmenter_result,
                 const MpImage& image, int64_t timestamp, char* error_msg) {
    ASSERT_NE(segmenter_result, nullptr);
    ASSERT_EQ(error_msg, nullptr);
    EXPECT_GT(image.image_frame.width, 0);
    EXPECT_GT(image.image_frame.height, 0);
    EXPECT_GT(timestamp, last_timestamp);
    ++last_timestamp;
  }
};
int64_t LiveStreamModeCallback::last_timestamp = -1;

TEST(ImageSegmenterTest, LiveStreamModeTest) {
  const auto image = DecodeImageFromFile(GetFullPath(kImageFile));
  ASSERT_TRUE(image.ok());

  const std::string model_path = GetFullPath(kModelName);

  ImageSegmenterOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::LIVE_STREAM,
      /* display_names_locale= */ "en",
      /* output_confidence_masks= */ true,
      /* output_category_mask= */ false,
      /* result_callback= */ LiveStreamModeCallback::Fn,
  };

  void* segmenter = image_segmenter_create(&options, /* error_msg */ nullptr);
  EXPECT_NE(segmenter, nullptr);

  const auto& image_frame = image->GetImageFrameSharedPtr();
  const MpImage mp_image = {
      .type = MpImage::IMAGE_FRAME,
      .image_frame = {.format = static_cast<ImageFormat>(image_frame->Format()),
                      .image_buffer = image_frame->PixelData(),
                      .width = image_frame->Width(),
                      .height = image_frame->Height()}};

  for (int i = 0; i < kIterations; ++i) {
    EXPECT_GE(image_segmenter_segment_async(segmenter, mp_image, i,
                                            /* error_msg */ nullptr),
              0);
  }
  image_segmenter_close(segmenter, /* error_msg */ nullptr);

  // Due to the flow limiter, the total of outputs might be smaller than the
  // number of iterations.
  EXPECT_LE(LiveStreamModeCallback::last_timestamp, kIterations);
  EXPECT_GT(LiveStreamModeCallback::last_timestamp, 0);
}

TEST(ImageSegmenterTest, InvalidArgumentHandling) {
  // It is an error to set neither the asset buffer nor the path.
  ImageSegmenterOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ nullptr},
      /* running_mode= */ RunningMode::IMAGE,
      /* display_names_locale= */ "en",
      /* output_confidence_masks= */ true,
      /* output_category_mask= */ false,
  };

  char* error_msg;
  void* segmenter = image_segmenter_create(&options, &error_msg);
  EXPECT_EQ(segmenter, nullptr);

  EXPECT_THAT(error_msg, HasSubstr("ExternalFile must specify"));

  free(error_msg);
}

TEST(ImageSegmenterTest, FailedRecognitionHandling) {
  const std::string model_path = GetFullPath(kModelName);
  ImageSegmenterOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::IMAGE,
      /* display_names_locale= */ "en",
      /* output_confidence_masks= */ true,
      /* output_category_mask= */ false,
  };

  void* segmenter = image_segmenter_create(&options, /* error_msg */
                                           nullptr);
  EXPECT_NE(segmenter, nullptr);

  const MpImage mp_image = {.type = MpImage::GPU_BUFFER, .gpu_buffer = {}};
  ImageSegmenterResult result;
  char* error_msg;
  image_segmenter_segment_image(segmenter, mp_image, &result, &error_msg);
  EXPECT_THAT(error_msg, HasSubstr("GPU Buffer not supported yet"));
  free(error_msg);
  image_segmenter_close(segmenter, /* error_msg */ nullptr);
}

}  // namespace
