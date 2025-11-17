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
#include <memory>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
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
using ::testing::HasSubstr;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kModelName[] = "deeplabv3.tflite";
constexpr char kImageFile[] = "segmentation_input_rotation0.jpg";
constexpr char kMaskImageFile[] = "segmentation_golden_rotation0.png";
constexpr char kImageRotatedFile[] = "segmentation_input_rotation90.jpg";
constexpr int kIterations = 2;
constexpr int kSleepBetweenFramesMilliseconds = 250;
constexpr float kGoldenMaskSimilarity = 0.98;
// Image rotation slightly lossy, so reduce golden similarity threshold a little
constexpr float kGoldenMaskSimilarityRotated = 0.96;

// Magnification factor used when creating the golden category masks to make
// them more human-friendly. Each pixel in the golden masks has its value
// multiplied by this factor, i.e. a value of 10 means class index 1, a value of
// 20 means class index 2, etc.
constexpr int kGoldenMaskMagnificationFactor = 10;

std::string GetFullPath(absl::string_view file_name) {
  return JoinPath("./", kTestDataDirectory, file_name);
}

TEST(ImageSegmenterTest, ImageModeTestSucceedsWithCategoryMask) {
  const auto image = GetImage(GetFullPath(kImageFile));

  const std::string model_path = GetFullPath(kModelName);
  ImageSegmenterOptions options = {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = model_path.c_str()},
      .running_mode = RunningMode::IMAGE,
      .display_names_locale = "en",
      .output_confidence_masks = false,
      .output_category_mask = true,
  };

  MpImageSegmenterPtr segmenter;
  EXPECT_EQ(MpImageSegmenterCreate(&options, &segmenter), kMpOk);

  ImageSegmenterResult result;
  EXPECT_EQ(MpImageSegmenterSegmentImage(
                segmenter, image.get(), /* image_processing_options= */ nullptr,
                &result),
            kMpOk);

  auto expected_mask_image = DecodeImageFromFile(GetFullPath(kMaskImageFile));
  const MpMask expected_mask = CreateCategoryMaskFromImage(expected_mask_image);
  const MpImagePtr actual_mask = result.category_mask;
  EXPECT_GT(SimilarToUint8Mask(actual_mask, &expected_mask,
                               kGoldenMaskMagnificationFactor),
            kGoldenMaskSimilarity);
  MpImageSegmenterCloseResult(&result);
  EXPECT_EQ(MpImageSegmenterClose(segmenter), kMpOk);

  delete[] expected_mask.image_frame.image_buffer;
}

TEST(ImageSegmenterTest, ImageModeWithRotationTestSucceedsWithCategoryMask) {
  const auto image = GetImage(GetFullPath(kImageRotatedFile));

  const std::string model_path = GetFullPath(kModelName);
  ImageSegmenterOptions options = {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = model_path.c_str()},
      .running_mode = RunningMode::IMAGE,
      .display_names_locale = "en",
      .output_confidence_masks = false,
      .output_category_mask = true,
  };

  MpImageSegmenterPtr segmenter;
  EXPECT_EQ(MpImageSegmenterCreate(&options, &segmenter), kMpOk);

  ImageProcessingOptions image_processing_options;
  image_processing_options.has_region_of_interest = 0;
  image_processing_options.rotation_degrees = 90;

  ImageSegmenterResult result;
  EXPECT_EQ(MpImageSegmenterSegmentImage(segmenter, image.get(),
                                         &image_processing_options, &result),
            kMpOk);

  auto expected_mask_image = DecodeImageFromFile(GetFullPath(kMaskImageFile));
  const MpMask expected_mask = CreateCategoryMaskFromImage(expected_mask_image);
  const MpImagePtr actual_mask = result.category_mask;
  EXPECT_GT(SimilarToUint8Mask(actual_mask, &expected_mask,
                               kGoldenMaskMagnificationFactor),
            kGoldenMaskSimilarityRotated);
  MpImageSegmenterCloseResult(&result);
  EXPECT_EQ(MpImageSegmenterClose(segmenter), kMpOk);

  delete[] expected_mask.image_frame.image_buffer;
}

TEST(ImageSegmenterTest, VideoModeTest) {
  const auto image = GetImage(GetFullPath(kImageFile));

  const std::string model_path = GetFullPath(kModelName);
  ImageSegmenterOptions options = {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = model_path.c_str()},
      .running_mode = RunningMode::VIDEO,
      .display_names_locale = "en",
      .output_confidence_masks = false,
      .output_category_mask = true,
  };

  MpImageSegmenterPtr segmenter;
  EXPECT_EQ(MpImageSegmenterCreate(&options, &segmenter), kMpOk);

  auto expected_mask_image = DecodeImageFromFile(GetFullPath(kMaskImageFile));
  const MpMask expected_mask = CreateCategoryMaskFromImage(expected_mask_image);

  for (int i = 0; i < kIterations; ++i) {
    ImageSegmenterResult result;
    EXPECT_EQ(MpImageSegmenterSegmentForVideo(
                  segmenter, image.get(),
                  /* image_processing_options= */ nullptr, i, &result),
              kMpOk);
    const MpImagePtr actual_mask = result.category_mask;
    EXPECT_GT(SimilarToUint8Mask(actual_mask, &expected_mask,
                                 kGoldenMaskMagnificationFactor),
              kGoldenMaskSimilarity);

    MpImageSegmenterCloseResult(&result);
  }
  EXPECT_EQ(MpImageSegmenterClose(segmenter), kMpOk);

  delete[] expected_mask.image_frame.image_buffer;
}

// A structure to support LiveStreamModeTest below. This structure holds a
// static method `Fn` for a callback function of C API. A `static` qualifier
// allows to take an address of the method to follow API style. Another static
// struct member is `last_timestamp` that is used to verify that current
// timestamp is greater than the previous one.
struct LiveStreamModeCallback {
  static int64_t last_timestamp;
  static absl::BlockingCounter* blocking_counter;
  static void Fn(MpStatus status, const ImageSegmenterResult* segmenter_result,
                 MpImagePtr image, int64_t timestamp) {
    ASSERT_EQ(status, kMpOk);
    ASSERT_NE(segmenter_result, nullptr);
    EXPECT_GT(MpImageGetWidth(image), 0);
    EXPECT_GT(MpImageGetHeight(image), 0);
    auto expected_mask_image = DecodeImageFromFile(GetFullPath(kMaskImageFile));
    const MpMask expected_mask =
        CreateCategoryMaskFromImage(expected_mask_image);
    const MpImagePtr actual_mask = segmenter_result->category_mask;
    EXPECT_GT(SimilarToUint8Mask(actual_mask, &expected_mask,
                                 kGoldenMaskMagnificationFactor),
              kGoldenMaskSimilarity);
    EXPECT_GT(timestamp, last_timestamp);
    ++last_timestamp;

    delete[] expected_mask.image_frame.image_buffer;
    if (blocking_counter) {
      blocking_counter->DecrementCount();
    }
  }
};
int64_t LiveStreamModeCallback::last_timestamp = -1;
absl::BlockingCounter* LiveStreamModeCallback::blocking_counter = nullptr;

TEST(ImageSegmenterTest, LiveStreamModeTest) {
  const auto image = GetImage(GetFullPath(kImageFile));

  const std::string model_path = GetFullPath(kModelName);

  ImageSegmenterOptions options = {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = model_path.c_str()},
      .running_mode = RunningMode::LIVE_STREAM,
      .display_names_locale = "en",
      .output_confidence_masks = false,
      .output_category_mask = true,
      .result_callback = LiveStreamModeCallback::Fn,
  };

  MpImageSegmenterPtr segmenter;
  EXPECT_EQ(MpImageSegmenterCreate(&options, &segmenter), kMpOk);

  absl::BlockingCounter counter(kIterations);
  LiveStreamModeCallback::blocking_counter = &counter;

  for (int i = 0; i < kIterations; ++i) {
    EXPECT_EQ(
        MpImageSegmenterSegmentAsync(
            segmenter, image.get(), /* image_processing_options= */ nullptr, i),
        kMpOk);
    // Short sleep so that MediaPipe does not drop frames.
    absl::SleepFor(absl::Milliseconds(kSleepBetweenFramesMilliseconds));
  }

  // Wait for all callbacks to be invoked.
  counter.Wait();
  LiveStreamModeCallback::blocking_counter = nullptr;

  EXPECT_EQ(MpImageSegmenterClose(segmenter), kMpOk);

  // Due to the flow limiter, the total of outputs might be smaller than the
  // number of iterations.
  EXPECT_LE(LiveStreamModeCallback::last_timestamp, kIterations);
  EXPECT_GT(LiveStreamModeCallback::last_timestamp, 0);
}

TEST(ImageSegmenterTest, InvalidArgumentHandling) {
  // It is an error to set neither the asset buffer nor the path.
  ImageSegmenterOptions options = {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = nullptr},
      .running_mode = RunningMode::IMAGE,
      .display_names_locale = "en",
      .output_confidence_masks = false,
      .output_category_mask = true,
  };

  MpImageSegmenterPtr segmenter = nullptr;
  MpStatus status = MpImageSegmenterCreate(&options, &segmenter);
  EXPECT_EQ(segmenter, nullptr);
  EXPECT_EQ(status, kMpInvalidArgument);
}

TEST(ImageSegmenterTest, FailedRecognitionHandling) {
  const std::string model_path = GetFullPath(kModelName);
  ImageSegmenterOptions options = {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = model_path.c_str()},
      .running_mode = RunningMode::IMAGE,
      .display_names_locale = "en",
      .output_confidence_masks = false,
      .output_category_mask = true,
  };

  MpImageSegmenterPtr segmenter;
  EXPECT_EQ(MpImageSegmenterCreate(&options, &segmenter), kMpOk);
  EXPECT_NE(segmenter, nullptr);

  const ScopedMpImage image = CreateEmptyGpuMpImage();
  ImageSegmenterResult result;
  EXPECT_EQ(MpImageSegmenterSegmentImage(
                segmenter, image.get(), /* image_processing_options= */ nullptr,
                &result),
            kMpInvalidArgument);
  EXPECT_EQ(MpImageSegmenterClose(segmenter), kMpOk);
}

TEST(ImageSegmenterTest, GetLabelsSucceeds) {
  const std::string model_path = GetFullPath(kModelName);
  ImageSegmenterOptions options = {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = model_path.c_str()},
      .running_mode = RunningMode::IMAGE,
      .display_names_locale = "en",
      .output_confidence_masks = false,
      .output_category_mask = true,
  };

  MpImageSegmenterPtr segmenter;
  EXPECT_EQ(MpImageSegmenterCreate(&options, &segmenter), kMpOk);
  EXPECT_NE(segmenter, nullptr);

  MpStringList labels;
  EXPECT_EQ(MpImageSegmenterGetLabels(segmenter, &labels), kMpOk);

  const std::vector<std::string> expected_labels = {
      "background", "aeroplane",    "bicycle", "bird",  "boat",
      "bottle",     "bus",          "car",     "cat",   "chair",
      "cow",        "dining table", "dog",     "horse", "motorbike",
      "person",     "potted plant", "sheep",   "sofa",  "train",
      "tv"};

  EXPECT_EQ(labels.num_strings, expected_labels.size());
  for (int i = 0; i < expected_labels.size(); ++i) {
    EXPECT_STREQ(labels.strings[i], expected_labels[i].c_str());
  }

  MpStringListFree(&labels);
  EXPECT_EQ(MpImageSegmenterClose(segmenter), kMpOk);
}

}  // namespace
