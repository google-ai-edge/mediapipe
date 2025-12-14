/* Copyright 2023 The MediaPipe Authors.

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

#include "mediapipe/tasks/c/vision/image_classifier/image_classifier.h"

#include <cstdint>
#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/category.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/core/image_test_util.h"

namespace {

using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::vision::core::CreateEmptyGpuMpImage;
using ::mediapipe::tasks::vision::core::GetImage;
using ::mediapipe::tasks::vision::core::ScopedMpImage;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kModelName[] = "mobilenet_v2_1.0_224.tflite";
constexpr float kPrecision = 1e-4;
constexpr int kIterations = 5;
constexpr int kSleepBetweenFramesMilliseconds = 100;

std::string GetFullPath(absl::string_view file_name) {
  return JoinPath("./", kTestDataDirectory, file_name);
}

TEST(ImageClassifierTest, ImageModeTest) {
  const ScopedMpImage image = GetImage(GetFullPath("burger.jpg"));

  const std::string model_path = GetFullPath(kModelName);
  ImageClassifierOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::IMAGE,
      /* classifier_options= */
      {/* display_names_locale= */ nullptr,
       /* max_results= */ -1,
       /* score_threshold= */ 0.0,
       /* category_allowlist= */ nullptr,
       /* category_allowlist_count= */ 0,
       /* category_denylist= */ nullptr,
       /* category_denylist_count= */ 0},
  };

  MpImageClassifierPtr classifier;
  ASSERT_EQ(
      MpImageClassifierCreate(&options, &classifier, /* error_msg= */ nullptr),
      kMpOk);
  ASSERT_NE(classifier, nullptr);

  ImageClassifierResult result;
  EXPECT_EQ(
      MpImageClassifierClassifyImage(classifier, image.get(),
                                     /* image_processing_options */ nullptr,
                                     &result, /* error_msg= */ nullptr),
      kMpOk);
  EXPECT_EQ(result.classifications_count, 1);
  EXPECT_EQ(result.classifications[0].categories_count, 1001);
  EXPECT_EQ(std::string{result.classifications[0].categories[0].category_name},
            "cheeseburger");
  EXPECT_NEAR(result.classifications[0].categories[0].score, 0.7939f,
              kPrecision);
  MpImageClassifierCloseResult(&result);
  EXPECT_EQ(MpImageClassifierClose(classifier, /* error_msg= */ nullptr),
            kMpOk);
}

TEST(ImageClassifierTest, ImageModeTestWithRotation) {
  const ScopedMpImage image = GetImage(GetFullPath("burger_rotated.jpg"));

  const std::string model_path = GetFullPath(kModelName);
  ImageClassifierOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::IMAGE,
      /* classifier_options= */
      {/* display_names_locale= */ nullptr,
       /* max_results= */ -1,
       /* score_threshold= */ 0.0,
       /* category_allowlist= */ nullptr,
       /* category_allowlist_count= */ 0,
       /* category_denylist= */ nullptr,
       /* category_denylist_count= */ 0},
  };

  MpImageClassifierPtr classifier;
  ASSERT_EQ(
      MpImageClassifierCreate(&options, &classifier, /* error_msg= */ nullptr),
      kMpOk);
  ASSERT_NE(classifier, nullptr);

  ImageProcessingOptions image_processing_options;
  image_processing_options.has_region_of_interest = 0;
  image_processing_options.rotation_degrees = -90;

  ImageClassifierResult result;
  EXPECT_EQ(MpImageClassifierClassifyImage(classifier, image.get(),
                                           &image_processing_options, &result,
                                           /* error_msg= */ nullptr),
            kMpOk);
  EXPECT_EQ(result.classifications_count, 1);
  EXPECT_EQ(result.classifications[0].categories_count, 1001);
  EXPECT_EQ(std::string{result.classifications[0].categories[0].category_name},
            "cheeseburger");
  EXPECT_NEAR(result.classifications[0].categories[0].score, 0.7545f,
              kPrecision);
  MpImageClassifierCloseResult(&result);
  EXPECT_EQ(MpImageClassifierClose(classifier, /* error_msg= */ nullptr),
            kMpOk);
}

TEST(ImageClassifierTest, VideoModeTest) {
  const ScopedMpImage image = GetImage(GetFullPath("burger.jpg"));

  const std::string model_path = GetFullPath(kModelName);
  ImageClassifierOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::VIDEO,
      /* classifier_options= */
      {/* display_names_locale= */ nullptr,
       /* max_results= */ 3,
       /* score_threshold= */ 0.0,
       /* category_allowlist= */ nullptr,
       /* category_allowlist_count= */ 0,
       /* category_denylist= */ nullptr,
       /* category_denylist_count= */ 0},
      /* result_callback= */ nullptr,
  };

  MpImageClassifierPtr classifier;
  ASSERT_EQ(
      MpImageClassifierCreate(&options, &classifier, /* error_msg= */ nullptr),
      kMpOk);
  ASSERT_NE(classifier, nullptr);

  for (int i = 0; i < kIterations; ++i) {
    ImageClassifierResult result;
    EXPECT_EQ(MpImageClassifierClassifyForVideo(
                  classifier, image.get(),
                  /* image_processing_options */ nullptr, i, &result,
                  /* error_msg= */ nullptr),
              kMpOk);
    EXPECT_EQ(result.classifications_count, 1);
    EXPECT_EQ(result.classifications[0].categories_count, 3);
    EXPECT_EQ(
        std::string{result.classifications[0].categories[0].category_name},
        "cheeseburger");
    EXPECT_NEAR(result.classifications[0].categories[0].score, 0.7939f,
                kPrecision);
    MpImageClassifierCloseResult(&result);
  }
  EXPECT_EQ(MpImageClassifierClose(classifier, /* error_msg= */ nullptr),
            kMpOk);
}

// A structure to support LiveStreamModeTest below. This structure holds a
// static method `Fn` for a callback function of C API. A `static` qualifier
// allows to take an address of the method to follow API style. Another static
// struct member is `last_timestamp` that is used to verify that current
// timestamp is greater than the previous one.
struct LiveStreamModeCallback {
  static int64_t last_timestamp;
  static absl::BlockingCounter* blocking_counter;
  static void Fn(MpStatus status,
                 const ImageClassifierResult* classifier_result,
                 const MpImagePtr image, int64_t timestamp) {
    ASSERT_EQ(status, kMpOk);
    ASSERT_NE(classifier_result, nullptr);
    EXPECT_EQ(
        std::string{
            classifier_result->classifications[0].categories[0].category_name},
        "cheeseburger");
    EXPECT_NEAR(classifier_result->classifications[0].categories[0].score,
                0.7939f, kPrecision);
    EXPECT_GT(MpImageGetWidth(image), 0);
    EXPECT_GT(MpImageGetHeight(image), 0);
    EXPECT_GT(timestamp, last_timestamp);
    last_timestamp++;

    if (blocking_counter) {
      blocking_counter->DecrementCount();
    }
  }
};
int64_t LiveStreamModeCallback::last_timestamp = -1;
absl::BlockingCounter* LiveStreamModeCallback::blocking_counter = nullptr;

TEST(ImageClassifierTest, LiveStreamModeTest) {
  const ScopedMpImage image = GetImage(GetFullPath("burger.jpg"));

  const std::string model_path = GetFullPath(kModelName);

  ImageClassifierOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::LIVE_STREAM,
      /* classifier_options= */
      {/* display_names_locale= */ nullptr,
       /* max_results= */ 3,
       /* score_threshold= */ 0.0,
       /* category_allowlist= */ nullptr,
       /* category_allowlist_count= */ 0,
       /* category_denylist= */ nullptr,
       /* category_denylist_count= */ 0},
      /* result_callback= */ LiveStreamModeCallback::Fn,
  };

  MpImageClassifierPtr classifier;
  ASSERT_EQ(
      MpImageClassifierCreate(&options, &classifier, /* error_msg= */ nullptr),
      kMpOk);
  ASSERT_NE(classifier, nullptr);

  absl::BlockingCounter counter(kIterations);
  LiveStreamModeCallback::blocking_counter = &counter;

  for (int i = 0; i < kIterations; ++i) {
    EXPECT_EQ(
        MpImageClassifierClassifyAsync(classifier, image.get(),
                                       /* image_processing_options */ nullptr,
                                       i, /* error_msg= */ nullptr),
        kMpOk);
    // Short sleep so that MediaPipe does not drop frames.
    absl::SleepFor(absl::Milliseconds(kSleepBetweenFramesMilliseconds));
  }

  // Wait for all callbacks to be invoked.
  counter.Wait();
  LiveStreamModeCallback::blocking_counter = nullptr;

  EXPECT_EQ(MpImageClassifierClose(classifier, /* error_msg= */ nullptr),
            kMpOk);

  // Due to the flow limiter, the total of outputs might be smaller than the
  // number of iterations.
  EXPECT_LE(LiveStreamModeCallback::last_timestamp, kIterations);
  EXPECT_GT(LiveStreamModeCallback::last_timestamp, 0);
}

TEST(ImageClassifierTest, InvalidArgumentHandling) {
  // It is an error to set neither the asset buffer nor the path.
  ImageClassifierOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ nullptr},
      /* classifier_options= */ {},
  };

  char* error_msg = nullptr;
  MpImageClassifierPtr classifier = nullptr;
  MpStatus status = MpImageClassifierCreate(&options, &classifier, &error_msg);
  EXPECT_EQ(status, kMpInvalidArgument);
  EXPECT_EQ(classifier, nullptr);
  EXPECT_THAT(error_msg,
              testing::HasSubstr("ExternalFile must specify at least one"));
  free(error_msg);
}

TEST(ImageClassifierTest, FailedClassificationHandling) {
  const std::string model_path = GetFullPath(kModelName);
  ImageClassifierOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::IMAGE,
      /* classifier_options= */
      {/* display_names_locale= */ nullptr,
       /* max_results= */ -1,
       /* score_threshold= */ 0.0,
       /* category_allowlist= */ nullptr,
       /* category_allowlist_count= */ 0,
       /* category_denylist= */ nullptr,
       /* category_denylist_count= */ 0},
  };

  MpImageClassifierPtr classifier;
  ASSERT_EQ(
      MpImageClassifierCreate(&options, &classifier, /* error_msg= */ nullptr),
      kMpOk);
  ASSERT_NE(classifier, nullptr);

  const ScopedMpImage image = CreateEmptyGpuMpImage();
  char* error_msg = nullptr;
  ImageClassifierResult result;
  MpStatus status = MpImageClassifierClassifyImage(
      classifier, image.get(),
      /* image_processing_options */ nullptr, &result, &error_msg);
  EXPECT_EQ(status, kMpInvalidArgument);
  EXPECT_THAT(error_msg, testing::HasSubstr(
                             "GPU input images are currently not supported."));
  free(error_msg);
  EXPECT_EQ(MpImageClassifierClose(classifier, /* error_msg= */ nullptr),
            kMpOk);
}

}  // namespace
