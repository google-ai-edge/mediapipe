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

#include "mediapipe/tasks/c/vision/object_detector/object_detector.h"

#include <cstdint>
#include <cstdlib>
#include <string>

#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/category.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace {

using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::vision::DecodeImageFromFile;
using testing::HasSubstr;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kImageFile[] = "cats_and_dogs.jpg";
constexpr char kModelName[] =
    "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.tflite";
constexpr float kPrecision = 1e-4;
constexpr int kIterations = 100;

std::string GetFullPath(absl::string_view file_name) {
  return JoinPath("./", kTestDataDirectory, file_name);
}

TEST(ObjectDetectorTest, ImageModeTest) {
  const auto image = DecodeImageFromFile(GetFullPath(kImageFile));
  ASSERT_TRUE(image.ok());

  const std::string model_path = GetFullPath(kModelName);
  ObjectDetectorOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::IMAGE,
      /* display_names_locale= */ nullptr,
      /* max_results= */ -1,
      /* score_threshold= */ 0.0,
      /* category_allowlist= */ nullptr,
      /* category_allowlist_count= */ 0,
      /* category_denylist= */ nullptr,
      /* category_denylist_count= */ 0,
  };

  void* detector = object_detector_create(&options, /* error_msg */ nullptr);
  EXPECT_NE(detector, nullptr);

  const auto& image_frame = image->GetImageFrameSharedPtr();
  const MpImage mp_image = {
      .type = MpImage::IMAGE_FRAME,
      .image_frame = {.format = static_cast<ImageFormat>(image_frame->Format()),
                      .image_buffer = image_frame->PixelData(),
                      .width = image_frame->Width(),
                      .height = image_frame->Height()}};

  ObjectDetectorResult result;
  object_detector_detect_image(detector, &mp_image, &result,
                               /* error_msg */ nullptr);
  EXPECT_EQ(result.detections_count, 10);
  EXPECT_EQ(result.detections[0].categories_count, 1);
  EXPECT_EQ(std::string{result.detections[0].categories[0].category_name},
            "cat");
  EXPECT_NEAR(result.detections[0].categories[0].score, 0.6992f, kPrecision);
  object_detector_close_result(&result);
  object_detector_close(detector, /* error_msg */ nullptr);
}

TEST(ObjectDetectorTest, VideoModeTest) {
  const auto image = DecodeImageFromFile(GetFullPath(kImageFile));
  ASSERT_TRUE(image.ok());

  const std::string model_path = GetFullPath(kModelName);
  ObjectDetectorOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::VIDEO,
      /* display_names_locale= */ nullptr,
      /* max_results= */ 3,
      /* score_threshold= */ 0.0,
      /* category_allowlist= */ nullptr,
      /* category_allowlist_count= */ 0,
      /* category_denylist= */ nullptr,
      /* category_denylist_count= */ 0,
  };

  void* detector = object_detector_create(&options, /* error_msg */ nullptr);
  EXPECT_NE(detector, nullptr);

  const auto& image_frame = image->GetImageFrameSharedPtr();
  const MpImage mp_image = {
      .type = MpImage::IMAGE_FRAME,
      .image_frame = {.format = static_cast<ImageFormat>(image_frame->Format()),
                      .image_buffer = image_frame->PixelData(),
                      .width = image_frame->Width(),
                      .height = image_frame->Height()}};

  for (int i = 0; i < kIterations; ++i) {
    ObjectDetectorResult result;
    object_detector_detect_for_video(detector, &mp_image, i, &result,
                                     /* error_msg */ nullptr);
    EXPECT_EQ(result.detections_count, 3);
    EXPECT_EQ(result.detections[0].categories_count, 1);
    EXPECT_EQ(std::string{result.detections[0].categories[0].category_name},
              "cat");
    EXPECT_NEAR(result.detections[0].categories[0].score, 0.6992f, kPrecision);
    object_detector_close_result(&result);
  }
  object_detector_close(detector, /* error_msg */ nullptr);
}

// A structure to support LiveStreamModeTest below. This structure holds a
// static method `Fn` for a callback function of C API. A `static` qualifier
// allows to take an address of the method to follow API style. Another static
// struct member is `last_timestamp` that is used to verify that current
// timestamp is greater than the previous one.
struct LiveStreamModeCallback {
  static int64_t last_timestamp;
  static void Fn(const ObjectDetectorResult* detector_result,
                 const MpImage& image, int64_t timestamp, char* error_msg) {
    ASSERT_NE(detector_result, nullptr);
    ASSERT_EQ(error_msg, nullptr);
    EXPECT_EQ(detector_result->detections_count, 3);
    EXPECT_EQ(detector_result->detections[0].categories_count, 1);
    EXPECT_EQ(
        std::string{detector_result->detections[0].categories[0].category_name},
        "cat");
    EXPECT_NEAR(detector_result->detections[0].categories[0].score, 0.6992f,
                kPrecision);
    EXPECT_GT(image.image_frame.width, 0);
    EXPECT_GT(image.image_frame.height, 0);
    EXPECT_GT(timestamp, last_timestamp);
    last_timestamp++;
  }
};
int64_t LiveStreamModeCallback::last_timestamp = -1;

TEST(ObjectDetectorTest, LiveStreamModeTest) {
  const auto image = DecodeImageFromFile(GetFullPath(kImageFile));
  ASSERT_TRUE(image.ok());

  const std::string model_path = GetFullPath(kModelName);

  ObjectDetectorOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::LIVE_STREAM,
      /* display_names_locale= */ nullptr,
      /* max_results= */ 3,
      /* score_threshold= */ 0.0,
      /* category_allowlist= */ nullptr,
      /* category_allowlist_count= */ 0,
      /* category_denylist= */ nullptr,
      /* category_denylist_count= */ 0,
      /* result_callback= */ LiveStreamModeCallback::Fn,
  };

  void* detector = object_detector_create(&options, /* error_msg */
                                          nullptr);
  EXPECT_NE(detector, nullptr);

  const auto& image_frame = image->GetImageFrameSharedPtr();
  const MpImage mp_image = {
      .type = MpImage::IMAGE_FRAME,
      .image_frame = {.format = static_cast<ImageFormat>(image_frame->Format()),
                      .image_buffer = image_frame->PixelData(),
                      .width = image_frame->Width(),
                      .height = image_frame->Height()}};

  for (int i = 0; i < kIterations; ++i) {
    EXPECT_GE(object_detector_detect_async(detector, &mp_image, i,
                                           /* error_msg */ nullptr),
              0);
  }
  object_detector_close(detector, /* error_msg */ nullptr);

  // Due to the flow limiter, the total of outputs might be smaller than the
  // number of iterations.
  EXPECT_LE(LiveStreamModeCallback::last_timestamp, kIterations);
  EXPECT_GT(LiveStreamModeCallback::last_timestamp, 0);
}

TEST(ObjectDetectorTest, InvalidArgumentHandling) {
  // It is an error to set neither the asset buffer nor the path.
  ObjectDetectorOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ nullptr},
  };

  char* error_msg;
  void* detector = object_detector_create(&options, &error_msg);
  EXPECT_EQ(detector, nullptr);

  EXPECT_THAT(error_msg, HasSubstr("ExternalFile must specify"));

  free(error_msg);
}

TEST(ObjectDetectorTest, FailedDetectionHandling) {
  const std::string model_path = GetFullPath(kModelName);
  ObjectDetectorOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::IMAGE,
      /* display_names_locale= */ nullptr,
      /* max_results= */ -1,
      /* score_threshold= */ 0.0,
      /* category_allowlist= */ nullptr,
      /* category_allowlist_count= */ 0,
      /* category_denylist= */ nullptr,
      /* category_denylist_count= */ 0,
  };

  void* detector = object_detector_create(&options, /* error_msg */
                                          nullptr);
  EXPECT_NE(detector, nullptr);

  const MpImage mp_image = {.type = MpImage::GPU_BUFFER, .gpu_buffer = {}};
  ObjectDetectorResult result;
  char* error_msg;
  object_detector_detect_image(detector, &mp_image, &result, &error_msg);
  EXPECT_THAT(error_msg, HasSubstr("GPU Buffer not supported yet"));
  free(error_msg);
  object_detector_close(detector, /* error_msg */ nullptr);
}

}  // namespace
