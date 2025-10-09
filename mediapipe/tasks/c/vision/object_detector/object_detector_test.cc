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
#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/category.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_frame_util.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"

namespace {

using ::mediapipe::file::JoinPath;
using testing::HasSubstr;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kImageFile[] = "cats_and_dogs.jpg";
constexpr char kImageRotatedFile[] = "cats_and_dogs_rotated.jpg";
constexpr char kModelName[] =
    "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.tflite";
constexpr float kPrecision = 1e-4;
constexpr int kIterations = 100;

std::string GetFullPath(absl::string_view file_name) {
  return JoinPath("./", kTestDataDirectory, file_name);
}

struct MpImageDeleter {
  void operator()(MpImagePtr image) const {
    if (image) {
      MpImageFree(image);
    }
  }
};
using ScopedMpImage = std::unique_ptr<MpImageInternal, MpImageDeleter>;

ScopedMpImage GetImage(const std::string& file_name) {
  MpImagePtr image_ptr = nullptr;
  MpStatus status = MpImageCreateFromFile(file_name.c_str(), &image_ptr);
  EXPECT_EQ(status, kMpOk);
  EXPECT_NE(image_ptr, nullptr);
  return ScopedMpImage(image_ptr);
}

TEST(ObjectDetectorTest, ImageModeTest) {
  const auto image = GetImage(GetFullPath(kImageFile));

  const std::string model_path = GetFullPath(kModelName);
  ObjectDetectorOptions options = {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = model_path.c_str()},
      .running_mode = RunningMode::IMAGE,
      .display_names_locale = nullptr,
      .max_results = -1,
      .score_threshold = 0.0,
      .category_allowlist = nullptr,
      .category_allowlist_count = 0,
      .category_denylist = nullptr,
      .category_denylist_count = 0,
  };

  MpObjectDetectorPtr detector =
      object_detector_create(&options, /* error_msg */ nullptr);
  EXPECT_NE(detector, nullptr);

  ObjectDetectorResult result;
  int error_code = object_detector_detect_image(detector, image.get(), &result,
                                                /* error_msg */ nullptr);
  EXPECT_EQ(error_code, 0);
  EXPECT_EQ(result.detections_count, 10);
  EXPECT_EQ(result.detections[0].categories_count, 1);
  EXPECT_EQ(std::string{result.detections[0].categories[0].category_name},
            "cat");
  EXPECT_NEAR(result.detections[0].categories[0].score, 0.6992f, kPrecision);
  object_detector_close_result(&result);
  object_detector_close(detector, /* error_msg */ nullptr);
}

TEST(ObjectDetectorTest, ImageModeWithOptionsTest) {
  const auto image = GetImage(GetFullPath(kImageRotatedFile));

  const std::string model_path = GetFullPath(kModelName);
  ObjectDetectorOptions options = {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = model_path.c_str()},
      .running_mode = RunningMode::IMAGE,
      .display_names_locale = nullptr,
      .max_results = -1,
      .score_threshold = 0.0,
      .category_allowlist = nullptr,
      .category_allowlist_count = 0,
      .category_denylist = nullptr,
      .category_denylist_count = 0,
  };

  MpObjectDetectorPtr detector =
      object_detector_create(&options, /* error_msg */ nullptr);
  EXPECT_NE(detector, nullptr);

  ImageProcessingOptions image_processing_options;
  image_processing_options.has_region_of_interest = 0;
  image_processing_options.rotation_degrees = -90;

  ObjectDetectorResult result;
  int error_code = object_detector_detect_image_with_options(
      detector, image.get(), &image_processing_options, &result,
      /* error_msg */ nullptr);
  EXPECT_EQ(error_code, 0);
  EXPECT_EQ(result.detections_count, 10);
  EXPECT_EQ(result.detections[0].categories_count, 1);
  EXPECT_EQ(std::string{result.detections[0].categories[0].category_name},
            "cat");
  EXPECT_NEAR(result.detections[0].categories[0].score, 0.6992f, kPrecision);
  object_detector_close_result(&result);
  object_detector_close(detector, /* error_msg */ nullptr);
}

TEST(ObjectDetectorTest, VideoModeTest) {
  const auto image = GetImage(GetFullPath(kImageFile));

  const std::string model_path = GetFullPath(kModelName);
  ObjectDetectorOptions options = {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = model_path.c_str()},
      .running_mode = RunningMode::VIDEO,
      .display_names_locale = nullptr,
      .max_results = 3,
      .score_threshold = 0.0,
      .category_allowlist = nullptr,
      .category_allowlist_count = 0,
      .category_denylist = nullptr,
      .category_denylist_count = 0,
  };

  MpObjectDetectorPtr detector =
      object_detector_create(&options, /* error_msg */ nullptr);
  EXPECT_NE(detector, nullptr);

  for (int i = 0; i < kIterations; ++i) {
    ObjectDetectorResult result;
    int error_code =
        object_detector_detect_for_video(detector, image.get(), i, &result,
                                         /* error_msg */ nullptr);
    EXPECT_EQ(error_code, 0);
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
  static void Fn(ObjectDetectorResult* detector_result, MpImagePtr image,
                 int64_t timestamp, char* error_msg) {
    ASSERT_NE(detector_result, nullptr);
    ASSERT_EQ(error_msg, nullptr);
    EXPECT_EQ(detector_result->detections_count, 3);
    EXPECT_EQ(detector_result->detections[0].categories_count, 1);
    EXPECT_EQ(
        std::string{detector_result->detections[0].categories[0].category_name},
        "cat");
    EXPECT_NEAR(detector_result->detections[0].categories[0].score, 0.6992f,
                kPrecision);
    EXPECT_GT(MpImageGetWidth(image), 0);
    EXPECT_GT(MpImageGetHeight(image), 0);
    EXPECT_GT(timestamp, last_timestamp);
    last_timestamp++;

    object_detector_close_result(detector_result);
  }
};
int64_t LiveStreamModeCallback::last_timestamp = -1;

// TODO: Await the callbacks and re-enable test
TEST(ObjectDetectorTest, DISABLED_LiveStreamModeTest) {
  const auto image = GetImage(GetFullPath(kImageFile));

  const std::string model_path = GetFullPath(kModelName);

  ObjectDetectorOptions options = {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = model_path.c_str()},
      .running_mode = RunningMode::LIVE_STREAM,
      .display_names_locale = nullptr,
      .max_results = 3,
      .score_threshold = 0.0,
      .category_allowlist = nullptr,
      .category_allowlist_count = 0,
      .category_denylist = nullptr,
      .category_denylist_count = 0,
      .result_callback = LiveStreamModeCallback::Fn,
  };

  MpObjectDetectorPtr detector =
      object_detector_create(&options, /* error_msg */
                             nullptr);
  EXPECT_NE(detector, nullptr);

  for (int i = 0; i < kIterations; ++i) {
    EXPECT_GE(object_detector_detect_async(detector, image.get(), i,
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
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = nullptr},
  };

  char* error_msg;
  MpObjectDetectorPtr detector = object_detector_create(&options, &error_msg);
  EXPECT_EQ(detector, nullptr);

  EXPECT_THAT(error_msg, HasSubstr("ExternalFile must specify"));

  free(error_msg);
}

}  // namespace
