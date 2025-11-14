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

#include "mediapipe/tasks/c/vision/face_detector/face_detector.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/detection_result.h"
#include "mediapipe/tasks/c/components/containers/keypoint.h"
#include "mediapipe/tasks/c/components/containers/rect.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/core/image_test_util.h"

namespace {

using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::vision::core::GetImage;
using ::mediapipe::tasks::vision::core::ScopedMpImage;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kModelName[] = "face_detection_short_range.tflite";
constexpr char kImageFile[] = "portrait.jpg";
constexpr char kImageRotatedFile[] = "portrait_rotated.jpg";
constexpr int kPixelDiffTolerance = 5;
constexpr float kKeypointErrorThreshold = 0.02;
constexpr int kIterations = 5;
constexpr int kKeypointCount = 2;
constexpr int kSleepBetweenFramesMilliseconds = 100;

// Expected results for portrait.jpg
const NormalizedKeypoint kExpectedKeypoints[] = {
    {0.4432f, 0.1792f, nullptr, 0},
    {0.5609f, 0.1800f, nullptr, 0},
};
constexpr MPRect kExpectedBoundingBox = {283, 115, 349, 517};

// Expected results for portrait_rotated.jpg
const NormalizedKeypoint kExpectedRotatedKeypoints[] = {
    {0.82075f, 0.44679f, nullptr, 0},
    {0.81965f, 0.56261f, nullptr, 0},
};
constexpr MPRect kExpectedRotatedBoundingBox = {674, 283, 519, 910};

std::string GetFullPath(absl::string_view file_name) {
  return JoinPath("./", kTestDataDirectory, file_name);
}

// Creates a Detection object with the given bounding box and keypoints.
Detection CreateExpectedDetection(const MPRect& bounding_box,
                                  const NormalizedKeypoint* keypoints,
                                  uint32_t keypoints_count) {
  return {/* categories= */ nullptr,
          /* categories_count= */ 0,
          /* bounding_box= */ bounding_box,
          /* keypoints= */ const_cast<NormalizedKeypoint*>(keypoints),
          /* keypoints_count= */ keypoints_count};
}

void AssertFaceDetectorResult(const FaceDetectorResult* result,
                              const Detection& expected_detection,
                              const int pixel_diff_tolerance,
                              const float keypoint_error_threshold) {
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->detections_count, 1);
  const auto& actual_bbox = result->detections[0].bounding_box;
  const auto& expected_bbox = expected_detection.bounding_box;
  EXPECT_NEAR(actual_bbox.bottom, expected_bbox.bottom, pixel_diff_tolerance);
  EXPECT_NEAR(actual_bbox.right, expected_bbox.right, pixel_diff_tolerance);
  EXPECT_NEAR(actual_bbox.top, expected_bbox.top, pixel_diff_tolerance);
  EXPECT_NEAR(actual_bbox.left, expected_bbox.left, pixel_diff_tolerance);
  EXPECT_EQ(result->detections[0].keypoints_count, 6);
  for (int j = 0; j < expected_detection.keypoints_count; j++) {
    EXPECT_NEAR(result->detections[0].keypoints[j].x,
                expected_detection.keypoints[j].x, keypoint_error_threshold);
    EXPECT_NEAR(result->detections[0].keypoints[j].y,
                expected_detection.keypoints[j].y, keypoint_error_threshold);
  }
}

TEST(FaceDetectorTest, ImageModeTest) {
  const auto image = GetImage(GetFullPath(kImageFile));

  const std::string model_path = GetFullPath(kModelName);
  FaceDetectorOptions options = {
      .base_options = {.model_asset_path = model_path.c_str()},
      .running_mode = RunningMode::IMAGE,
      .min_detection_confidence = 0.5,
      .min_suppression_threshold = 0.5,
  };

  MpFaceDetectorPtr detector;
  EXPECT_EQ(MpFaceDetectorCreate(&options, &detector), kMpOk);
  EXPECT_NE(detector, nullptr);

  FaceDetectorResult result;
  EXPECT_EQ(MpFaceDetectorDetectImage(detector, image.get(),
                                      /* image_processing_options */ nullptr,
                                      &result),
            kMpOk);

  Detection expected_detection = CreateExpectedDetection(
      kExpectedBoundingBox, kExpectedKeypoints, kKeypointCount);

  AssertFaceDetectorResult(&result, expected_detection, kPixelDiffTolerance,
                           kKeypointErrorThreshold);

  MpFaceDetectorCloseResult(&result);
  EXPECT_EQ(MpFaceDetectorClose(detector), kMpOk);
}

TEST(FaceDetectorTest, ImageModeWithRotationTest) {
  const auto image = GetImage(GetFullPath(kImageRotatedFile));

  const std::string model_path = GetFullPath(kModelName);
  FaceDetectorOptions options = {
      .base_options = {.model_asset_path = model_path.c_str()},
      .running_mode = RunningMode::IMAGE,
      .min_detection_confidence = 0.5,
      .min_suppression_threshold = 0.5,
  };

  MpFaceDetectorPtr detector;
  EXPECT_EQ(MpFaceDetectorCreate(&options, &detector), kMpOk);
  EXPECT_NE(detector, nullptr);

  ImageProcessingOptions image_processing_options;
  image_processing_options.has_region_of_interest = 0;
  image_processing_options.rotation_degrees = -90;

  FaceDetectorResult result;
  EXPECT_EQ(MpFaceDetectorDetectImage(detector, image.get(),
                                      &image_processing_options, &result),
            kMpOk);

  Detection expected_detection = CreateExpectedDetection(
      kExpectedRotatedBoundingBox, kExpectedRotatedKeypoints, kKeypointCount);
  AssertFaceDetectorResult(&result, expected_detection, kPixelDiffTolerance,
                           kKeypointErrorThreshold);

  MpFaceDetectorCloseResult(&result);
  EXPECT_EQ(MpFaceDetectorClose(detector), kMpOk);
}

TEST(FaceDetectorTest, VideoModeTest) {
  const auto image = GetImage(GetFullPath(kImageFile));

  const std::string model_path = GetFullPath(kModelName);
  FaceDetectorOptions options = {
      .base_options = {.model_asset_path = model_path.c_str()},
      .running_mode = RunningMode::VIDEO,
      .min_detection_confidence = 0.5,
      .min_suppression_threshold = 0.5,
  };

  MpFaceDetectorPtr detector;
  EXPECT_EQ(MpFaceDetectorCreate(&options, &detector), kMpOk);
  EXPECT_NE(detector, nullptr);

  Detection expected_detection = CreateExpectedDetection(
      kExpectedBoundingBox, kExpectedKeypoints, kKeypointCount);
  for (int i = 0; i < kIterations; ++i) {
    FaceDetectorResult result;
    EXPECT_EQ(MpFaceDetectorDetectForVideo(
                  detector, image.get(),
                  /* image_processing_options */ nullptr, i, &result),
              kMpOk);
    AssertFaceDetectorResult(&result, expected_detection, kPixelDiffTolerance,
                             kKeypointErrorThreshold);
    MpFaceDetectorCloseResult(&result);
  }
  EXPECT_EQ(MpFaceDetectorClose(detector), kMpOk);
}

// A structure to support LiveStreamModeTest below. This structure holds a
// static method `Fn` for a callback function of C API. A `static` qualifier
// allows to take an address of the method to follow API style. Another static
// struct member is `last_timestamp` that is used to verify that current
// timestamp is greater than the previous one.
struct LiveStreamModeCallback {
  static int64_t last_timestamp;
  static absl::BlockingCounter* blocking_counter;
  static void Fn(MpStatus status, const FaceDetectorResult* detector_result,
                 const MpImagePtr image, int64_t timestamp) {
    ASSERT_EQ(status, kMpOk);
    ASSERT_NE(detector_result, nullptr);
    Detection expected_detection = CreateExpectedDetection(
        kExpectedBoundingBox, kExpectedKeypoints, kKeypointCount);
    AssertFaceDetectorResult(detector_result, expected_detection,
                             kPixelDiffTolerance, kKeypointErrorThreshold);
    EXPECT_GT(MpImageGetWidth(image), 0);
    EXPECT_GT(MpImageGetHeight(image), 0);
    EXPECT_GT(timestamp, last_timestamp);
    ++last_timestamp;

    if (blocking_counter) {
      blocking_counter->DecrementCount();
    }
  }
};
int64_t LiveStreamModeCallback::last_timestamp = -1;
absl::BlockingCounter* LiveStreamModeCallback::blocking_counter = nullptr;

TEST(FaceDetectorTest, LiveStreamModeTest) {
  const auto image = GetImage(GetFullPath(kImageFile));

  const std::string model_path = GetFullPath(kModelName);

  FaceDetectorOptions options = {
      .base_options = {.model_asset_path = model_path.c_str()},
      .running_mode = RunningMode::LIVE_STREAM,
      .min_detection_confidence = 0.5,
      .min_suppression_threshold = 0.5,
      .result_callback = LiveStreamModeCallback::Fn,
  };

  MpFaceDetectorPtr detector;
  EXPECT_EQ(MpFaceDetectorCreate(&options, &detector), kMpOk);
  EXPECT_NE(detector, nullptr);

  absl::BlockingCounter counter(kIterations);
  LiveStreamModeCallback::blocking_counter = &counter;

  for (int i = 0; i < kIterations; ++i) {
    EXPECT_EQ(
        MpFaceDetectorDetectAsync(detector, image.get(),
                                  /* image_processing_options */ nullptr, i),
        kMpOk);
    // Short sleep so that MediaPipe does not drop frames.
    absl::SleepFor(absl::Milliseconds(kSleepBetweenFramesMilliseconds));
  }

  // Wait for all callbacks to be invoked.
  counter.Wait();
  LiveStreamModeCallback::blocking_counter = nullptr;

  EXPECT_EQ(MpFaceDetectorClose(detector), kMpOk);

  // Due to the flow limiter, the total of outputs might be smaller than the
  // number of iterations.
  EXPECT_LE(LiveStreamModeCallback::last_timestamp, kIterations);
  EXPECT_GT(LiveStreamModeCallback::last_timestamp, 0);
}

TEST(FaceDetectorTest, InvalidArgumentHandling) {
  // It is an error to set neither the asset buffer nor the path.
  FaceDetectorOptions options = {
      .base_options = {.model_asset_path = nullptr},
      .running_mode = RunningMode::IMAGE,
      .min_detection_confidence = 0.5,
      .min_suppression_threshold = 0.5,
  };

  MpFaceDetectorPtr detector = nullptr;
  EXPECT_EQ(MpFaceDetectorCreate(&options, &detector), kMpInvalidArgument);
  EXPECT_EQ(detector, nullptr);
}

}  // namespace
