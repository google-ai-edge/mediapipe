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
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/detection_result.h"
#include "mediapipe/tasks/c/components/containers/keypoint.h"
#include "mediapipe/tasks/c/components/containers/rect.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"

namespace {

using ::mediapipe::file::JoinPath;
using testing::HasSubstr;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kModelName[] = "face_detection_short_range.tflite";
constexpr char kImageFile[] = "portrait.jpg";
constexpr char kImageRotatedFile[] = "portrait_rotated.jpg";
constexpr int kPixelDiffTolerance = 5;
constexpr float kKeypointErrorThreshold = 0.02;
constexpr int kIterations = 100;
constexpr int kKeypointCount = 2;

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

void AssertFaceDetectorResult(const FaceDetectorResult* result,
                              const int error_code,
                              const Detection& expected_detection,
                              const int pixel_diff_tolerance,
                              const float keypoint_error_threshold) {
  EXPECT_EQ(error_code, 0);
  AssertFaceDetectorResult(result, expected_detection, pixel_diff_tolerance,
                           keypoint_error_threshold);
}

TEST(FaceDetectorTest, ImageModeTest) {
  const auto image = GetImage(GetFullPath(kImageFile));

  const std::string model_path = GetFullPath(kModelName);
  FaceDetectorOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::IMAGE,
      /* min_detection_confidence= */ 0.5,
      /* min_suppression_threshold= */ 0.5,
  };

  MpFaceDetectorPtr detector =
      face_detector_create(&options, /* error_msg */ nullptr);
  EXPECT_NE(detector, nullptr);

  FaceDetectorResult result;
  int error_code = face_detector_detect_image(detector, image.get(), &result,
                                              /* error_msg */ nullptr);

  Detection expected_detection = CreateExpectedDetection(
      kExpectedBoundingBox, kExpectedKeypoints, kKeypointCount);

  AssertFaceDetectorResult(&result, error_code, expected_detection,
                           kPixelDiffTolerance, kKeypointErrorThreshold);

  face_detector_close_result(&result);
  face_detector_close(detector, /* error_msg */ nullptr);
}

TEST(FaceDetectorTest, ImageModeWithImageProcessingOptionsTest) {
  const auto image = GetImage(GetFullPath(kImageRotatedFile));

  const std::string model_path = GetFullPath(kModelName);
  FaceDetectorOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::IMAGE,
      /* min_detection_confidence= */ 0.5,
      /* min_suppression_threshold= */ 0.5,
  };

  MpFaceDetectorPtr detector =
      face_detector_create(&options, /* error_msg */ nullptr);
  EXPECT_NE(detector, nullptr);

  ImageProcessingOptions image_processing_options;
  image_processing_options.has_region_of_interest = 0;
  image_processing_options.rotation_degrees = -90;

  FaceDetectorResult result;
  int error_code = face_detector_detect_image_with_options(
      detector, image.get(), &image_processing_options, &result,
      /* error_msg */ nullptr);

  Detection expected_detection = CreateExpectedDetection(
      kExpectedRotatedBoundingBox, kExpectedRotatedKeypoints, kKeypointCount);
  AssertFaceDetectorResult(&result, error_code, expected_detection,
                           kPixelDiffTolerance, kKeypointErrorThreshold);

  face_detector_close_result(&result);
  face_detector_close(detector, /* error_msg */ nullptr);
}

TEST(FaceDetectorTest, VideoModeTest) {
  const auto image = GetImage(GetFullPath(kImageFile));

  const std::string model_path = GetFullPath(kModelName);
  FaceDetectorOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::VIDEO,
      /* min_detection_confidence= */ 0.5,
      /* min_suppression_threshold= */ 0.5,
  };

  MpFaceDetectorPtr detector = face_detector_create(&options,
                                                    /* error_msg */ nullptr);
  EXPECT_NE(detector, nullptr);

  Detection expected_detection = CreateExpectedDetection(
      kExpectedBoundingBox, kExpectedKeypoints, kKeypointCount);
  for (int i = 0; i < kIterations; ++i) {
    FaceDetectorResult result;
    int error_code =
        face_detector_detect_for_video(detector, image.get(), i, &result,
                                       /* error_msg */ nullptr);
    AssertFaceDetectorResult(&result, error_code, expected_detection,
                             kPixelDiffTolerance, kKeypointErrorThreshold);
    face_detector_close_result(&result);
  }
  face_detector_close(detector, /* error_msg */ nullptr);
}

// A structure to support LiveStreamModeTest below. This structure holds a
// static method `Fn` for a callback function of C API. A `static` qualifier
// allows to take an address of the method to follow API style. Another static
// struct member is `last_timestamp` that is used to verify that current
// timestamp is greater than the previous one.
struct LiveStreamModeCallback {
  static int64_t last_timestamp;
  static void Fn(FaceDetectorResult* detector_result, const MpImagePtr image,
                 int64_t timestamp, char* error_msg) {
    ASSERT_NE(detector_result, nullptr);
    ASSERT_EQ(error_msg, nullptr);
    Detection expected_detection = CreateExpectedDetection(
        kExpectedBoundingBox, kExpectedKeypoints, kKeypointCount);
    AssertFaceDetectorResult(detector_result, expected_detection,
                             kPixelDiffTolerance, kKeypointErrorThreshold);
    EXPECT_GT(MpImageGetWidth(image), 0);
    EXPECT_GT(MpImageGetHeight(image), 0);
    EXPECT_GT(timestamp, last_timestamp);
    ++last_timestamp;

    face_detector_close_result(detector_result);
  }
};
int64_t LiveStreamModeCallback::last_timestamp = -1;

// TODO: Await the callbacks and re-enable test
TEST(FaceDetectorTest, DISABLED_LiveStreamModeTest) {
  const auto image = GetImage(GetFullPath(kImageFile));

  const std::string model_path = GetFullPath(kModelName);

  FaceDetectorOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::LIVE_STREAM,
      /* min_detection_confidence= */ 0.5,
      /* min_suppression_threshold= */ 0.5,
      /* result_callback= */ LiveStreamModeCallback::Fn,
  };

  MpFaceDetectorPtr detector = face_detector_create(&options, /* error_msg */
                                                    nullptr);
  EXPECT_NE(detector, nullptr);

  for (int i = 0; i < kIterations; ++i) {
    EXPECT_GE(face_detector_detect_async(detector, image.get(), i,
                                         /* error_msg */ nullptr),
              0);
  }
  face_detector_close(detector, /* error_msg */ nullptr);

  // Due to the flow limiter, the total of outputs might be smaller than the
  // number of iterations.
  EXPECT_LE(LiveStreamModeCallback::last_timestamp, kIterations);
  EXPECT_GT(LiveStreamModeCallback::last_timestamp, 0);
}

TEST(FaceDetectorTest, InvalidArgumentHandling) {
  // It is an error to set neither the asset buffer nor the path.
  FaceDetectorOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ nullptr},
      /* running_mode= */ RunningMode::IMAGE,
      /* min_detection_confidence= */ 0.5,
      /* min_suppression_threshold= */ 0.5,
  };

  char* error_msg;
  MpFaceDetectorPtr detector = face_detector_create(&options, &error_msg);
  EXPECT_EQ(detector, nullptr);

  EXPECT_THAT(error_msg, HasSubstr("INVALID_ARGUMENT"));

  free(error_msg);
}

}  // namespace
