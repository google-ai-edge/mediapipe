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
#include <string>

#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/detection_result.h"
#include "mediapipe/tasks/c/components/containers/keypoint.h"
#include "mediapipe/tasks/c/components/containers/rect.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace {

using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::vision::DecodeImageFromFile;
using testing::HasSubstr;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kModelName[] = "face_detection_short_range.tflite";
constexpr char kImageFile[] = "portrait.jpg";
constexpr int kPixelDiffTolerance = 5;
constexpr float kKeypointErrorThreshold = 0.02;
constexpr int kIterations = 100;
constexpr int kKeypointCount = 2;

std::string GetFullPath(absl::string_view file_name) {
  return JoinPath("./", kTestDataDirectory, file_name);
}

void AssertFaceDetectorResult(const FaceDetectorResult* result,
                              const int pixel_diff_tolerance,
                              const float keypoint_error_threshold) {
  NormalizedKeypoint keypoints[] = {
      {0.4432f, 0.1792f, nullptr, 0},
      {0.5609f, 0.1800f, nullptr, 0},
  };

  MPRect bounding_box = {283, 115, 349, 517};

  // Create a single detection with expected results
  Detection expected_detection = {/* categories= */ nullptr,
                                  /* categories_count= */ 0,
                                  /* bounding_box= */ bounding_box,
                                  /* keypoints= */ keypoints,
                                  /* keypoints_count= */ kKeypointCount};

  // Create DetectionResult with one Detection
  DetectionResult expected_results = {/* detections= */ &expected_detection,
                                      /* detections_count= */ 1};

  EXPECT_EQ(result->detections_count, expected_results.detections_count);
  for (int i = 0; i < result->detections_count; i++) {
    const auto& actual_bbox = result->detections[i].bounding_box;
    const auto& expected_bbox = expected_results.detections[i].bounding_box;
    EXPECT_NEAR(actual_bbox.bottom, expected_bbox.bottom, pixel_diff_tolerance);
    EXPECT_NEAR(actual_bbox.right, expected_bbox.right, pixel_diff_tolerance);
    EXPECT_NEAR(actual_bbox.top, expected_bbox.top, pixel_diff_tolerance);
    EXPECT_NEAR(actual_bbox.left, expected_bbox.left, pixel_diff_tolerance);
    EXPECT_EQ(result->detections[i].keypoints_count, 6);
    for (int j = 0; j < expected_results.detections[i].keypoints_count; j++) {
      EXPECT_NEAR(result->detections[i].keypoints[j].x,
                  expected_results.detections[i].keypoints[j].x,
                  keypoint_error_threshold);
      EXPECT_NEAR(result->detections[i].keypoints[j].y,
                  expected_results.detections[i].keypoints[j].y,
                  keypoint_error_threshold);
    }
  }
}

TEST(FaceDetectorTest, ImageModeTest) {
  const auto image = DecodeImageFromFile(GetFullPath(kImageFile));
  ASSERT_TRUE(image.ok());

  const std::string model_path = GetFullPath(kModelName);
  FaceDetectorOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::IMAGE,
      /* min_detection_confidence= */ 0.5,
      /* min_suppression_threshold= */ 0.5,
  };

  void* detector = face_detector_create(&options, /* error_msg */ nullptr);
  EXPECT_NE(detector, nullptr);

  const auto& image_frame = image->GetImageFrameSharedPtr();
  MpImage mp_image = {
      .type = MpImage::IMAGE_FRAME,
      .image_frame = {.format = static_cast<ImageFormat>(image_frame->Format()),
                      .image_buffer = image_frame->PixelData(),
                      .width = image_frame->Width(),
                      .height = image_frame->Height()}};

  FaceDetectorResult result;
  face_detector_detect_image(detector, &mp_image, &result,
                             /* error_msg */ nullptr);
  AssertFaceDetectorResult(&result, kPixelDiffTolerance,
                           kKeypointErrorThreshold);
  face_detector_close_result(&result);
  face_detector_close(detector, /* error_msg */ nullptr);
}

TEST(FaceDetectorTest, VideoModeTest) {
  const auto image = DecodeImageFromFile(GetFullPath(kImageFile));
  ASSERT_TRUE(image.ok());

  const std::string model_path = GetFullPath(kModelName);
  FaceDetectorOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::VIDEO,
      /* min_detection_confidence= */ 0.5,
      /* min_suppression_threshold= */ 0.5,
  };

  void* detector = face_detector_create(&options,
                                        /* error_msg */ nullptr);
  EXPECT_NE(detector, nullptr);

  const auto& image_frame = image->GetImageFrameSharedPtr();
  MpImage mp_image = {
      .type = MpImage::IMAGE_FRAME,
      .image_frame = {.format = static_cast<ImageFormat>(image_frame->Format()),
                      .image_buffer = image_frame->PixelData(),
                      .width = image_frame->Width(),
                      .height = image_frame->Height()}};

  for (int i = 0; i < kIterations; ++i) {
    FaceDetectorResult result;
    face_detector_detect_for_video(detector, &mp_image, i, &result,
                                   /* error_msg */ nullptr);

    AssertFaceDetectorResult(&result, kPixelDiffTolerance,
                             kKeypointErrorThreshold);
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
  static void Fn(FaceDetectorResult* detector_result, const MpImage* image,
                 int64_t timestamp, char* error_msg) {
    ASSERT_NE(detector_result, nullptr);
    ASSERT_EQ(error_msg, nullptr);
    AssertFaceDetectorResult(detector_result, kPixelDiffTolerance,
                             kKeypointErrorThreshold);
    EXPECT_GT(image->image_frame.width, 0);
    EXPECT_GT(image->image_frame.height, 0);
    EXPECT_GT(timestamp, last_timestamp);
    ++last_timestamp;

    face_detector_close_result(detector_result);
  }
};
int64_t LiveStreamModeCallback::last_timestamp = -1;

// TODO: Await the callbacks and re-enable test
TEST(FaceDetectorTest, DISABLED_LiveStreamModeTest) {
  const auto image = DecodeImageFromFile(GetFullPath(kImageFile));
  ASSERT_TRUE(image.ok());

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

  void* detector = face_detector_create(&options, /* error_msg */
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
    EXPECT_GE(face_detector_detect_async(detector, &mp_image, i,
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
  void* detector = face_detector_create(&options, &error_msg);
  EXPECT_EQ(detector, nullptr);

  EXPECT_THAT(error_msg, HasSubstr("INVALID_ARGUMENT"));

  free(error_msg);
}

TEST(FaceDetectorTest, FailedRecognitionHandling) {
  const std::string model_path = GetFullPath(kModelName);
  FaceDetectorOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::IMAGE,
      /* min_detection_confidence= */ 0.5,
      /* min_suppression_threshold= */ 0.5,
  };

  void* detector = face_detector_create(&options, /* error_msg */
                                        nullptr);
  EXPECT_NE(detector, nullptr);

  const MpImage mp_image = {.type = MpImage::GPU_BUFFER, .gpu_buffer = {}};
  FaceDetectorResult result;
  char* error_msg;
  face_detector_detect_image(detector, &mp_image, &result, &error_msg);
  EXPECT_THAT(error_msg, HasSubstr("GPU Buffer not supported yet"));
  free(error_msg);
  face_detector_close(detector, /* error_msg */ nullptr);
}

}  // namespace
