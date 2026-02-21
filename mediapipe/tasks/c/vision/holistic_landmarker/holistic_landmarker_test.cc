/* Copyright 2026 The MediaPipe Authors.

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

#include "mediapipe/tasks/c/vision/holistic_landmarker/holistic_landmarker.h"

#include <cstdint>
#include <string>

#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "file/base/helpers.h"
#include "file/base/options.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/category.h"
#include "mediapipe/tasks/c/components/containers/landmark.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_frame_util.h"
#include "mediapipe/tasks/c/vision/core/image_test_util.h"
#include "mediapipe/tasks/c/vision/holistic_landmarker/holistic_landmarker_result.h"
#include "mediapipe/tasks/cc/vision/holistic_landmarker/proto/holistic_result.pb.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"

using ::file::Defaults;
using ::file::GetTextProto;
using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::vision::core::GetImage;
using HolisticResultProto =
    ::mediapipe::tasks::vision::holistic_landmarker::proto::HolisticResult;

constexpr char kTestDataDirectory[] = "mediapipe/tasks/testdata/vision/";
constexpr char kModelFile[] = "holistic_landmarker.task";
constexpr char kTestImageFile[] = "male_full_height_hands.jpg";
constexpr int kMicroSecondsPerMilliSecond = 1000;
constexpr char kHolisticResultProto[] =
    "male_full_height_hands_result_cpu.pbtxt";
constexpr float kLandmarksAbsMargin = 0.03;
constexpr float kBlendshapesAbsMargin = 0.3f;
constexpr int kIterations = 5;
constexpr int kSleepBetweenFramesMilliseconds = 100;
constexpr int kExpectedPoseWorldLandmarksCount = 33;
constexpr int kExpectedHandWorldLandmarksCount = 21;
constexpr int kExpectedSegmentationMaskWidth = 638;
constexpr int kExpectedSegmentationMaskHeight = 1000;

std::string GetFullPath(absl::string_view file_name) {
  return JoinPath("./", kTestDataDirectory, file_name);
}

HolisticResultProto GetExpectedHolisticResult(absl::string_view result_file) {
  HolisticResultProto result;
  MP_EXPECT_OK(GetTextProto(GetFullPath(result_file), &result, Defaults()));
  return result;
}

void AssertFaceBlendshapes(
    const Categories& face_blendshapes,
    const mediapipe::ClassificationList& expected_face_blendshapes,
    float margin = kBlendshapesAbsMargin) {
  ASSERT_EQ(face_blendshapes.categories_count,
            expected_face_blendshapes.classification_size());
  for (int i = 0; i < face_blendshapes.categories_count; ++i) {
    EXPECT_EQ(face_blendshapes.categories[i].index,
              expected_face_blendshapes.classification(i).index());
    EXPECT_NEAR(face_blendshapes.categories[i].score,
                expected_face_blendshapes.classification(i).score(), margin);
    EXPECT_EQ(std::string(face_blendshapes.categories[i].category_name),
              expected_face_blendshapes.classification(i).label());
  }
}

void AssertLandmarks(
    const NormalizedLandmarks& landmarks,
    const mediapipe::NormalizedLandmarkList& expected_landmark_list,
    float margin = kLandmarksAbsMargin) {
  ASSERT_EQ(landmarks.landmarks_count, expected_landmark_list.landmark_size());
  for (int i = 0; i < landmarks.landmarks_count; ++i) {
    EXPECT_NEAR(landmarks.landmarks[i].x,
                expected_landmark_list.landmark(i).x(), margin);
    EXPECT_NEAR(landmarks.landmarks[i].y,
                expected_landmark_list.landmark(i).y(), margin);
  }
}

void AssertHolisticLandmarkerResult(
    const HolisticLandmarkerResult* result,
    const HolisticResultProto& expected_result_proto) {
  EXPECT_TRUE(result);

  // Holistic landmarks
  AssertLandmarks(result->face_landmarks,
                  expected_result_proto.face_landmarks());

  // Pose landmarks
  AssertLandmarks(result->pose_landmarks,
                  expected_result_proto.pose_landmarks());

  // Hand landmarks
  AssertLandmarks(result->left_hand_landmarks,
                  expected_result_proto.left_hand_landmarks());
  AssertLandmarks(result->right_hand_landmarks,
                  expected_result_proto.right_hand_landmarks());

  EXPECT_EQ(result->pose_world_landmarks.landmarks_count,
            kExpectedPoseWorldLandmarksCount);
  EXPECT_EQ(result->left_hand_world_landmarks.landmarks_count,
            kExpectedHandWorldLandmarksCount);
  EXPECT_EQ(result->right_hand_world_landmarks.landmarks_count,
            kExpectedHandWorldLandmarksCount);
  AssertFaceBlendshapes(result->face_blendshapes,
                        expected_result_proto.face_blendshapes());
  EXPECT_TRUE(result->pose_segmentation_mask);
  EXPECT_EQ(
      result->pose_segmentation_mask->image.GetImageFrameSharedPtr()->Width(),
      kExpectedSegmentationMaskWidth);
  EXPECT_EQ(
      result->pose_segmentation_mask->image.GetImageFrameSharedPtr()->Height(),
      kExpectedSegmentationMaskHeight);
}

TEST(HolisticLandmarkerTest, ImageModeSucceeds) {
  const auto expected_result = GetExpectedHolisticResult(kHolisticResultProto);
  const auto image = GetImage(GetFullPath(kTestImageFile));
  const std::string model_path = GetFullPath(kModelFile);
  HolisticLandmarkerOptions options;
  options.base_options = {.model_asset_path = model_path.c_str()};
  options.running_mode = RunningMode::IMAGE;
  options.output_face_blendshapes = true;
  options.output_pose_segmentation_masks = true;

  MpHolisticLandmarkerPtr landmarker_ptr;
  char* error_msg = nullptr;
  MpStatus status =
      MpHolisticLandmarkerCreate(&options, &landmarker_ptr, &error_msg);

  ASSERT_EQ(status, kMpOk);
  ASSERT_EQ(error_msg, nullptr);

  HolisticLandmarkerResult result;
  status = MpHolisticLandmarkerDetectImage(landmarker_ptr, image.get(), nullptr,
                                           &result, &error_msg);
  ASSERT_EQ(status, kMpOk);
  ASSERT_EQ(error_msg, nullptr);
  AssertHolisticLandmarkerResult(&result, expected_result);
  MpHolisticLandmarkerCloseResult(&result);
  status = MpHolisticLandmarkerClose(landmarker_ptr, &error_msg);
  EXPECT_EQ(status, kMpOk);
  EXPECT_EQ(error_msg, nullptr);
}

TEST(HolisticLandmarkerTest, VideoModeSucceeds) {
  const auto expected_result = GetExpectedHolisticResult(kHolisticResultProto);
  const auto image = GetImage(GetFullPath(kTestImageFile));
  const std::string model_path = GetFullPath(kModelFile);
  HolisticLandmarkerOptions options;
  options.base_options = {.model_asset_path = model_path.c_str()};
  options.running_mode = RunningMode::VIDEO;
  options.output_face_blendshapes = true;
  options.output_pose_segmentation_masks = true;

  MpHolisticLandmarkerPtr landmarker_ptr;
  char* error_msg = nullptr;
  MpStatus status =
      MpHolisticLandmarkerCreate(&options, &landmarker_ptr, &error_msg);

  ASSERT_EQ(status, kMpOk);
  ASSERT_EQ(error_msg, nullptr);

  int iterations = 3;
  for (int i = 0; i < iterations; ++i) {
    HolisticLandmarkerResult result;
    status = MpHolisticLandmarkerDetectForVideo(
        landmarker_ptr, image.get(), nullptr, i * kMicroSecondsPerMilliSecond,
        &result, &error_msg);
    ASSERT_EQ(status, kMpOk);
    ASSERT_EQ(error_msg, nullptr);
    AssertHolisticLandmarkerResult(&result, expected_result);
    MpHolisticLandmarkerCloseResult(&result);
  }
  status = MpHolisticLandmarkerClose(landmarker_ptr, &error_msg);
  EXPECT_EQ(status, kMpOk);
  EXPECT_EQ(error_msg, nullptr);
}

TEST(HolisticLandmarkerTest, ReturnsEmptyResultsWithHighThresholds) {
  const auto image = GetImage(GetFullPath(kTestImageFile));
  const std::string model_path = GetFullPath(kModelFile);
  HolisticLandmarkerOptions options;
  options.base_options = {.model_asset_path = model_path.c_str()};
  options.running_mode = RunningMode::IMAGE;
  options.output_face_blendshapes = true;
  options.output_pose_segmentation_masks = true;
  options.min_face_detection_confidence = 1.0f;
  options.min_face_presence_confidence = 1.0f;
  options.min_hand_landmarks_confidence = 1.0f;
  options.min_pose_detection_confidence = 1.0f;
  options.min_pose_presence_confidence = 1.0f;
  options.min_face_suppression_threshold = 1.0f;
  options.min_pose_suppression_threshold = 1.0f;

  MpHolisticLandmarkerPtr landmarker_ptr;
  char* error_msg = nullptr;
  MpStatus status =
      MpHolisticLandmarkerCreate(&options, &landmarker_ptr, &error_msg);

  ASSERT_EQ(status, kMpOk);
  ASSERT_EQ(error_msg, nullptr);

  HolisticLandmarkerResult result;
  status = MpHolisticLandmarkerDetectImage(landmarker_ptr, image.get(), nullptr,
                                           &result, &error_msg);
  ASSERT_EQ(status, kMpOk);
  ASSERT_EQ(error_msg, nullptr);
  EXPECT_EQ(result.face_landmarks.landmarks_count, 0);
  EXPECT_EQ(result.pose_landmarks.landmarks_count, 0);
  EXPECT_EQ(result.pose_world_landmarks.landmarks_count, 0);
  EXPECT_EQ(result.left_hand_landmarks.landmarks_count, 0);
  EXPECT_EQ(result.right_hand_landmarks.landmarks_count, 0);
  EXPECT_EQ(result.left_hand_world_landmarks.landmarks_count, 0);
  EXPECT_EQ(result.right_hand_world_landmarks.landmarks_count, 0);
  EXPECT_EQ(result.face_blendshapes.categories_count, 0);
  EXPECT_FALSE(result.pose_segmentation_mask);

  MpHolisticLandmarkerCloseResult(&result);
  status = MpHolisticLandmarkerClose(landmarker_ptr, &error_msg);
  EXPECT_EQ(status, kMpOk);
  EXPECT_EQ(error_msg, nullptr);
}

// A structure to support LiveStreamModeTest below. This structure holds a
// static method `Fn` for a callback function of C API. A `static` qualifier
// allows to take an address of the method to follow API style. Another static
// struct member is `last_timestamp` that is used to verify that current
// timestamp is greater than the previous one.
struct LiveStreamModeCallback {
  static int64_t last_timestamp;
  static absl::BlockingCounter* blocking_counter;
  static void LiveStreamCallback(MpStatus status_code,
                                 const HolisticLandmarkerResult* result,
                                 MpImagePtr image, int64_t timestamp) {
    ASSERT_EQ(status_code, kMpOk);
    const HolisticResultProto expected_result =
        GetExpectedHolisticResult(kHolisticResultProto);
    AssertHolisticLandmarkerResult(result, expected_result);
    EXPECT_GT(timestamp, last_timestamp);
    ++last_timestamp;

    if (blocking_counter) {
      blocking_counter->DecrementCount();
    }
  }
};

int64_t LiveStreamModeCallback::last_timestamp = -1;
absl::BlockingCounter* LiveStreamModeCallback::blocking_counter = nullptr;

TEST(HolisticLandmarkerTest, LiveStreamModeSucceeds) {
  const auto image = GetImage(GetFullPath(kTestImageFile));
  const std::string model_path = GetFullPath(kModelFile);
  HolisticLandmarkerOptions options;
  options.base_options = {.model_asset_path = model_path.c_str()};
  options.running_mode = RunningMode::LIVE_STREAM;
  options.output_face_blendshapes = true;
  options.output_pose_segmentation_masks = true;
  options.result_callback = &LiveStreamModeCallback::LiveStreamCallback;

  MpHolisticLandmarkerPtr landmarker_ptr;
  char* error_msg = nullptr;
  MpStatus status =
      MpHolisticLandmarkerCreate(&options, &landmarker_ptr, &error_msg);

  ASSERT_EQ(status, kMpOk);
  ASSERT_EQ(error_msg, nullptr);

  absl::BlockingCounter counter(kIterations);
  LiveStreamModeCallback::blocking_counter = &counter;

  for (int i = 0; i < kIterations; ++i) {
    ASSERT_EQ(MpHolisticLandmarkerDetectAsync(
                  landmarker_ptr, image.get(),
                  /* image_processing_options= */ nullptr, i,
                  /* error_msg= */ nullptr),
              kMpOk);
    // Short sleep so that MediaPipe does not drop frames.
    absl::SleepFor(absl::Milliseconds(kSleepBetweenFramesMilliseconds));
  }

  // Wait for all callbacks to be invoked.
  counter.Wait();
  LiveStreamModeCallback::blocking_counter = nullptr;

  EXPECT_EQ(MpHolisticLandmarkerClose(landmarker_ptr, /* error_msg= */ nullptr),
            kMpOk);

  // Due to the flow limiter, the total of outputs might be smaller than the
  // number of iterations.
  EXPECT_LE(LiveStreamModeCallback::last_timestamp, kIterations);
  EXPECT_GT(LiveStreamModeCallback::last_timestamp, 0);
}
