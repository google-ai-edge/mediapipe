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

#include "mediapipe/tasks/c/vision/pose_landmarker/pose_landmarker.h"

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
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/landmark.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_frame_util.h"
#include "mediapipe/tasks/c/vision/pose_landmarker/pose_landmarker_result.h"

namespace {

using ::mediapipe::file::JoinPath;
using testing::HasSubstr;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kModelName[] = "pose_landmarker.task";
constexpr char kImageFile[] = "pose.jpg";
constexpr float kLandmarkPrecision = 1e-1;
constexpr int kIterations = 5;
constexpr int kSleepBetweenFramesMilliseconds = 100;

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

void MatchesPoseLandmarkerResult(const PoseLandmarkerResult* result,
                                 const float landmark_precision) {
  // Expects to have the same number of poses detected.
  EXPECT_EQ(result->pose_landmarks_count, 1);

  // Expects to have the same number of segmentation_masks detected.
  EXPECT_EQ(result->segmentation_masks_count, 1);
  EXPECT_EQ(MpImageGetWidth(result->segmentation_masks[0]), 1000);
  EXPECT_EQ(MpImageGetHeight(result->segmentation_masks[0]), 667);

  // Actual landmarks match expected landmarks.
  EXPECT_NEAR(result->pose_landmarks[0].landmarks[0].x, 0.4649f,
              landmark_precision);
  EXPECT_NEAR(result->pose_landmarks[0].landmarks[0].y, 0.4228f,
              landmark_precision);
  EXPECT_NEAR(result->pose_landmarks[0].landmarks[0].z, -0.1500f,
              landmark_precision);
  EXPECT_NEAR(result->pose_world_landmarks[0].landmarks[0].x, -0.0852f,
              landmark_precision);
  EXPECT_NEAR(result->pose_world_landmarks[0].landmarks[0].y, -0.6153f,
              landmark_precision);
  EXPECT_NEAR(result->pose_world_landmarks[0].landmarks[0].z, -0.1469f,
              landmark_precision);
}

TEST(PoseLandmarkerTest, ImageModeTest) {
  const auto image = GetImage(GetFullPath(kImageFile));

  const std::string model_path = GetFullPath(kModelName);
  PoseLandmarkerOptions options = {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = model_path.c_str()},
      .running_mode = RunningMode::IMAGE,
      .num_poses = 1,
      .min_pose_detection_confidence = 0.5,
      .min_pose_presence_confidence = 0.5,
      .min_tracking_confidence = 0.5,
      .output_segmentation_masks = true,
  };

  MpPoseLandmarkerPtr landmarker;
  MpStatus status = MpPoseLandmarkerCreate(&options, &landmarker);
  EXPECT_EQ(status, kMpOk);
  EXPECT_NE(landmarker, nullptr);

  PoseLandmarkerResult result;
  status = MpPoseLandmarkerDetectImage(landmarker, image.get(),
                                       /* image_processing_options= */ nullptr,
                                       &result);
  EXPECT_EQ(status, kMpOk);
  MatchesPoseLandmarkerResult(&result, kLandmarkPrecision);
  MpPoseLandmarkerCloseResult(&result);
  EXPECT_EQ(MpPoseLandmarkerClose(landmarker), kMpOk);
}

TEST(PoseLandmarkerTest, VideoModeTest) {
  const auto image = GetImage(GetFullPath(kImageFile));

  const std::string model_path = GetFullPath(kModelName);
  PoseLandmarkerOptions options = {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = model_path.c_str()},
      .running_mode = RunningMode::VIDEO,
      .num_poses = 1,
      .min_pose_detection_confidence = 0.5,
      .min_pose_presence_confidence = 0.5,
      .min_tracking_confidence = 0.5,
      .output_segmentation_masks = true,
  };

  MpPoseLandmarkerPtr landmarker;
  MpStatus status = MpPoseLandmarkerCreate(&options, &landmarker);
  EXPECT_EQ(status, kMpOk);
  EXPECT_NE(landmarker, nullptr);

  for (int i = 0; i < kIterations; ++i) {
    PoseLandmarkerResult result;
    status = MpPoseLandmarkerDetectForVideo(
        landmarker, image.get(),
        /* image_processing_options= */ nullptr, i, &result);
    EXPECT_EQ(status, kMpOk);

    MatchesPoseLandmarkerResult(&result, kLandmarkPrecision);
    MpPoseLandmarkerCloseResult(&result);
  }
  EXPECT_EQ(MpPoseLandmarkerClose(landmarker), kMpOk);
}

// A structure to support LiveStreamModeTest below. This structure holds a
// static method `Fn` for a callback function of C API. A `static` qualifier
// allows to take an address of the method to follow API style. Another static
// struct member is `last_timestamp` that is used to verify that current
// timestamp is greater than the previous one.
struct LiveStreamModeCallback {
  static int64_t last_timestamp;
  static absl::BlockingCounter* blocking_counter;
  static void Fn(MpStatus status, const PoseLandmarkerResult* landmarker_result,
                 const MpImagePtr image, int64_t timestamp) {
    ASSERT_EQ(status, kMpOk);
    ASSERT_NE(landmarker_result, nullptr);
    MatchesPoseLandmarkerResult(landmarker_result, kLandmarkPrecision);
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

TEST(PoseLandmarkerTest, LiveStreamModeTest) {
  const auto image = GetImage(GetFullPath(kImageFile));

  const std::string model_path = GetFullPath(kModelName);

  PoseLandmarkerOptions options = {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = model_path.c_str()},
      .running_mode = RunningMode::LIVE_STREAM,
      .num_poses = 1,
      .min_pose_detection_confidence = 0.5,
      .min_pose_presence_confidence = 0.5,
      .min_tracking_confidence = 0.5,
      .output_segmentation_masks = true,
      .result_callback = LiveStreamModeCallback::Fn,
  };

  MpPoseLandmarkerPtr landmarker;
  MpStatus status = MpPoseLandmarkerCreate(&options, &landmarker);
  EXPECT_EQ(status, kMpOk);
  EXPECT_NE(landmarker, nullptr);

  absl::BlockingCounter counter(kIterations);
  LiveStreamModeCallback::blocking_counter = &counter;

  for (int i = 0; i < kIterations; ++i) {
    EXPECT_EQ(
        MpPoseLandmarkerDetectAsync(landmarker, image.get(),
                                    /* image_processing_options= */ nullptr, i),
        kMpOk);
    // Short sleep so that MediaPipe does not drop frames.
    absl::SleepFor(absl::Milliseconds(kSleepBetweenFramesMilliseconds));
  }

  // Wait for all callbacks to be invoked.
  counter.Wait();
  LiveStreamModeCallback::blocking_counter = nullptr;

  EXPECT_EQ(MpPoseLandmarkerClose(landmarker), kMpOk);

  // Due to the flow limiter, the total of outputs might be smaller than the
  // number of iterations.
  EXPECT_LE(LiveStreamModeCallback::last_timestamp, kIterations);
  EXPECT_GT(LiveStreamModeCallback::last_timestamp, 0);
}

TEST(PoseLandmarkerTest, InvalidArgumentHandling) {
  // It is an error to set neither the asset buffer nor the path.
  PoseLandmarkerOptions options = {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = nullptr},
      .running_mode = RunningMode::IMAGE,
      .num_poses = 1,
      .min_pose_detection_confidence = 0.5,
      .min_pose_presence_confidence = 0.5,
      .min_tracking_confidence = 0.5,
      .output_segmentation_masks = true,
  };

  MpPoseLandmarkerPtr landmarker = nullptr;
  MpStatus status = MpPoseLandmarkerCreate(&options, &landmarker);
  EXPECT_EQ(status, kMpInvalidArgument);
  EXPECT_EQ(landmarker, nullptr);
}

}  // namespace
