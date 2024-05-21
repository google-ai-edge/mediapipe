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

#include "mediapipe/tasks/c/vision/hand_landmarker/hand_landmarker.h"

#include <cstdint>
#include <cstdlib>
#include <string>

#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/landmark.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/c/vision/hand_landmarker/hand_landmarker_result.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace {

using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::vision::DecodeImageFromFile;
using testing::HasSubstr;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kModelName[] = "hand_landmarker.task";
constexpr char kImageFile[] = "fist.jpg";
constexpr float kScorePrecision = 1e-2;
constexpr float kLandmarkPrecision = 1e-1;
constexpr int kIterations = 100;

std::string GetFullPath(absl::string_view file_name) {
  return JoinPath("./", kTestDataDirectory, file_name);
}

void AssertHandLandmarkerResult(const HandLandmarkerResult* result,
                                const float score_precision,
                                const float landmark_precision) {
  // Expects to have the same number of hands detected.
  EXPECT_EQ(result->handedness_count, 1);

  // Actual handedness matches expected handedness.
  EXPECT_EQ(std::string{result->handedness[0].categories[0].category_name},
            "Right");
  EXPECT_NEAR(result->handedness[0].categories[0].score, 0.9893f,
              score_precision);

  // Actual landmarks match expected landmarks.
  EXPECT_NEAR(result->hand_landmarks[0].landmarks[0].x, 0.477f,
              landmark_precision);
  EXPECT_NEAR(result->hand_landmarks[0].landmarks[0].y, 0.661f,
              landmark_precision);
  EXPECT_NEAR(result->hand_landmarks[0].landmarks[0].z, 0.0f,
              landmark_precision);
  EXPECT_NEAR(result->hand_world_landmarks[0].landmarks[0].x, -0.009f,
              landmark_precision);
  EXPECT_NEAR(result->hand_world_landmarks[0].landmarks[0].y, 0.082f,
              landmark_precision);
  EXPECT_NEAR(result->hand_world_landmarks[0].landmarks[0].z, 0.006f,
              landmark_precision);
}

TEST(HandLandmarkerTest, ImageModeTest) {
  const auto image = DecodeImageFromFile(GetFullPath(kImageFile));
  ASSERT_TRUE(image.ok());

  const std::string model_path = GetFullPath(kModelName);
  HandLandmarkerOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::IMAGE,
      /* num_hands= */ 1,
      /* min_hand_detection_confidence= */ 0.5,
      /* min_hand_presence_confidence= */ 0.5,
      /* min_tracking_confidence= */ 0.5,
  };

  void* landmarker = hand_landmarker_create(&options, /* error_msg */ nullptr);
  EXPECT_NE(landmarker, nullptr);

  const auto& image_frame = image->GetImageFrameSharedPtr();
  const MpImage mp_image = {
      .type = MpImage::IMAGE_FRAME,
      .image_frame = {.format = static_cast<ImageFormat>(image_frame->Format()),
                      .image_buffer = image_frame->PixelData(),
                      .width = image_frame->Width(),
                      .height = image_frame->Height()}};

  HandLandmarkerResult result;
  hand_landmarker_detect_image(landmarker, &mp_image, &result,
                               /* error_msg */ nullptr);
  AssertHandLandmarkerResult(&result, kScorePrecision, kLandmarkPrecision);
  hand_landmarker_close_result(&result);
  hand_landmarker_close(landmarker, /* error_msg */ nullptr);
}

TEST(HandLandmarkerTest, VideoModeTest) {
  const auto image = DecodeImageFromFile(GetFullPath(kImageFile));
  ASSERT_TRUE(image.ok());

  const std::string model_path = GetFullPath(kModelName);
  HandLandmarkerOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::VIDEO,
      /* num_hands= */ 1,
      /* min_hand_detection_confidence= */ 0.5,
      /* min_hand_presence_confidence= */ 0.5,
      /* min_tracking_confidence= */ 0.5,
  };

  void* landmarker = hand_landmarker_create(&options, /* error_msg */ nullptr);
  EXPECT_NE(landmarker, nullptr);

  const auto& image_frame = image->GetImageFrameSharedPtr();
  const MpImage mp_image = {
      .type = MpImage::IMAGE_FRAME,
      .image_frame = {.format = static_cast<ImageFormat>(image_frame->Format()),
                      .image_buffer = image_frame->PixelData(),
                      .width = image_frame->Width(),
                      .height = image_frame->Height()}};

  for (int i = 0; i < kIterations; ++i) {
    HandLandmarkerResult result;
    hand_landmarker_detect_for_video(landmarker, &mp_image, i, &result,
                                     /* error_msg */ nullptr);

    AssertHandLandmarkerResult(&result, kScorePrecision, kLandmarkPrecision);
    hand_landmarker_close_result(&result);
  }
  hand_landmarker_close(landmarker, /* error_msg */ nullptr);
}

// A structure to support LiveStreamModeTest below. This structure holds a
// static method `Fn` for a callback function of C API. A `static` qualifier
// allows to take an address of the method to follow API style. Another static
// struct member is `last_timestamp` that is used to verify that current
// timestamp is greater than the previous one.
struct LiveStreamModeCallback {
  static int64_t last_timestamp;
  static void Fn(HandLandmarkerResult* landmarker_result, const MpImage* image,
                 int64_t timestamp, char* error_msg) {
    ASSERT_NE(landmarker_result, nullptr);
    ASSERT_EQ(error_msg, nullptr);
    AssertHandLandmarkerResult(landmarker_result, kScorePrecision,
                               kLandmarkPrecision);
    EXPECT_GT(image->image_frame.width, 0);
    EXPECT_GT(image->image_frame.height, 0);
    EXPECT_GT(timestamp, last_timestamp);
    ++last_timestamp;
  }
};
int64_t LiveStreamModeCallback::last_timestamp = -1;

// TODO: Await the callbacks and re-enable test
TEST(HandLandmarkerTest, DISABLED_LiveStreamModeTest) {
  const auto image = DecodeImageFromFile(GetFullPath(kImageFile));
  ASSERT_TRUE(image.ok());

  const std::string model_path = GetFullPath(kModelName);

  HandLandmarkerOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::LIVE_STREAM,
      /* num_hands= */ 1,
      /* min_hand_detection_confidence= */ 0.5,
      /* min_hand_presence_confidence= */ 0.5,
      /* min_tracking_confidence= */ 0.5,
      /* result_callback_fn= */ LiveStreamModeCallback::Fn,
  };

  void* landmarker = hand_landmarker_create(&options, /* error_msg */ nullptr);
  EXPECT_NE(landmarker, nullptr);

  const auto& image_frame = image->GetImageFrameSharedPtr();
  const MpImage mp_image = {
      .type = MpImage::IMAGE_FRAME,
      .image_frame = {.format = static_cast<ImageFormat>(image_frame->Format()),
                      .image_buffer = image_frame->PixelData(),
                      .width = image_frame->Width(),
                      .height = image_frame->Height()}};

  for (int i = 0; i < kIterations; ++i) {
    EXPECT_GE(hand_landmarker_detect_async(landmarker, &mp_image, i,
                                           /* error_msg */ nullptr),
              0);
  }
  hand_landmarker_close(landmarker, /* error_msg */ nullptr);

  // Due to the flow limiter, the total of outputs might be smaller than the
  // number of iterations.
  EXPECT_LE(LiveStreamModeCallback::last_timestamp, kIterations);
  EXPECT_GT(LiveStreamModeCallback::last_timestamp, 0);
}

TEST(HandLandmarkerTest, InvalidArgumentHandling) {
  // It is an error to set neither the asset buffer nor the path.
  HandLandmarkerOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ nullptr},
      /* running_mode= */ RunningMode::IMAGE,
      /* num_hands= */ 1,
      /* min_hand_detection_confidence= */ 0.5,
      /* min_hand_presence_confidence= */ 0.5,
      /* min_tracking_confidence= */ 0.5,
  };

  char* error_msg;
  void* landmarker = hand_landmarker_create(&options, &error_msg);
  EXPECT_EQ(landmarker, nullptr);

  EXPECT_THAT(error_msg, HasSubstr("ExternalFile must specify"));

  free(error_msg);
}

TEST(HandLandmarkerTest, FailedRecognitionHandling) {
  const std::string model_path = GetFullPath(kModelName);
  HandLandmarkerOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::IMAGE,
      /* num_hands= */ 1,
      /* min_hand_detection_confidence= */ 0.5,
      /* min_hand_presence_confidence= */ 0.5,
      /* min_tracking_confidence= */ 0.5,
  };

  void* landmarker = hand_landmarker_create(&options, /* error_msg */
                                            nullptr);
  EXPECT_NE(landmarker, nullptr);

  const MpImage mp_image = {.type = MpImage::GPU_BUFFER, .gpu_buffer = {}};
  HandLandmarkerResult result;
  char* error_msg;
  hand_landmarker_detect_image(landmarker, &mp_image, &result, &error_msg);
  EXPECT_THAT(error_msg, HasSubstr("GPU Buffer not supported yet"));
  free(error_msg);
  hand_landmarker_close(landmarker, /* error_msg */ nullptr);
}

}  // namespace
