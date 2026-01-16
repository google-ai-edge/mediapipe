/* Copyright 2022 The MediaPipe Authors.

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

#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker.h"

#include <cmath>
#include <memory>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/containers/classification_result.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"
#include "mediapipe/tasks/cc/components/containers/proto/landmarks_detection_result.pb.h"
#include "mediapipe/tasks/cc/components/containers/rect.h"
#include "mediapipe/tasks/cc/components/processors/proto/classifier_options.pb.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker_result.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"
#include "tensorflow/lite/test_util.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace hand_landmarker {

namespace {

using ::file::Defaults;
using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::components::containers::ConvertToClassifications;
using ::mediapipe::tasks::components::containers::ConvertToNormalizedLandmarks;
using ::mediapipe::tasks::components::containers::RectF;
using ::mediapipe::tasks::containers::proto::LandmarksDetectionResult;
using ::mediapipe::tasks::vision::core::ImageProcessingOptions;
using ::testing::HasSubstr;
using ::testing::Optional;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::Values;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kHandLandmarkerBundleAsset[] = "hand_landmarker.task";
constexpr char kThumbUpLandmarksFilename[] = "thumb_up_landmarks.pbtxt";
constexpr char kPointingUpLandmarksFilename[] = "pointing_up_landmarks.pbtxt";
constexpr char kPointingUpRotatedLandmarksFilename[] =
    "pointing_up_rotated_landmarks.pbtxt";
constexpr char kThumbUpImage[] = "thumb_up.jpg";
constexpr char kPointingUpImage[] = "pointing_up.jpg";
constexpr char kPointingUpRotatedImage[] = "pointing_up_rotated.jpg";
constexpr char kNoHandsImage[] = "cats_and_dogs.jpg";

constexpr float kLandmarksAbsMargin = 0.03;
constexpr float kHandednessMargin = 0.05;

LandmarksDetectionResult GetLandmarksDetectionResult(
    absl::string_view landmarks_file_name) {
  LandmarksDetectionResult result;
  MP_EXPECT_OK(GetTextProto(
      file::JoinPath("./", kTestDataDirectory, landmarks_file_name), &result,
      Defaults()));
  // Remove z position of landmarks, because they are not used in correctness
  // testing. For video or live stream mode, the z positions varies a lot during
  // tracking from frame to frame.
  for (int i = 0; i < result.landmarks().landmark().size(); i++) {
    auto& landmark = *result.mutable_landmarks()->mutable_landmark(i);
    landmark.clear_z();
  }
  return result;
}

HandLandmarkerResult GetExpectedHandLandmarkerResult(
    const std::vector<absl::string_view>& landmarks_file_names) {
  HandLandmarkerResult expected_results;
  for (const auto& file_name : landmarks_file_names) {
    const auto landmarks_detection_result =
        GetLandmarksDetectionResult(file_name);
    expected_results.hand_landmarks.push_back(
        ConvertToNormalizedLandmarks(landmarks_detection_result.landmarks()));
    expected_results.handedness.push_back(
        ConvertToClassifications(landmarks_detection_result.classifications()));
  }
  return expected_results;
}

MATCHER_P2(HandednessMatches, expected_handedness, tolerance, "") {
  for (int i = 0; i < arg.size(); i++) {
    for (int j = 0; j < arg[i].categories.size(); j++) {
      if (arg[i].categories[j].index !=
          expected_handedness[i].categories[j].index) {
        return false;
      }
      if (std::abs(arg[i].categories[j].score -
                   expected_handedness[i].categories[j].score) > tolerance) {
        return false;
      }
      if (arg[i].categories[j].category_name !=
          expected_handedness[i].categories[j].category_name) {
        return false;
      }
    }
  }
  return true;
}

MATCHER_P2(LandmarksMatches, expected_landmarks, toleration, "") {
  for (int i = 0; i < arg.size(); i++) {
    for (int j = 0; j < arg[i].landmarks.size(); j++) {
      if (std::abs(arg[i].landmarks[j].x -
                   expected_landmarks[i].landmarks[j].x) > toleration ||
          std::abs(arg[i].landmarks[j].y -
                   expected_landmarks[i].landmarks[j].y) > toleration) {
        return false;
      }
    }
  }
  return true;
}

void ExpectHandLandmarkerResultsCorrect(
    const HandLandmarkerResult& actual_results,
    const HandLandmarkerResult& expected_results) {
  const auto& actual_landmarks = actual_results.hand_landmarks;
  const auto& actual_handedness = actual_results.handedness;

  const auto& expected_landmarks = expected_results.hand_landmarks;
  const auto& expected_handedness = expected_results.handedness;

  ASSERT_EQ(actual_landmarks.size(), expected_landmarks.size());
  ASSERT_EQ(actual_handedness.size(), expected_handedness.size());
  if (actual_landmarks.empty()) {
    return;
  }
  ASSERT_GE(actual_landmarks.size(), 1);

  EXPECT_THAT(actual_handedness,
              HandednessMatches(expected_handedness, kHandednessMargin));
  EXPECT_THAT(actual_landmarks,
              LandmarksMatches(expected_landmarks, kLandmarksAbsMargin));
}

}  // namespace

struct TestParams {
  // The name of this test, for convenience when displaying test results.
  std::string test_name;
  // The filename of test image.
  std::string test_image_name;
  // The filename of test model.
  std::string test_model_file;
  // The rotation to apply to the test image before processing, in degrees
  // clockwise.
  int rotation;
  // Expected results from the hand landmarker model output.
  HandLandmarkerResult expected_results;
};

class ImageModeTest : public testing::TestWithParam<TestParams> {};

TEST_F(ImageModeTest, FailsWithCallingWrongMethod) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, kThumbUpImage)));
  auto options = std::make_unique<HandLandmarkerOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kHandLandmarkerBundleAsset);
  options->running_mode = core::RunningMode::IMAGE;

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HandLandmarker> hand_landmarker,
                          HandLandmarker::Create(std::move(options)));
  auto results = hand_landmarker->DetectForVideo(image, 0);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the video mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));

  results = hand_landmarker->DetectAsync(image, 0);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the live stream mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));
  MP_ASSERT_OK(hand_landmarker->Close());
}

TEST_F(ImageModeTest, FailsWithRegionOfInterest) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, kThumbUpImage)));
  auto options = std::make_unique<HandLandmarkerOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kHandLandmarkerBundleAsset);
  options->running_mode = core::RunningMode::IMAGE;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HandLandmarker> hand_landmarker,
                          HandLandmarker::Create(std::move(options)));
  RectF roi{/*left=*/0.1, /*top=*/0, /*right=*/0.9, /*bottom=*/1};
  ImageProcessingOptions image_processing_options{roi, /*rotation_degrees=*/0};

  auto results = hand_landmarker->Detect(image, image_processing_options);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("This task doesn't support region-of-interest"));
  EXPECT_THAT(
      results.status().GetPayload(kMediaPipeTasksPayload),
      Optional(absl::Cord(absl::StrCat(
          MediaPipeTasksStatus::kImageProcessingInvalidArgumentError))));
}

TEST_P(ImageModeTest, Succeeds) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image, DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                                GetParam().test_image_name)));
  auto options = std::make_unique<HandLandmarkerOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, GetParam().test_model_file);
  options->running_mode = core::RunningMode::IMAGE;

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HandLandmarker> hand_landmarker,
                          HandLandmarker::Create(std::move(options)));
  HandLandmarkerResult hand_landmarker_results;
  if (GetParam().rotation != 0) {
    ImageProcessingOptions image_processing_options;
    image_processing_options.rotation_degrees = GetParam().rotation;
    MP_ASSERT_OK_AND_ASSIGN(
        hand_landmarker_results,
        hand_landmarker->Detect(image, image_processing_options));
  } else {
    MP_ASSERT_OK_AND_ASSIGN(hand_landmarker_results,
                            hand_landmarker->Detect(image));
  }
  ExpectHandLandmarkerResultsCorrect(hand_landmarker_results,
                                     GetParam().expected_results);
  MP_ASSERT_OK(hand_landmarker->Close());
}

INSTANTIATE_TEST_SUITE_P(
    HandGestureTest, ImageModeTest,
    Values(TestParams{
               /* test_name= */ "LandmarksThumbUp",
               /* test_image_name= */ kThumbUpImage,
               /* test_model_file= */ kHandLandmarkerBundleAsset,
               /* rotation= */ 0,
               /* expected_results = */
               GetExpectedHandLandmarkerResult({kThumbUpLandmarksFilename}),
           },
           TestParams{
               /* test_name= */ "LandmarksPointingUp",
               /* test_image_name= */ kPointingUpImage,
               /* test_model_file= */ kHandLandmarkerBundleAsset,
               /* rotation= */ 0,
               /* expected_results = */
               GetExpectedHandLandmarkerResult({kPointingUpLandmarksFilename}),
           },
           TestParams{
               /* test_name= */ "LandmarksPointingUpRotated",
               /* test_image_name= */ kPointingUpRotatedImage,
               /* test_model_file= */ kHandLandmarkerBundleAsset,
               /* rotation= */ -90,
               /* expected_results = */
               GetExpectedHandLandmarkerResult(
                   {kPointingUpRotatedLandmarksFilename}),
           },
           TestParams{
               /* test_name= */ "NoHands",
               /* test_image_name= */ kNoHandsImage,
               /* test_model_file= */ kHandLandmarkerBundleAsset,
               /* rotation= */ 0,
               /* expected_results = */
               {{}, {}, {}},
           }),
    [](const TestParamInfo<ImageModeTest::ParamType>& info) {
      return info.param.test_name;
    });

class VideoModeTest : public testing::TestWithParam<TestParams> {};

TEST_F(VideoModeTest, FailsWithCallingWrongMethod) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, kThumbUpImage)));
  auto options = std::make_unique<HandLandmarkerOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kHandLandmarkerBundleAsset);
  options->running_mode = core::RunningMode::VIDEO;

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HandLandmarker> hand_landmarker,
                          HandLandmarker::Create(std::move(options)));
  auto results = hand_landmarker->Detect(image);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the image mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));

  results = hand_landmarker->DetectAsync(image, 0);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the live stream mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));
  MP_ASSERT_OK(hand_landmarker->Close());
}

TEST_P(VideoModeTest, Succeeds) {
  const int iterations = 100;
  MP_ASSERT_OK_AND_ASSIGN(
      Image image, DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                                GetParam().test_image_name)));
  auto options = std::make_unique<HandLandmarkerOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, GetParam().test_model_file);
  options->running_mode = core::RunningMode::VIDEO;

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HandLandmarker> hand_landmarker,
                          HandLandmarker::Create(std::move(options)));
  const auto expected_results = GetParam().expected_results;
  for (int i = 0; i < iterations; ++i) {
    HandLandmarkerResult hand_landmarker_results;
    if (GetParam().rotation != 0) {
      ImageProcessingOptions image_processing_options;
      image_processing_options.rotation_degrees = GetParam().rotation;
      MP_ASSERT_OK_AND_ASSIGN(
          hand_landmarker_results,
          hand_landmarker->DetectForVideo(image, i, image_processing_options));
    } else {
      MP_ASSERT_OK_AND_ASSIGN(hand_landmarker_results,
                              hand_landmarker->DetectForVideo(image, i));
    }
    ExpectHandLandmarkerResultsCorrect(hand_landmarker_results,
                                       expected_results);
  }
  MP_ASSERT_OK(hand_landmarker->Close());
}

INSTANTIATE_TEST_SUITE_P(
    HandGestureTest, VideoModeTest,
    Values(TestParams{
               /* test_name= */ "LandmarksThumbUp",
               /* test_image_name= */ kThumbUpImage,
               /* test_model_file= */ kHandLandmarkerBundleAsset,
               /* rotation= */ 0,
               /* expected_results = */
               GetExpectedHandLandmarkerResult({kThumbUpLandmarksFilename}),
           },
           TestParams{
               /* test_name= */ "LandmarksPointingUp",
               /* test_image_name= */ kPointingUpImage,
               /* test_model_file= */ kHandLandmarkerBundleAsset,
               /* rotation= */ 0,
               /* expected_results = */
               GetExpectedHandLandmarkerResult({kPointingUpLandmarksFilename}),
           },
           TestParams{
               /* test_name= */ "LandmarksPointingUpRotated",
               /* test_image_name= */ kPointingUpRotatedImage,
               /* test_model_file= */ kHandLandmarkerBundleAsset,
               /* rotation= */ -90,
               /* expected_results = */
               GetExpectedHandLandmarkerResult(
                   {kPointingUpRotatedLandmarksFilename}),
           },
           TestParams{
               /* test_name= */ "NoHands",
               /* test_image_name= */ kNoHandsImage,
               /* test_model_file= */ kHandLandmarkerBundleAsset,
               /* rotation= */ 0,
               /* expected_results = */
               {{}, {}, {}},
           }),
    [](const TestParamInfo<ImageModeTest::ParamType>& info) {
      return info.param.test_name;
    });

class LiveStreamModeTest : public testing::TestWithParam<TestParams> {};

TEST_F(LiveStreamModeTest, FailsWithCallingWrongMethod) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, kThumbUpImage)));
  auto options = std::make_unique<HandLandmarkerOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kHandLandmarkerBundleAsset);
  options->running_mode = core::RunningMode::LIVE_STREAM;
  options->result_callback = [](absl::StatusOr<HandLandmarkerResult> results,
                                const Image& image, int64_t timestamp_ms) {};

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HandLandmarker> hand_landmarker,
                          HandLandmarker::Create(std::move(options)));
  auto results = hand_landmarker->Detect(image);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the image mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));

  results = hand_landmarker->DetectForVideo(image, 0);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the video mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));
  MP_ASSERT_OK(hand_landmarker->Close());
}

TEST_P(LiveStreamModeTest, Succeeds) {
  const int iterations = 100;
  MP_ASSERT_OK_AND_ASSIGN(
      Image image, DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                                GetParam().test_image_name)));
  auto options = std::make_unique<HandLandmarkerOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, GetParam().test_model_file);
  options->running_mode = core::RunningMode::LIVE_STREAM;
  std::vector<HandLandmarkerResult> hand_landmarker_results;
  std::vector<std::pair<int, int>> image_sizes;
  std::vector<int64_t> timestamps;
  options->result_callback = [&hand_landmarker_results, &image_sizes,
                              &timestamps](
                                 absl::StatusOr<HandLandmarkerResult> results,
                                 const Image& image, int64_t timestamp_ms) {
    MP_ASSERT_OK(results.status());
    hand_landmarker_results.push_back(std::move(results.value()));
    image_sizes.push_back({image.width(), image.height()});
    timestamps.push_back(timestamp_ms);
  };

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HandLandmarker> hand_landmarker,
                          HandLandmarker::Create(std::move(options)));
  for (int i = 0; i < iterations; ++i) {
    HandLandmarkerResult hand_landmarker_results;
    if (GetParam().rotation != 0) {
      ImageProcessingOptions image_processing_options;
      image_processing_options.rotation_degrees = GetParam().rotation;
      MP_ASSERT_OK(
          hand_landmarker->DetectAsync(image, i, image_processing_options));
    } else {
      MP_ASSERT_OK(hand_landmarker->DetectAsync(image, i));
    }
  }
  MP_ASSERT_OK(hand_landmarker->Close());
  // Due to the flow limiter, the total of outputs will be smaller than the
  // number of iterations.
  ASSERT_LE(hand_landmarker_results.size(), iterations);
  ASSERT_GT(hand_landmarker_results.size(), 0);

  const auto expected_results = GetParam().expected_results;
  for (int i = 0; i < hand_landmarker_results.size(); ++i) {
    ExpectHandLandmarkerResultsCorrect(hand_landmarker_results[i],
                                       expected_results);
  }
  for (const auto& image_size : image_sizes) {
    EXPECT_EQ(image_size.first, image.width());
    EXPECT_EQ(image_size.second, image.height());
  }
  int64_t timestamp_ms = -1;
  for (const auto& timestamp : timestamps) {
    EXPECT_GT(timestamp, timestamp_ms);
    timestamp_ms = timestamp;
  }
}

INSTANTIATE_TEST_SUITE_P(
    HandGestureTest, LiveStreamModeTest,
    Values(TestParams{
               /* test_name= */ "LandmarksThumbUp",
               /* test_image_name= */ kThumbUpImage,
               /* test_model_file= */ kHandLandmarkerBundleAsset,
               /* rotation= */ 0,
               /* expected_results = */
               GetExpectedHandLandmarkerResult({kThumbUpLandmarksFilename}),
           },
           TestParams{
               /* test_name= */ "LandmarksPointingUp",
               /* test_image_name= */ kPointingUpImage,
               /* test_model_file= */ kHandLandmarkerBundleAsset,
               /* rotation= */ 0,
               /* expected_results = */
               GetExpectedHandLandmarkerResult({kPointingUpLandmarksFilename}),
           },
           TestParams{
               /* test_name= */ "LandmarksPointingUpRotated",
               /* test_image_name= */ kPointingUpRotatedImage,
               /* test_model_file= */ kHandLandmarkerBundleAsset,
               /* rotation= */ -90,
               /* expected_results = */
               GetExpectedHandLandmarkerResult(
                   {kPointingUpRotatedLandmarksFilename}),
           },
           TestParams{
               /* test_name= */ "NoHands",
               /* test_image_name= */ kNoHandsImage,
               /* test_model_file= */ kHandLandmarkerBundleAsset,
               /* rotation= */ 0,
               /* expected_results = */
               {{}, {}, {}},
           }),
    [](const TestParamInfo<ImageModeTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace hand_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
