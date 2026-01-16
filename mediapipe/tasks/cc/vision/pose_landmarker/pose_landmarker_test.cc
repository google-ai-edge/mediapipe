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

#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarker.h"

#include <cmath>
#include <memory>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"
#include "mediapipe/tasks/cc/components/containers/proto/landmarks_detection_result.pb.h"
#include "mediapipe/tasks/cc/components/containers/rect.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarker_result.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"
#include "tensorflow/lite/test_util.h"
#include "util/tuple/dump_vars.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace pose_landmarker {

namespace {

using ::file::Defaults;
using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::components::containers::ConvertToLandmarks;
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
constexpr char kPoseLandmarkerBundleAsset[] = "pose_landmarker.task";
constexpr char kPoseLandmarksFilename[] = "pose_landmarks.pbtxt";

constexpr char kPoseImage[] = "pose.jpg";
constexpr char kBurgerImage[] = "burger.jpg";

constexpr float kLandmarksAbsMargin = 0.03;
constexpr float kLandmarksOnVideoAbsMargin = 0.03;

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

PoseLandmarkerResult GetExpectedPoseLandmarkerResult(
    const std::vector<absl::string_view>& landmarks_file_names) {
  PoseLandmarkerResult expected_results;
  for (const auto& file_name : landmarks_file_names) {
    const auto landmarks_detection_result =
        GetLandmarksDetectionResult(file_name);
    expected_results.pose_landmarks.push_back(
        ConvertToNormalizedLandmarks(landmarks_detection_result.landmarks()));
    expected_results.pose_world_landmarks.push_back(
        ConvertToLandmarks(landmarks_detection_result.world_landmarks()));
  }
  return expected_results;
}

MATCHER_P2(LandmarksMatches, expected_landmarks, toleration, "") {
  for (int i = 0; i < arg.size(); i++) {
    for (int j = 0; j < arg[i].landmarks.size(); j++) {
      if (arg[i].landmarks.size() != expected_landmarks[i].landmarks.size()) {
        ABSL_LOG(INFO) << "sizes not equal";
        return false;
      }
      if (std::abs(arg[i].landmarks[j].x -
                   expected_landmarks[i].landmarks[j].x) > toleration ||
          std::abs(arg[i].landmarks[j].y -
                   expected_landmarks[i].landmarks[j].y) > toleration) {
        ABSL_LOG(INFO) << DUMP_VARS(arg[i].landmarks[j].x,
                                    expected_landmarks[i].landmarks[j].x);
        ABSL_LOG(INFO) << DUMP_VARS(arg[i].landmarks[j].y,
                                    expected_landmarks[i].landmarks[j].y);
        return false;
      }
    }
  }
  return true;
}

void ExpectPoseLandmarkerResultsCorrect(
    const PoseLandmarkerResult& actual_results,
    const PoseLandmarkerResult& expected_results, float margin) {
  const auto& actual_landmarks = actual_results.pose_landmarks;

  const auto& expected_landmarks = expected_results.pose_landmarks;

  ASSERT_EQ(actual_landmarks.size(), expected_landmarks.size());

  if (actual_landmarks.empty()) {
    return;
  }

  ASSERT_GE(actual_landmarks.size(), 1);
  EXPECT_THAT(actual_landmarks, LandmarksMatches(expected_landmarks, margin));
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
  // Expected results from the pose landmarker model output.
  PoseLandmarkerResult expected_results;
};

class ImageModeTest : public testing::TestWithParam<TestParams> {};

TEST_F(ImageModeTest, FailsWithCallingWrongMethod) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, kPoseImage)));
  auto options = std::make_unique<PoseLandmarkerOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kPoseLandmarkerBundleAsset);
  options->running_mode = core::RunningMode::IMAGE;

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PoseLandmarker> pose_landmarker,
                          PoseLandmarker::Create(std::move(options)));
  auto results = pose_landmarker->DetectForVideo(image, 0);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the video mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));

  results = pose_landmarker->DetectAsync(image, 0);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the live stream mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));
  MP_ASSERT_OK(pose_landmarker->Close());
}

TEST_F(ImageModeTest, FailsWithRegionOfInterest) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, kPoseImage)));
  auto options = std::make_unique<PoseLandmarkerOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kPoseLandmarkerBundleAsset);
  options->running_mode = core::RunningMode::IMAGE;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PoseLandmarker> pose_landmarker,
                          PoseLandmarker::Create(std::move(options)));
  RectF roi{/*left=*/0.1, /*top=*/0, /*right=*/0.9, /*bottom=*/1};
  ImageProcessingOptions image_processing_options{roi, /*rotation_degrees=*/0};

  auto results = pose_landmarker->Detect(image, image_processing_options);
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
  auto options = std::make_unique<PoseLandmarkerOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, GetParam().test_model_file);
  options->running_mode = core::RunningMode::IMAGE;

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PoseLandmarker> pose_landmarker,
                          PoseLandmarker::Create(std::move(options)));
  PoseLandmarkerResult pose_landmarker_results;
  if (GetParam().rotation != 0) {
    ImageProcessingOptions image_processing_options;
    image_processing_options.rotation_degrees = GetParam().rotation;
    MP_ASSERT_OK_AND_ASSIGN(
        pose_landmarker_results,
        pose_landmarker->Detect(image, image_processing_options));
  } else {
    MP_ASSERT_OK_AND_ASSIGN(pose_landmarker_results,
                            pose_landmarker->Detect(image));
  }
  ExpectPoseLandmarkerResultsCorrect(pose_landmarker_results,
                                     GetParam().expected_results,
                                     kLandmarksAbsMargin);
  MP_ASSERT_OK(pose_landmarker->Close());
}

INSTANTIATE_TEST_SUITE_P(
    PoseTest, ImageModeTest,
    Values(TestParams{
               /* test_name= */ "Pose",
               /* test_image_name= */ kPoseImage,
               /* test_model_file= */ kPoseLandmarkerBundleAsset,
               /* rotation= */ 0,
               /* expected_results = */
               GetExpectedPoseLandmarkerResult({kPoseLandmarksFilename}),
           },
           TestParams{
               /* test_name= */ "NoPoses",
               /* test_image_name= */ kBurgerImage,
               /* test_model_file= */ kPoseLandmarkerBundleAsset,
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
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, kPoseImage)));
  auto options = std::make_unique<PoseLandmarkerOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kPoseLandmarkerBundleAsset);
  options->running_mode = core::RunningMode::VIDEO;

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PoseLandmarker> pose_landmarker,
                          PoseLandmarker::Create(std::move(options)));
  auto results = pose_landmarker->Detect(image);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the image mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));

  results = pose_landmarker->DetectAsync(image, 0);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the live stream mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));
  MP_ASSERT_OK(pose_landmarker->Close());
}

TEST_P(VideoModeTest, Succeeds) {
  const int iterations = 100;
  MP_ASSERT_OK_AND_ASSIGN(
      Image image, DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                                GetParam().test_image_name)));
  auto options = std::make_unique<PoseLandmarkerOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, GetParam().test_model_file);
  options->running_mode = core::RunningMode::VIDEO;

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PoseLandmarker> pose_landmarker,
                          PoseLandmarker::Create(std::move(options)));
  const auto expected_results = GetParam().expected_results;
  for (int i = 0; i < iterations; ++i) {
    PoseLandmarkerResult pose_landmarker_results;
    if (GetParam().rotation != 0) {
      ImageProcessingOptions image_processing_options;
      image_processing_options.rotation_degrees = GetParam().rotation;
      MP_ASSERT_OK_AND_ASSIGN(
          pose_landmarker_results,
          pose_landmarker->DetectForVideo(image, i, image_processing_options));
    } else {
      MP_ASSERT_OK_AND_ASSIGN(pose_landmarker_results,
                              pose_landmarker->DetectForVideo(image, i));
    }
    ABSL_LOG(INFO) << i;
    ExpectPoseLandmarkerResultsCorrect(
        pose_landmarker_results, expected_results, kLandmarksOnVideoAbsMargin);
  }
  MP_ASSERT_OK(pose_landmarker->Close());
}

// TODO Add additional tests for MP Tasks Pose Graphs
// TODO Investigate PoseLandmarker performance in VideoMode.

INSTANTIATE_TEST_SUITE_P(
    PoseTest, VideoModeTest,
    Values(TestParams{
               /* test_name= */ "Pose",
               /* test_image_name= */ kPoseImage,
               /* test_model_file= */ kPoseLandmarkerBundleAsset,
               /* rotation= */ 0,
               /* expected_results = */
               GetExpectedPoseLandmarkerResult({kPoseLandmarksFilename}),
           },
           TestParams{
               /* test_name= */ "NoPoses",
               /* test_image_name= */ kBurgerImage,
               /* test_model_file= */ kPoseLandmarkerBundleAsset,
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
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, kPoseImage)));
  auto options = std::make_unique<PoseLandmarkerOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kPoseLandmarkerBundleAsset);
  options->running_mode = core::RunningMode::LIVE_STREAM;
  options->result_callback = [](absl::StatusOr<PoseLandmarkerResult> results,
                                const Image& image, int64_t timestamp_ms) {};

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PoseLandmarker> pose_landmarker,
                          PoseLandmarker::Create(std::move(options)));
  auto results = pose_landmarker->Detect(image);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the image mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));

  results = pose_landmarker->DetectForVideo(image, 0);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the video mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));
  MP_ASSERT_OK(pose_landmarker->Close());
}

TEST_P(LiveStreamModeTest, Succeeds) {
  const int iterations = 100;
  MP_ASSERT_OK_AND_ASSIGN(
      Image image, DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                                GetParam().test_image_name)));
  auto options = std::make_unique<PoseLandmarkerOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, GetParam().test_model_file);
  options->running_mode = core::RunningMode::LIVE_STREAM;
  std::vector<PoseLandmarkerResult> pose_landmarker_results;
  std::vector<std::pair<int, int>> image_sizes;
  std::vector<int64_t> timestamps;
  options->result_callback = [&pose_landmarker_results, &image_sizes,
                              &timestamps](
                                 absl::StatusOr<PoseLandmarkerResult> results,
                                 const Image& image, int64_t timestamp_ms) {
    MP_ASSERT_OK(results.status());
    pose_landmarker_results.push_back(std::move(results.value()));
    image_sizes.push_back({image.width(), image.height()});
    timestamps.push_back(timestamp_ms);
  };

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PoseLandmarker> pose_landmarker,
                          PoseLandmarker::Create(std::move(options)));
  for (int i = 0; i < iterations; ++i) {
    PoseLandmarkerResult pose_landmarker_results;
    if (GetParam().rotation != 0) {
      ImageProcessingOptions image_processing_options;
      image_processing_options.rotation_degrees = GetParam().rotation;
      MP_ASSERT_OK(
          pose_landmarker->DetectAsync(image, i, image_processing_options));
    } else {
      MP_ASSERT_OK(pose_landmarker->DetectAsync(image, i));
    }
  }
  MP_ASSERT_OK(pose_landmarker->Close());
  // Due to the flow limiter, the total of outputs will be smaller than the
  // number of iterations.
  ASSERT_LE(pose_landmarker_results.size(), iterations);
  ASSERT_GT(pose_landmarker_results.size(), 0);

  const auto expected_results = GetParam().expected_results;
  for (int i = 0; i < pose_landmarker_results.size(); ++i) {
    ExpectPoseLandmarkerResultsCorrect(pose_landmarker_results[i],
                                       expected_results,
                                       kLandmarksOnVideoAbsMargin);
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

// TODO Add additional tests for MP Tasks Pose Graphs
// Investigate PoseLandmarker performance in LiveStreamMode.

INSTANTIATE_TEST_SUITE_P(
    PoseTest, LiveStreamModeTest,
    Values(TestParams{
               /* test_name= */ "Pose",
               /* test_image_name= */ kPoseImage,
               /* test_model_file= */ kPoseLandmarkerBundleAsset,
               /* rotation= */ 0,
               /* expected_results = */
               GetExpectedPoseLandmarkerResult({kPoseLandmarksFilename}),
           },
           TestParams{
               /* test_name= */ "NoPoses",
               /* test_image_name= */ kBurgerImage,
               /* test_model_file= */ kPoseLandmarkerBundleAsset,
               /* rotation= */ 0,
               /* expected_results = */
               {{}, {}, {}},
           }),
    [](const TestParamInfo<ImageModeTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace pose_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
