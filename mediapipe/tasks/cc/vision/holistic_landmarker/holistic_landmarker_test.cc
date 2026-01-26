
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

#include "mediapipe/tasks/cc/vision/holistic_landmarker/holistic_landmarker.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "file/base/helpers.h"
#include "file/base/options.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/containers/proto/landmarks_detection_result.pb.h"
#include "mediapipe/tasks/cc/components/containers/rect.h"
#include "mediapipe/tasks/cc/components/processors/proto/classifier_options.pb.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/holistic_landmarker/holistic_landmarker_result.h"
#include "mediapipe/tasks/cc/vision/holistic_landmarker/proto/holistic_result.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace holistic_landmarker {

namespace {

using ::file::Defaults;
using ::file::GetTextProto;
using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::components::containers::RectF;
using ::mediapipe::tasks::vision::core::ImageProcessingOptions;
using ::mediapipe::tasks::vision::holistic_landmarker::proto::HolisticResult;
using ::testing::HasSubstr;
using ::testing::Optional;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kHolisticLandmarkerBundleAsset[] = "holistic_landmarker.task";
constexpr char kPoseImage[] = "male_full_height_hands.jpg";
constexpr char kCatImage[] = "cat.jpg";
constexpr char kFaceImage[] = "portrait.jpg";
constexpr char kHolisticResultProto[] =
    "male_full_height_hands_result_cpu.pbtxt";
constexpr float kLandmarksAbsMargin = 0.03;
constexpr int kMaskWidth = 638;
constexpr int kMaskHeight = 1000;

proto::HolisticResult GetExpectedHolisticResult(absl::string_view result_file) {
  proto::HolisticResult result;
  MP_EXPECT_OK(GetTextProto(JoinPath("./", kTestDataDirectory, result_file),
                            &result, Defaults()));
  return result;
}

MATCHER_P2(LandmarksMatch, expected_landmarks_proto, tolerance, "") {
  if (arg.landmarks.size() != expected_landmarks_proto.landmark_size()) {
    *result_listener << "landmark lists have different size: "
                     << arg.landmarks.size() << " vs "
                     << expected_landmarks_proto.landmark_size();
    return false;
  }
  for (int i = 0; i < arg.landmarks.size(); ++i) {
    if (std::abs(arg.landmarks[i].x -
                 expected_landmarks_proto.landmark(i).x()) > tolerance ||
        std::abs(arg.landmarks[i].y -
                 expected_landmarks_proto.landmark(i).y()) > tolerance) {
      *result_listener << "landmark " << i << " mismatch: got {"
                       << arg.landmarks[i].x << ", " << arg.landmarks[i].y
                       << "}, expected {"
                       << expected_landmarks_proto.landmark(i).x() << ", "
                       << expected_landmarks_proto.landmark(i).y() << "}";
      return false;
    }
  }
  return true;
}

void AssertHolisticLandmarkerResultCorrect(
    const HolisticLandmarkerResult& actual_result,
    const proto::HolisticResult& expected_result_proto,
    bool has_segmentation_masks = false) {
  // Face landmarks
  EXPECT_THAT(actual_result.face_landmarks,
              LandmarksMatch(expected_result_proto.face_landmarks(),
                             kLandmarksAbsMargin));

  // Pose landmarks
  EXPECT_THAT(actual_result.pose_landmarks,
              LandmarksMatch(expected_result_proto.pose_landmarks(),
                             kLandmarksAbsMargin));

  // Hand landmarks
  EXPECT_THAT(actual_result.left_hand_landmarks,
              LandmarksMatch(expected_result_proto.left_hand_landmarks(),
                             kLandmarksAbsMargin));
  EXPECT_THAT(actual_result.right_hand_landmarks,
              LandmarksMatch(expected_result_proto.right_hand_landmarks(),
                             kLandmarksAbsMargin));

  if (has_segmentation_masks) {
    ASSERT_TRUE(actual_result.pose_segmentation_masks.has_value());
    EXPECT_EQ(actual_result.pose_segmentation_masks->width(), kMaskWidth);
    EXPECT_EQ(actual_result.pose_segmentation_masks->height(), kMaskHeight);
  } else {
    ASSERT_FALSE(actual_result.pose_segmentation_masks.has_value());
  }
}

}  // namespace

class ImageModeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    expected_result_ = GetExpectedHolisticResult(kHolisticResultProto);
  }
  proto::HolisticResult expected_result_;
};

TEST_F(ImageModeTest, Succeeds) {
  MP_ASSERT_OK_AND_ASSIGN(Image image,
                          DecodeImageFromFile(mediapipe::file::JoinPath(
                              "./", kTestDataDirectory, kPoseImage)));
  auto options = std::make_unique<HolisticLandmarkerOptions>();
  options->base_options.model_asset_path = mediapipe::file::JoinPath(
      "./", kTestDataDirectory, kHolisticLandmarkerBundleAsset);
  options->running_mode = core::RunningMode::IMAGE;

  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HolisticLandmarker> holistic_landmarker,
      HolisticLandmarker::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(auto results, holistic_landmarker->Detect(image));
  AssertHolisticLandmarkerResultCorrect(results, expected_result_);
  MP_ASSERT_OK(holistic_landmarker->Close());
}

TEST_F(ImageModeTest, SucceedsWithSegmentationMask) {
  MP_ASSERT_OK_AND_ASSIGN(Image image,
                          DecodeImageFromFile(mediapipe::file::JoinPath(
                              "./", kTestDataDirectory, kPoseImage)));
  auto options = std::make_unique<HolisticLandmarkerOptions>();
  options->base_options.model_asset_path = mediapipe::file::JoinPath(
      "./", kTestDataDirectory, kHolisticLandmarkerBundleAsset);
  options->running_mode = core::RunningMode::IMAGE;
  options->output_pose_segmentation_masks = true;

  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HolisticLandmarker> holistic_landmarker,
      HolisticLandmarker::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(auto results, holistic_landmarker->Detect(image));
  AssertHolisticLandmarkerResultCorrect(results, expected_result_,
                                        /* has_segmentation_masks= */ true);
  MP_ASSERT_OK(holistic_landmarker->Close());
}

TEST_F(ImageModeTest, SucceedsWithFaceOnly) {
  MP_ASSERT_OK_AND_ASSIGN(Image image,
                          DecodeImageFromFile(mediapipe::file::JoinPath(
                              "./", kTestDataDirectory, kFaceImage)));
  auto options = std::make_unique<HolisticLandmarkerOptions>();
  options->base_options.model_asset_path = mediapipe::file::JoinPath(
      "./", kTestDataDirectory, kHolisticLandmarkerBundleAsset);
  options->running_mode = core::RunningMode::IMAGE;

  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HolisticLandmarker> holistic_landmarker,
      HolisticLandmarker::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(auto results, holistic_landmarker->Detect(image));
  EXPECT_FALSE(results.face_landmarks.landmarks.empty());
  MP_ASSERT_OK(holistic_landmarker->Close());
}

TEST_F(ImageModeTest, SucceedsWithEmptyResult) {
  MP_ASSERT_OK_AND_ASSIGN(Image image,
                          DecodeImageFromFile(mediapipe::file::JoinPath(
                              "./", kTestDataDirectory, kCatImage)));
  auto options = std::make_unique<HolisticLandmarkerOptions>();
  options->base_options.model_asset_path = mediapipe::file::JoinPath(
      "./", kTestDataDirectory, kHolisticLandmarkerBundleAsset);
  options->running_mode = core::RunningMode::IMAGE;

  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HolisticLandmarker> holistic_landmarker,
      HolisticLandmarker::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(auto results, holistic_landmarker->Detect(image));
  EXPECT_TRUE(results.face_landmarks.landmarks.empty());
  MP_ASSERT_OK(holistic_landmarker->Close());
}

TEST_F(ImageModeTest, FailsWithCallingWrongMethod) {
  MP_ASSERT_OK_AND_ASSIGN(Image image,
                          DecodeImageFromFile(mediapipe::file::JoinPath(
                              "./", kTestDataDirectory, kPoseImage)));
  auto options = std::make_unique<HolisticLandmarkerOptions>();
  options->base_options.model_asset_path = mediapipe::file::JoinPath(
      "./", kTestDataDirectory, kHolisticLandmarkerBundleAsset);
  options->running_mode = core::RunningMode::IMAGE;

  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HolisticLandmarker> holistic_landmarker,
      HolisticLandmarker::Create(std::move(options)));
  auto results = holistic_landmarker->DetectForVideo(image, 0);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the video mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError)));

  auto status = holistic_landmarker->DetectAsync(image, 0);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              HasSubstr("not initialized with the live stream mode"));
  EXPECT_THAT(status.GetPayload(kMediaPipeTasksPayload),
              Optional(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError)));
  MP_ASSERT_OK(holistic_landmarker->Close());
}

TEST_F(ImageModeTest, FailsWithRegionOfInterest) {
  MP_ASSERT_OK_AND_ASSIGN(Image image,
                          DecodeImageFromFile(mediapipe::file::JoinPath(
                              "./", kTestDataDirectory, kPoseImage)));
  auto options = std::make_unique<HolisticLandmarkerOptions>();
  options->base_options.model_asset_path = mediapipe::file::JoinPath(
      "./", kTestDataDirectory, kHolisticLandmarkerBundleAsset);
  options->running_mode = core::RunningMode::IMAGE;
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HolisticLandmarker> holistic_landmarker,
      HolisticLandmarker::Create(std::move(options)));
  RectF roi{/*left=*/0.1, /*top=*/0, /*right=*/0.9, /*bottom=*/1};
  ImageProcessingOptions image_processing_options{roi, /*rotation_degrees=*/0};

  auto results = holistic_landmarker->Detect(image, image_processing_options);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("This task doesn't support region-of-interest"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::StrCat(
                  MediaPipeTasksStatus::kImageProcessingInvalidArgumentError)));
}

class VideoModeTest : public testing::Test {
 protected:
  void SetUp() override {
    expected_result_ = GetExpectedHolisticResult(kHolisticResultProto);
  }
  proto::HolisticResult expected_result_;
};

TEST_F(VideoModeTest, FailsWithCallingWrongMethod) {
  MP_ASSERT_OK_AND_ASSIGN(Image image,
                          DecodeImageFromFile(mediapipe::file::JoinPath(
                              "./", kTestDataDirectory, kPoseImage)));
  auto options = std::make_unique<HolisticLandmarkerOptions>();
  options->base_options.model_asset_path = mediapipe::file::JoinPath(
      "./", kTestDataDirectory, kHolisticLandmarkerBundleAsset);
  options->running_mode = core::RunningMode::VIDEO;

  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HolisticLandmarker> holistic_landmarker,
      HolisticLandmarker::Create(std::move(options)));
  auto results = holistic_landmarker->Detect(image);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the image mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError)));

  auto status = holistic_landmarker->DetectAsync(image, 0);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              HasSubstr("not initialized with the live stream mode"));
  EXPECT_THAT(status.GetPayload(kMediaPipeTasksPayload),
              Optional(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError)));
  MP_ASSERT_OK(holistic_landmarker->Close());
}

TEST_F(VideoModeTest, Succeeds) {
  MP_ASSERT_OK_AND_ASSIGN(Image image,
                          DecodeImageFromFile(mediapipe::file::JoinPath(
                              "./", kTestDataDirectory, kPoseImage)));
  auto options = std::make_unique<HolisticLandmarkerOptions>();
  options->base_options.model_asset_path = mediapipe::file::JoinPath(
      "./", kTestDataDirectory, kHolisticLandmarkerBundleAsset);
  options->running_mode = core::RunningMode::VIDEO;
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HolisticLandmarker> holistic_landmarker,
      HolisticLandmarker::Create(std::move(options)));
  for (int i = 0; i < 3; ++i) {
    MP_ASSERT_OK_AND_ASSIGN(auto results,
                            holistic_landmarker->DetectForVideo(image, i));
    AssertHolisticLandmarkerResultCorrect(results, expected_result_);
  }
  MP_ASSERT_OK(holistic_landmarker->Close());
}

class LiveStreamModeTest : public testing::Test {
 protected:
  void SetUp() override {
    expected_result_ = GetExpectedHolisticResult(kHolisticResultProto);
  }
  proto::HolisticResult expected_result_;
};

TEST_F(LiveStreamModeTest, FailsWithCallingWrongMethod) {
  MP_ASSERT_OK_AND_ASSIGN(Image image,
                          DecodeImageFromFile(mediapipe::file::JoinPath(
                              "./", kTestDataDirectory, kPoseImage)));
  auto options = std::make_unique<HolisticLandmarkerOptions>();
  options->base_options.model_asset_path = mediapipe::file::JoinPath(
      "./", kTestDataDirectory, kHolisticLandmarkerBundleAsset);
  options->running_mode = core::RunningMode::LIVE_STREAM;
  options->result_callback =
      [](absl::StatusOr<HolisticLandmarkerResult> results, const Image& image,
         int64_t timestamp_ms) {};

  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HolisticLandmarker> holistic_landmarker,
      HolisticLandmarker::Create(std::move(options)));
  auto results = holistic_landmarker->Detect(image);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the image mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError)));

  results = holistic_landmarker->DetectForVideo(image, 0);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the video mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError)));
  MP_ASSERT_OK(holistic_landmarker->Close());
}

TEST_F(LiveStreamModeTest, FailsWithOutOfOrderInputTimestamps) {
  MP_ASSERT_OK_AND_ASSIGN(Image image,
                          DecodeImageFromFile(mediapipe::file::JoinPath(
                              "./", kTestDataDirectory, kPoseImage)));
  auto options = std::make_unique<HolisticLandmarkerOptions>();
  options->base_options.model_asset_path = mediapipe::file::JoinPath(
      "./", kTestDataDirectory, kHolisticLandmarkerBundleAsset);
  options->running_mode = core::RunningMode::LIVE_STREAM;
  options->result_callback = [](absl::StatusOr<HolisticLandmarkerResult>,
                                const Image&, int64_t) {};
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HolisticLandmarker> holistic_landmarker,
      HolisticLandmarker::Create(std::move(options)));
  MP_EXPECT_OK(holistic_landmarker->DetectAsync(image, 1));
  auto status = holistic_landmarker->DetectAsync(image, 0);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              HasSubstr("Input timestamp must be monotonically increasing"));
  MP_ASSERT_OK(holistic_landmarker->Close());
}

TEST_F(LiveStreamModeTest, Succeeds) {
  static constexpr int kTimestampMs = 1337;
  MP_ASSERT_OK_AND_ASSIGN(Image image,
                          DecodeImageFromFile(mediapipe::file::JoinPath(
                              "./", kTestDataDirectory, kPoseImage)));
  std::vector<HolisticLandmarkerResult> results_list;
  proto::HolisticResult expected_result = expected_result_;
  auto options = std::make_unique<HolisticLandmarkerOptions>();
  options->base_options.model_asset_path = mediapipe::file::JoinPath(
      "./", kTestDataDirectory, kHolisticLandmarkerBundleAsset);
  options->running_mode = core::RunningMode::LIVE_STREAM;
  options->result_callback =
      [&results_list, expected_result](
          absl::StatusOr<HolisticLandmarkerResult> results, const Image& image,
          int64_t timestamp_ms) {
        MP_ASSERT_OK(results);
        results_list.push_back(std::move(results.value()));
        ASSERT_EQ(timestamp_ms, kTimestampMs);
        AssertHolisticLandmarkerResultCorrect(results_list.back(),
                                              expected_result);
      };

  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HolisticLandmarker> holistic_landmarker,
      HolisticLandmarker::Create(std::move(options)));
  MP_ASSERT_OK(holistic_landmarker->DetectAsync(image, kTimestampMs));
  MP_ASSERT_OK(holistic_landmarker->Close());
  EXPECT_GE(results_list.size(), 1);
}

TEST_F(LiveStreamModeTest, SucceedsWithFlowLimiting) {
  static constexpr int kNumFrames = 100;
  MP_ASSERT_OK_AND_ASSIGN(Image image,
                          DecodeImageFromFile(mediapipe::file::JoinPath(
                              "./", kTestDataDirectory, kPoseImage)));
  std::vector<HolisticLandmarkerResult> results_list;
  proto::HolisticResult expected_result = expected_result_;
  auto options = std::make_unique<HolisticLandmarkerOptions>();
  options->base_options.model_asset_path = mediapipe::file::JoinPath(
      "./", kTestDataDirectory, kHolisticLandmarkerBundleAsset);
  options->running_mode = core::RunningMode::LIVE_STREAM;
  options->output_pose_segmentation_masks = true;
  options->result_callback =
      [&results_list, expected_result](
          absl::StatusOr<HolisticLandmarkerResult> results, const Image& image,
          int64_t timestamp_ms) {
        MP_ASSERT_OK(results);
        results_list.push_back(std::move(results.value()));
        AssertHolisticLandmarkerResultCorrect(
            results_list.back(), expected_result,
            /* has_segmentation_masks= */ true);
      };

  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HolisticLandmarker> holistic_landmarker,
      HolisticLandmarker::Create(std::move(options)));
  for (int i = 0; i < kNumFrames; ++i) {
    MP_ASSERT_OK(holistic_landmarker->DetectAsync(image, i));
  }
  MP_ASSERT_OK(holistic_landmarker->Close());
  EXPECT_LE(results_list.size(), kNumFrames);
}

}  // namespace holistic_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
