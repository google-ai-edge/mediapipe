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

#include "mediapipe/tasks/cc/vision/face_detector/face_detector.h"

#include <vector>

#include "absl/log/absl_check.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/containers/detection_result.h"
#include "mediapipe/tasks/cc/components/containers/keypoint.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"
#include "testing/base/public/gunit.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace face_detector {
namespace {

using ::file::Defaults;
using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::components::containers::NormalizedKeypoint;
using ::mediapipe::tasks::vision::core::ImageProcessingOptions;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::Values;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kShortRangeBlazeFaceModel[] =
    "face_detection_short_range.tflite";
constexpr char kPortraitImage[] = "portrait.jpg";
constexpr char kPortraitRotatedImage[] = "portrait_rotated.jpg";
constexpr char kPortraitExpectedDetection[] =
    "portrait_expected_detection.pbtxt";
constexpr char kPortraitRotatedExpectedDetection[] =
    "portrait_rotated_expected_detection.pbtxt";
constexpr char kCatImageName[] = "cat.jpg";
constexpr float kKeypointErrorThreshold = 1e-2;

FaceDetectorResult GetExpectedFaceDetectorResult(absl::string_view file_name) {
  mediapipe::Detection detection;
  ABSL_CHECK_OK(GetTextProto(
      file::JoinPath(::testing::SrcDir(), kTestDataDirectory, file_name),
      &detection, Defaults()))
      << "Expected face detection result does not exist.";
  return components::containers::ConvertToDetectionResult({detection});
}

void ExpectKeypointsCorrect(
    const std::vector<NormalizedKeypoint> actual_keypoints,
    const std::vector<NormalizedKeypoint> expected_keypoints) {
  ASSERT_EQ(actual_keypoints.size(), expected_keypoints.size());
  for (int i = 0; i < actual_keypoints.size(); i++) {
    EXPECT_NEAR(actual_keypoints[i].x, expected_keypoints[i].x,
                kKeypointErrorThreshold);
    EXPECT_NEAR(actual_keypoints[i].y, expected_keypoints[i].y,
                kKeypointErrorThreshold);
  }
}

void ExpectFaceDetectorResultsCorrect(
    const FaceDetectorResult& actual_results,
    const FaceDetectorResult& expected_results) {
  EXPECT_EQ(actual_results.detections.size(),
            expected_results.detections.size());
  for (int i = 0; i < actual_results.detections.size(); i++) {
    const auto& actual_bbox = actual_results.detections[i].bounding_box;
    const auto& expected_bbox = expected_results.detections[i].bounding_box;
    EXPECT_EQ(actual_bbox, expected_bbox);
    ASSERT_TRUE(actual_results.detections[i].keypoints.has_value());
    ExpectKeypointsCorrect(actual_results.detections[i].keypoints.value(),
                           expected_results.detections[i].keypoints.value());
  }
}

struct TestParams {
  // The name of this test, for convenience when displaying test results.
  std::string test_name;
  // The filename of test image.
  std::string test_image_name;
  // The filename of face landmark detection model.
  std::string face_detection_model_name;
  // The rotation to apply to the test image before processing, in degrees
  // clockwise.
  int rotation;
  // Expected face detector results.
  FaceDetectorResult expected_result;
};

class ImageModeTest : public testing::TestWithParam<TestParams> {};

TEST_P(ImageModeTest, Succeeds) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath(::testing::SrcDir(), kTestDataDirectory,
                                   GetParam().test_image_name)));
  auto options = std::make_unique<FaceDetectorOptions>();
  options->base_options.model_asset_path =
      JoinPath(::testing::SrcDir(), kTestDataDirectory,
               GetParam().face_detection_model_name);
  options->running_mode = core::RunningMode::IMAGE;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<FaceDetector> face_detector,
                          FaceDetector::Create(std::move(options)));
  FaceDetectorResult face_detector_result;
  if (GetParam().rotation != 0) {
    ImageProcessingOptions image_processing_options;
    image_processing_options.rotation_degrees = GetParam().rotation;
    MP_ASSERT_OK_AND_ASSIGN(
        face_detector_result,
        face_detector->Detect(image, image_processing_options));
  } else {
    MP_ASSERT_OK_AND_ASSIGN(face_detector_result, face_detector->Detect(image));
  }
  ExpectFaceDetectorResultsCorrect(face_detector_result,
                                   GetParam().expected_result);
  MP_ASSERT_OK(face_detector->Close());
}

INSTANTIATE_TEST_SUITE_P(
    FaceDetectorTest, ImageModeTest,
    Values(
        TestParams{/* test_name= */ "PortraitShortRange",
                   /* test_image_name= */ kPortraitImage,
                   /* face_detection_model_name= */ kShortRangeBlazeFaceModel,
                   /* rotation= */ 0,
                   /* expected_result = */
                   GetExpectedFaceDetectorResult(kPortraitExpectedDetection)},
        TestParams{
            /* test_name= */ "PortraitRotatedShortRange",
            /* test_image_name= */ kPortraitRotatedImage,
            /* face_detection_model_name= */ kShortRangeBlazeFaceModel,
            /* rotation= */ -90,
            /* expected_result = */
            GetExpectedFaceDetectorResult(kPortraitRotatedExpectedDetection)},
        TestParams{/* test_name= */ "NoFace",
                   /* test_image_name= */ kCatImageName,
                   /* face_detection_model_name= */ kShortRangeBlazeFaceModel,
                   /* rotation= */ 0,
                   /* expected_result = */
                   {}}),
    [](const TestParamInfo<ImageModeTest::ParamType>& info) {
      return info.param.test_name;
    });

class VideoModeTest : public testing::TestWithParam<TestParams> {};

TEST_P(VideoModeTest, Succeeds) {
  const int iterations = 100;
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath(::testing::SrcDir(), kTestDataDirectory,
                                   GetParam().test_image_name)));
  auto options = std::make_unique<FaceDetectorOptions>();
  options->base_options.model_asset_path =
      JoinPath(::testing::SrcDir(), kTestDataDirectory,
               GetParam().face_detection_model_name);
  options->running_mode = core::RunningMode::VIDEO;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<FaceDetector> face_detector,
                          FaceDetector::Create(std::move(options)));
  const FaceDetectorResult& expected_result = GetParam().expected_result;

  for (int i = 0; i < iterations; i++) {
    FaceDetectorResult face_detector_result;
    if (GetParam().rotation != 0) {
      ImageProcessingOptions image_processing_options;
      image_processing_options.rotation_degrees = GetParam().rotation;
      MP_ASSERT_OK_AND_ASSIGN(
          face_detector_result,
          face_detector->DetectForVideo(image, i, image_processing_options));
    } else {
      MP_ASSERT_OK_AND_ASSIGN(face_detector_result,
                              face_detector->DetectForVideo(image, i));
    }
    ExpectFaceDetectorResultsCorrect(face_detector_result, expected_result);
  }
  MP_ASSERT_OK(face_detector->Close());
}

INSTANTIATE_TEST_SUITE_P(
    FaceDetectorTest, VideoModeTest,
    Values(
        TestParams{/* test_name= */ "PortraitShortRange",
                   /* test_image_name= */ kPortraitImage,
                   /* face_detection_model_name= */ kShortRangeBlazeFaceModel,
                   /* rotation= */ 0,
                   /* expected_result = */
                   GetExpectedFaceDetectorResult(kPortraitExpectedDetection)},
        TestParams{
            /* test_name= */ "PortraitRotatedShortRange",
            /* test_image_name= */ kPortraitRotatedImage,
            /* face_detection_model_name= */ kShortRangeBlazeFaceModel,
            /* rotation= */ -90,
            /* expected_result = */
            GetExpectedFaceDetectorResult(kPortraitRotatedExpectedDetection)},
        TestParams{/* test_name= */ "NoFace",
                   /* test_image_name= */ kCatImageName,
                   /* face_detection_model_name= */ kShortRangeBlazeFaceModel,
                   /* rotation= */ 0,
                   /* expected_result = */
                   {}}),
    [](const TestParamInfo<ImageModeTest::ParamType>& info) {
      return info.param.test_name;
    });

class LiveStreamModeTest : public testing::TestWithParam<TestParams> {};

TEST_P(LiveStreamModeTest, Succeeds) {
  const int iterations = 100;
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath(::testing::SrcDir(), kTestDataDirectory,
                                   GetParam().test_image_name)));
  auto options = std::make_unique<FaceDetectorOptions>();
  options->base_options.model_asset_path =
      JoinPath(::testing::SrcDir(), kTestDataDirectory,
               GetParam().face_detection_model_name);
  options->running_mode = core::RunningMode::LIVE_STREAM;
  std::vector<FaceDetectorResult> face_detector_results;
  std::vector<std::pair<int, int>> image_sizes;
  std::vector<uint64_t> timestamps;
  options->result_callback = [&face_detector_results, &image_sizes,
                              &timestamps](
                                 absl::StatusOr<FaceDetectorResult> results,
                                 const Image& image, uint64_t timestamp_ms) {
    MP_ASSERT_OK(results.status());
    face_detector_results.push_back(std::move(results.value()));
    image_sizes.push_back({image.width(), image.height()});
    timestamps.push_back(timestamp_ms);
  };
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<FaceDetector> face_detector,
                          FaceDetector::Create(std::move(options)));
  for (int i = 0; i < iterations; ++i) {
    if (GetParam().rotation != 0) {
      ImageProcessingOptions image_processing_options;
      image_processing_options.rotation_degrees = GetParam().rotation;
      MP_ASSERT_OK(
          face_detector->DetectAsync(image, i + 1, image_processing_options));
    } else {
      MP_ASSERT_OK(face_detector->DetectAsync(image, i + 1));
    }
  }
  MP_ASSERT_OK(face_detector->Close());

  ASSERT_LE(face_detector_results.size(), iterations);
  ASSERT_GT(face_detector_results.size(), 0);
  const FaceDetectorResult& expected_results = GetParam().expected_result;
  for (int i = 0; i < face_detector_results.size(); ++i) {
    ExpectFaceDetectorResultsCorrect(face_detector_results[i],
                                     expected_results);
  }
  for (const auto& image_size : image_sizes) {
    EXPECT_EQ(image_size.first, image.width());
    EXPECT_EQ(image_size.second, image.height());
  }
  uint64_t timestamp_ms = 0;
  for (const auto& timestamp : timestamps) {
    EXPECT_GT(timestamp, timestamp_ms);
    timestamp_ms = timestamp;
  }
}

INSTANTIATE_TEST_SUITE_P(
    FaceDetectorTest, LiveStreamModeTest,
    Values(
        TestParams{/* test_name= */ "PortraitShortRange",
                   /* test_image_name= */ kPortraitImage,
                   /* face_detection_model_name= */ kShortRangeBlazeFaceModel,
                   /* rotation= */ 0,
                   /* expected_result = */
                   GetExpectedFaceDetectorResult(kPortraitExpectedDetection)},
        TestParams{
            /* test_name= */ "PortraitRotatedShortRange",
            /* test_image_name= */ kPortraitRotatedImage,
            /* face_detection_model_name= */ kShortRangeBlazeFaceModel,
            /* rotation= */ -90,
            /* expected_result = */
            GetExpectedFaceDetectorResult(kPortraitRotatedExpectedDetection)},
        TestParams{/* test_name= */ "NoFace",
                   /* test_image_name= */ kCatImageName,
                   /* face_detection_model_name= */ kShortRangeBlazeFaceModel,
                   /* rotation= */ 0,
                   /* expected_result = */
                   {}}),
    [](const TestParamInfo<ImageModeTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace face_detector
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
