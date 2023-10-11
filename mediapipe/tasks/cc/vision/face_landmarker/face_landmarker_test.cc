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

#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
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
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/matrix_data.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/containers/category.h"
#include "mediapipe/tasks/cc/components/containers/classification_result.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"
#include "mediapipe/tasks/cc/components/containers/rect.h"
#include "mediapipe/tasks/cc/components/processors/proto/classifier_options.pb.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/face_geometry.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker_result.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace face_landmarker {
namespace {

using ::file::Defaults;
using ::mediapipe::tasks::vision::core::ImageProcessingOptions;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::Values;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kFaceLandmarkerWithBlendshapesModelBundleName[] =
    "face_landmarker_v2_with_blendshapes.task";
constexpr char kPortraitImageName[] = "portrait.jpg";
constexpr char kPortraitExpectedFaceLandmarksName[] =
    "portrait_expected_face_landmarks.pbtxt";
constexpr char kPortraitExpectedBlendshapesName[] =
    "portrait_expected_blendshapes.pbtxt";
constexpr char kPortraitExpectedFaceGeometryName[] =
    "portrait_expected_face_geometry.pbtxt";

constexpr float kLandmarksDiffMargin = 0.03;
constexpr float kBlendshapesDiffMargin = 0.12;
constexpr float kFacialTransformationMatrixDiffMargin = 0.02;

template <typename ProtoT>
ProtoT GetExpectedProto(absl::string_view filename) {
  ProtoT expected_proto;
  MP_EXPECT_OK(GetTextProto(file::JoinPath("./", kTestDataDirectory, filename),
                            &expected_proto, Defaults()));
  return expected_proto;
}

// Struct holding the parameters for parameterized FaceLandmarkerGraphTest
// class.
struct FaceLandmarkerTestParams {
  // The name of this test, for convenience when displaying test results.
  std::string test_name;
  // The filename of the model to test.
  std::string input_model_name;
  // The filename of the test image.
  std::string test_image_name;
  // The rotation to apply to the test image before processing, in degrees
  // clockwise.
  int rotation;
  // The expected output face landmarker result.
  FaceLandmarkerResult expected_result;
};

mediapipe::MatrixData MakePortraitExpectedFacialTransformationMatrix() {
  auto face_geometry = GetExpectedProto<face_geometry::proto::FaceGeometry>(
      kPortraitExpectedFaceGeometryName);
  return face_geometry.pose_transform_matrix();
}

testing::Matcher<components::containers::NormalizedLandmark> LandmarkIs(
    const components::containers::NormalizedLandmark& landmark) {
  return testing::AllOf(
      testing::Field(&components::containers::NormalizedLandmark::x,
                     testing::FloatNear(landmark.x, kLandmarksDiffMargin)),
      testing::Field(&components::containers::NormalizedLandmark::y,
                     testing::FloatNear(landmark.y, kLandmarksDiffMargin)));
}

void ExpectLandmarksCorrect(
    const std::vector<components::containers::NormalizedLandmarks>
        actual_landmarks,
    const std::vector<components::containers::NormalizedLandmarks>
        expected_landmarks) {
  ASSERT_EQ(actual_landmarks.size(), expected_landmarks.size());
  for (int i = 0; i < actual_landmarks.size(); ++i) {
    ASSERT_EQ(actual_landmarks[i].landmarks.size(),
              expected_landmarks[i].landmarks.size());
    for (int j = 0; j < actual_landmarks[i].landmarks.size(); ++j) {
      EXPECT_THAT(actual_landmarks[i].landmarks[j],
                  LandmarkIs(expected_landmarks[i].landmarks[j]));
    }
  }
}

testing::Matcher<components::containers::Category> CategoryIs(
    const components::containers::Category& category) {
  return testing::AllOf(
      testing::Field(&components::containers::Category::index,
                     testing::Eq(category.index)),
      testing::Field(
          &components::containers::Category::score,
          testing::FloatNear(category.score, kBlendshapesDiffMargin)));
}

void ExpectBlendshapesCorrect(
    const std::vector<components::containers::Classifications>&
        actual_blendshapes,
    const std::vector<components::containers::Classifications>&
        expected_blendshapes) {
  ASSERT_EQ(actual_blendshapes.size(), expected_blendshapes.size());
  for (int i = 0; i < actual_blendshapes.size(); ++i) {
    ASSERT_EQ(actual_blendshapes[i].categories.size(),
              expected_blendshapes[i].categories.size());
    for (int j = 0; j < actual_blendshapes[i].categories.size(); ++j) {
      EXPECT_THAT(actual_blendshapes[i].categories[j],
                  CategoryIs(expected_blendshapes[i].categories[j]));
    }
  }
}

void ExpectFacialTransformationMatrixCorrect(
    const std::vector<Matrix>& actual_matrix_list,
    const std::vector<Matrix>& expected_matrix_list) {
  ASSERT_EQ(actual_matrix_list.size(), expected_matrix_list.size());
  for (int i = 0; i < actual_matrix_list.size(); ++i) {
    const Matrix& actual_matrix = actual_matrix_list[i];
    const Matrix& expected_matrix = expected_matrix_list[i];
    ASSERT_EQ(actual_matrix.cols(), expected_matrix.cols());
    ASSERT_EQ(actual_matrix.rows(), expected_matrix.rows());
    for (int i = 0; i < actual_matrix.size(); ++i) {
      EXPECT_NEAR(actual_matrix.data()[i], expected_matrix.data()[i],
                  kFacialTransformationMatrixDiffMargin);
    }
  }
}

void ExpectFaceLandmarkerResultCorrect(
    const FaceLandmarkerResult& actual_result,
    const FaceLandmarkerResult& expected_result) {
  ExpectLandmarksCorrect(actual_result.face_landmarks,
                         expected_result.face_landmarks);

  ASSERT_EQ(actual_result.face_blendshapes.has_value(),
            expected_result.face_blendshapes.has_value());
  if (expected_result.face_blendshapes.has_value()) {
    ASSERT_TRUE(actual_result.face_blendshapes.has_value());
    ExpectBlendshapesCorrect(*actual_result.face_blendshapes,
                             *expected_result.face_blendshapes);
  }

  ASSERT_EQ(actual_result.facial_transformation_matrixes.has_value(),
            expected_result.facial_transformation_matrixes.has_value());
  if (expected_result.facial_transformation_matrixes.has_value()) {
    ASSERT_TRUE(actual_result.facial_transformation_matrixes.has_value());
    ExpectFacialTransformationMatrixCorrect(
        *actual_result.facial_transformation_matrixes,
        *expected_result.facial_transformation_matrixes);
  }
}

class ImageModeTest : public TestWithParam<FaceLandmarkerTestParams> {};

TEST_P(ImageModeTest, Succeeds) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image, DecodeImageFromFile(file::JoinPath(
                       "./", kTestDataDirectory, GetParam().test_image_name)));
  auto options = std::make_unique<FaceLandmarkerOptions>();
  options->base_options.model_asset_path =
      file::JoinPath("./", kTestDataDirectory, GetParam().input_model_name);
  options->running_mode = core::RunningMode::IMAGE;
  options->output_face_blendshapes =
      GetParam().expected_result.face_blendshapes.has_value();
  options->output_facial_transformation_matrixes =
      GetParam().expected_result.facial_transformation_matrixes.has_value();

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<FaceLandmarker> face_landmarker,
                          FaceLandmarker::Create(std::move(options)));
  FaceLandmarkerResult actual_result;
  if (GetParam().rotation != 0) {
    ImageProcessingOptions image_processing_options;
    image_processing_options.rotation_degrees = GetParam().rotation;
    MP_ASSERT_OK_AND_ASSIGN(
        actual_result,
        face_landmarker->Detect(image, image_processing_options));
  } else {
    MP_ASSERT_OK_AND_ASSIGN(actual_result, face_landmarker->Detect(image));
  }
  ExpectFaceLandmarkerResultCorrect(actual_result, GetParam().expected_result);
  MP_ASSERT_OK(face_landmarker->Close());
}

INSTANTIATE_TEST_SUITE_P(
    FaceLandmarkerTest, ImageModeTest,
    Values(
        FaceLandmarkerTestParams{/* test_name= */ "PortraitV2",
                                 /* input_model_name= */
                                 kFaceLandmarkerWithBlendshapesModelBundleName,
                                 /* test_image_name= */ kPortraitImageName,
                                 /* rotation= */ 0,
                                 /* expected_result= */
                                 ConvertToFaceLandmarkerResult(
                                     {GetExpectedProto<NormalizedLandmarkList>(
                                         kPortraitExpectedFaceLandmarksName)})},
        FaceLandmarkerTestParams{/* test_name= */ "PortraitWithBlendshapes",
                                 /* input_model_name= */
                                 kFaceLandmarkerWithBlendshapesModelBundleName,
                                 /* test_image_name= */ kPortraitImageName,
                                 /* rotation= */ 0,
                                 /* expected_result= */
                                 ConvertToFaceLandmarkerResult(
                                     {GetExpectedProto<NormalizedLandmarkList>(
                                         kPortraitExpectedFaceLandmarksName)},
                                     {{GetExpectedProto<ClassificationList>(
                                         kPortraitExpectedBlendshapesName)}})},
        FaceLandmarkerTestParams{
            /* test_name= */ "PortraitWithBlendshapesWithFacialTransformatio"
                             "nMatrix",
            /* input_model_name= */
            kFaceLandmarkerWithBlendshapesModelBundleName,
            /* test_image_name= */ kPortraitImageName,
            /* rotation= */ 0,
            /* expected_result= */
            ConvertToFaceLandmarkerResult(
                {GetExpectedProto<NormalizedLandmarkList>(
                    kPortraitExpectedFaceLandmarksName)},
                {{GetExpectedProto<ClassificationList>(
                    kPortraitExpectedBlendshapesName)}},
                {{MakePortraitExpectedFacialTransformationMatrix()}})}),
    [](const TestParamInfo<ImageModeTest::ParamType>& info) {
      return info.param.test_name;
    });

class VideoModeTest : public TestWithParam<FaceLandmarkerTestParams> {};

TEST_P(VideoModeTest, Succeeds) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image, DecodeImageFromFile(file::JoinPath(
                       "./", kTestDataDirectory, GetParam().test_image_name)));
  auto options = std::make_unique<FaceLandmarkerOptions>();
  options->base_options.model_asset_path =
      file::JoinPath("./", kTestDataDirectory, GetParam().input_model_name);
  options->running_mode = core::RunningMode::VIDEO;
  options->output_face_blendshapes =
      GetParam().expected_result.face_blendshapes.has_value();
  options->output_facial_transformation_matrixes =
      GetParam().expected_result.facial_transformation_matrixes.has_value();

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<FaceLandmarker> face_landmarker,
                          FaceLandmarker::Create(std::move(options)));
  for (int i = 0; i < 3; ++i) {
    FaceLandmarkerResult actual_result;
    if (GetParam().rotation != 0) {
      ImageProcessingOptions image_processing_options;
      image_processing_options.rotation_degrees = GetParam().rotation;
      MP_ASSERT_OK_AND_ASSIGN(
          actual_result,
          face_landmarker->DetectForVideo(image, i, image_processing_options));
    } else {
      MP_ASSERT_OK_AND_ASSIGN(actual_result,
                              face_landmarker->DetectForVideo(image, i));
    }
    ExpectFaceLandmarkerResultCorrect(actual_result,
                                      GetParam().expected_result);
  }
  MP_ASSERT_OK(face_landmarker->Close());
}

INSTANTIATE_TEST_SUITE_P(
    FaceLandmarkerTest, VideoModeTest,
    Values(

        FaceLandmarkerTestParams{/* test_name= */ "Portrait",
                                 /* input_model_name= */
                                 kFaceLandmarkerWithBlendshapesModelBundleName,
                                 /* test_image_name= */ kPortraitImageName,
                                 /* rotation= */ 0,
                                 /* expected_result= */
                                 ConvertToFaceLandmarkerResult(
                                     {GetExpectedProto<NormalizedLandmarkList>(
                                         kPortraitExpectedFaceLandmarksName)})},
        FaceLandmarkerTestParams{/* test_name= */ "PortraitWithBlendshapes",
                                 /* input_model_name= */
                                 kFaceLandmarkerWithBlendshapesModelBundleName,
                                 /* test_image_name= */ kPortraitImageName,
                                 /* rotation= */ 0,
                                 /* expected_result= */
                                 ConvertToFaceLandmarkerResult(
                                     {GetExpectedProto<NormalizedLandmarkList>(
                                         kPortraitExpectedFaceLandmarksName)},
                                     {{GetExpectedProto<ClassificationList>(
                                         kPortraitExpectedBlendshapesName)}})}),
    [](const TestParamInfo<VideoModeTest::ParamType>& info) {
      return info.param.test_name;
    });

class LiveStreamModeTest : public TestWithParam<FaceLandmarkerTestParams> {};

TEST_P(LiveStreamModeTest, Succeeds) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image, DecodeImageFromFile(file::JoinPath(
                       "./", kTestDataDirectory, GetParam().test_image_name)));
  auto options = std::make_unique<FaceLandmarkerOptions>();
  options->base_options.model_asset_path =
      file::JoinPath("./", kTestDataDirectory, GetParam().input_model_name);
  options->running_mode = core::RunningMode::LIVE_STREAM;
  options->output_face_blendshapes =
      GetParam().expected_result.face_blendshapes.has_value();
  options->output_facial_transformation_matrixes =
      GetParam().expected_result.facial_transformation_matrixes.has_value();

  std::vector<FaceLandmarkerResult> face_landmarker_results;
  std::vector<int64_t> timestamps;
  options->result_callback = [&face_landmarker_results, &timestamps](
                                 absl::StatusOr<FaceLandmarkerResult> result,
                                 const Image& image, int64_t timestamp_ms) {
    MP_ASSERT_OK(result.status());
    face_landmarker_results.push_back(std::move(result.value()));
    timestamps.push_back(timestamp_ms);
  };

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<FaceLandmarker> face_landmarker,
                          FaceLandmarker::Create(std::move(options)));

  const int iterations = 100;
  for (int i = 0; i < iterations; ++i) {
    FaceLandmarkerResult actual_result;
    if (GetParam().rotation != 0) {
      ImageProcessingOptions image_processing_options;
      image_processing_options.rotation_degrees = GetParam().rotation;
      MP_ASSERT_OK(
          face_landmarker->DetectAsync(image, i, image_processing_options));
    } else {
      MP_ASSERT_OK(face_landmarker->DetectAsync(image, i));
    }
  }
  MP_ASSERT_OK(face_landmarker->Close());

  // Due to the flow limiter, the total of outputs will be smaller than the
  // number of iterations.
  ASSERT_LE(face_landmarker_results.size(), iterations);
  ASSERT_GT(face_landmarker_results.size(), 0);

  for (int i = 0; i < face_landmarker_results.size(); ++i) {
    ExpectFaceLandmarkerResultCorrect(face_landmarker_results[i],
                                      GetParam().expected_result);
  }
  int64_t timestamp_ms = -1;
  for (const auto& timestamp : timestamps) {
    EXPECT_GT(timestamp, timestamp_ms);
    timestamp_ms = timestamp;
  }
}

INSTANTIATE_TEST_SUITE_P(
    FaceLandmarkerTest, LiveStreamModeTest,
    Values(
        FaceLandmarkerTestParams{/* test_name= */ "Portrait",
                                 /* input_model_name= */
                                 kFaceLandmarkerWithBlendshapesModelBundleName,
                                 /* test_image_name= */ kPortraitImageName,
                                 /* rotation= */ 0,
                                 /* expected_result= */
                                 ConvertToFaceLandmarkerResult(
                                     {GetExpectedProto<NormalizedLandmarkList>(
                                         kPortraitExpectedFaceLandmarksName)})},
        FaceLandmarkerTestParams{/* test_name= */ "PortraitWithBlendshapes",
                                 /* input_model_name= */
                                 kFaceLandmarkerWithBlendshapesModelBundleName,
                                 /* test_image_name= */ kPortraitImageName,
                                 /* rotation= */ 0,
                                 /* expected_result= */
                                 ConvertToFaceLandmarkerResult(
                                     {GetExpectedProto<NormalizedLandmarkList>(
                                         kPortraitExpectedFaceLandmarksName)},
                                     {{GetExpectedProto<ClassificationList>(
                                         kPortraitExpectedBlendshapesName)}})}),
    [](const TestParamInfo<LiveStreamModeTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace face_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
