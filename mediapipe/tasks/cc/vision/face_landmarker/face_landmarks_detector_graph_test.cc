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

#include <optional>

#include "absl/flags/flag.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/cc/core/mediapipe_builtin_op_resolver.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_blendshapes_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarks_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace face_landmarker {
namespace {

using ::file::Defaults;
using ::file::GetTextProto;
using ::mediapipe::NormalizedRect;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::core::TaskRunner;
using ::mediapipe::tasks::vision::DecodeImageFromFile;
using ::testing::ElementsAreArray;
using ::testing::EqualsProto;
using ::testing::Pointwise;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::proto::Approximately;
using ::testing::proto::Partially;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kFaceLandmarksV2Model[] =
    "facemesh2_lite_iris_faceflag_2023_02_14.tflite";
constexpr char kPortraitImageName[] = "portrait.jpg";
constexpr char kCatImageName[] = "cat.jpg";
constexpr char kPortraitExpectedFaceLandmarksName[] =
    "portrait_expected_face_landmarks.pbtxt";
constexpr char kFaceBlendshapesModel[] = "face_blendshapes.tflite";
constexpr char kPortraitExpectedBlendshapesName[] =
    "portrait_expected_blendshapes.pbtxt";

constexpr char kImageTag[] = "IMAGE";
constexpr char kImageName[] = "image";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kNormRectName[] = "norm_rect";

constexpr char kNormLandmarksTag[] = "NORM_LANDMARKS";
constexpr char kNormLandmarksName[] = "norm_landmarks";
constexpr char kFaceRectNextFrameTag[] = "FACE_RECT_NEXT_FRAME";
constexpr char kFaceRectNextFrameName[] = "face_rect_next_frame";
constexpr char kFaceRectsNextFrameTag[] = "FACE_RECTS_NEXT_FRAME";
constexpr char kFaceRectsNextFrameName[] = "face_rects_next_frame";
constexpr char kPresenceTag[] = "PRESENCE";
constexpr char kPresenceName[] = "presence";
constexpr char kPresenceScoreTag[] = "PRESENCE_SCORE";
constexpr char kPresenceScoreName[] = "presence_score";
constexpr char kBlendshapesTag[] = "BLENDSHAPES";
constexpr char kBlendshapesName[] = "blendshapes";

constexpr float kFractionDiff = 0.05;  // percentage
constexpr float kAbsMargin = 0.03;
constexpr float kBlendshapesDiffMargin = 0.1;

// Helper function to create a Single Face Landmark TaskRunner.
absl::StatusOr<std::unique_ptr<TaskRunner>> CreateSingleFaceLandmarksTaskRunner(
    absl::string_view landmarks_model_name) {
  Graph graph;

  auto& face_landmark_detection = graph.AddNode(
      "mediapipe.tasks.vision.face_landmarker."
      "SingleFaceLandmarksDetectorGraph");

  auto options = std::make_unique<proto::FaceLandmarksDetectorGraphOptions>();
  options->mutable_base_options()->mutable_model_asset()->set_file_name(
      JoinPath("./", kTestDataDirectory, landmarks_model_name));
  options->set_min_detection_confidence(0.5);

  face_landmark_detection.GetOptions<proto::FaceLandmarksDetectorGraphOptions>()
      .Swap(options.get());

  graph[Input<Image>(kImageTag)].SetName(kImageName) >>
      face_landmark_detection.In(kImageTag);
  graph[Input<NormalizedRect>(kNormRectTag)].SetName(kNormRectName) >>
      face_landmark_detection.In(kNormRectTag);

  face_landmark_detection.Out(kNormLandmarksTag).SetName(kNormLandmarksName) >>
      graph[Output<NormalizedLandmarkList>(kNormLandmarksTag)];
  face_landmark_detection.Out(kPresenceTag).SetName(kPresenceName) >>
      graph[Output<bool>(kPresenceTag)];
  face_landmark_detection.Out(kPresenceScoreTag).SetName(kPresenceScoreName) >>
      graph[Output<float>(kPresenceScoreTag)];
  face_landmark_detection.Out(kFaceRectNextFrameTag)
          .SetName(kFaceRectNextFrameName) >>
      graph[Output<NormalizedRect>(kFaceRectNextFrameTag)];
  return TaskRunner::Create(
      graph.GetConfig(), absl::make_unique<core::MediaPipeBuiltinOpResolver>());
}

// Helper function to create a Multi Face Landmark TaskRunner.
absl::StatusOr<std::unique_ptr<TaskRunner>> CreateMultiFaceLandmarksTaskRunner(
    absl::string_view landmarks_model_name,
    std::optional<absl::string_view> blendshapes_model_name) {
  Graph graph;

  auto& face_landmark_detection = graph.AddNode(
      "mediapipe.tasks.vision.face_landmarker."
      "MultiFaceLandmarksDetectorGraph");

  auto options = std::make_unique<proto::FaceLandmarksDetectorGraphOptions>();
  options->mutable_base_options()->mutable_model_asset()->set_file_name(
      JoinPath("./", kTestDataDirectory, landmarks_model_name));
  options->set_min_detection_confidence(0.5);
  if (blendshapes_model_name.has_value()) {
    options->mutable_face_blendshapes_graph_options()
        ->mutable_base_options()
        ->mutable_model_asset()
        ->set_file_name(
            JoinPath("./", kTestDataDirectory, *blendshapes_model_name));
  }
  face_landmark_detection.GetOptions<proto::FaceLandmarksDetectorGraphOptions>()
      .Swap(options.get());

  graph[Input<Image>(kImageTag)].SetName(kImageName) >>
      face_landmark_detection.In(kImageTag);
  graph[Input<std::vector<NormalizedRect>>(kNormRectTag)].SetName(
      kNormRectName) >>
      face_landmark_detection.In(kNormRectTag);

  face_landmark_detection.Out(kNormLandmarksTag).SetName(kNormLandmarksName) >>
      graph[Output<std::vector<NormalizedLandmarkList>>(kNormLandmarksTag)];
  face_landmark_detection.Out(kPresenceTag).SetName(kPresenceName) >>
      graph[Output<std::vector<bool>>(kPresenceTag)];
  face_landmark_detection.Out(kPresenceScoreTag).SetName(kPresenceScoreName) >>
      graph[Output<std::vector<float>>(kPresenceScoreTag)];
  face_landmark_detection.Out(kFaceRectsNextFrameTag)
          .SetName(kFaceRectsNextFrameName) >>
      graph[Output<std::vector<NormalizedRect>>(kFaceRectsNextFrameTag)];
  if (blendshapes_model_name.has_value()) {
    face_landmark_detection.Out(kBlendshapesTag).SetName(kBlendshapesName) >>
        graph[Output<ClassificationList>(kBlendshapesTag)];
  }

  return TaskRunner::Create(
      graph.GetConfig(), absl::make_unique<core::MediaPipeBuiltinOpResolver>());
}

NormalizedLandmarkList GetExpectedLandmarkList(absl::string_view filename) {
  NormalizedLandmarkList expected_landmark_list;
  MP_EXPECT_OK(GetTextProto(file::JoinPath("./", kTestDataDirectory, filename),
                            &expected_landmark_list, Defaults()));
  return expected_landmark_list;
}

ClassificationList GetBlendshapes(absl::string_view filename) {
  ClassificationList blendshapes;
  MP_EXPECT_OK(GetTextProto(file::JoinPath("./", kTestDataDirectory, filename),
                            &blendshapes, Defaults()));
  return blendshapes;
}

// Helper function to construct NormalizeRect proto.
NormalizedRect MakeNormRect(float x_center, float y_center, float width,
                            float height, float rotation) {
  NormalizedRect face_rect;
  face_rect.set_x_center(x_center);
  face_rect.set_y_center(y_center);
  face_rect.set_width(width);
  face_rect.set_height(height);
  face_rect.set_rotation(rotation);
  return face_rect;
}

// Struct holding the parameters for parameterized FaceLandmarksDetectionTest
// class.
struct SingeFaceTestParams {
  // The name of this test, for convenience when displaying test results.
  std::string test_name;
  // The filename of landmarks model name.
  std::string landmarks_model_name;
  // The filename of the test image.
  std::string test_image_name;
  // RoI on image to detect faces.
  NormalizedRect norm_rect;
  // Expected face presence value.
  bool expected_presence;
  // The expected output landmarks positions.
  NormalizedLandmarkList expected_landmarks;
  // The max value difference between expected_positions and detected positions.
  float landmarks_diff_threshold;
};

struct MultiFaceTestParams {
  // The name of this test, for convenience when displaying test results.
  std::string test_name;
  // The filename of landmarks model name.
  std::string landmarks_model_name;
  // The filename of blendshape model name.
  std::optional<std::string> blendshape_model_name;
  // The filename of the test image.
  std::string test_image_name;
  // RoI on image to detect faces.
  std::vector<NormalizedRect> norm_rects;
  // Expected face presence value.
  std::vector<bool> expected_presence;
  // The expected output landmarks positions.
  std::optional<std::vector<NormalizedLandmarkList>> expected_landmarks_lists;
  // The expected output blendshape classification;
  std::optional<std::vector<ClassificationList>> expected_blendshapes;
  // The max value difference between expected_positions and detected positions.
  float landmarks_diff_threshold;
  // The max value difference between expected blendshapes and actual
  // blendshapes.
  float blendshapes_diff_threshold;
};

class SingleFaceLandmarksDetectionTest
    : public testing::TestWithParam<SingeFaceTestParams> {};

TEST_P(SingleFaceLandmarksDetectionTest, Succeeds) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image, DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                                GetParam().test_image_name)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto task_runner,
      CreateSingleFaceLandmarksTaskRunner(GetParam().landmarks_model_name));

  auto output_packets = task_runner->Process(
      {{kImageName, MakePacket<Image>(std::move(image))},
       {kNormRectName,
        MakePacket<NormalizedRect>(std::move(GetParam().norm_rect))}});
  MP_ASSERT_OK(output_packets);

  const bool presence = (*output_packets)[kPresenceName].Get<bool>();
  ASSERT_EQ(presence, GetParam().expected_presence);

  if (presence) {
    const NormalizedLandmarkList landmarks =
        (*output_packets)[kNormLandmarksName].Get<NormalizedLandmarkList>();
    const NormalizedLandmarkList& expected_landmarks =
        GetParam().expected_landmarks;
    EXPECT_THAT(
        landmarks,
        Approximately(Partially(EqualsProto(expected_landmarks)),
                      /*margin=*/kAbsMargin,
                      /*fraction=*/GetParam().landmarks_diff_threshold));
  }
}

class MultiFaceLandmarksDetectionTest
    : public testing::TestWithParam<MultiFaceTestParams> {};

TEST_P(MultiFaceLandmarksDetectionTest, Succeeds) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image, DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                                GetParam().test_image_name)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto task_runner,
      CreateMultiFaceLandmarksTaskRunner(GetParam().landmarks_model_name,
                                         GetParam().blendshape_model_name));

  auto output_packets = task_runner->Process(
      {{kImageName, MakePacket<Image>(std::move(image))},
       {kNormRectName, MakePacket<std::vector<NormalizedRect>>(
                           std::move(GetParam().norm_rects))}});
  MP_ASSERT_OK(output_packets);

  const std::vector<bool>& presences =
      (*output_packets)[kPresenceName].Get<std::vector<bool>>();
  EXPECT_THAT(presences, ElementsAreArray(GetParam().expected_presence));
  if (GetParam().expected_landmarks_lists) {
    const std::vector<NormalizedLandmarkList>& landmarks_lists =
        (*output_packets)[kNormLandmarksName]
            .Get<std::vector<NormalizedLandmarkList>>();
    EXPECT_THAT(landmarks_lists,
                Pointwise(Approximately(
                              Partially(EqualsProto()), /*margin=*/kAbsMargin,
                              /*fraction=*/GetParam().landmarks_diff_threshold),
                          *GetParam().expected_landmarks_lists));
  }
  if (GetParam().expected_blendshapes) {
    const std::vector<ClassificationList>& actual_blendshapes =
        (*output_packets)[kBlendshapesName]
            .Get<std::vector<ClassificationList>>();
    const std::vector<ClassificationList>& expected_blendshapes =
        *GetParam().expected_blendshapes;
    EXPECT_THAT(actual_blendshapes,
                Pointwise(Approximately(EqualsProto(),
                                        GetParam().blendshapes_diff_threshold),
                          expected_blendshapes));
  }
}

INSTANTIATE_TEST_SUITE_P(
    FaceLandmarksDetectionTest, SingleFaceLandmarksDetectionTest,
    Values(SingeFaceTestParams{
        /* test_name= */ "PortraitV2",
        /* landmarks_model_name= */
        kFaceLandmarksV2Model,
        /* test_image_name= */ kPortraitImageName,
        /* norm_rect= */ MakeNormRect(0.4987, 0.2211, 0.2877, 0.2303, 0),
        /* expected_presence= */ true,
        /* expected_landmarks= */
        GetExpectedLandmarkList(kPortraitExpectedFaceLandmarksName),
        /* landmarks_diff_threshold= */ kFractionDiff}),

    [](const TestParamInfo<SingleFaceLandmarksDetectionTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    FaceLandmarksDetectionTest, MultiFaceLandmarksDetectionTest,
    Values(
        MultiFaceTestParams{
            /* test_name= */ "PortraitWithV2",
            /* landmarks_model_name= */
            kFaceLandmarksV2Model,
            /* blendshape_model_name= */ std::nullopt,
            /* test_image_name= */ kPortraitImageName,
            /* norm_rects= */ {MakeNormRect(0.4987, 0.2211, 0.2877, 0.2303, 0)},
            /* expected_presence= */ {true},
            /* expected_landmarks_list= */
            {{GetExpectedLandmarkList(kPortraitExpectedFaceLandmarksName)}},
            /* expected_blendshapes= */ std::nullopt,
            /* landmarks_diff_threshold= */ kFractionDiff,
            /* blendshapes_diff_threshold= */ kBlendshapesDiffMargin},
        MultiFaceTestParams{
            /* test_name= */ "PortraitWithV2WithBlendshapes",
            /* landmarks_model_name= */
            kFaceLandmarksV2Model,
            /* blendshape_model_name= */ kFaceBlendshapesModel,
            /* test_image_name= */ kPortraitImageName,
            /* norm_rects= */
            {MakeNormRect(0.48906386, 0.22731927, 0.42905223, 0.34357703,
                          0.008304443)},
            /* expected_presence= */ {true},
            /* expected_landmarks_list= */
            {{GetExpectedLandmarkList(kPortraitExpectedFaceLandmarksName)}},
            /* expected_blendshapes= */
            {{GetBlendshapes(kPortraitExpectedBlendshapesName)}},
            /* landmarks_diff_threshold= */ kFractionDiff,
            /* blendshapes_diff_threshold= */ kBlendshapesDiffMargin},
        MultiFaceTestParams{
            /* test_name= */ "NoFace",
            /* landmarks_model_name= */
            kFaceLandmarksV2Model,
            /* blendshape_model_name= */ std::nullopt,
            /* test_image_name= */ kCatImageName,
            /* norm_rects= */ {MakeNormRect(0.5, 0.5, 1.0, 1.0, 0)},
            /* expected_presence= */ {false},
            /* expected_landmarks_list= */ std::nullopt,
            /* expected_blendshapes= */ std::nullopt,
            /*landmarks_diff_threshold= */ kFractionDiff,
            /*blendshapes_diff_threshold= */ kBlendshapesDiffMargin}),
    [](const TestParamInfo<MultiFaceLandmarksDetectionTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace

}  // namespace face_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
