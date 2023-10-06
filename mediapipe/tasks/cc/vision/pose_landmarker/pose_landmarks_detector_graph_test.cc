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

#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/proto/pose_landmarks_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace pose_landmarker {
namespace {

using ::file::Defaults;
using ::file::GetTextProto;
using ::mediapipe::NormalizedRect;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::core::TaskRunner;
using ::mediapipe::tasks::vision::DecodeImageFromFile;
using ::mediapipe::tasks::vision::pose_landmarker::proto::
    PoseLandmarksDetectorGraphOptions;
using ::testing::ElementsAreArray;
using ::testing::EqualsProto;
using ::testing::Pointwise;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::proto::Approximately;
using ::testing::proto::Partially;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kPoseLandmarkerLiteModel[] = "pose_landmark_lite.tflite";
constexpr char kPoseImage[] = "pose.jpg";
constexpr char kBurgerImage[] = "burger.jpg";

constexpr char kImageTag[] = "IMAGE";
constexpr char kImageName[] = "image_in";
constexpr char kNormRectTag[] = "NORM_RECT";

constexpr char kPoseRectName[] = "pose_rect_in";

constexpr char kLandmarksTag[] = "LANDMARKS";
constexpr char kLandmarksName[] = "landmarks";
constexpr char kWorldLandmarksTag[] = "WORLD_LANDMARKS";
constexpr char kWorldLandmarksName[] = "world_landmarks";
constexpr char kAuxLandmarksTag[] = "AUXILIARY_LANDMARKS";
constexpr char kAuxLandmarksName[] = "auxiliary_landmarks";
constexpr char kPoseRectNextFrameTag[] = "POSE_RECT_NEXT_FRAME";
constexpr char kPoseRectNextFrameName[] = "pose_rect_next_frame";
constexpr char kPoseRectsNextFrameTag[] = "POSE_RECTS_NEXT_FRAME";
constexpr char kPoseRectsNextFrameName[] = "pose_rects_next_frame";
constexpr char kPresenceTag[] = "PRESENCE";
constexpr char kPresenceName[] = "presence";
constexpr char kPresenceScoreTag[] = "PRESENCE_SCORE";
constexpr char kPresenceScoreName[] = "presence_score";
constexpr char kSegmentationMaskTag[] = "SEGMENTATION_MASK";
constexpr char kSegmentationMaskName[] = "segmentation_mask";

// Expected pose landmarks positions, in text proto format.
constexpr char kExpectedPoseLandmarksFilename[] =
    "expected_pose_landmarks.prototxt";

constexpr float kLiteModelFractionDiff = 0.05;  // percentage
constexpr float kAbsMargin = 0.03;

// Helper function to create a Single Pose Landmark TaskRunner.
absl::StatusOr<std::unique_ptr<TaskRunner>> CreateSinglePoseTaskRunner(
    absl::string_view model_name) {
  Graph graph;

  auto& pose_landmark_detection = graph.AddNode(
      "mediapipe.tasks.vision.pose_landmarker."
      "SinglePoseLandmarksDetectorGraph");

  auto options = std::make_unique<PoseLandmarksDetectorGraphOptions>();
  options->mutable_base_options()->mutable_model_asset()->set_file_name(
      JoinPath("./", kTestDataDirectory, model_name));
  pose_landmark_detection.GetOptions<PoseLandmarksDetectorGraphOptions>().Swap(
      options.get());

  graph[Input<Image>(kImageTag)].SetName(kImageName) >>
      pose_landmark_detection.In(kImageTag);
  graph[Input<NormalizedRect>(kNormRectTag)].SetName(kPoseRectName) >>
      pose_landmark_detection.In(kNormRectTag);

  pose_landmark_detection.Out(kLandmarksTag).SetName(kLandmarksName) >>
      graph[Output<NormalizedLandmarkList>(kLandmarksTag)];
  pose_landmark_detection.Out(kWorldLandmarksTag)
          .SetName(kWorldLandmarksName) >>
      graph[Output<LandmarkList>(kWorldLandmarksTag)];
  pose_landmark_detection.Out(kAuxLandmarksTag).SetName(kAuxLandmarksName) >>
      graph[Output<LandmarkList>(kAuxLandmarksTag)];
  pose_landmark_detection.Out(kPresenceTag).SetName(kPresenceName) >>
      graph[Output<bool>(kPresenceTag)];
  pose_landmark_detection.Out(kPresenceScoreTag).SetName(kPresenceScoreName) >>
      graph[Output<float>(kPresenceScoreTag)];
  pose_landmark_detection.Out(kSegmentationMaskTag)
          .SetName(kSegmentationMaskName) >>
      graph[Output<Image>(kSegmentationMaskTag)];
  pose_landmark_detection.Out(kPoseRectNextFrameTag)
          .SetName(kPoseRectNextFrameName) >>
      graph[Output<NormalizedRect>(kPoseRectNextFrameTag)];

  return TaskRunner::Create(
      graph.GetConfig(),
      absl::make_unique<tflite::ops::builtin::BuiltinOpResolver>());
}

// Helper function to create a Multi Pose Landmark TaskRunner.
absl::StatusOr<std::unique_ptr<TaskRunner>> CreateMultiPoseTaskRunner(
    absl::string_view model_name) {
  Graph graph;

  auto& multi_pose_landmark_detection = graph.AddNode(
      "mediapipe.tasks.vision.pose_landmarker."
      "MultiplePoseLandmarksDetectorGraph");

  auto options = std::make_unique<PoseLandmarksDetectorGraphOptions>();
  options->mutable_base_options()->mutable_model_asset()->set_file_name(
      JoinPath("./", kTestDataDirectory, model_name));
  multi_pose_landmark_detection.GetOptions<PoseLandmarksDetectorGraphOptions>()
      .Swap(options.get());

  graph[Input<Image>(kImageTag)].SetName(kImageName) >>
      multi_pose_landmark_detection.In(kImageTag);
  graph[Input<std::vector<NormalizedRect>>(kNormRectTag)].SetName(
      kPoseRectName) >>
      multi_pose_landmark_detection.In(kNormRectTag);

  multi_pose_landmark_detection.Out(kLandmarksTag).SetName(kLandmarksName) >>
      graph[Output<std::vector<NormalizedLandmarkList>>(kLandmarksTag)];
  multi_pose_landmark_detection.Out(kWorldLandmarksTag)
          .SetName(kWorldLandmarksName) >>
      graph[Output<std::vector<LandmarkList>>(kWorldLandmarksTag)];
  multi_pose_landmark_detection.Out(kAuxLandmarksTag)
          .SetName(kAuxLandmarksName) >>
      graph[Output<std::vector<NormalizedLandmarkList>>(kAuxLandmarksTag)];
  multi_pose_landmark_detection.Out(kPresenceTag).SetName(kPresenceName) >>
      graph[Output<std::vector<bool>>(kPresenceTag)];
  multi_pose_landmark_detection.Out(kPresenceScoreTag)
          .SetName(kPresenceScoreName) >>
      graph[Output<std::vector<float>>(kPresenceScoreTag)];
  multi_pose_landmark_detection.Out(kSegmentationMaskTag)
          .SetName(kSegmentationMaskName) >>
      graph[Output<std::vector<Image>>(kSegmentationMaskTag)];
  multi_pose_landmark_detection.Out(kPoseRectsNextFrameTag)
          .SetName(kPoseRectsNextFrameName) >>
      graph[Output<std::vector<NormalizedRect>>(kPoseRectsNextFrameTag)];

  return TaskRunner::Create(
      graph.GetConfig(),
      absl::make_unique<tflite::ops::builtin::BuiltinOpResolver>());
}

NormalizedLandmarkList GetExpectedLandmarkList(absl::string_view filename) {
  NormalizedLandmarkList expected_landmark_list;
  MP_EXPECT_OK(GetTextProto(file::JoinPath("./", kTestDataDirectory, filename),
                            &expected_landmark_list, Defaults()));
  return expected_landmark_list;
}

// Struct holding the parameters for parameterized PoseLandmarkerTest
// class.
struct SinglePoseTestParams {
  // The name of this test, for convenience when displaying test results.
  std::string test_name;
  // The filename of the model to test.
  std::string input_model_name;
  // The filename of the test image.
  std::string test_image_name;
  // RoI on image to detect pose.
  NormalizedRect pose_rect;
  // Expected pose presence value.
  bool expected_presence;
  // The expected output landmarks positions in pixels coornidates.
  std::optional<NormalizedLandmarkList> expected_landmarks;
  // The expected segmentation mask.
  Image expected_segmentation_mask;
  // The max value difference between expected_positions and detected positions.
  float landmarks_diff_threshold;
};

struct MultiPoseTestParams {
  // The name of this test, for convenience when displaying test results.
  std::string test_name;
  // The filename of the model to test.
  std::string input_model_name;
  // The filename of the test image.
  std::string test_image_name;
  // RoIs on image to detect poses.
  std::vector<NormalizedRect> pose_rects;
  // Expected pose presence values.
  std::vector<bool> expected_presences;
  // The expected output landmarks positions in pixels coornidates.
  std::vector<NormalizedLandmarkList> expected_landmark_lists;
  // The expected segmentation_mask Image.
  std::vector<Image> expected_segmentation_masks;
  // The max value difference between expected_positions and detected positions.
  float landmarks_diff_threshold;
};

// Helper function to construct NormalizeRect proto.
NormalizedRect MakePoseRect(float x_center, float y_center, float width,
                            float height, float rotation) {
  NormalizedRect pose_rect;
  pose_rect.set_x_center(x_center);
  pose_rect.set_y_center(y_center);
  pose_rect.set_width(width);
  pose_rect.set_height(height);
  pose_rect.set_rotation(rotation);
  return pose_rect;
}

class PoseLandmarkerTest : public testing::TestWithParam<SinglePoseTestParams> {
};

TEST_P(PoseLandmarkerTest, Succeeds) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image, DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                                GetParam().test_image_name)));
  MP_ASSERT_OK_AND_ASSIGN(auto task_runner, CreateSinglePoseTaskRunner(
                                                GetParam().input_model_name));

  auto output_packets = task_runner->Process(
      {{kImageName, MakePacket<Image>(std::move(image))},
       {kPoseRectName,
        MakePacket<NormalizedRect>(std::move(GetParam().pose_rect))}});
  MP_ASSERT_OK(output_packets);

  const bool presence = (*output_packets)[kPresenceName].Get<bool>();
  ASSERT_EQ(presence, GetParam().expected_presence);

  if (presence) {
    const NormalizedLandmarkList landmarks =
        (*output_packets)[kLandmarksName].Get<NormalizedLandmarkList>();

    if (GetParam().expected_landmarks.has_value()) {
      const NormalizedLandmarkList& expected_landmarks =
          GetParam().expected_landmarks.value();

      EXPECT_THAT(
          landmarks,
          Approximately(Partially(EqualsProto(expected_landmarks)),
                        /*margin=*/kAbsMargin,
                        /*fraction=*/GetParam().landmarks_diff_threshold));
    }
  }
}

class MultiPoseLandmarkerTest
    : public testing::TestWithParam<MultiPoseTestParams> {};

TEST_P(MultiPoseLandmarkerTest, Succeeds) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image, DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                                GetParam().test_image_name)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto task_runner, CreateMultiPoseTaskRunner(GetParam().input_model_name));

  auto output_packets = task_runner->Process(
      {{kImageName, MakePacket<Image>(std::move(image))},
       {kPoseRectName, MakePacket<std::vector<NormalizedRect>>(
                           std::move(GetParam().pose_rects))}});
  MP_ASSERT_OK(output_packets);

  const std::vector<bool>& presences =
      (*output_packets)[kPresenceName].Get<std::vector<bool>>();
  const std::vector<NormalizedLandmarkList>& landmark_lists =
      (*output_packets)[kLandmarksName]
          .Get<std::vector<NormalizedLandmarkList>>();

  EXPECT_THAT(presences, ElementsAreArray(GetParam().expected_presences));

  EXPECT_THAT(
      landmark_lists,
      Pointwise(Approximately(Partially(EqualsProto()),
                              /*margin=*/kAbsMargin,
                              /*fraction=*/GetParam().landmarks_diff_threshold),
                GetParam().expected_landmark_lists));
}
// TODO: Add additional tests for MP Tasks Pose Graphs.
// PoseRects below are based on result from PoseDetectorGraph,
// mediapipe/tasks/testdata/vision/pose_expected_expanded_rect.pbtxt.
INSTANTIATE_TEST_SUITE_P(
    PoseLandmarkerTest, PoseLandmarkerTest,
    Values(
        SinglePoseTestParams{
            .test_name = "PoseLandmarkerLiteModel",
            .input_model_name = kPoseLandmarkerLiteModel,
            .test_image_name = kPoseImage,
            .pose_rect = MakePoseRect(0.49192297, 0.7013345, 0.6317167,
                                      0.9471016, -0.029253244),
            .expected_presence = true,
            .expected_landmarks =
                GetExpectedLandmarkList(kExpectedPoseLandmarksFilename),
            .landmarks_diff_threshold = kLiteModelFractionDiff},
        SinglePoseTestParams{
            .test_name = "PoseLandmarkerLiteModelNoPose",
            .input_model_name = kPoseLandmarkerLiteModel,
            .test_image_name = kBurgerImage,
            .pose_rect = MakePoseRect(0.49192297, 0.7013345, 0.6317167,
                                      0.9471016, -0.029253244),
            .expected_presence = false,
            .expected_landmarks = std::nullopt,
            .landmarks_diff_threshold = kLiteModelFractionDiff}),
    [](const TestParamInfo<PoseLandmarkerTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    MultiPoseLandmarkerTest, MultiPoseLandmarkerTest,
    Values(MultiPoseTestParams{
        .test_name = "MultiPoseLandmarkerLiteModel",
        .input_model_name = kPoseLandmarkerLiteModel,
        .test_image_name = kPoseImage,
        .pose_rects = {MakePoseRect(0.49192297, 0.7013345, 0.6317167, 0.9471016,
                                    -0.029253244)},
        .expected_presences = {true},
        .expected_landmark_lists = {GetExpectedLandmarkList(
            kExpectedPoseLandmarksFilename)},
        .landmarks_diff_threshold = kLiteModelFractionDiff,
    }),
    [](const TestParamInfo<MultiPoseLandmarkerTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace pose_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
