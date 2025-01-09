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
#include "mediapipe/framework/formats/classification.pb.h"
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
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_landmarks_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace hand_landmarker {
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
using ::mediapipe::tasks::vision::hand_landmarker::proto::
    HandLandmarksDetectorGraphOptions;
using ::testing::ElementsAreArray;
using ::testing::EqualsProto;
using ::testing::Pointwise;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::proto::Approximately;
using ::testing::proto::Partially;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kHandLandmarkerLiteModel[] = "hand_landmark_lite.tflite";
constexpr char kHandLandmarkerFullModel[] = "hand_landmark_full.tflite";
constexpr char kRightHandsImage[] = "right_hands.jpg";
constexpr char kLeftHandsImage[] = "left_hands.jpg";

constexpr char kTestModelResourcesTag[] = "test_model_resources";

constexpr char kImageTag[] = "IMAGE";
constexpr char kImageName[] = "image_in";
constexpr char kHandRectTag[] = "HAND_RECT";
constexpr char kHandRectName[] = "hand_rect_in";

constexpr char kLandmarksTag[] = "LANDMARKS";
constexpr char kLandmarksName[] = "landmarks";
constexpr char kWorldLandmarksTag[] = "WORLD_LANDMARKS";
constexpr char kWorldLandmarksName[] = "world_landmarks";
constexpr char kHandRectNextFrameTag[] = "HAND_RECT_NEXT_FRAME";
constexpr char kHandRectNextFrameName[] = "hand_rect_next_frame";
constexpr char kPresenceTag[] = "PRESENCE";
constexpr char kPresenceName[] = "presence";
constexpr char kPresenceScoreTag[] = "PRESENCE_SCORE";
constexpr char kPresenceScoreName[] = "presence_score";
constexpr char kHandednessTag[] = "HANDEDNESS";
constexpr char kHandednessName[] = "handedness";

// Expected hand landmarks positions, in text proto format.
constexpr char kExpectedRightUpHandLandmarksFilename[] =
    "expected_right_up_hand_landmarks.prototxt";
constexpr char kExpectedRightDownHandLandmarksFilename[] =
    "expected_right_down_hand_landmarks.prototxt";
constexpr char kExpectedLeftUpHandLandmarksFilename[] =
    "expected_left_up_hand_landmarks.prototxt";
constexpr char kExpectedLeftDownHandLandmarksFilename[] =
    "expected_left_down_hand_landmarks.prototxt";

constexpr float kLiteModelFractionDiff = 0.05;  // percentage
constexpr float kFullModelFractionDiff = 0.03;  // percentage
constexpr float kAbsMargin = 0.03;

// Helper function to create a Single Hand Landmark TaskRunner.
absl::StatusOr<std::unique_ptr<TaskRunner>> CreateSingleHandTaskRunner(
    absl::string_view model_name) {
  Graph graph;

  auto& hand_landmark_detection = graph.AddNode(
      "mediapipe.tasks.vision.hand_landmarker."
      "SingleHandLandmarksDetectorGraph");

  auto options = std::make_unique<HandLandmarksDetectorGraphOptions>();
  options->mutable_base_options()->mutable_model_asset()->set_file_name(
      JoinPath("./", kTestDataDirectory, model_name));
  hand_landmark_detection.GetOptions<HandLandmarksDetectorGraphOptions>().Swap(
      options.get());

  graph[Input<Image>(kImageTag)].SetName(kImageName) >>
      hand_landmark_detection.In(kImageTag);
  graph[Input<NormalizedRect>(kHandRectTag)].SetName(kHandRectName) >>
      hand_landmark_detection.In(kHandRectTag);

  hand_landmark_detection.Out(kLandmarksTag).SetName(kLandmarksName) >>
      graph[Output<NormalizedLandmarkList>(kLandmarksTag)];
  hand_landmark_detection.Out(kWorldLandmarksTag)
          .SetName(kWorldLandmarksName) >>
      graph[Output<LandmarkList>(kWorldLandmarksTag)];
  hand_landmark_detection.Out(kPresenceTag).SetName(kPresenceName) >>
      graph[Output<bool>(kPresenceTag)];
  hand_landmark_detection.Out(kPresenceScoreTag).SetName(kPresenceScoreName) >>
      graph[Output<float>(kPresenceScoreTag)];
  hand_landmark_detection.Out(kHandednessTag).SetName(kHandednessName) >>
      graph[Output<ClassificationList>(kHandednessTag)];
  hand_landmark_detection.Out(kHandRectNextFrameTag)
          .SetName(kHandRectNextFrameName) >>
      graph[Output<NormalizedRect>(kHandRectNextFrameTag)];

  return TaskRunner::Create(
      graph.GetConfig(),
      absl::make_unique<tflite::ops::builtin::BuiltinOpResolver>());
}

// Helper function to create a Multi Hand Landmark TaskRunner.
absl::StatusOr<std::unique_ptr<TaskRunner>> CreateMultiHandTaskRunner(
    absl::string_view model_name) {
  Graph graph;

  auto& multi_hand_landmark_detection = graph.AddNode(
      "mediapipe.tasks.vision.hand_landmarker."
      "MultipleHandLandmarksDetectorGraph");

  auto options = std::make_unique<HandLandmarksDetectorGraphOptions>();
  options->mutable_base_options()->mutable_model_asset()->set_file_name(
      JoinPath("./", kTestDataDirectory, model_name));
  multi_hand_landmark_detection.GetOptions<HandLandmarksDetectorGraphOptions>()
      .Swap(options.get());

  graph[Input<Image>(kImageTag)].SetName(kImageName) >>
      multi_hand_landmark_detection.In(kImageTag);
  graph[Input<std::vector<NormalizedRect>>(kHandRectTag)].SetName(
      kHandRectName) >>
      multi_hand_landmark_detection.In(kHandRectTag);

  multi_hand_landmark_detection.Out(kLandmarksTag).SetName(kLandmarksName) >>
      graph[Output<std::vector<NormalizedLandmarkList>>(kLandmarksTag)];
  multi_hand_landmark_detection.Out(kWorldLandmarksTag)
          .SetName(kWorldLandmarksName) >>
      graph[Output<std::vector<LandmarkList>>(kWorldLandmarksTag)];
  multi_hand_landmark_detection.Out(kPresenceTag).SetName(kPresenceName) >>
      graph[Output<std::vector<bool>>(kPresenceTag)];
  multi_hand_landmark_detection.Out(kPresenceScoreTag)
          .SetName(kPresenceScoreName) >>
      graph[Output<std::vector<float>>(kPresenceScoreTag)];
  multi_hand_landmark_detection.Out(kHandednessTag).SetName(kHandednessName) >>
      graph[Output<std::vector<ClassificationList>>(kHandednessTag)];
  multi_hand_landmark_detection.Out(kHandRectNextFrameTag)
          .SetName(kHandRectNextFrameName) >>
      graph[Output<std::vector<NormalizedRect>>(kHandRectNextFrameTag)];

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

ClassificationList GetExpectedHandedness(
    const std::vector<absl::string_view>& handedness_labels) {
  ClassificationList expected_handedness;
  for (const auto& handedness_label : handedness_labels) {
    auto& classification = *expected_handedness.add_classification();
    classification.set_label(handedness_label);
    classification.set_display_name(handedness_label);
  }
  return expected_handedness;
}

// Struct holding the parameters for parameterized HandLandmarkerTest
// class.
struct SingeHandTestParams {
  // The name of this test, for convenience when displaying test results.
  std::string test_name;
  // The filename of the model to test.
  std::string input_model_name;
  // The filename of the test image.
  std::string test_image_name;
  // RoI on image to detect hands.
  NormalizedRect hand_rect;
  // Expected hand presence value.
  bool expected_presence;
  // The expected output landmarks positions in pixels coornidates.
  NormalizedLandmarkList expected_landmarks;
  // The expected handedness ClassificationList.
  ClassificationList expected_handedness;
  // The max value difference between expected_positions and detected positions.
  float landmarks_diff_threshold;
};

struct MultiHandTestParams {
  // The name of this test, for convenience when displaying test results.
  std::string test_name;
  // The filename of the model to test.
  std::string input_model_name;
  // The filename of the test image.
  std::string test_image_name;
  // RoIs on image to detect hands.
  std::vector<NormalizedRect> hand_rects;
  // Expected hand presence values.
  std::vector<bool> expected_presences;
  // The expected output landmarks positions in pixels coornidates.
  std::vector<NormalizedLandmarkList> expected_landmark_lists;
  // The expected handedness ClassificationList.
  std::vector<ClassificationList> expected_handedness;
  // The max value difference between expected_positions and detected positions.
  float landmarks_diff_threshold;
};

// Helper function to construct NormalizeRect proto.
NormalizedRect MakeHandRect(float x_center, float y_center, float width,
                            float height, float rotation) {
  NormalizedRect hand_rect;
  hand_rect.set_x_center(x_center);
  hand_rect.set_y_center(y_center);
  hand_rect.set_width(width);
  hand_rect.set_height(height);
  hand_rect.set_rotation(rotation);
  return hand_rect;
}

class HandLandmarkerTest : public testing::TestWithParam<SingeHandTestParams> {
};

TEST_P(HandLandmarkerTest, Succeeds) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image, DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                                GetParam().test_image_name)));
  MP_ASSERT_OK_AND_ASSIGN(auto task_runner, CreateSingleHandTaskRunner(
                                                GetParam().input_model_name));

  auto output_packets = task_runner->Process(
      {{kImageName, MakePacket<Image>(std::move(image))},
       {kHandRectName,
        MakePacket<NormalizedRect>(std::move(GetParam().hand_rect))}});
  MP_ASSERT_OK(output_packets);

  const bool presence = (*output_packets)[kPresenceName].Get<bool>();
  ASSERT_EQ(presence, GetParam().expected_presence);

  if (presence) {
    const ClassificationList& handedness =
        (*output_packets)[kHandednessName].Get<ClassificationList>();
    const ClassificationList expected_handedness =
        GetParam().expected_handedness;
    EXPECT_THAT(handedness, Partially(EqualsProto(expected_handedness)));

    const NormalizedLandmarkList landmarks =
        (*output_packets)[kLandmarksName].Get<NormalizedLandmarkList>();

    const NormalizedLandmarkList& expected_landmarks =
        GetParam().expected_landmarks;

    EXPECT_THAT(
        landmarks,
        Approximately(Partially(EqualsProto(expected_landmarks)),
                      /*margin=*/kAbsMargin,
                      /*fraction=*/GetParam().landmarks_diff_threshold));
  }
}

class MultiHandLandmarkerTest
    : public testing::TestWithParam<MultiHandTestParams> {};

TEST_P(MultiHandLandmarkerTest, Succeeds) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image, DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                                GetParam().test_image_name)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto task_runner, CreateMultiHandTaskRunner(GetParam().input_model_name));

  auto output_packets = task_runner->Process(
      {{kImageName, MakePacket<Image>(std::move(image))},
       {kHandRectName, MakePacket<std::vector<NormalizedRect>>(
                           std::move(GetParam().hand_rects))}});
  MP_ASSERT_OK(output_packets);

  const std::vector<bool>& presences =
      (*output_packets)[kPresenceName].Get<std::vector<bool>>();
  const std::vector<ClassificationList>& handedness =
      (*output_packets)[kHandednessName].Get<std::vector<ClassificationList>>();
  const std::vector<NormalizedLandmarkList>& landmark_lists =
      (*output_packets)[kLandmarksName]
          .Get<std::vector<NormalizedLandmarkList>>();

  EXPECT_THAT(presences, ElementsAreArray(GetParam().expected_presences));
  EXPECT_THAT(handedness, Pointwise(Partially(EqualsProto()),
                                    GetParam().expected_handedness));
  EXPECT_THAT(
      landmark_lists,
      Pointwise(Approximately(Partially(EqualsProto()), /*margin=*/kAbsMargin,
                              /*fraction=*/GetParam().landmarks_diff_threshold),
                GetParam().expected_landmark_lists));
}

INSTANTIATE_TEST_SUITE_P(
    HandLandmarkerTest, HandLandmarkerTest,
    Values(
        SingeHandTestParams{
            .test_name = "HandLandmarkerLiteModelRightUpHand",
            .input_model_name = kHandLandmarkerLiteModel,
            .test_image_name = kRightHandsImage,
            .hand_rect = MakeHandRect(0.75, 0.5, 0.5, 1.0, 0),
            .expected_presence = true,
            .expected_landmarks =
                GetExpectedLandmarkList(kExpectedRightUpHandLandmarksFilename),
            .expected_handedness = GetExpectedHandedness({"Right"}),
            .landmarks_diff_threshold = kLiteModelFractionDiff},
        SingeHandTestParams{
            .test_name = "HandLandmarkerLiteModelRightDownHand",
            .input_model_name = kHandLandmarkerLiteModel,
            .test_image_name = kRightHandsImage,
            .hand_rect = MakeHandRect(0.25, 0.5, 0.5, 1.0, M_PI),
            .expected_presence = true,
            .expected_landmarks = GetExpectedLandmarkList(
                kExpectedRightDownHandLandmarksFilename),
            .expected_handedness = GetExpectedHandedness({"Right"}),
            .landmarks_diff_threshold = kLiteModelFractionDiff},
        SingeHandTestParams{
            .test_name = "HandLandmarkerFullModelRightUpHand",
            .input_model_name = kHandLandmarkerFullModel,
            .test_image_name = kRightHandsImage,
            .hand_rect = MakeHandRect(0.75, 0.5, 0.5, 1.0, 0),
            .expected_presence = true,
            .expected_landmarks =
                GetExpectedLandmarkList(kExpectedRightUpHandLandmarksFilename),
            .expected_handedness = GetExpectedHandedness({"Right"}),
            .landmarks_diff_threshold = kFullModelFractionDiff},
        SingeHandTestParams{
            .test_name = "HandLandmarkerFullModelRightDownHand",
            .input_model_name = kHandLandmarkerFullModel,
            .test_image_name = kRightHandsImage,
            .hand_rect = MakeHandRect(0.25, 0.5, 0.5, 1.0, M_PI),
            .expected_presence = true,
            .expected_landmarks = GetExpectedLandmarkList(
                kExpectedRightDownHandLandmarksFilename),
            .expected_handedness = GetExpectedHandedness({"Right"}),
            .landmarks_diff_threshold = kFullModelFractionDiff},
        SingeHandTestParams{
            .test_name = "HandLandmarkerLiteModelLeftUpHand",
            .input_model_name = kHandLandmarkerLiteModel,
            .test_image_name = kLeftHandsImage,
            .hand_rect = MakeHandRect(0.25, 0.5, 0.5, 1.0, 0),
            .expected_presence = true,
            .expected_landmarks =
                GetExpectedLandmarkList(kExpectedLeftUpHandLandmarksFilename),
            .expected_handedness = GetExpectedHandedness({"Left"}),
            .landmarks_diff_threshold = kLiteModelFractionDiff},
        SingeHandTestParams{
            .test_name = "HandLandmarkerLiteModelLeftDownHand",
            .input_model_name = kHandLandmarkerLiteModel,
            .test_image_name = kLeftHandsImage,
            .hand_rect = MakeHandRect(0.75, 0.5, 0.5, 1.0, M_PI),
            .expected_presence = true,
            .expected_landmarks =
                GetExpectedLandmarkList(kExpectedLeftDownHandLandmarksFilename),
            .expected_handedness = GetExpectedHandedness({"Left"}),
            .landmarks_diff_threshold = kLiteModelFractionDiff},
        SingeHandTestParams{
            .test_name = "HandLandmarkerFullModelLeftUpHand",
            .input_model_name = kHandLandmarkerFullModel,
            .test_image_name = kLeftHandsImage,
            .hand_rect = MakeHandRect(0.25, 0.5, 0.5, 1.0, 0),
            .expected_presence = true,
            .expected_landmarks =
                GetExpectedLandmarkList(kExpectedLeftUpHandLandmarksFilename),
            .expected_handedness = GetExpectedHandedness({"Left"}),
            .landmarks_diff_threshold = kFullModelFractionDiff},
        SingeHandTestParams{
            .test_name = "HandLandmarkerFullModelLeftDownHand",
            .input_model_name = kHandLandmarkerFullModel,
            .test_image_name = kLeftHandsImage,
            .hand_rect = MakeHandRect(0.75, 0.5, 0.5, 1.0, M_PI),
            .expected_presence = true,
            .expected_landmarks =
                GetExpectedLandmarkList(kExpectedLeftDownHandLandmarksFilename),
            .expected_handedness = GetExpectedHandedness({"Left"}),
            .landmarks_diff_threshold = kFullModelFractionDiff}),
    [](const TestParamInfo<HandLandmarkerTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    MultiHandLandmarkerTest, MultiHandLandmarkerTest,
    Values(
        MultiHandTestParams{
            .test_name = "MultiHandLandmarkerRightHands",
            .input_model_name = kHandLandmarkerLiteModel,
            .test_image_name = kRightHandsImage,
            .hand_rects =
                {
                    MakeHandRect(0.75, 0.5, 0.5, 1.0, 0),
                    MakeHandRect(0.25, 0.5, 0.5, 1.0, M_PI),
                },
            .expected_presences = {true, true},
            .expected_landmark_lists =
                {GetExpectedLandmarkList(kExpectedRightUpHandLandmarksFilename),
                 GetExpectedLandmarkList(
                     kExpectedRightDownHandLandmarksFilename)},
            .expected_handedness = {GetExpectedHandedness({"Right"}),
                                    GetExpectedHandedness({"Right"})},
            .landmarks_diff_threshold = kLiteModelFractionDiff,
        },
        MultiHandTestParams{
            .test_name = "MultiHandLandmarkerLeftHands",
            .input_model_name = kHandLandmarkerLiteModel,
            .test_image_name = kLeftHandsImage,
            .hand_rects =
                {
                    MakeHandRect(0.25, 0.5, 0.5, 1.0, 0),
                    MakeHandRect(0.75, 0.5, 0.5, 1.0, M_PI),
                },
            .expected_presences = {true, true},
            .expected_landmark_lists =
                {GetExpectedLandmarkList(kExpectedLeftUpHandLandmarksFilename),
                 GetExpectedLandmarkList(
                     kExpectedLeftDownHandLandmarksFilename)},
            .expected_handedness = {GetExpectedHandedness({"Left"}),
                                    GetExpectedHandedness({"Left"})},
            .landmarks_diff_threshold = kLiteModelFractionDiff,
        }),
    [](const TestParamInfo<MultiHandLandmarkerTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace hand_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
