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

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/cc/core/mediapipe_builtin_op_resolver.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/vision/hand_detector/proto/hand_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_landmarks_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/test_util.h"

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
using ::mediapipe::tasks::vision::hand_landmarker::proto::
    HandLandmarkerGraphOptions;
using ::testing::EqualsProto;
using ::testing::proto::Approximately;
using ::testing::proto::Partially;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kHandLandmarkerModelBundle[] = "hand_landmarker.task";
constexpr char kRightHandsImage[] = "right_hands.jpg";
constexpr char kRightHandsRotatedImage[] = "right_hands_rotated.jpg";

constexpr char kImageTag[] = "IMAGE";
constexpr char kImageName[] = "image_in";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kNormRectName[] = "norm_rect_in";
constexpr char kLandmarksTag[] = "LANDMARKS";
constexpr char kLandmarksName[] = "landmarks";
constexpr char kWorldLandmarksTag[] = "WORLD_LANDMARKS";
constexpr char kWorldLandmarksName[] = "world_landmarks";
constexpr char kHandRectNextFrameTag[] = "HAND_RECT_NEXT_FRAME";
constexpr char kHandRectNextFrameName[] = "hand_rect_next_frame";
constexpr char kHandednessTag[] = "HANDEDNESS";
constexpr char kHandednessName[] = "handedness";

// Expected hand landmarks positions, in text proto format.
constexpr char kExpectedRightUpHandLandmarksFilename[] =
    "expected_right_up_hand_landmarks.prototxt";
constexpr char kExpectedRightDownHandLandmarksFilename[] =
    "expected_right_down_hand_landmarks.prototxt";
// Same but for the rotated image.
constexpr char kExpectedRightUpHandRotatedLandmarksFilename[] =
    "expected_right_up_hand_rotated_landmarks.prototxt";
constexpr char kExpectedRightDownHandRotatedLandmarksFilename[] =
    "expected_right_down_hand_rotated_landmarks.prototxt";

constexpr float kFullModelFractionDiff = 0.03;  // percentage
constexpr float kAbsMargin = 0.03;
constexpr int kMaxNumHands = 2;
constexpr float kMinTrackingConfidence = 0.5;

NormalizedLandmarkList GetExpectedLandmarkList(absl::string_view filename) {
  NormalizedLandmarkList expected_landmark_list;
  MP_EXPECT_OK(GetTextProto(file::JoinPath("./", kTestDataDirectory, filename),
                            &expected_landmark_list, Defaults()));
  return expected_landmark_list;
}

// Helper function to create a Hand Landmarker TaskRunner.
absl::StatusOr<std::unique_ptr<TaskRunner>> CreateTaskRunner() {
  Graph graph;
  auto& hand_landmarker_graph = graph.AddNode(
      "mediapipe.tasks.vision.hand_landmarker.HandLandmarkerGraph");
  auto& options =
      hand_landmarker_graph.GetOptions<HandLandmarkerGraphOptions>();
  options.mutable_base_options()->mutable_model_asset()->set_file_name(
      JoinPath("./", kTestDataDirectory, kHandLandmarkerModelBundle));
  options.mutable_hand_detector_graph_options()->set_num_hands(kMaxNumHands);
  options.set_min_tracking_confidence(kMinTrackingConfidence);

  graph[Input<Image>(kImageTag)].SetName(kImageName) >>
      hand_landmarker_graph.In(kImageTag);
  graph[Input<NormalizedRect>(kNormRectTag)].SetName(kNormRectName) >>
      hand_landmarker_graph.In(kNormRectTag);
  hand_landmarker_graph.Out(kLandmarksTag).SetName(kLandmarksName) >>
      graph[Output<std::vector<NormalizedLandmarkList>>(kLandmarksTag)];
  hand_landmarker_graph.Out(kWorldLandmarksTag).SetName(kWorldLandmarksName) >>
      graph[Output<std::vector<LandmarkList>>(kWorldLandmarksTag)];
  hand_landmarker_graph.Out(kHandednessTag).SetName(kHandednessName) >>
      graph[Output<std::vector<ClassificationList>>(kHandednessTag)];
  hand_landmarker_graph.Out(kHandRectNextFrameTag)
          .SetName(kHandRectNextFrameName) >>
      graph[Output<std::vector<NormalizedRect>>(kHandRectNextFrameTag)];
  return TaskRunner::Create(
      graph.GetConfig(), absl::make_unique<core::MediaPipeBuiltinOpResolver>());
}

class HandLandmarkerTest : public tflite::testing::Test {};

TEST_F(HandLandmarkerTest, Succeeds) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image, DecodeImageFromFile(
                       JoinPath("./", kTestDataDirectory, kRightHandsImage)));
  NormalizedRect input_norm_rect;
  input_norm_rect.set_x_center(0.5);
  input_norm_rect.set_y_center(0.5);
  input_norm_rect.set_width(1.0);
  input_norm_rect.set_height(1.0);
  MP_ASSERT_OK_AND_ASSIGN(auto task_runner, CreateTaskRunner());
  auto output_packets = task_runner->Process(
      {{kImageName, MakePacket<Image>(std::move(image))},
       {kNormRectName,
        MakePacket<NormalizedRect>(std::move(input_norm_rect))}});
  const auto& landmarks = (*output_packets)[kLandmarksName]
                              .Get<std::vector<NormalizedLandmarkList>>();
  ASSERT_EQ(landmarks.size(), kMaxNumHands);
  std::vector<NormalizedLandmarkList> expected_landmarks = {
      GetExpectedLandmarkList(kExpectedRightUpHandLandmarksFilename),
      GetExpectedLandmarkList(kExpectedRightDownHandLandmarksFilename)};

  EXPECT_THAT(landmarks[0],
              Approximately(Partially(EqualsProto(expected_landmarks[0])),
                            /*margin=*/kAbsMargin,
                            /*fraction=*/kFullModelFractionDiff));
  EXPECT_THAT(landmarks[1],
              Approximately(Partially(EqualsProto(expected_landmarks[1])),
                            /*margin=*/kAbsMargin,
                            /*fraction=*/kFullModelFractionDiff));
}

TEST_F(HandLandmarkerTest, SucceedsWithRotation) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image, DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                                kRightHandsRotatedImage)));
  NormalizedRect input_norm_rect;
  input_norm_rect.set_x_center(0.5);
  input_norm_rect.set_y_center(0.5);
  input_norm_rect.set_width(1.0);
  input_norm_rect.set_height(1.0);
  input_norm_rect.set_rotation(M_PI / 2.0);
  MP_ASSERT_OK_AND_ASSIGN(auto task_runner, CreateTaskRunner());
  auto output_packets = task_runner->Process(
      {{kImageName, MakePacket<Image>(std::move(image))},
       {kNormRectName,
        MakePacket<NormalizedRect>(std::move(input_norm_rect))}});
  const auto& landmarks = (*output_packets)[kLandmarksName]
                              .Get<std::vector<NormalizedLandmarkList>>();
  ASSERT_EQ(landmarks.size(), kMaxNumHands);
  std::vector<NormalizedLandmarkList> expected_landmarks = {
      GetExpectedLandmarkList(kExpectedRightUpHandRotatedLandmarksFilename),
      GetExpectedLandmarkList(kExpectedRightDownHandRotatedLandmarksFilename)};

  EXPECT_THAT(landmarks[0],
              Approximately(Partially(EqualsProto(expected_landmarks[0])),
                            /*margin=*/kAbsMargin,
                            /*fraction=*/kFullModelFractionDiff));
  EXPECT_THAT(landmarks[1],
              Approximately(Partially(EqualsProto(expected_landmarks[1])),
                            /*margin=*/kAbsMargin,
                            /*fraction=*/kFullModelFractionDiff));
}

}  // namespace

}  // namespace hand_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
