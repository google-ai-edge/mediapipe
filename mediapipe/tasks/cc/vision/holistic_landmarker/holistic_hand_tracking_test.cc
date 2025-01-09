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

#include "mediapipe/tasks/cc/vision/holistic_landmarker/holistic_hand_tracking.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "file/base/helpers.h"
#include "file/base/options.h"
#include "mediapipe/calculators/util/landmarks_to_render_data_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/stream/image_size.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/tool/test_util.h"
#include "mediapipe/tasks/cc/core/mediapipe_builtin_op_resolver.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarks_connections.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_landmarks_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_roi_refinement_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/holistic_landmarker/proto/holistic_result.pb.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_topology.h"
#include "mediapipe/tasks/cc/vision/utils/data_renderer.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"
#include "mediapipe/util/color.pb.h"
#include "mediapipe/util/render_data.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace holistic_landmarker {

namespace {

using ::file::Defaults;
using ::file::GetTextProto;
using ::mediapipe::Image;
using ::mediapipe::api2::builder::GetImageSize;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Stream;
using ::mediapipe::tasks::core::TaskRunner;
using ::testing::proto::Approximately;
using ::testing::proto::Partially;

constexpr float kAbsMargin = 0.018;
constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kHolisticHandTrackingLeft[] =
    "holistic_hand_tracking_left_hand_graph.pbtxt";
constexpr char kTestImageFile[] = "male_full_height_hands.jpg";
constexpr char kHolisticResultFile[] =
    "male_full_height_hands_result_cpu.pbtxt";
constexpr char kImageInStream[] = "image_in";
constexpr char kPoseLandmarksInStream[] = "pose_landmarks_in";
constexpr char kPoseWorldLandmarksInStream[] = "pose_world_landmarks_in";
constexpr char kLeftHandLandmarksOutStream[] = "left_hand_landmarks_out";
constexpr char kLeftHandWorldLandmarksOutStream[] =
    "left_hand_world_landmarks_out";
constexpr char kRightHandLandmarksOutStream[] = "right_hand_landmarks_out";
constexpr char kRenderedImageOutStream[] = "rendered_image_out";
constexpr char kHandLandmarksModelFile[] = "hand_landmark_full.tflite";
constexpr char kHandRoiRefinementModelFile[] =
    "handrecrop_2020_07_21_v0.f16.tflite";

std::string GetFilePath(const std::string& filename) {
  return file::JoinPath("./", kTestDataDirectory, filename);
}

mediapipe::LandmarksToRenderDataCalculatorOptions GetHandRendererOptions() {
  mediapipe::LandmarksToRenderDataCalculatorOptions renderer_options;
  for (const auto& connection : hand_landmarker::kHandConnections) {
    renderer_options.add_landmark_connections(connection[0]);
    renderer_options.add_landmark_connections(connection[1]);
  }
  renderer_options.mutable_landmark_color()->set_r(255);
  renderer_options.mutable_landmark_color()->set_g(255);
  renderer_options.mutable_landmark_color()->set_b(255);
  renderer_options.mutable_connection_color()->set_r(255);
  renderer_options.mutable_connection_color()->set_g(255);
  renderer_options.mutable_connection_color()->set_b(255);
  renderer_options.set_thickness(0.5);
  renderer_options.set_visualize_landmark_depth(false);
  return renderer_options;
}

void ConfigHandTrackingModelsOptions(
    hand_landmarker::proto::HandLandmarksDetectorGraphOptions&
        hand_landmarks_detector_graph_options,
    hand_landmarker::proto::HandRoiRefinementGraphOptions&
        hand_roi_refinement_options) {
  hand_landmarks_detector_graph_options.mutable_base_options()
      ->mutable_model_asset()
      ->set_file_name(GetFilePath(kHandLandmarksModelFile));

  hand_roi_refinement_options.mutable_base_options()
      ->mutable_model_asset()
      ->set_file_name(GetFilePath(kHandRoiRefinementModelFile));
}

// Helper function to create a TaskRunner.
absl::StatusOr<std::unique_ptr<tasks::core::TaskRunner>> CreateTaskRunner() {
  Graph graph;
  Stream<Image> image = graph.In("IMAGE").Cast<Image>().SetName(kImageInStream);
  Stream<mediapipe::NormalizedLandmarkList> pose_landmarks =
      graph.In("POSE_LANDMARKS")
          .Cast<mediapipe::NormalizedLandmarkList>()
          .SetName(kPoseLandmarksInStream);
  Stream<mediapipe::LandmarkList> pose_world_landmarks =
      graph.In("POSE_WORLD_LANDMARKS")
          .Cast<mediapipe::LandmarkList>()
          .SetName(kPoseWorldLandmarksInStream);
  hand_landmarker::proto::HandLandmarksDetectorGraphOptions
      hand_landmarks_detector_options;
  hand_landmarker::proto::HandRoiRefinementGraphOptions
      hand_roi_refinement_options;
  ConfigHandTrackingModelsOptions(hand_landmarks_detector_options,
                                  hand_roi_refinement_options);
  HolisticHandTrackingRequest request;
  request.landmarks = true;
  MP_ASSIGN_OR_RETURN(
      HolisticHandTrackingOutput left_hand_result,
      TrackHolisticHand(
          image, pose_landmarks, pose_world_landmarks,
          hand_landmarks_detector_options, hand_roi_refinement_options,
          PoseIndices{
              /*wrist_idx=*/static_cast<int>(
                  pose_landmarker::PoseLandmarkName::kLeftWrist),
              /*pinky_idx=*/
              static_cast<int>(pose_landmarker::PoseLandmarkName::kLeftPinky1),
              /*index_idx=*/
              static_cast<int>(pose_landmarker::PoseLandmarkName::kLeftIndex1)},
          request, graph));
  MP_ASSIGN_OR_RETURN(
      HolisticHandTrackingOutput right_hand_result,
      TrackHolisticHand(
          image, pose_landmarks, pose_world_landmarks,
          hand_landmarks_detector_options, hand_roi_refinement_options,
          PoseIndices{
              /*wrist_idx=*/static_cast<int>(
                  pose_landmarker::PoseLandmarkName::kRightWrist),
              /*pinky_idx=*/
              static_cast<int>(pose_landmarker::PoseLandmarkName::kRightPinky1),
              /*index_idx=*/
              static_cast<int>(
                  pose_landmarker::PoseLandmarkName::kRightIndex1)},
          request, graph));

  auto image_size = GetImageSize(image, graph);
  auto left_hand_landmarks_render_data = utils::RenderLandmarks(
      *left_hand_result.landmarks,
      utils::GetRenderScale(image_size,
                            left_hand_result.debug_output.roi_from_pose, 0.0001,
                            graph),
      GetHandRendererOptions(), graph);
  auto right_hand_landmarks_render_data = utils::RenderLandmarks(
      *right_hand_result.landmarks,
      utils::GetRenderScale(image_size,
                            right_hand_result.debug_output.roi_from_pose,
                            0.0001, graph),
      GetHandRendererOptions(), graph);
  std::vector<Stream<mediapipe::RenderData>> render_list = {
      left_hand_landmarks_render_data, right_hand_landmarks_render_data};
  auto rendered_image =
      utils::Render(
          image, absl::Span<Stream<mediapipe::RenderData>>(render_list), graph)
          .SetName(kRenderedImageOutStream);
  left_hand_result.landmarks->SetName(kLeftHandLandmarksOutStream) >>
      graph.Out("LEFT_HAND_LANDMARKS");
  right_hand_result.landmarks->SetName(kRightHandLandmarksOutStream) >>
      graph.Out("RIGHT_HAND_LANDMARKS");
  rendered_image >> graph.Out("RENDERED_IMAGE");

  auto config = graph.GetConfig();
  core::FixGraphBackEdges(config);

  return TaskRunner::Create(
      config, std::make_unique<core::MediaPipeBuiltinOpResolver>());
}

class HolisticHandTrackingTest : public ::testing::Test {};

TEST_F(HolisticHandTrackingTest, VerifyGraph) {
  Graph graph;
  Stream<Image> image = graph.In("IMAGE").Cast<Image>().SetName(kImageInStream);
  Stream<mediapipe::NormalizedLandmarkList> pose_landmarks =
      graph.In("POSE_LANDMARKS")
          .Cast<mediapipe::NormalizedLandmarkList>()
          .SetName(kPoseLandmarksInStream);
  Stream<mediapipe::LandmarkList> pose_world_landmarks =
      graph.In("POSE_WORLD_LANDMARKS")
          .Cast<mediapipe::LandmarkList>()
          .SetName(kPoseWorldLandmarksInStream);
  hand_landmarker::proto::HandLandmarksDetectorGraphOptions
      hand_landmarks_detector_options;
  hand_landmarker::proto::HandRoiRefinementGraphOptions
      hand_roi_refinement_options;
  ConfigHandTrackingModelsOptions(hand_landmarks_detector_options,
                                  hand_roi_refinement_options);
  HolisticHandTrackingRequest request;
  request.landmarks = true;
  request.world_landmarks = true;
  MP_ASSERT_OK_AND_ASSIGN(
      HolisticHandTrackingOutput left_hand_result,
      TrackHolisticHand(
          image, pose_landmarks, pose_world_landmarks,
          hand_landmarks_detector_options, hand_roi_refinement_options,
          PoseIndices{
              /*wrist_idx=*/static_cast<int>(
                  pose_landmarker::PoseLandmarkName::kLeftWrist),
              /*pinky_idx=*/
              static_cast<int>(pose_landmarker::PoseLandmarkName::kLeftPinky1),
              /*index_idx=*/
              static_cast<int>(pose_landmarker::PoseLandmarkName::kLeftIndex1)},
          request, graph));
  left_hand_result.landmarks->SetName(kLeftHandLandmarksOutStream) >>
      graph.Out("LEFT_HAND_LANDMARKS");
  left_hand_result.world_landmarks->SetName(kLeftHandWorldLandmarksOutStream) >>
      graph.Out("LEFT_HAND_WORLD_LANDMARKS");

  // Read the expected graph config.
  std::string expected_graph_contents;
  MP_ASSERT_OK(file::GetContents(
      file::JoinPath("./", kTestDataDirectory, kHolisticHandTrackingLeft),
      &expected_graph_contents));

  // Need to replace the expected graph config with the test srcdir, because
  // each run has different test dir on TAP.
  expected_graph_contents = absl::Substitute(
      expected_graph_contents, FLAGS_test_srcdir, FLAGS_test_srcdir);
  CalculatorGraphConfig expected_graph =
      ParseTextProtoOrDie<CalculatorGraphConfig>(expected_graph_contents);

  EXPECT_THAT(graph.GetConfig(), testing::proto::IgnoringRepeatedFieldOrdering(
                                     testing::EqualsProto(expected_graph)));
}

TEST_F(HolisticHandTrackingTest, SmokeTest) {
  MP_ASSERT_OK_AND_ASSIGN(Image image,
                          DecodeImageFromFile(GetFilePath(kTestImageFile)));

  proto::HolisticResult holistic_result;
  MP_ASSERT_OK(GetTextProto(GetFilePath(kHolisticResultFile), &holistic_result,
                            Defaults()));
  MP_ASSERT_OK_AND_ASSIGN(auto task_runner, CreateTaskRunner());
  MP_ASSERT_OK_AND_ASSIGN(
      auto output_packets,
      task_runner->Process(
          {{kImageInStream, MakePacket<Image>(image)},
           {kPoseLandmarksInStream, MakePacket<NormalizedLandmarkList>(
                                        holistic_result.pose_landmarks())},
           {kPoseWorldLandmarksInStream,
            MakePacket<LandmarkList>(
                holistic_result.pose_world_landmarks())}}));
  auto left_hand_landmarks = output_packets.at(kLeftHandLandmarksOutStream)
                                 .Get<NormalizedLandmarkList>();
  auto right_hand_landmarks = output_packets.at(kRightHandLandmarksOutStream)
                                  .Get<NormalizedLandmarkList>();
  EXPECT_THAT(left_hand_landmarks,
              Approximately(
                  Partially(EqualsProto(holistic_result.left_hand_landmarks())),
                  /*margin=*/kAbsMargin));
  EXPECT_THAT(
      right_hand_landmarks,
      Approximately(
          Partially(EqualsProto(holistic_result.right_hand_landmarks())),
          /*margin=*/kAbsMargin));
  auto rendered_image = output_packets.at(kRenderedImageOutStream).Get<Image>();
  MP_EXPECT_OK(SavePngTestOutput(*rendered_image.GetImageFrameSharedPtr(),
                                 "holistic_hand_landmarks"));
}

}  // namespace
}  // namespace holistic_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
