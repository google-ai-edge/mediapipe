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

#include "mediapipe/tasks/cc/vision/holistic_landmarker/holistic_pose_tracking.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
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
#include "mediapipe/tasks/cc/vision/holistic_landmarker/proto/holistic_result.pb.h"
#include "mediapipe/tasks/cc/vision/pose_detector/proto/pose_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarks_connections.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/proto/pose_landmarks_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/data_renderer.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"
#include "mediapipe/util/color.pb.h"
#include "mediapipe/util/render_data.pb.h"
#include "testing/base/public/googletest.h"

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

constexpr float kAbsMargin = 0.025;
constexpr absl::string_view kTestDataDirectory =
    "/mediapipe/tasks/testdata/vision/";
constexpr absl::string_view kTestImageFile = "male_full_height_hands.jpg";
constexpr absl::string_view kImageInStream = "image_in";
constexpr absl::string_view kPoseLandmarksOutStream = "pose_landmarks_out";
constexpr absl::string_view kPoseWorldLandmarksOutStream =
    "pose_world_landmarks_out";
constexpr absl::string_view kRenderedImageOutStream = "rendered_image_out";
constexpr absl::string_view kHolisticResultFile =
    "male_full_height_hands_result_cpu.pbtxt";
constexpr absl::string_view kHolisticPoseTrackingGraph =
    "holistic_pose_tracking_graph.pbtxt";

std::string GetFilePath(absl::string_view filename) {
  return file::JoinPath("./", kTestDataDirectory, filename);
}

mediapipe::LandmarksToRenderDataCalculatorOptions GetPoseRendererOptions() {
  mediapipe::LandmarksToRenderDataCalculatorOptions renderer_options;
  for (const auto& connection : pose_landmarker::kPoseLandmarksConnections) {
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

// Helper function to create a TaskRunner.
absl::StatusOr<std::unique_ptr<tasks::core::TaskRunner>> CreateTaskRunner() {
  Graph graph;
  Stream<Image> image = graph.In("IMAGE").Cast<Image>().SetName(kImageInStream);
  pose_detector::proto::PoseDetectorGraphOptions pose_detector_graph_options;
  pose_detector_graph_options.mutable_base_options()
      ->mutable_model_asset()
      ->set_file_name(GetFilePath("pose_detection.tflite"));
  pose_detector_graph_options.set_num_poses(1);
  pose_landmarker::proto::PoseLandmarksDetectorGraphOptions
      pose_landmarks_detector_graph_options;
  pose_landmarks_detector_graph_options.mutable_base_options()
      ->mutable_model_asset()
      ->set_file_name(GetFilePath("pose_landmark_lite.tflite"));

  HolisticPoseTrackingRequest request;
  request.landmarks = true;
  request.world_landmarks = true;
  MP_ASSIGN_OR_RETURN(
      HolisticPoseTrackingOutput result,
      TrackHolisticPose(image, pose_detector_graph_options,
                        pose_landmarks_detector_graph_options, request, graph));

  auto image_size = GetImageSize(image, graph);
  auto render_data = utils::RenderLandmarks(
      *result.landmarks,
      utils::GetRenderScale(image_size, result.debug_output.roi_from_landmarks,
                            0.0001, graph),
      GetPoseRendererOptions(), graph);
  std::vector<Stream<mediapipe::RenderData>> render_list = {render_data};
  auto rendered_image =
      utils::Render(
          image, absl::Span<Stream<mediapipe::RenderData>>(render_list), graph)
          .SetName(kRenderedImageOutStream);

  rendered_image >> graph.Out("RENDERED_IMAGE");
  result.landmarks->SetName(kPoseLandmarksOutStream) >>
      graph.Out("POSE_LANDMARKS");
  result.world_landmarks->SetName(kPoseWorldLandmarksOutStream) >>
      graph.Out("POSE_WORLD_LANDMARKS");

  auto config = graph.GetConfig();
  core::FixGraphBackEdges(config);

  return TaskRunner::Create(
      config, std::make_unique<core::MediaPipeBuiltinOpResolver>());
}

// Remove fields not to be checked in the result, since the model
// generating expected result is different from the testing model.
void RemoveUncheckedResult(proto::HolisticResult& holistic_result) {
  for (auto& landmark :
       *holistic_result.mutable_pose_landmarks()->mutable_landmark()) {
    landmark.clear_z();
    landmark.clear_visibility();
    landmark.clear_presence();
  }
}

class HolisticPoseTrackingTest : public testing::Test {};

TEST_F(HolisticPoseTrackingTest, VerifyGraph) {
  Graph graph;
  Stream<Image> image = graph.In("IMAGE").Cast<Image>().SetName(kImageInStream);
  pose_detector::proto::PoseDetectorGraphOptions pose_detector_graph_options;
  pose_detector_graph_options.mutable_base_options()
      ->mutable_model_asset()
      ->set_file_name(GetFilePath("pose_detection.tflite"));
  pose_detector_graph_options.set_num_poses(1);
  pose_landmarker::proto::PoseLandmarksDetectorGraphOptions
      pose_landmarks_detector_graph_options;
  pose_landmarks_detector_graph_options.mutable_base_options()
      ->mutable_model_asset()
      ->set_file_name(GetFilePath("pose_landmark_lite.tflite"));
  HolisticPoseTrackingRequest request;
  request.landmarks = true;
  request.world_landmarks = true;
  MP_ASSERT_OK_AND_ASSIGN(
      HolisticPoseTrackingOutput result,
      TrackHolisticPose(image, pose_detector_graph_options,
                        pose_landmarks_detector_graph_options, request, graph));
  result.landmarks->SetName(kPoseLandmarksOutStream) >>
      graph.Out("POSE_LANDMARKS");
  result.world_landmarks->SetName(kPoseWorldLandmarksOutStream) >>
      graph.Out("POSE_WORLD_LANDMARKS");

  auto config = graph.GetConfig();
  core::FixGraphBackEdges(config);

  // Read the expected graph config.
  std::string expected_graph_contents;
  MP_ASSERT_OK(file::GetContents(
      file::JoinPath("./", kTestDataDirectory, kHolisticPoseTrackingGraph),
      &expected_graph_contents));

  // Need to replace the expected graph config with the test srcdir, because
  // each run has different test dir on TAP.
  expected_graph_contents = absl::Substitute(
      expected_graph_contents, FLAGS_test_srcdir, FLAGS_test_srcdir);
  CalculatorGraphConfig expected_graph =
      ParseTextProtoOrDie<CalculatorGraphConfig>(expected_graph_contents);

  EXPECT_THAT(config, testing::proto::IgnoringRepeatedFieldOrdering(
                          testing::EqualsProto(expected_graph)));
}

TEST_F(HolisticPoseTrackingTest, SmokeTest) {
  MP_ASSERT_OK_AND_ASSIGN(Image image,
                          DecodeImageFromFile(GetFilePath(kTestImageFile)));

  proto::HolisticResult holistic_result;
  MP_ASSERT_OK(GetTextProto(GetFilePath(kHolisticResultFile), &holistic_result,
                            Defaults()));
  RemoveUncheckedResult(holistic_result);
  MP_ASSERT_OK_AND_ASSIGN(auto task_runner, CreateTaskRunner());
  MP_ASSERT_OK_AND_ASSIGN(auto output_packets,
                          task_runner->Process({{std::string(kImageInStream),
                                                 MakePacket<Image>(image)}}));
  auto pose_landmarks = output_packets.at(std::string(kPoseLandmarksOutStream))
                            .Get<NormalizedLandmarkList>();
  EXPECT_THAT(
      pose_landmarks,
      Approximately(Partially(EqualsProto(holistic_result.pose_landmarks())),
                    /*margin=*/kAbsMargin));
  auto rendered_image =
      output_packets.at(std::string(kRenderedImageOutStream)).Get<Image>();
  MP_EXPECT_OK(SavePngTestOutput(*rendered_image.GetImageFrameSharedPtr(),
                                 "pose_landmarks"));
}

}  // namespace
}  // namespace holistic_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
