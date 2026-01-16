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

#include "mediapipe/tasks/cc/vision/holistic_landmarker/holistic_face_tracking.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mediapipe/calculators/util/landmarks_to_render_data_calculator.pb.h"
#include "mediapipe/calculators/util/rect_to_render_data_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/stream/image_size.h"
#include "mediapipe/framework/api2/stream/split.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/test_util.h"
#include "mediapipe/tasks/cc/core/mediapipe_builtin_op_resolver.h"
#include "mediapipe/tasks/cc/core/model_asset_bundle_resources.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/vision/face_detector/proto/face_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarks_connections.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarks_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/holistic_landmarker/proto/holistic_result.pb.h"
#include "mediapipe/tasks/cc/vision/utils/data_renderer.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"
#include "mediapipe/util/color.pb.h"
#include "mediapipe/util/render_data.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace holistic_landmarker {

namespace {

using ::mediapipe::Image;
using ::mediapipe::api2::builder::GetImageSize;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::SplitToRanges;
using ::mediapipe::api2::builder::Stream;
using ::mediapipe::tasks::core::ModelAssetBundleResources;
using ::mediapipe::tasks::core::TaskRunner;
using ::mediapipe::tasks::core::proto::ExternalFile;
using ::testing::proto::Approximately;
using ::testing::proto::Partially;

constexpr float kAbsMargin = 0.015;
constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kTestImageFile[] = "male_full_height_hands.jpg";
constexpr char kHolisticResultFile[] =
    "male_full_height_hands_result_cpu.pbtxt";
constexpr char kImageInStream[] = "image_in";
constexpr char kPoseLandmarksInStream[] = "pose_landmarks_in";
constexpr char kFaceLandmarksOutStream[] = "face_landmarks_out";
constexpr char kRenderedImageOutStream[] = "rendered_image_out";
constexpr char kFaceDetectionModelFile[] = "face_detection_short_range.tflite";
constexpr char kFaceLandmarksModelFile[] =
    "facemesh2_lite_iris_faceflag_2023_02_14.tflite";

std::string GetFilePath(absl::string_view filename) {
  return file::JoinPath("./", kTestDataDirectory, filename);
}

mediapipe::LandmarksToRenderDataCalculatorOptions GetFaceRendererOptions() {
  mediapipe::LandmarksToRenderDataCalculatorOptions render_options;
  for (const auto& connection :
       face_landmarker::FaceLandmarksConnections::kFaceLandmarksConnectors) {
    render_options.add_landmark_connections(connection[0]);
    render_options.add_landmark_connections(connection[1]);
  }
  render_options.mutable_landmark_color()->set_r(255);
  render_options.mutable_landmark_color()->set_g(255);
  render_options.mutable_landmark_color()->set_b(255);
  render_options.mutable_connection_color()->set_r(255);
  render_options.mutable_connection_color()->set_g(255);
  render_options.mutable_connection_color()->set_b(255);
  render_options.set_thickness(0.5);
  render_options.set_visualize_landmark_depth(false);
  return render_options;
}

mediapipe::RectToRenderDataCalculatorOptions GetRectRendererOptions() {
  mediapipe::RectToRenderDataCalculatorOptions render_options;
  render_options.set_filled(false);
  render_options.mutable_color()->set_r(255);
  render_options.mutable_color()->set_g(0);
  render_options.mutable_color()->set_b(0);
  render_options.set_thickness(2);
  return render_options;
}

absl::StatusOr<std::unique_ptr<ModelAssetBundleResources>>
CreateModelAssetBundleResources(const std::string& model_asset_filename) {
  auto external_model_bundle = std::make_unique<ExternalFile>();
  external_model_bundle->set_file_name(model_asset_filename);
  return ModelAssetBundleResources::Create("",
                                           std::move(external_model_bundle));
}

// Helper function to create a TaskRunner.
absl::StatusOr<std::unique_ptr<tasks::core::TaskRunner>> CreateTaskRunner() {
  Graph graph;
  Stream<Image> image = graph.In("IMAGE").Cast<Image>().SetName(kImageInStream);
  Stream<mediapipe::NormalizedLandmarkList> pose_landmarks =
      graph.In("POSE_LANDMARKS")
          .Cast<mediapipe::NormalizedLandmarkList>()
          .SetName(kPoseLandmarksInStream);
  Stream<NormalizedLandmarkList> face_landmarks_from_pose =
      SplitToRanges(pose_landmarks, {{0, 11}}, graph)[0];
  // Create face landmarker model bundle.
  face_detector::proto::FaceDetectorGraphOptions detector_options;
  face_landmarker::proto::FaceLandmarksDetectorGraphOptions
      landmarks_detector_options;

  // Set face detection model.
  detector_options.set_num_faces(1);
  detector_options.mutable_base_options()->mutable_model_asset()->set_file_name(
      GetFilePath(kFaceDetectionModelFile));

  // Set face landmarks model.
  landmarks_detector_options.mutable_base_options()
      ->mutable_model_asset()
      ->set_file_name(GetFilePath(kFaceLandmarksModelFile));

  // Track holistic face.
  HolisticFaceTrackingRequest request;
  MP_ASSIGN_OR_RETURN(
      HolisticFaceTrackingOutput result,
      TrackHolisticFace(image, face_landmarks_from_pose, detector_options,
                        landmarks_detector_options, request, graph));
  auto face_landmarks =
      result.landmarks.value().SetName(kFaceLandmarksOutStream);

  auto image_size = GetImageSize(image, graph);
  auto render_scale = utils::GetRenderScale(
      image_size, result.debug_output.roi_from_pose, 0.0001, graph);

  std::vector<Stream<mediapipe::RenderData>> render_list = {
      utils::RenderLandmarks(face_landmarks, render_scale,
                             GetFaceRendererOptions(), graph),
      utils::RenderRect(result.debug_output.roi_from_pose,
                        GetRectRendererOptions(), graph)};

  auto rendered_image =
      utils::Render(
          image, absl::Span<Stream<mediapipe::RenderData>>(render_list), graph)
          .SetName(kRenderedImageOutStream);
  face_landmarks >> graph.Out("FACE_LANDMARKS");
  rendered_image >> graph.Out("RENDERED_IMAGE");

  auto config = graph.GetConfig();
  core::FixGraphBackEdges(config);
  return TaskRunner::Create(
      config, std::make_unique<core::MediaPipeBuiltinOpResolver>());
}

class HolisticFaceTrackingTest : public ::testing::Test {};

TEST_F(HolisticFaceTrackingTest, SmokeTest) {
  MP_ASSERT_OK_AND_ASSIGN(Image image,
                          DecodeImageFromFile(GetFilePath(kTestImageFile)));

  proto::HolisticResult holistic_result;
  MP_ASSERT_OK(GetTextProto(GetFilePath(kHolisticResultFile), &holistic_result,
                            ::file::Defaults()));
  MP_ASSERT_OK_AND_ASSIGN(auto task_runner, CreateTaskRunner());
  MP_ASSERT_OK_AND_ASSIGN(
      auto output_packets,
      task_runner->Process(
          {{kImageInStream, MakePacket<Image>(image)},
           {kPoseLandmarksInStream, MakePacket<NormalizedLandmarkList>(
                                        holistic_result.pose_landmarks())}}));
  ASSERT_TRUE(output_packets.find(kFaceLandmarksOutStream) !=
              output_packets.end());
  ASSERT_FALSE(output_packets.find(kFaceLandmarksOutStream)->second.IsEmpty());
  auto face_landmarks = output_packets.find(kFaceLandmarksOutStream)
                            ->second.Get<NormalizedLandmarkList>();
  EXPECT_THAT(
      face_landmarks,
      Approximately(Partially(EqualsProto(holistic_result.face_landmarks())),
                    /*margin=*/kAbsMargin));
  auto rendered_image = output_packets.at(kRenderedImageOutStream).Get<Image>();
  MP_EXPECT_OK(SavePngTestOutput(*rendered_image.GetImageFrameSharedPtr(),
                                 "holistic_face_landmarks"));
}

}  // namespace
}  // namespace holistic_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
