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

#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/tool/sink.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/environment.pb.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/face_geometry.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace face_geometry {
namespace {

using ::file::Defaults;
using ::mediapipe::tasks::vision::face_geometry::proto::Environment;
// using ::mediapipe::face_geometry::Environment;
using ::mediapipe::tasks::vision::face_geometry::proto::FaceGeometry;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kFaceLandmarksFileName[] =
    "face_blendshapes_in_landmarks.prototxt";
constexpr char kFaceGeometryFileName[] = "face_geometry_expected_out.pbtxt";
constexpr char kGeometryPipelineMetadataPath[] =
    "mediapipe/tasks/cc/vision/face_geometry/data/"
    "geometry_pipeline_metadata_landmarks.binarypb";

std::vector<NormalizedLandmarkList> GetLandmarks(absl::string_view filename) {
  NormalizedLandmarkList landmarks;
  MP_EXPECT_OK(GetTextProto(file::JoinPath("./", kTestDataDirectory, filename),
                            &landmarks, Defaults()));
  return {landmarks};
}

FaceGeometry GetExpectedFaceGeometry(absl::string_view filename) {
  FaceGeometry face_geometry;
  MP_EXPECT_OK(GetTextProto(file::JoinPath("./", kTestDataDirectory, filename),
                            &face_geometry, Defaults()));
  return face_geometry;
}

Environment CreateEnvironment() {
  Environment environment;
  environment.set_origin_point_location(
      proto::OriginPointLocation::TOP_LEFT_CORNER);
  auto& perspective_camera = *environment.mutable_perspective_camera();
  perspective_camera.set_vertical_fov_degrees(63.0 /*degrees*/);
  perspective_camera.set_near(1.0 /* 1cm */);
  perspective_camera.set_far(10000.0 /* 100m */);
  return environment;
}

void MakeInputPacketsAndRunGraph(CalculatorGraph& graph) {
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "face_landmarks", MakePacket<std::vector<NormalizedLandmarkList>>(
                            GetLandmarks(kFaceLandmarksFileName))
                            .At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "image_size", MakePacket<std::pair<int, int>>(std::make_pair(820, 1024))
                        .At(Timestamp(0))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
}

TEST(FaceGeometryFromLandmarksGraphTest, DefaultEnvironment) {
  CalculatorGraphConfig graph_config = ParseTextProtoOrDie<
      CalculatorGraphConfig>(absl::Substitute(
      R"pb(
        input_stream: "FACE_LANDMARKS:face_landmarks"
        input_stream: "IMAGE_SIZE:image_size"
        output_stream: "FACE_GEOMETRY:face_geometry"
        node {
          calculator: "mediapipe.tasks.vision.face_geometry.FaceGeometryFromLandmarksGraph"
          input_stream: "FACE_LANDMARKS:face_landmarks"
          input_stream: "IMAGE_SIZE:image_size"
          output_stream: "FACE_GEOMETRY:face_geometry"
          options: {
            [mediapipe.tasks.vision.face_geometry.proto.FaceGeometryGraphOptions
                 .ext] {
              geometry_pipeline_options { metadata_file { file_name: "$0" } }
            }
          }
        }
      )pb",
      kGeometryPipelineMetadataPath));
  std::vector<Packet> output_packets;
  tool::AddVectorSink("face_geometry", &graph_config, &output_packets);

  // Run the graph.
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MakeInputPacketsAndRunGraph(graph);
  ASSERT_THAT(output_packets, testing::SizeIs(1));
  auto& face_geometry = output_packets[0].Get<std::vector<FaceGeometry>>()[0];
  EXPECT_THAT(
      face_geometry,
      testing::EqualsProto(GetExpectedFaceGeometry(kFaceGeometryFileName)));
}

TEST(FaceGeometryFromLandmarksGraphTest, SideInEnvironment) {
  CalculatorGraphConfig graph_config = ParseTextProtoOrDie<
      CalculatorGraphConfig>(absl::Substitute(
      R"pb(
        input_stream: "FACE_LANDMARKS:face_landmarks"
        input_stream: "IMAGE_SIZE:image_size"
        input_side_packet: "ENVIRONMENT:environment"
        output_stream: "FACE_GEOMETRY:face_geometry"
        node {
          calculator: "mediapipe.tasks.vision.face_geometry.FaceGeometryFromLandmarksGraph"
          input_stream: "FACE_LANDMARKS:face_landmarks"
          input_stream: "IMAGE_SIZE:image_size"
          input_side_packet: "ENVIRONMENT:environment"
          output_stream: "FACE_GEOMETRY:face_geometry"
          options: {
            [mediapipe.tasks.vision.face_geometry.proto.FaceGeometryGraphOptions
                 .ext] {
              geometry_pipeline_options { metadata_file { file_name: "$0" } }
            }
          }
        }
      )pb",
      kGeometryPipelineMetadataPath));
  std::vector<Packet> output_packets;
  tool::AddVectorSink("face_geometry", &graph_config, &output_packets);

  // Run the graph.
  CalculatorGraph graph;
  std::map<std::string, Packet> input_side_packets;
  input_side_packets["environment"] =
      MakePacket<Environment>(CreateEnvironment());
  MP_ASSERT_OK(graph.Initialize(graph_config, input_side_packets));
  MakeInputPacketsAndRunGraph(graph);
  ASSERT_THAT(output_packets, testing::SizeIs(1));
  auto& face_geometry = output_packets[0].Get<std::vector<FaceGeometry>>()[0];
  EXPECT_THAT(
      face_geometry,
      testing::EqualsProto(GetExpectedFaceGeometry(kFaceGeometryFileName)));
}

}  // namespace
}  // namespace face_geometry
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
