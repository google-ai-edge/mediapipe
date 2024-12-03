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

#include <memory>
#include <string_view>
#include <utility>

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/tasks/cc/core/mediapipe_builtin_op_resolver.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_blendshapes_graph_options.pb.h"
#include "tensorflow/lite/test_util.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace face_landmarker {
namespace {

using ::file::Defaults;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::core::TaskRunner;
using ::mediapipe::tasks::vision::face_landmarker::proto::
    FaceBlendshapesGraphOptions;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kFaceBlendshapesModel[] = "face_blendshapes.tflite";
constexpr char kInLandmarks[] = "face_blendshapes_in_landmarks.prototxt";
constexpr char kOutBlendshapes[] = "face_blendshapes_out.prototxt";
constexpr float kSimilarityThreshold = 0.1;
constexpr std::string_view kGeneratedGraph =
    "face_blendshapes_generated_graph.pbtxt";

constexpr char kLandmarksTag[] = "LANDMARKS";
constexpr char kLandmarksName[] = "landmarks";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kImageSizeName[] = "image_size";
constexpr char kBlendshapesTag[] = "BLENDSHAPES";
constexpr char kBlendshapesName[] = "blendshapes";

absl::StatusOr<CalculatorGraphConfig> ExpandConfig(
    const std::string& config_str) {
  auto config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(config_str);
  CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));
  return graph.Config();
}

NormalizedLandmarkList GetLandmarks(absl::string_view filename) {
  NormalizedLandmarkList landmarks;
  MP_EXPECT_OK(GetTextProto(file::JoinPath("./", kTestDataDirectory, filename),
                            &landmarks, Defaults()));
  return landmarks;
}

ClassificationList GetBlendshapes(absl::string_view filename) {
  ClassificationList blendshapes;
  MP_EXPECT_OK(GetTextProto(file::JoinPath("./", kTestDataDirectory, filename),
                            &blendshapes, Defaults()));
  return blendshapes;
}

// Helper function to create a Face Blendshapes TaskRunner.
absl::StatusOr<std::unique_ptr<TaskRunner>> CreateTaskRunner() {
  Graph graph;
  auto& face_blendshapes_graph = graph.AddNode(
      "mediapipe.tasks.vision.face_landmarker.FaceBlendshapesGraph");
  auto& options =
      face_blendshapes_graph.GetOptions<FaceBlendshapesGraphOptions>();
  options.mutable_base_options()->mutable_model_asset()->set_file_name(
      JoinPath("./", kTestDataDirectory, kFaceBlendshapesModel));

  graph[Input<NormalizedLandmarkList>(kLandmarksTag)].SetName(kLandmarksName) >>
      face_blendshapes_graph.In(kLandmarksTag);
  graph[Input<std::pair<int, int>>(kImageSizeTag)].SetName(kImageSizeName) >>
      face_blendshapes_graph.In(kImageSizeTag);
  face_blendshapes_graph.Out(kBlendshapesTag).SetName(kBlendshapesName) >>
      graph[Output<ClassificationList>(kBlendshapesTag)];

  return TaskRunner::Create(
      graph.GetConfig(), std::make_unique<core::MediaPipeBuiltinOpResolver>());
}

class FaceBlendshapesTest : public tflite::testing::Test {};

TEST_F(FaceBlendshapesTest, SmokeTest) {
  // Prepare graph inputs.
  auto in_landmarks = GetLandmarks(kInLandmarks);
  std::pair<int, int> in_image_size = {820, 1024};

  // Run graph.
  MP_ASSERT_OK_AND_ASSIGN(auto task_runner, CreateTaskRunner());
  auto output_packets = task_runner->Process(
      {{kLandmarksName,
        MakePacket<NormalizedLandmarkList>(std::move(in_landmarks))},
       {kImageSizeName,
        MakePacket<std::pair<int, int>>(std::move(in_image_size))}});
  MP_ASSERT_OK(output_packets);

  // Compare with expected result.
  const auto& actual_blendshapes =
      (*output_packets)[kBlendshapesName].Get<ClassificationList>();
  auto expected_blendshapes = GetBlendshapes(kOutBlendshapes);
  EXPECT_THAT(
      actual_blendshapes,
      testing::proto::Approximately(testing::EqualsProto(expected_blendshapes),
                                    kSimilarityThreshold));
}

TEST(FaceRigGhumGpuTest, VerifyGraph) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto actual_graph,
      ExpandConfig(
          R"pb(
            node {
              calculator: "mediapipe.tasks.vision.face_landmarker.FaceBlendshapesGraph"
              input_stream: "LANDMARKS:landmarks"
              input_stream: "IMAGE_SIZE:image_size"
              output_stream: "BLENDSHAPES:blendshapes"
              options {
                [mediapipe.tasks.vision.face_landmarker.proto
                     .FaceBlendshapesGraphOptions.ext] {
                  base_options {
                    model_asset {
                      file_name: "mediapipe/tasks/testdata/vision/face_blendshapes.tflite"
                    }
                  }
                }
              }
            }
            input_stream: "LANDMARKS:landmarks"
            input_stream: "IMAGE_SIZE:image_size"
          )pb"));

  std::string expected_graph_contents;
  MP_ASSERT_OK(file::GetContents(
      file::JoinPath("./", kTestDataDirectory, kGeneratedGraph),
      &expected_graph_contents));
  auto expected_graph = mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
      expected_graph_contents);
  EXPECT_THAT(actual_graph, testing::proto::IgnoringRepeatedFieldOrdering(
                                testing::EqualsProto(expected_graph)));
}

}  // namespace
}  // namespace face_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
