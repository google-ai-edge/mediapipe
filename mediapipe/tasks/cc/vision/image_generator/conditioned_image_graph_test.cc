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
#include <utility>

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/tool/test_util.h"
#include "mediapipe/tasks/cc/core/mediapipe_builtin_op_resolver.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/vision/face_detector/proto/face_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/image_generator/proto/conditioned_image_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/proto/image_segmenter_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace image_generator {

namespace {

using ::mediapipe::Image;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::core::TaskRunner;
using ::mediapipe::tasks::vision::DecodeImageFromFile;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kFaceLandmarkerModel[] = "face_landmarker_v2.task";
constexpr char kDepthModel[] =
    "mobilenetsweep_dptrigmqn384_unit_384_384_fp16quant_fp32input_opt.tflite";
constexpr char kPortraitImage[] = "portrait.jpg";
constexpr char kImageTag[] = "IMAGE";
constexpr char kImageInStream[] = "image_in";
constexpr char kImageOutStream[] = "image_out";

// Helper function to create a ConditionedImageGraphTaskRunner TaskRunner.
absl::StatusOr<std::unique_ptr<TaskRunner>>
CreateConditionedImageGraphTaskRunner(
    std::unique_ptr<proto::ConditionedImageGraphOptions> options) {
  Graph graph;
  auto& conditioned_image_graph = graph.AddNode(
      "mediapipe.tasks.vision.image_generator.ConditionedImageGraph");
  conditioned_image_graph.GetOptions<proto::ConditionedImageGraphOptions>()
      .Swap(options.get());
  graph.In(kImageTag).Cast<Image>().SetName(kImageInStream) >>
      conditioned_image_graph.In(kImageTag);
  conditioned_image_graph.Out(kImageTag).SetName(kImageOutStream) >>
      graph.Out(kImageTag).Cast<Image>();
  return core::TaskRunner::Create(
      graph.GetConfig(),
      absl::make_unique<tasks::core::MediaPipeBuiltinOpResolver>());
}

TEST(ConditionedImageGraphTest, SucceedsFaceLandmarkerConditionType) {
  auto options = std::make_unique<proto::ConditionedImageGraphOptions>();
  options->mutable_face_condition_type_options()
      ->mutable_face_landmarker_graph_options()
      ->mutable_base_options()
      ->mutable_model_asset()
      ->set_file_name(
          file::JoinPath("./", kTestDataDirectory, kFaceLandmarkerModel));
  options->mutable_face_condition_type_options()
      ->mutable_face_landmarker_graph_options()
      ->mutable_face_detector_graph_options()
      ->set_num_faces(1);
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner, CreateConditionedImageGraphTaskRunner(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      Image image, DecodeImageFromFile(file::JoinPath("./", kTestDataDirectory,
                                                      kPortraitImage)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto output_packets,
      runner->Process({{kImageInStream, MakePacket<Image>(std::move(image))}}));
  const auto& output_image = output_packets[kImageOutStream].Get<Image>();
  MP_EXPECT_OK(SavePngTestOutput(*output_image.GetImageFrameSharedPtr(),
                                 "face_landmarks_image"));
}

TEST(ConditionedImageGraphTest, SucceedsDepthConditionType) {
  auto options = std::make_unique<proto::ConditionedImageGraphOptions>();
  options->mutable_depth_condition_type_options()
      ->mutable_image_segmenter_graph_options()
      ->mutable_base_options()
      ->mutable_model_asset()
      ->set_file_name(file::JoinPath("./", kTestDataDirectory, kDepthModel));
  MP_ASSERT_OK_AND_ASSIGN(
      Image image, DecodeImageFromFile(file::JoinPath("./", kTestDataDirectory,
                                                      kPortraitImage)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner, CreateConditionedImageGraphTaskRunner(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto output_packets,
      runner->Process({{kImageInStream, MakePacket<Image>(std::move(image))}}));
  const auto& output_image = output_packets[kImageOutStream].Get<Image>();
  MP_EXPECT_OK(
      SavePngTestOutput(*output_image.GetImageFrameSharedPtr(), "depth_image"));
}

TEST(ConditionedImageGraphTest, SucceedsEdgeConditionType) {
  auto options = std::make_unique<proto::ConditionedImageGraphOptions>();
  auto edge_condition_type_options =
      options->mutable_edge_condition_type_options();
  edge_condition_type_options->set_threshold_1(100);
  edge_condition_type_options->set_threshold_2(200);
  edge_condition_type_options->set_aperture_size(3);
  MP_ASSERT_OK_AND_ASSIGN(
      Image image, DecodeImageFromFile(file::JoinPath("./", kTestDataDirectory,
                                                      kPortraitImage)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner, CreateConditionedImageGraphTaskRunner(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto output_packets,
      runner->Process({{kImageInStream, MakePacket<Image>(std::move(image))}}));
  const auto& output_image = output_packets[kImageOutStream].Get<Image>();
  MP_EXPECT_OK(
      SavePngTestOutput(*output_image.GetImageFrameSharedPtr(), "edges_image"));
}

}  // namespace
}  // namespace image_generator
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
