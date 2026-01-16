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

#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/cc/core/mediapipe_builtin_op_resolver.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/vision/pose_detector/proto/pose_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace pose_detector {
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
using ::mediapipe::tasks::vision::pose_detector::proto::
    PoseDetectorGraphOptions;
using ::testing::EqualsProto;
using ::testing::Pointwise;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::proto::Approximately;
using ::testing::proto::Partially;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kPoseDetectionModel[] = "pose_detection.tflite";
constexpr char kPortraitImage[] = "pose.jpg";
constexpr char kPoseExpectedDetection[] = "pose_expected_detection.pbtxt";
constexpr char kPoseExpectedExpandedRect[] =
    "pose_expected_expanded_rect.pbtxt";

constexpr char kImageTag[] = "IMAGE";
constexpr char kImageName[] = "image";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kNormRectName[] = "norm_rect";
constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kDetectionsName[] = "detections";
constexpr char kExpandedPoseRectsTag[] = "EXPANDED_POSE_RECTS";
constexpr char kExpandedPoseRectsName[] = "expanded_pose_rects";

constexpr float kPoseDetectionMaxDiff = 0.01;
constexpr float kExpandedPoseRectMaxDiff = 0.01;

// Helper function to create a TaskRunner.
absl::StatusOr<std::unique_ptr<TaskRunner>> CreateTaskRunner(
    absl::string_view model_name) {
  Graph graph;

  auto& pose_detector_graph =
      graph.AddNode("mediapipe.tasks.vision.pose_detector.PoseDetectorGraph");

  auto options = std::make_unique<PoseDetectorGraphOptions>();
  options->mutable_base_options()->mutable_model_asset()->set_file_name(
      JoinPath("./", kTestDataDirectory, model_name));
  options->set_min_detection_confidence(0.6);
  options->set_min_suppression_threshold(0.3);
  pose_detector_graph.GetOptions<PoseDetectorGraphOptions>().Swap(
      options.get());

  graph[Input<Image>(kImageTag)].SetName(kImageName) >>
      pose_detector_graph.In(kImageTag);
  graph[Input<NormalizedRect>(kNormRectTag)].SetName(kNormRectName) >>
      pose_detector_graph.In(kNormRectTag);

  pose_detector_graph.Out(kDetectionsTag).SetName(kDetectionsName) >>
      graph[Output<std::vector<Detection>>(kDetectionsTag)];

  pose_detector_graph.Out(kExpandedPoseRectsTag)
          .SetName(kExpandedPoseRectsName) >>
      graph[Output<std::vector<NormalizedRect>>(kExpandedPoseRectsTag)];

  return TaskRunner::Create(
      graph.GetConfig(), std::make_unique<core::MediaPipeBuiltinOpResolver>());
}

Detection GetExpectedPoseDetectionResult(absl::string_view file_name) {
  Detection detection;
  ABSL_CHECK_OK(
      GetTextProto(file::JoinPath("./", kTestDataDirectory, file_name),
                   &detection, Defaults()))
      << "Expected pose detection result does not exist.";
  return detection;
}

NormalizedRect GetExpectedExpandedPoseRect(absl::string_view file_name) {
  NormalizedRect expanded_rect;
  ABSL_CHECK_OK(
      GetTextProto(file::JoinPath("./", kTestDataDirectory, file_name),
                   &expanded_rect, Defaults()))
      << "Expected expanded pose rect does not exist.";
  return expanded_rect;
}

struct TestParams {
  // The name of this test, for convenience when displaying test results.
  std::string test_name;
  // The filename of pose landmark detection model.
  std::string pose_detection_model_name;
  // The filename of test image.
  std::string test_image_name;
  // Expected pose detection results.
  std::vector<Detection> expected_detection;
  // Expected expanded pose rects.
  std::vector<NormalizedRect> expected_expanded_pose_rect;
};

class PoseDetectorGraphTest : public testing::TestWithParam<TestParams> {};

TEST_P(PoseDetectorGraphTest, Succeed) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image, DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                                GetParam().test_image_name)));
  NormalizedRect input_norm_rect;
  input_norm_rect.set_x_center(0.5);
  input_norm_rect.set_y_center(0.5);
  input_norm_rect.set_width(1.0);
  input_norm_rect.set_height(1.0);
  MP_ASSERT_OK_AND_ASSIGN(
      auto task_runner, CreateTaskRunner(GetParam().pose_detection_model_name));
  auto output_packets = task_runner->Process(
      {{kImageName, MakePacket<Image>(std::move(image))},
       {kNormRectName,
        MakePacket<NormalizedRect>(std::move(input_norm_rect))}});
  MP_ASSERT_OK(output_packets);
  const std::vector<Detection>& pose_detections =
      (*output_packets)[kDetectionsName].Get<std::vector<Detection>>();
  EXPECT_THAT(pose_detections, Pointwise(Approximately(Partially(EqualsProto()),
                                                       kPoseDetectionMaxDiff),
                                         GetParam().expected_detection));

  const std::vector<NormalizedRect>& expanded_pose_rects =
      (*output_packets)[kExpandedPoseRectsName]
          .Get<std::vector<NormalizedRect>>();
  EXPECT_THAT(expanded_pose_rects,
              Pointwise(Approximately(Partially(EqualsProto()),
                                      kExpandedPoseRectMaxDiff),
                        GetParam().expected_expanded_pose_rect));
}

INSTANTIATE_TEST_SUITE_P(
    PoseDetectorGraphTest, PoseDetectorGraphTest,
    Values(TestParams{
        .test_name = "DetectPose",
        .pose_detection_model_name = kPoseDetectionModel,
        .test_image_name = kPortraitImage,
        .expected_detection = {GetExpectedPoseDetectionResult(
            kPoseExpectedDetection)},
        .expected_expanded_pose_rect = {GetExpectedExpandedPoseRect(
            kPoseExpectedExpandedRect)}}),
    [](const TestParamInfo<PoseDetectorGraphTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace pose_detector
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
