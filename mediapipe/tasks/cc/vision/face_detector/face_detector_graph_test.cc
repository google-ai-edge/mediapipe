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

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
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
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/tasks/cc/core/mediapipe_builtin_op_resolver.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/vision/face_detector/proto/face_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"
#include "testing/base/public/gunit.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace face_detector {
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
using ::mediapipe::tasks::vision::face_detector::proto::
    FaceDetectorGraphOptions;
using ::testing::EqualsProto;
using ::testing::Pointwise;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::proto::Approximately;
using ::testing::proto::Partially;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kFullRangeBlazeFaceModel[] = "face_detection_full_range.tflite";
constexpr char kFullRangeSparseBlazeFaceModel[] =
    "face_detection_full_range_sparse.tflite";
constexpr char kShortRangeBlazeFaceModel[] =
    "face_detection_short_range.tflite";
constexpr char kPortraitImage[] = "portrait.jpg";
constexpr char kPortraitExpectedDetection[] =
    "portrait_expected_detection.pbtxt";

constexpr char kImageTag[] = "IMAGE";
constexpr char kImageName[] = "image";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kNormRectName[] = "norm_rect";
constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kDetectionsName[] = "detections";

constexpr float kFaceDetectionMaxDiff = 0.01;

// Helper function to create a TaskRunner.
absl::StatusOr<std::unique_ptr<TaskRunner>> CreateTaskRunner(
    absl::string_view model_name, std::string graph_name) {
  Graph graph;

  auto& face_detector_graph = graph.AddNode(graph_name);

  auto options = std::make_unique<FaceDetectorGraphOptions>();
  options->mutable_base_options()->mutable_model_asset()->set_file_name(
      JoinPath(::testing::SrcDir(), kTestDataDirectory, model_name));
  options->set_min_detection_confidence(0.6);
  options->set_min_suppression_threshold(0.3);
  face_detector_graph.GetOptions<FaceDetectorGraphOptions>().Swap(
      options.get());

  graph[Input<Image>(kImageTag)].SetName(kImageName) >>
      face_detector_graph.In(kImageTag);
  graph[Input<NormalizedRect>(kNormRectTag)].SetName(kNormRectName) >>
      face_detector_graph.In(kNormRectTag);

  face_detector_graph.Out(kDetectionsTag).SetName(kDetectionsName) >>
      graph[Output<std::vector<Detection>>(kDetectionsTag)];

  return TaskRunner::Create(
      graph.GetConfig(), std::make_unique<core::MediaPipeBuiltinOpResolver>());
}

Detection GetExpectedFaceDetectionResult(absl::string_view file_name) {
  Detection detection;
  ABSL_CHECK_OK(GetTextProto(
      file::JoinPath(::testing::SrcDir(), kTestDataDirectory, file_name),
      &detection, Defaults()))
      << "Expected face detection result does not exist.";
  return detection;
}

struct TestParams {
  // The name of this test, for convenience when displaying test results.
  std::string test_name;
  // The filename of face landmark detection model.
  std::string face_detection_model_name;
  // The filename of test image.
  std::string test_image_name;
  // Expected face detection results.
  std::vector<Detection> expected_result;
  // The name of the mediapipe graph to run.
  std::string graph_name;
};

class FaceDetectorGraphTest : public testing::TestWithParam<TestParams> {};

TEST_P(FaceDetectorGraphTest, Succeed) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath(::testing::SrcDir(), kTestDataDirectory,
                                   GetParam().test_image_name)));
  NormalizedRect input_norm_rect;
  input_norm_rect.set_x_center(0.5);
  input_norm_rect.set_y_center(0.5);
  input_norm_rect.set_width(1.0);
  input_norm_rect.set_height(1.0);
  MP_ASSERT_OK_AND_ASSIGN(auto task_runner,
                          CreateTaskRunner(GetParam().face_detection_model_name,
                                           GetParam().graph_name));
  auto output_packets = task_runner->Process(
      {{kImageName, MakePacket<Image>(std::move(image))},
       {kNormRectName,
        MakePacket<NormalizedRect>(std::move(input_norm_rect))}});
  MP_ASSERT_OK(output_packets);
  const std::vector<Detection>& face_detections =
      (*output_packets)[kDetectionsName].Get<std::vector<Detection>>();
  EXPECT_THAT(face_detections, Pointwise(Approximately(Partially(EqualsProto()),
                                                       kFaceDetectionMaxDiff),
                                         GetParam().expected_result));
}

INSTANTIATE_TEST_SUITE_P(
    FaceDetectorGraphTest, FaceDetectorGraphTest,
    Values(TestParams{
        .test_name = "ShortRange",
        .face_detection_model_name = kShortRangeBlazeFaceModel,
        .test_image_name = kPortraitImage,
        .expected_result = {GetExpectedFaceDetectionResult(
            kPortraitExpectedDetection)},
        .graph_name =
            "mediapipe.tasks.vision.face_detector.FaceDetectorGraph"}),
    [](const TestParamInfo<FaceDetectorGraphTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace face_detector
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
