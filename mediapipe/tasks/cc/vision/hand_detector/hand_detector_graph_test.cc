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
#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
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
#include "mediapipe/tasks/cc/vision/hand_detector/proto/hand_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_detector/proto/hand_detector_result.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace hand_detector {
namespace {

using ::file::Defaults;
using ::file::GetTextProto;
using ::mediapipe::NormalizedRect;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::core::ModelResources;
using ::mediapipe::tasks::core::TaskRunner;
using ::mediapipe::tasks::core::proto::ExternalFile;
using ::mediapipe::tasks::vision::DecodeImageFromFile;
using ::mediapipe::tasks::vision::hand_detector::proto::
    HandDetectorGraphOptions;
using ::mediapipe::tasks::vision::hand_detector::proto::HandDetectorResult;
using ::testing::EqualsProto;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::UnorderedPointwise;
using ::testing::Values;
using ::testing::proto::Approximately;
using ::testing::proto::Partially;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kPalmDetectionModel[] = "palm_detection_full.tflite";
constexpr char kTestLeftHandsImage[] = "left_hands.jpg";
constexpr char kTestLeftHandsRotatedImage[] = "left_hands_rotated.jpg";
constexpr char kTestModelResourcesTag[] = "test_model_resources";

constexpr char kOneHandResultFile[] = "hand_detector_result_one_hand.pbtxt";
constexpr char kOneHandRotatedResultFile[] =
    "hand_detector_result_one_hand_rotated.pbtxt";
constexpr char kTwoHandsResultFile[] = "hand_detector_result_two_hands.pbtxt";

constexpr char kImageTag[] = "IMAGE";
constexpr char kImageName[] = "image";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kNormRectName[] = "norm_rect";
constexpr char kPalmDetectionsTag[] = "PALM_DETECTIONS";
constexpr char kPalmDetectionsName[] = "palm_detections";
constexpr char kHandRectsTag[] = "HAND_RECTS";
constexpr char kHandNormRectsName[] = "hand_norm_rects";

constexpr float kPalmDetectionBboxMaxDiff = 0.01;
constexpr float kHandRectMaxDiff = 0.02;

// Helper function to get ModelResources.
absl::StatusOr<std::unique_ptr<ModelResources>> CreateModelResourcesForModel(
    absl::string_view model_name) {
  auto external_file = std::make_unique<ExternalFile>();
  external_file->set_file_name(JoinPath("./", kTestDataDirectory, model_name));
  return ModelResources::Create(kTestModelResourcesTag,
                                std::move(external_file));
}

// Helper function to create a TaskRunner from ModelResources.
absl::StatusOr<std::unique_ptr<TaskRunner>> CreateTaskRunner(
    const ModelResources& model_resources, absl::string_view model_name,
    int num_hands) {
  Graph graph;

  auto& hand_detection =
      graph.AddNode("mediapipe.tasks.vision.hand_detector.HandDetectorGraph");

  auto options = std::make_unique<HandDetectorGraphOptions>();
  options->mutable_base_options()->mutable_model_asset()->set_file_name(
      JoinPath("./", kTestDataDirectory, model_name));
  options->set_min_detection_confidence(0.5);
  options->set_num_hands(num_hands);
  hand_detection.GetOptions<HandDetectorGraphOptions>().Swap(options.get());

  graph[Input<Image>(kImageTag)].SetName(kImageName) >>
      hand_detection.In(kImageTag);
  graph[Input<NormalizedRect>(kNormRectTag)].SetName(kNormRectName) >>
      hand_detection.In(kNormRectTag);

  hand_detection.Out(kPalmDetectionsTag).SetName(kPalmDetectionsName) >>
      graph[Output<std::vector<Detection>>(kPalmDetectionsTag)];
  hand_detection.Out(kHandRectsTag).SetName(kHandNormRectsName) >>
      graph[Output<std::vector<NormalizedRect>>(kHandRectsTag)];

  return TaskRunner::Create(
      graph.GetConfig(), std::make_unique<core::MediaPipeBuiltinOpResolver>());
}

HandDetectorResult GetExpectedHandDetectorResult(absl::string_view file_name) {
  HandDetectorResult result;
  ABSL_CHECK_OK(GetTextProto(
      file::JoinPath("./", kTestDataDirectory, file_name), &result, Defaults()))
      << "Expected hand detector result does not exist.";
  return result;
}

struct TestParams {
  // The name of this test, for convenience when displaying test results.
  std::string test_name;
  // The filename of hand landmark detection model.
  std::string hand_detection_model_name;
  // The filename of test image.
  std::string test_image_name;
  // The rotation to apply to the test image before processing, in radians
  // counter-clockwise.
  float rotation;
  // The number of maximum detected hands.
  int num_hands;
  // The expected hand detector result.
  HandDetectorResult expected_result;
};

class HandDetectionTest : public testing::TestWithParam<TestParams> {};

TEST_P(HandDetectionTest, DetectTwoHands) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image, DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                                GetParam().test_image_name)));
  NormalizedRect input_norm_rect;
  input_norm_rect.set_rotation(GetParam().rotation);
  input_norm_rect.set_x_center(0.5);
  input_norm_rect.set_y_center(0.5);
  input_norm_rect.set_width(1.0);
  input_norm_rect.set_height(1.0);
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateModelResourcesForModel(GetParam().hand_detection_model_name));
  MP_ASSERT_OK_AND_ASSIGN(
      auto task_runner, CreateTaskRunner(*model_resources, kPalmDetectionModel,
                                         GetParam().num_hands));
  auto output_packets = task_runner->Process(
      {{kImageName, MakePacket<Image>(std::move(image))},
       {kNormRectName,
        MakePacket<NormalizedRect>(std::move(input_norm_rect))}});
  MP_ASSERT_OK(output_packets);
  const std::vector<Detection>& palm_detections =
      (*output_packets)[kPalmDetectionsName].Get<std::vector<Detection>>();
  const std::vector<Detection> expected_palm_detections(
      GetParam().expected_result.detections().begin(),
      GetParam().expected_result.detections().end());
  EXPECT_THAT(palm_detections,
              UnorderedPointwise(Approximately(Partially(EqualsProto()),
                                               kPalmDetectionBboxMaxDiff),
                                 expected_palm_detections));
  const std::vector<NormalizedRect>& hand_rects =
      (*output_packets)[kHandNormRectsName].Get<std::vector<NormalizedRect>>();
  const std::vector<NormalizedRect> expected_hand_rects(
      GetParam().expected_result.hand_rects().begin(),
      GetParam().expected_result.hand_rects().end());
  EXPECT_THAT(hand_rects,
              UnorderedPointwise(
                  Approximately(Partially(EqualsProto()), kHandRectMaxDiff),
                  expected_hand_rects));
}

INSTANTIATE_TEST_SUITE_P(
    HandDetectionTest, HandDetectionTest,
    Values(TestParams{.test_name = "DetectOneHand",
                      .hand_detection_model_name = kPalmDetectionModel,
                      .test_image_name = kTestLeftHandsImage,
                      .rotation = 0,
                      .num_hands = 1,
                      .expected_result =
                          GetExpectedHandDetectorResult(kOneHandResultFile)},
           TestParams{.test_name = "DetectTwoHands",
                      .hand_detection_model_name = kPalmDetectionModel,
                      .test_image_name = kTestLeftHandsImage,
                      .rotation = 0,
                      .num_hands = 2,
                      .expected_result =
                          GetExpectedHandDetectorResult(kTwoHandsResultFile)},
           TestParams{.test_name = "DetectOneHandWithRotation",
                      .hand_detection_model_name = kPalmDetectionModel,
                      .test_image_name = kTestLeftHandsRotatedImage,
                      .rotation = M_PI / 2.0f,
                      .num_hands = 1,
                      .expected_result = GetExpectedHandDetectorResult(
                          kOneHandRotatedResultFile)}),
    [](const TestParamInfo<HandDetectionTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace hand_detector
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
