// Copyright 2018 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"

namespace mediapipe {

namespace tf = ::tensorflow;
namespace {
const char kNumDetections[] = "NUM_DETECTIONS";
const char kBoxes[] = "BOXES";
const char kScores[] = "SCORES";
const char kClasses[] = "CLASSES";
const char kKeypoints[] = "KEYPOINTS";
const char kDetections[] = "DETECTIONS";
const int kNumBoxes = 3;
const int kNumClasses = 4;
const int kNumCoordsPerBox = 4;
const int kNumKeypointsPerBox = 2;
const int kNumCoordsPerKeypoint = 2;

class ObjectDetectionTensorsToDetectionsCalculatorTest
    : public ::testing::Test {
 protected:
  void SetUp() override { SetUpInputs(); }

  void SetUpInputs() {
    input_num_detections_ = tf::test::AsTensor<float>({kNumBoxes}, {1});
    // {ymin, xmin, ymax, xmax} format.
    input_boxes_ =
        tf::test::AsTensor<float>({0.1f, 0.2f, 0.3f, 0.4f, 0.1f, 0.2f, 0.3f,
                                   0.4f, 0.1f, 0.2f, 0.3f, 0.4f},
                                  {kNumBoxes, kNumCoordsPerBox});
    input_scores_ = tf::test::AsTensor<float>({0.1f, 0.5f, 1.0f}, {kNumBoxes});
    input_scores_for_all_classes_ =
        tf::test::AsTensor<float>({0.0f, 0.1f, 0.05f, 0.02f, 0.0f, 0.1f, 0.5f,
                                   0.2f, 0.0f, 0.5f, 0.8f, 1.0f},
                                  {kNumBoxes, kNumClasses});
    input_classes_ = tf::test::AsTensor<float>({1.0, 2.0, 3.0}, {kNumBoxes});
    input_keypoints_ = tf::test::AsTensor<float>(
        {0.6f, 0.5f, 0.6f, 0.5f, 0.6f, 0.5f, 0.6f, 0.5f, 0.6f, 0.5f, 0.6f,
         0.5f},
        {kNumBoxes, kNumKeypointsPerBox, kNumCoordsPerKeypoint});
  }

  void CreateNodeConfig(CalculatorGraphConfig::Node* node_config) const {
    ASSERT_NE(nullptr, node_config);
    *node_config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
      calculator: "ObjectDetectionTensorsToDetectionsCalculator"
      input_stream: "NUM_DETECTIONS:num_detections"
      input_stream: "BOXES:boxes"
      input_stream: "SCORES:scores"
      input_stream: "CLASSES:classes"
      output_stream: "DETECTIONS:detections"
    )");
  }

  void CreateNodeConfigRawTensors(
      CalculatorGraphConfig::Node* node_config) const {
    ASSERT_NE(nullptr, node_config);
    *node_config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
      calculator: "ObjectDetectionTensorsToDetectionsCalculator"
      input_stream: "BOXES:raw_detection_boxes"
      input_stream: "SCORES:raw_detection_scores"
      output_stream: "DETECTIONS:detections"
    )");
  }

  void CreateNodeConfigWithKeypoints(
      CalculatorGraphConfig::Node* node_config) const {
    ASSERT_NE(nullptr, node_config);
    *node_config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
      calculator: "ObjectDetectionTensorsToDetectionsCalculator"
      input_stream: "NUM_DETECTIONS:num_detections"
      input_stream: "BOXES:boxes"
      input_stream: "SCORES:scores"
      input_stream: "CLASSES:classes"
      input_stream: "KEYPOINTS:keypoints"
      output_stream: "DETECTIONS:detections"
    )");
  }

  void SetUpCalculatorRunner() {
    CalculatorGraphConfig::Node node_config;
    CreateNodeConfig(&node_config);
    runner_ = absl::make_unique<CalculatorRunner>(node_config);
  }

  void SetUpCalculatorRunnerRawTensors() {
    CalculatorGraphConfig::Node node_config;
    CreateNodeConfigRawTensors(&node_config);
    runner_ = absl::make_unique<CalculatorRunner>(node_config);
  }

  void SetUpCalculatorRunnerWithKeypoints() {
    CalculatorGraphConfig::Node node_config;
    CreateNodeConfigWithKeypoints(&node_config);
    runner_ = absl::make_unique<CalculatorRunner>(node_config);
  }

  void RunCalculator() {
    SetUpCalculatorRunner();
    runner_->MutableInputs()
        ->Tag(kNumDetections)
        .packets.push_back(
            PointToForeign(&input_num_detections_).At(Timestamp::PostStream()));
    runner_->MutableInputs()->Tag(kBoxes).packets.push_back(
        PointToForeign(&input_boxes_).At(Timestamp::PostStream()));
    runner_->MutableInputs()->Tag(kScores).packets.push_back(
        PointToForeign(&input_scores_).At(Timestamp::PostStream()));
    runner_->MutableInputs()->Tag(kClasses).packets.push_back(
        PointToForeign(&input_classes_).At(Timestamp::PostStream()));

    MP_ASSERT_OK(runner_->Run());
    ASSERT_EQ(1, runner_->Outputs().Tag(kDetections).packets.size());
  }

  void RunCalculatorRawTensors() {
    SetUpCalculatorRunnerRawTensors();
    runner_->MutableInputs()->Tag(kBoxes).packets.push_back(
        PointToForeign(&input_boxes_).At(Timestamp::PostStream()));
    runner_->MutableInputs()->Tag(kScores).packets.push_back(
        PointToForeign(&input_scores_for_all_classes_)
            .At(Timestamp::PostStream()));

    MP_ASSERT_OK(runner_->Run());
    ASSERT_EQ(1, runner_->Outputs().Tag(kDetections).packets.size());
  }

  void RunCalculatorWithKeypoints() {
    SetUpCalculatorRunnerWithKeypoints();
    runner_->MutableInputs()
        ->Tag(kNumDetections)
        .packets.push_back(
            PointToForeign(&input_num_detections_).At(Timestamp::PostStream()));
    runner_->MutableInputs()->Tag(kBoxes).packets.push_back(
        PointToForeign(&input_boxes_).At(Timestamp::PostStream()));
    runner_->MutableInputs()->Tag(kScores).packets.push_back(
        PointToForeign(&input_scores_).At(Timestamp::PostStream()));
    runner_->MutableInputs()->Tag(kClasses).packets.push_back(
        PointToForeign(&input_classes_).At(Timestamp::PostStream()));
    runner_->MutableInputs()
        ->Tag(kKeypoints)
        .packets.push_back(
            PointToForeign(&input_keypoints_).At(Timestamp::PostStream()));

    MP_ASSERT_OK(runner_->Run());
    ASSERT_EQ(1, runner_->Outputs().Tag(kDetections).packets.size());
  }

  void RunCalculatorWithTensorDimensionSqueezing() {
    InsertExtraSingltonDim(&input_num_detections_);
    InsertExtraSingltonDim(&input_boxes_);
    InsertExtraSingltonDim(&input_scores_);
    InsertExtraSingltonDim(&input_classes_);
    CalculatorGraphConfig::Node node_config =
        ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
          calculator: "ObjectDetectionTensorsToDetectionsCalculator"
          input_stream: "NUM_DETECTIONS:num_detections"
          input_stream: "BOXES:boxes"
          input_stream: "SCORES:scores"
          input_stream: "CLASSES:classes"
          output_stream: "DETECTIONS:detections"
          options: {
            [mediapipe.ObjectDetectionsTensorToDetectionsCalculatorOptions
                 .ext]: { tensor_dim_to_squeeze: 0 }
          }
        )");
    runner_ = absl::make_unique<CalculatorRunner>(node_config);
    runner_->MutableInputs()
        ->Tag(kNumDetections)
        .packets.push_back(
            PointToForeign(&input_num_detections_).At(Timestamp::PostStream()));
    runner_->MutableInputs()->Tag(kBoxes).packets.push_back(
        PointToForeign(&input_boxes_).At(Timestamp::PostStream()));
    runner_->MutableInputs()->Tag(kScores).packets.push_back(
        PointToForeign(&input_scores_).At(Timestamp::PostStream()));
    runner_->MutableInputs()->Tag(kClasses).packets.push_back(
        PointToForeign(&input_classes_).At(Timestamp::PostStream()));

    MP_ASSERT_OK(runner_->Run());
    ASSERT_EQ(1, runner_->Outputs().Tag(kDetections).packets.size());
  }

  void InsertExtraSingltonDim(tf::Tensor* tensor) {
    tf::TensorShape new_shape(tensor->shape());
    new_shape.InsertDim(0, 1);
    ASSERT_TRUE(tensor->CopyFrom(*tensor, new_shape));
  }

  std::unique_ptr<CalculatorRunner> runner_;

  tf::Tensor input_num_detections_;
  tf::Tensor input_boxes_;
  tf::Tensor input_scores_;
  tf::Tensor input_scores_for_all_classes_;
  tf::Tensor input_classes_;
  tf::Tensor input_keypoints_;
};

TEST_F(ObjectDetectionTensorsToDetectionsCalculatorTest, OutputsDetections) {
  RunCalculator();
  EXPECT_EQ(kNumBoxes, runner_->Outputs()
                           .Tag(kDetections)
                           .packets[0]
                           .Get<std::vector<Detection>>()
                           .size());
}

TEST_F(ObjectDetectionTensorsToDetectionsCalculatorTest,
       OutputsDetectionsFromRawTensors) {
  RunCalculatorRawTensors();
  EXPECT_EQ(kNumBoxes, runner_->Outputs()
                           .Tag(kDetections)
                           .packets[0]
                           .Get<std::vector<Detection>>()
                           .size());
}

TEST_F(ObjectDetectionTensorsToDetectionsCalculatorTest,
       OutputsDetectionsWithKeypoints) {
  RunCalculatorWithKeypoints();
  EXPECT_EQ(kNumBoxes, runner_->Outputs()
                           .Tag(kDetections)
                           .packets[0]
                           .Get<std::vector<Detection>>()
                           .size());
}

TEST_F(ObjectDetectionTensorsToDetectionsCalculatorTest,
       OutputsDetectionsWithCorrectValues) {
  RunCalculator();
  const std::vector<Detection> detections = runner_->Outputs()
                                                .Tag(kDetections)
                                                .packets[0]
                                                .Get<std::vector<Detection>>();
  EXPECT_EQ(kNumBoxes, detections.size());
  for (const auto& detection : detections) {
    LocationData::RelativeBoundingBox relative_bbox =
        detection.location_data().relative_bounding_box();
    EXPECT_FLOAT_EQ(0.2, relative_bbox.xmin());
    EXPECT_FLOAT_EQ(0.1, relative_bbox.ymin());
    EXPECT_FLOAT_EQ(0.2, relative_bbox.width());
    EXPECT_FLOAT_EQ(0.2, relative_bbox.height());
  }
  EXPECT_FLOAT_EQ(0.1f, detections[0].score(0));
  EXPECT_FLOAT_EQ(0.5f, detections[1].score(0));
  EXPECT_FLOAT_EQ(1.0f, detections[2].score(0));
  EXPECT_EQ(1, detections[0].label_id(0));
  EXPECT_EQ(2, detections[1].label_id(0));
  EXPECT_EQ(3, detections[2].label_id(0));
}

TEST_F(ObjectDetectionTensorsToDetectionsCalculatorTest,
       OutputsDetectionsFromRawTensorsWithCorrectValues) {
  RunCalculatorRawTensors();
  const std::vector<Detection> detections = runner_->Outputs()
                                                .Tag(kDetections)
                                                .packets[0]
                                                .Get<std::vector<Detection>>();
  EXPECT_EQ(kNumBoxes, detections.size());
  for (const auto& detection : detections) {
    LocationData::RelativeBoundingBox relative_bbox =
        detection.location_data().relative_bounding_box();
    EXPECT_FLOAT_EQ(0.2, relative_bbox.xmin());
    EXPECT_FLOAT_EQ(0.1, relative_bbox.ymin());
    EXPECT_FLOAT_EQ(0.2, relative_bbox.width());
    EXPECT_FLOAT_EQ(0.2, relative_bbox.height());
  }
  EXPECT_FLOAT_EQ(0.1f, detections[0].score(0));
  EXPECT_FLOAT_EQ(0.5f, detections[1].score(0));
  EXPECT_FLOAT_EQ(1.0f, detections[2].score(0));
  EXPECT_EQ(1, detections[0].label_id(0));
  EXPECT_EQ(2, detections[1].label_id(0));
  EXPECT_EQ(3, detections[2].label_id(0));
}

TEST_F(ObjectDetectionTensorsToDetectionsCalculatorTest,
       OutputsDetectionsWithKeypointsAndCorrectValues) {
  RunCalculatorWithKeypoints();
  const std::vector<Detection> detections = runner_->Outputs()
                                                .Tag(kDetections)
                                                .packets[0]
                                                .Get<std::vector<Detection>>();
  EXPECT_EQ(kNumBoxes, detections.size());
  for (const auto& detection : detections) {
    LocationData::RelativeBoundingBox relative_bbox =
        detection.location_data().relative_bounding_box();
    EXPECT_FLOAT_EQ(0.2, relative_bbox.xmin());
    EXPECT_FLOAT_EQ(0.1, relative_bbox.ymin());
    EXPECT_FLOAT_EQ(0.2, relative_bbox.width());
    EXPECT_FLOAT_EQ(0.2, relative_bbox.height());
    for (const auto& relative_keypoint :
         detection.location_data().relative_keypoints()) {
      EXPECT_FLOAT_EQ(0.5, relative_keypoint.x());
      EXPECT_FLOAT_EQ(0.6, relative_keypoint.y());
    }
  }
  EXPECT_FLOAT_EQ(0.1f, detections[0].score(0));
  EXPECT_FLOAT_EQ(0.5f, detections[1].score(0));
  EXPECT_FLOAT_EQ(1.0f, detections[2].score(0));
  EXPECT_EQ(1, detections[0].label_id(0));
  EXPECT_EQ(2, detections[1].label_id(0));
  EXPECT_EQ(3, detections[2].label_id(0));
}

TEST_F(ObjectDetectionTensorsToDetectionsCalculatorTest,
       SqueezesInputTensorDimensionAndOutputsDetectionsWithCorrectValues) {
  RunCalculatorWithTensorDimensionSqueezing();
  const std::vector<Detection> detections = runner_->Outputs()
                                                .Tag(kDetections)
                                                .packets[0]
                                                .Get<std::vector<Detection>>();
  EXPECT_EQ(kNumBoxes, detections.size());
  for (const auto& detection : detections) {
    LocationData::RelativeBoundingBox relative_bbox =
        detection.location_data().relative_bounding_box();
    EXPECT_FLOAT_EQ(0.2, relative_bbox.xmin());
    EXPECT_FLOAT_EQ(0.1, relative_bbox.ymin());
    EXPECT_FLOAT_EQ(0.2, relative_bbox.width());
    EXPECT_FLOAT_EQ(0.2, relative_bbox.height());
  }
  EXPECT_FLOAT_EQ(0.1f, detections[0].score(0));
  EXPECT_FLOAT_EQ(0.5f, detections[1].score(0));
  EXPECT_FLOAT_EQ(1.0f, detections[2].score(0));
  EXPECT_EQ(1, detections[0].label_id(0));
  EXPECT_EQ(2, detections[1].label_id(0));
  EXPECT_EQ(3, detections[2].label_id(0));
}

}  // namespace
}  // namespace mediapipe
