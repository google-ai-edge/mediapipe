/* Copyright 2023 The MediaPipe Authors. All Rights Reserved.

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

#include "mediapipe/tasks/cc/components/processors/detection_postprocessing_graph.h"

#include <vector>

#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/util/detection_label_id_to_text_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/graph_runner.h"
#include "mediapipe/framework/output_stream_poller.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/tasks/cc/components/calculators/score_calibration_calculator.pb.h"
#include "mediapipe/tasks/cc/components/processors/proto/detection_postprocessing_graph_options.pb.h"
#include "mediapipe/tasks/cc/components/processors/proto/detector_options.pb.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "tensorflow/lite/test_util.h"

namespace mediapipe {
namespace tasks {
namespace components {
namespace processors {
namespace {

using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::core::ModelResources;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Pointwise;
using ::testing::proto::Approximately;
using ::testing::proto::Partially;

constexpr absl::string_view kTestDataDirectory =
    "/mediapipe/tasks/testdata/vision";
constexpr absl::string_view kMobileSsdWithMetadata =
    "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.tflite";
constexpr absl::string_view kMobileSsdWithDummyScoreCalibration =
    "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29_with_dummy_score_calibration."
    "tflite";
constexpr absl::string_view kEfficientDetWithoutNms =
    "efficientdet_lite0_fp16_no_nms.tflite";

constexpr char kTestModelResourcesTag[] = "test_model_resources";

constexpr absl::string_view kTensorsTag = "TENSORS";
constexpr absl::string_view kDetectionsTag = "DETECTIONS";
constexpr absl::string_view kTensorsName = "tensors";
constexpr absl::string_view kDetectionsName = "detections";

// Helper function to get ModelResources.
absl::StatusOr<std::unique_ptr<ModelResources>> CreateModelResourcesForModel(
    absl::string_view model_name) {
  auto external_file = std::make_unique<core::proto::ExternalFile>();
  external_file->set_file_name(JoinPath("./", kTestDataDirectory, model_name));
  return ModelResources::Create(kTestModelResourcesTag,
                                std::move(external_file));
}

class ConfigureTest : public tflite::testing::Test {};

TEST_F(ConfigureTest, FailsWithInvalidMaxResults) {
  MP_ASSERT_OK_AND_ASSIGN(auto model_resources,
                          CreateModelResourcesForModel(kMobileSsdWithMetadata));
  proto::DetectorOptions options_in;
  options_in.set_max_results(0);

  proto::DetectionPostprocessingGraphOptions options_out;
  auto status = ConfigureDetectionPostprocessingGraph(*model_resources,
                                                      options_in, options_out);

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("Invalid `max_results` option"));
}

TEST_F(ConfigureTest, FailsWithBothAllowlistAndDenylist) {
  MP_ASSERT_OK_AND_ASSIGN(auto model_resources,
                          CreateModelResourcesForModel(kMobileSsdWithMetadata));
  proto::DetectorOptions options_in;
  options_in.add_category_allowlist("foo");
  options_in.add_category_denylist("bar");

  proto::DetectionPostprocessingGraphOptions options_out;
  auto status = ConfigureDetectionPostprocessingGraph(*model_resources,
                                                      options_in, options_out);

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("mutually exclusive options"));
}

TEST_F(ConfigureTest, SucceedsWithMaxResults) {
  MP_ASSERT_OK_AND_ASSIGN(auto model_resources,
                          CreateModelResourcesForModel(kMobileSsdWithMetadata));
  proto::DetectorOptions options_in;
  options_in.set_max_results(3);

  proto::DetectionPostprocessingGraphOptions options_out;
  MP_ASSERT_OK(ConfigureDetectionPostprocessingGraph(*model_resources,
                                                     options_in, options_out));

  EXPECT_THAT(
      options_out,
      Approximately(Partially(EqualsProto(
          R"pb(tensors_to_detections_options {
                 min_score_thresh: -3.4028235e+38
                 num_classes: 90
                 num_coords: 4
                 max_results: 3
                 tensor_mapping {
                   detections_tensor_index: 0
                   classes_tensor_index: 1
                   scores_tensor_index: 2
                   num_detections_tensor_index: 3
                 }
                 box_boundaries_indices { ymin: 0 xmin: 1 ymax: 2 xmax: 3 }
               }
          )pb"))));
}

TEST_F(ConfigureTest, SucceedsWithMaxResultsWithoutModelNms) {
  MP_ASSERT_OK_AND_ASSIGN(auto model_resources, CreateModelResourcesForModel(
                                                    kEfficientDetWithoutNms));
  proto::DetectorOptions options_in;
  options_in.set_max_results(3);

  proto::DetectionPostprocessingGraphOptions options_out;
  MP_ASSERT_OK(ConfigureDetectionPostprocessingGraph(*model_resources,
                                                     options_in, options_out));
  EXPECT_THAT(options_out, Approximately(Partially(EqualsProto(
                               R"pb(tensors_to_detections_options {
                                      min_score_thresh: -3.4028235e+38
                                      num_classes: 90
                                      num_boxes: 19206
                                      num_coords: 4
                                      x_scale: 1
                                      y_scale: 1
                                      w_scale: 1
                                      h_scale: 1
                                      keypoint_coord_offset: 0
                                      num_keypoints: 0
                                      num_values_per_keypoint: 2
                                      apply_exponential_on_box_size: true
                                      sigmoid_score: false
                                      tensor_mapping {
                                        detections_tensor_index: 1
                                        scores_tensor_index: 0
                                      }
                                      box_format: YXHW
                                    }
                                    non_max_suppression_options {
                                      max_num_detections: 3
                                      min_suppression_threshold: 0
                                      overlap_type: INTERSECTION_OVER_UNION
                                      algorithm: DEFAULT
                                    }
                               )pb"))));
  EXPECT_THAT(
      options_out.detection_label_ids_to_text_options().label_items_size(), 90);
}

TEST_F(ConfigureTest, SucceedsWithScoreThreshold) {
  MP_ASSERT_OK_AND_ASSIGN(auto model_resources,
                          CreateModelResourcesForModel(kMobileSsdWithMetadata));
  proto::DetectorOptions options_in;
  options_in.set_score_threshold(0.5);

  proto::DetectionPostprocessingGraphOptions options_out;
  MP_ASSERT_OK(ConfigureDetectionPostprocessingGraph(*model_resources,
                                                     options_in, options_out));
  EXPECT_THAT(
      options_out,
      Approximately(Partially(EqualsProto(
          R"pb(tensors_to_detections_options {
                 min_score_thresh: 0.5
                 num_classes: 90
                 num_coords: 4
                 tensor_mapping {
                   detections_tensor_index: 0
                   classes_tensor_index: 1
                   scores_tensor_index: 2
                   num_detections_tensor_index: 3
                 }
                 box_boundaries_indices { ymin: 0 xmin: 1 ymax: 2 xmax: 3 }
               }
               has_quantized_outputs: false
          )pb"))));
  EXPECT_THAT(
      options_out.detection_label_ids_to_text_options().label_items_size(), 90);
}

TEST_F(ConfigureTest, SucceedsWithAllowlist) {
  MP_ASSERT_OK_AND_ASSIGN(auto model_resources,
                          CreateModelResourcesForModel(kMobileSsdWithMetadata));
  proto::DetectorOptions options_in;
  options_in.add_category_allowlist("bicycle");
  proto::DetectionPostprocessingGraphOptions options_out;
  MP_ASSERT_OK(ConfigureDetectionPostprocessingGraph(*model_resources,
                                                     options_in, options_out));
  // Clear labels ids to text and compare the rest of the options.
  options_out.clear_detection_label_ids_to_text_options();
  EXPECT_THAT(
      options_out,
      Approximately(EqualsProto(
          R"pb(tensors_to_detections_options {
                 min_score_thresh: -3.4028235e+38
                 num_classes: 90
                 num_coords: 4
                 allow_classes: 1
                 tensor_mapping {
                   detections_tensor_index: 0
                   classes_tensor_index: 1
                   scores_tensor_index: 2
                   num_detections_tensor_index: 3
                 }
                 box_boundaries_indices { ymin: 0 xmin: 1 ymax: 2 xmax: 3 }
                 max_classes_per_detection: 1
               }
               has_quantized_outputs: false
          )pb")));
}

TEST_F(ConfigureTest, SucceedsWithDenylist) {
  MP_ASSERT_OK_AND_ASSIGN(auto model_resources,
                          CreateModelResourcesForModel(kMobileSsdWithMetadata));
  proto::DetectorOptions options_in;
  options_in.add_category_denylist("person");
  proto::DetectionPostprocessingGraphOptions options_out;
  MP_ASSERT_OK(ConfigureDetectionPostprocessingGraph(*model_resources,
                                                     options_in, options_out));
  // Clear labels ids to text and compare the rest of the options.
  options_out.clear_detection_label_ids_to_text_options();
  EXPECT_THAT(
      options_out,
      Approximately(EqualsProto(
          R"pb(tensors_to_detections_options {
                 min_score_thresh: -3.4028235e+38
                 num_classes: 90
                 num_coords: 4
                 ignore_classes: 0
                 tensor_mapping {
                   detections_tensor_index: 0
                   classes_tensor_index: 1
                   scores_tensor_index: 2
                   num_detections_tensor_index: 3
                 }
                 box_boundaries_indices { ymin: 0 xmin: 1 ymax: 2 xmax: 3 }
                 max_classes_per_detection: 1
               }
               has_quantized_outputs: false
          )pb")));
}

TEST_F(ConfigureTest, SucceedsWithScoreCalibration) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateModelResourcesForModel(kMobileSsdWithDummyScoreCalibration));
  proto::DetectorOptions options_in;
  proto::DetectionPostprocessingGraphOptions options_out;
  MP_ASSERT_OK(ConfigureDetectionPostprocessingGraph(*model_resources,
                                                     options_in, options_out));
  // Clear labels ids to text.
  options_out.clear_detection_label_ids_to_text_options();
  // Check sigmoids size and first element.
  ASSERT_EQ(options_out.score_calibration_options().sigmoids_size(), 89);
  EXPECT_THAT(options_out.score_calibration_options().sigmoids()[0],
              EqualsProto(R"pb(scale: 1.0 slope: 1.0 offset: 0.0)pb"));
  options_out.mutable_score_calibration_options()->clear_sigmoids();
  // Compare the rest of the option.
  EXPECT_THAT(
      options_out,
      Approximately(EqualsProto(
          R"pb(tensors_to_detections_options {
                 min_score_thresh: -3.4028235e+38
                 num_classes: 90
                 num_coords: 4
                 tensor_mapping {
                   detections_tensor_index: 0
                   classes_tensor_index: 1
                   scores_tensor_index: 2
                   num_detections_tensor_index: 3
                 }
                 box_boundaries_indices { ymin: 0 xmin: 1 ymax: 2 xmax: 3 }
                 max_classes_per_detection: 1
               }
               score_calibration_options {
                 score_transformation: IDENTITY
                 default_score: 0.5
               }
               has_quantized_outputs: false
          )pb")));
}

class PostprocessingTest : public tflite::testing::Test {
 protected:
  absl::StatusOr<OutputStreamPoller> BuildGraph(
      absl::string_view model_name, const proto::DetectorOptions& options) {
    MP_ASSIGN_OR_RETURN(auto model_resources,
                        CreateModelResourcesForModel(model_name));

    Graph graph;
    auto& postprocessing = graph.AddNode(
        "mediapipe.tasks.components.processors."
        "DetectionPostprocessingGraph");
    MP_RETURN_IF_ERROR(ConfigureDetectionPostprocessingGraph(
        *model_resources, options,
        postprocessing
            .GetOptions<proto::DetectionPostprocessingGraphOptions>()));
    graph[Input<std::vector<Tensor>>(kTensorsTag)].SetName(
        std::string(kTensorsName)) >>
        postprocessing.In(kTensorsTag);
    postprocessing.Out(kDetectionsTag).SetName(std::string(kDetectionsName)) >>
        graph[Output<std::vector<Detection>>(kDetectionsTag)];
    MP_RETURN_IF_ERROR(calculator_graph_.Initialize(graph.GetConfig()));
    MP_ASSIGN_OR_RETURN(auto poller, calculator_graph_.AddOutputStreamPoller(
                                         std::string(kDetectionsName)));
    MP_RETURN_IF_ERROR(calculator_graph_.StartRun(/*extra_side_packets=*/{}));
    return poller;
  }

  template <typename T>
  void AddTensor(const std::vector<T>& tensor,
                 const Tensor::ElementType& element_type,
                 const Tensor::Shape& shape) {
    tensors_->emplace_back(element_type, shape);
    auto view = tensors_->back().GetCpuWriteView();
    T* buffer = view.buffer<T>();
    std::copy(tensor.begin(), tensor.end(), buffer);
  }

  absl::Status Run(int timestamp = 0) {
    MP_RETURN_IF_ERROR(calculator_graph_.AddPacketToInputStream(
        std::string(kTensorsName),
        Adopt(tensors_.release()).At(Timestamp(timestamp))));
    // Reset tensors for future calls.
    tensors_ = absl::make_unique<std::vector<Tensor>>();
    return absl::OkStatus();
  }

  template <typename T>
  absl::StatusOr<T> GetResult(OutputStreamPoller& poller) {
    MP_RETURN_IF_ERROR(calculator_graph_.WaitUntilIdle());
    MP_RETURN_IF_ERROR(calculator_graph_.CloseAllInputStreams());

    Packet packet;
    if (!poller.Next(&packet)) {
      return absl::InternalError("Unable to get output packet");
    }
    auto result = packet.Get<T>();
    MP_RETURN_IF_ERROR(calculator_graph_.WaitUntilDone());
    return result;
  }

 private:
  CalculatorGraph calculator_graph_;
  std::unique_ptr<std::vector<Tensor>> tensors_ =
      absl::make_unique<std::vector<Tensor>>();
};

TEST_F(PostprocessingTest, SucceedsWithMetadata) {
  // Build graph.
  proto::DetectorOptions options;
  options.set_max_results(3);
  MP_ASSERT_OK_AND_ASSIGN(auto poller,
                          BuildGraph(kMobileSsdWithMetadata, options));

  // Build input tensors.
  constexpr int kBboxesNum = 5;
  // Location tensor.
  std::vector<float> location_tensor(kBboxesNum * 4, 0);
  for (int i = 0; i < kBboxesNum; ++i) {
    location_tensor[i * 4] = 0.1f;
    location_tensor[i * 4 + 1] = 0.1f;
    location_tensor[i * 4 + 2] = 0.4f;
    location_tensor[i * 4 + 3] = 0.5f;
  }
  // Category tensor.
  std::vector<float> category_tensor(kBboxesNum, 0);
  for (int i = 0; i < kBboxesNum; ++i) {
    category_tensor[i] = i + 1;
  }

  // Score tensor. Post processed tensor scores are in descending order.
  std::vector<float> score_tensor(kBboxesNum, 0);
  for (int i = 0; i < kBboxesNum; ++i) {
    score_tensor[i] = static_cast<float>(kBboxesNum - i) / kBboxesNum;
  }

  // Number of detections tensor.
  std::vector<float> num_detections_tensor(1, 0);
  num_detections_tensor[0] = kBboxesNum;

  // Send tensors and get results.
  AddTensor(location_tensor, Tensor::ElementType::kFloat32, {1, kBboxesNum, 4});
  AddTensor(category_tensor, Tensor::ElementType::kFloat32, {1, kBboxesNum});
  AddTensor(score_tensor, Tensor::ElementType::kFloat32, {1, kBboxesNum});
  AddTensor(num_detections_tensor, Tensor::ElementType::kFloat32, {1});
  MP_ASSERT_OK(Run());

  // Validate results.
  EXPECT_THAT(GetResult<std::vector<Detection>>(poller),
              IsOkAndHolds(ElementsAre(Approximately(EqualsProto(
                                           R"pb(
                                             label: "bicycle"
                                             score: 1
                                             location_data {
                                               format: RELATIVE_BOUNDING_BOX
                                               relative_bounding_box {
                                                 xmin: 0.1
                                                 ymin: 0.1
                                                 width: 0.4
                                                 height: 0.3
                                               }
                                             }
                                           )pb")),
                                       Approximately(EqualsProto(
                                           R"pb(
                                             label: "car"
                                             score: 0.8
                                             location_data {
                                               format: RELATIVE_BOUNDING_BOX
                                               relative_bounding_box {
                                                 xmin: 0.1
                                                 ymin: 0.1
                                                 width: 0.4
                                                 height: 0.3
                                               }
                                             }
                                           )pb")),
                                       Approximately(EqualsProto(
                                           R"pb(
                                             label: "motorcycle"
                                             score: 0.6
                                             location_data {
                                               format: RELATIVE_BOUNDING_BOX
                                               relative_bounding_box {
                                                 xmin: 0.1
                                                 ymin: 0.1
                                                 width: 0.4
                                                 height: 0.3
                                               }
                                             }
                                           )pb")))));
}

TEST_F(PostprocessingTest, SucceedsWithOutModelNms) {
  // Build graph.
  proto::DetectorOptions options;
  options.set_max_results(3);
  MP_ASSERT_OK_AND_ASSIGN(auto poller,
                          BuildGraph(kEfficientDetWithoutNms, options));

  // Build input tensors.
  constexpr int kBboxesNum = 19206;
  constexpr int kBicycleBboxIdx = 1000;
  constexpr int kCarBboxIdx = 2000;
  constexpr int kMotoCycleBboxIdx = 4000;
  // Location tensor.
  std::vector<float> location_tensor(kBboxesNum * 4, 0);
  for (int i = 0; i < kBboxesNum; ++i) {
    location_tensor[i * 4] = 0.5f;
    location_tensor[i * 4 + 1] = 0.5f;
    location_tensor[i * 4 + 2] = 0.001f;
    location_tensor[i * 4 + 3] = 0.001f;
  }

  // Detected three objects.
  location_tensor[kBicycleBboxIdx * 4] = 0.7f;
  location_tensor[kBicycleBboxIdx * 4 + 1] = 0.8f;
  location_tensor[kBicycleBboxIdx * 4 + 2] = 0.2f;
  location_tensor[kBicycleBboxIdx * 4 + 3] = 0.1f;

  location_tensor[kCarBboxIdx * 4] = 0.1f;
  location_tensor[kCarBboxIdx * 4 + 1] = 0.1f;
  location_tensor[kCarBboxIdx * 4 + 2] = 0.1f;
  location_tensor[kCarBboxIdx * 4 + 3] = 0.1f;

  location_tensor[kMotoCycleBboxIdx * 4] = 0.2f;
  location_tensor[kMotoCycleBboxIdx * 4 + 1] = 0.8f;
  location_tensor[kMotoCycleBboxIdx * 4 + 2] = 0.1f;
  location_tensor[kMotoCycleBboxIdx * 4 + 3] = 0.2f;

  // Score tensor.
  constexpr int kClassesNum = 90;
  std::vector<float> score_tensor(kBboxesNum * kClassesNum, 1.f / kClassesNum);

  // Detected three objects.
  score_tensor[kBicycleBboxIdx * kClassesNum + 1] = 1.0f;    // bicycle.
  score_tensor[kCarBboxIdx * kClassesNum + 2] = 0.9f;        // car.
  score_tensor[kMotoCycleBboxIdx * kClassesNum + 3] = 0.8f;  // motorcycle.

  // Send tensors and get results.
  AddTensor(score_tensor, Tensor::ElementType::kFloat32, {1, kBboxesNum, 90});
  AddTensor(location_tensor, Tensor::ElementType::kFloat32, {1, kBboxesNum, 4});
  MP_ASSERT_OK(Run());

  // Validate results.
  EXPECT_THAT(GetResult<std::vector<Detection>>(poller),
              IsOkAndHolds(ElementsAre(Approximately(EqualsProto(
                                           R"pb(
                                             label: "bicycle"
                                             score: 1
                                             location_data {
                                               format: RELATIVE_BOUNDING_BOX
                                               relative_bounding_box {
                                                 xmin: 0.8137423
                                                 ymin: 0.067235775
                                                 width: 0.117221
                                                 height: 0.064774655
                                               }
                                             }
                                           )pb")),
                                       Approximately(EqualsProto(
                                           R"pb(
                                             label: "car"
                                             score: 0.9
                                             location_data {
                                               format: RELATIVE_BOUNDING_BOX
                                               relative_bounding_box {
                                                 xmin: 0.53849804
                                                 ymin: 0.08949606
                                                 width: 0.05861056
                                                 height: 0.11722109
                                               }
                                             }
                                           )pb")),
                                       Approximately(EqualsProto(
                                           R"pb(
                                             label: "motorcycle"
                                             score: 0.8
                                             location_data {
                                               format: RELATIVE_BOUNDING_BOX
                                               relative_bounding_box {
                                                 xmin: 0.13779688
                                                 ymin: 0.26394117
                                                 width: 0.16322193
                                                 height: 0.07384467
                                               }
                                             }
                                           )pb")))));
}

}  // namespace
}  // namespace processors
}  // namespace components
}  // namespace tasks
}  // namespace mediapipe
