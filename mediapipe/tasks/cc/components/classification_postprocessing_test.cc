/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

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

#include "mediapipe/tasks/cc/components/classification_postprocessing.h"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/tensor/tensors_to_classification_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/graph_runner.h"
#include "mediapipe/framework/output_stream_poller.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/tasks/cc/components/calculators/classification_aggregation_calculator.pb.h"
#include "mediapipe/tasks/cc/components/classification_postprocessing_options.pb.h"
#include "mediapipe/tasks/cc/components/classifier_options.pb.h"
#include "mediapipe/tasks/cc/components/containers/classifications.pb.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/util/label_map.pb.h"
#include "tensorflow/lite/core/shims/cc/shims_test_util.h"

namespace mediapipe {
namespace tasks {
namespace {

using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::core::ModelResources;
using ::testing::HasSubstr;
using ::testing::proto::Approximately;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/";
constexpr char kQuantizedImageClassifierWithMetadata[] =
    "vision/mobilenet_v1_0.25_224_quant.tflite";
constexpr char kQuantizedImageClassifierWithoutMetadata[] =
    "vision/mobilenet_v1_0.25_192_quantized_1_default_1.tflite";
constexpr char kFloatTwoHeadsAudioClassifierWithMetadata[] =
    "audio/two_heads.tflite";

constexpr char kTestModelResourcesTag[] = "test_model_resources";
constexpr int kMobileNetNumClasses = 1001;
constexpr int kTwoHeadsNumClasses[] = {521, 5};

constexpr char kTensorsTag[] = "TENSORS";
constexpr char kTensorsName[] = "tensors";
constexpr char kTimestampsTag[] = "TIMESTAMPS";
constexpr char kTimestampsName[] = "timestamps";
constexpr char kClassificationResultTag[] = "CLASSIFICATION_RESULT";
constexpr char kClassificationResultName[] = "classification_result";

// Helper function to get ModelResources.
absl::StatusOr<std::unique_ptr<ModelResources>> CreateModelResourcesForModel(
    absl::string_view model_name) {
  auto external_file = std::make_unique<core::proto::ExternalFile>();
  external_file->set_file_name(JoinPath("./", kTestDataDirectory, model_name));
  return ModelResources::Create(kTestModelResourcesTag,
                                std::move(external_file));
}

class ConfigureTest : public tflite_shims::testing::Test {};

TEST_F(ConfigureTest, FailsWithInvalidMaxResults) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateModelResourcesForModel(kQuantizedImageClassifierWithMetadata));
  ClassifierOptions options_in;
  options_in.set_max_results(0);

  ClassificationPostprocessingOptions options_out;
  auto status = ConfigureClassificationPostprocessing(*model_resources,
                                                      options_in, &options_out);

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("Invalid `max_results` option"));
}

TEST_F(ConfigureTest, FailsWithBothAllowlistAndDenylist) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateModelResourcesForModel(kQuantizedImageClassifierWithMetadata));
  ClassifierOptions options_in;
  options_in.add_category_allowlist("foo");
  options_in.add_category_denylist("bar");

  ClassificationPostprocessingOptions options_out;
  auto status = ConfigureClassificationPostprocessing(*model_resources,
                                                      options_in, &options_out);

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("mutually exclusive options"));
}

TEST_F(ConfigureTest, FailsWithAllowlistAndNoMetadata) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateModelResourcesForModel(kQuantizedImageClassifierWithoutMetadata));
  ClassifierOptions options_in;
  options_in.add_category_allowlist("foo");

  ClassificationPostprocessingOptions options_out;
  auto status = ConfigureClassificationPostprocessing(*model_resources,
                                                      options_in, &options_out);

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      status.message(),
      HasSubstr("requires labels to be present in the TFLite Model Metadata"));
}

TEST_F(ConfigureTest, SucceedsWithoutMetadata) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateModelResourcesForModel(kQuantizedImageClassifierWithoutMetadata));
  ClassifierOptions options_in;

  ClassificationPostprocessingOptions options_out;
  MP_EXPECT_OK(ConfigureClassificationPostprocessing(*model_resources,
                                                     options_in, &options_out));

  EXPECT_THAT(options_out, Approximately(EqualsProto(
                               R"pb(tensors_to_classifications_options {
                                      min_score_threshold: -3.4028235e+38
                                      top_k: -1
                                      sort_by_descending_score: true
                                    }
                                    classification_aggregation_options {}
                                    has_quantized_outputs: true
                               )pb")));
}

TEST_F(ConfigureTest, SucceedsWithMaxResults) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateModelResourcesForModel(kQuantizedImageClassifierWithoutMetadata));
  ClassifierOptions options_in;
  options_in.set_max_results(3);

  ClassificationPostprocessingOptions options_out;
  MP_EXPECT_OK(ConfigureClassificationPostprocessing(*model_resources,
                                                     options_in, &options_out));

  EXPECT_THAT(options_out, Approximately(EqualsProto(
                               R"pb(tensors_to_classifications_options {
                                      min_score_threshold: -3.4028235e+38
                                      top_k: 3
                                      sort_by_descending_score: true
                                    }
                                    classification_aggregation_options {}
                                    has_quantized_outputs: true
                               )pb")));
}

TEST_F(ConfigureTest, SucceedsWithScoreThreshold) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateModelResourcesForModel(kQuantizedImageClassifierWithoutMetadata));
  ClassifierOptions options_in;
  options_in.set_score_threshold(0.5);

  ClassificationPostprocessingOptions options_out;
  MP_EXPECT_OK(ConfigureClassificationPostprocessing(*model_resources,
                                                     options_in, &options_out));

  EXPECT_THAT(options_out, Approximately(EqualsProto(
                               R"pb(tensors_to_classifications_options {
                                      min_score_threshold: 0.5
                                      top_k: -1
                                      sort_by_descending_score: true
                                    }
                                    classification_aggregation_options {}
                                    has_quantized_outputs: true
                               )pb")));
}

TEST_F(ConfigureTest, SucceedsWithMetadata) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateModelResourcesForModel(kQuantizedImageClassifierWithMetadata));
  ClassifierOptions options_in;

  ClassificationPostprocessingOptions options_out;
  MP_EXPECT_OK(ConfigureClassificationPostprocessing(*model_resources,
                                                     options_in, &options_out));

  // Check label map size and two first elements.
  EXPECT_EQ(
      options_out.tensors_to_classifications_options(0).label_items_size(),
      kMobileNetNumClasses);
  EXPECT_THAT(
      options_out.tensors_to_classifications_options(0).label_items().at(0),
      EqualsProto(R"pb(name: "background")pb"));
  EXPECT_THAT(
      options_out.tensors_to_classifications_options(0).label_items().at(1),
      EqualsProto(R"pb(name: "tench")pb"));
  // Clear label map and compare the rest of the options.
  options_out.mutable_tensors_to_classifications_options(0)
      ->clear_label_items();
  EXPECT_THAT(options_out, Approximately(EqualsProto(
                               R"pb(tensors_to_classifications_options {
                                      min_score_threshold: -3.4028235e+38
                                      top_k: -1
                                      sort_by_descending_score: true
                                    }
                                    classification_aggregation_options {
                                      head_names: "probability"
                                    }
                                    has_quantized_outputs: true
                               )pb")));
}

TEST_F(ConfigureTest, SucceedsWithAllowlist) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateModelResourcesForModel(kQuantizedImageClassifierWithMetadata));
  ClassifierOptions options_in;
  options_in.add_category_allowlist("tench");

  ClassificationPostprocessingOptions options_out;
  MP_EXPECT_OK(ConfigureClassificationPostprocessing(*model_resources,
                                                     options_in, &options_out));

  // Clear label map and compare the rest of the options.
  options_out.mutable_tensors_to_classifications_options(0)
      ->clear_label_items();
  EXPECT_THAT(options_out, Approximately(EqualsProto(
                               R"pb(tensors_to_classifications_options {
                                      min_score_threshold: -3.4028235e+38
                                      top_k: -1
                                      sort_by_descending_score: true
                                      allow_classes: 1
                                    }
                                    classification_aggregation_options {
                                      head_names: "probability"
                                    }
                                    has_quantized_outputs: true
                               )pb")));
}

TEST_F(ConfigureTest, SucceedsWithDenylist) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateModelResourcesForModel(kQuantizedImageClassifierWithMetadata));
  ClassifierOptions options_in;
  options_in.add_category_denylist("background");

  ClassificationPostprocessingOptions options_out;
  MP_EXPECT_OK(ConfigureClassificationPostprocessing(*model_resources,
                                                     options_in, &options_out));

  // Clear label map and compare the rest of the options.
  options_out.mutable_tensors_to_classifications_options(0)
      ->clear_label_items();
  EXPECT_THAT(options_out, Approximately(EqualsProto(
                               R"pb(tensors_to_classifications_options {
                                      min_score_threshold: -3.4028235e+38
                                      top_k: -1
                                      sort_by_descending_score: true
                                      ignore_classes: 0
                                    }
                                    classification_aggregation_options {
                                      head_names: "probability"
                                    }
                                    has_quantized_outputs: true
                               )pb")));
}

TEST_F(ConfigureTest, SucceedsWithMultipleHeads) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateModelResourcesForModel(kFloatTwoHeadsAudioClassifierWithMetadata));
  ClassifierOptions options_in;

  ClassificationPostprocessingOptions options_out;
  MP_EXPECT_OK(ConfigureClassificationPostprocessing(*model_resources,
                                                     options_in, &options_out));
  // Check label maps sizes and first two elements.
  EXPECT_EQ(
      options_out.tensors_to_classifications_options(0).label_items_size(),
      kTwoHeadsNumClasses[0]);
  EXPECT_THAT(
      options_out.tensors_to_classifications_options(0).label_items().at(0),
      EqualsProto(R"pb(name: "Speech")pb"));
  EXPECT_THAT(
      options_out.tensors_to_classifications_options(0).label_items().at(1),
      EqualsProto(R"pb(name: "Child speech, kid speaking")pb"));
  EXPECT_EQ(
      options_out.tensors_to_classifications_options(1).label_items_size(),
      kTwoHeadsNumClasses[1]);
  EXPECT_THAT(
      options_out.tensors_to_classifications_options(1).label_items().at(0),
      EqualsProto(R"pb(name: "Red Crossbill")pb"));
  EXPECT_THAT(
      options_out.tensors_to_classifications_options(1).label_items().at(1),
      EqualsProto(R"pb(name: "White-breasted Wood-Wren")pb"));
  // Clear label maps and compare the rest of the options.
  options_out.mutable_tensors_to_classifications_options(0)
      ->clear_label_items();
  options_out.mutable_tensors_to_classifications_options(1)
      ->clear_label_items();
  EXPECT_THAT(options_out, Approximately(EqualsProto(
                               R"pb(tensors_to_classifications_options {
                                      min_score_threshold: -3.4028235e+38
                                      top_k: -1
                                      sort_by_descending_score: true
                                    }
                                    tensors_to_classifications_options {
                                      min_score_threshold: -3.4028235e+38
                                      top_k: -1
                                      sort_by_descending_score: true
                                    }
                                    classification_aggregation_options {
                                      head_names: "yamnet_classification"
                                      head_names: "bird_classification"
                                    }
                                    has_quantized_outputs: false
                               )pb")));
}

class PostprocessingTest : public tflite_shims::testing::Test {
 protected:
  absl::StatusOr<OutputStreamPoller> BuildGraph(
      absl::string_view model_name, const ClassifierOptions& options,
      bool connect_timestamps = false) {
    ASSIGN_OR_RETURN(auto model_resources,
                     CreateModelResourcesForModel(model_name));

    Graph graph;
    auto& postprocessing =
        graph.AddNode("mediapipe.tasks.ClassificationPostprocessingSubgraph");
    MP_RETURN_IF_ERROR(ConfigureClassificationPostprocessing(
        *model_resources, options,
        &postprocessing.GetOptions<ClassificationPostprocessingOptions>()));
    graph[Input<std::vector<Tensor>>(kTensorsTag)].SetName(kTensorsName) >>
        postprocessing.In(kTensorsTag);
    if (connect_timestamps) {
      graph[Input<std::vector<Timestamp>>(kTimestampsTag)].SetName(
          kTimestampsName) >>
          postprocessing.In(kTimestampsTag);
    }
    postprocessing.Out(kClassificationResultTag)
            .SetName(kClassificationResultName) >>
        graph[Output<ClassificationResult>(kClassificationResultTag)];

    MP_RETURN_IF_ERROR(calculator_graph_.Initialize(graph.GetConfig()));
    ASSIGN_OR_RETURN(auto poller, calculator_graph_.AddOutputStreamPoller(
                                      kClassificationResultName));
    MP_RETURN_IF_ERROR(calculator_graph_.StartRun(/*extra_side_packets=*/{}));
    return poller;
  }

  template <typename T>
  void AddTensor(
      const std::vector<T>& tensor, const Tensor::ElementType& element_type,
      const Tensor::QuantizationParameters& quantization_parameters = {}) {
    tensors_->emplace_back(element_type,
                           Tensor::Shape{1, static_cast<int>(tensor.size())},
                           quantization_parameters);
    auto view = tensors_->back().GetCpuWriteView();
    T* buffer = view.buffer<T>();
    std::copy(tensor.begin(), tensor.end(), buffer);
  }

  absl::Status Run(
      std::optional<std::vector<int>> aggregation_timestamps = std::nullopt,
      int timestamp = 0) {
    MP_RETURN_IF_ERROR(calculator_graph_.AddPacketToInputStream(
        kTensorsName, Adopt(tensors_.release()).At(Timestamp(timestamp))));
    // Reset tensors for future calls.
    tensors_ = absl::make_unique<std::vector<Tensor>>();
    if (aggregation_timestamps.has_value()) {
      auto packet = absl::make_unique<std::vector<Timestamp>>();
      for (const auto& timestamp : *aggregation_timestamps) {
        packet->emplace_back(Timestamp(timestamp));
      }
      MP_RETURN_IF_ERROR(calculator_graph_.AddPacketToInputStream(
          kTimestampsName, Adopt(packet.release()).At(Timestamp(timestamp))));
    }
    return absl::OkStatus();
  }

  absl::StatusOr<ClassificationResult> GetClassificationResult(
      OutputStreamPoller& poller) {
    MP_RETURN_IF_ERROR(calculator_graph_.WaitUntilIdle());
    MP_RETURN_IF_ERROR(calculator_graph_.CloseAllInputStreams());

    Packet packet;
    if (!poller.Next(&packet)) {
      return absl::InternalError("Unable to get output packet");
    }
    auto result = packet.Get<ClassificationResult>();
    MP_RETURN_IF_ERROR(calculator_graph_.WaitUntilDone());
    return result;
  }

 private:
  CalculatorGraph calculator_graph_;
  std::unique_ptr<std::vector<Tensor>> tensors_ =
      absl::make_unique<std::vector<Tensor>>();
};

TEST_F(PostprocessingTest, SucceedsWithoutMetadata) {
  // Build graph.
  ClassifierOptions options;
  options.set_max_results(3);
  options.set_score_threshold(0.5);
  MP_ASSERT_OK_AND_ASSIGN(
      auto poller,
      BuildGraph(kQuantizedImageClassifierWithoutMetadata, options));
  // Build input tensors.
  std::vector<uint8> tensor(kMobileNetNumClasses, 0);
  tensor[1] = 18;
  tensor[2] = 16;

  // Send tensors and get results.
  AddTensor(tensor, Tensor::ElementType::kUInt8,
            /*quantization_parameters=*/{0.1, 10});
  MP_ASSERT_OK(Run());
  MP_ASSERT_OK_AND_ASSIGN(auto results, GetClassificationResult(poller));

  // Validate results.
  EXPECT_THAT(results, EqualsProto(R"pb(classifications {
                                          entries {
                                            categories { index: 1 score: 0.8 }
                                            categories { index: 2 score: 0.6 }
                                            timestamp_ms: 0
                                          }
                                        })pb"));
}

TEST_F(PostprocessingTest, SucceedsWithMetadata) {
  // Build graph.
  ClassifierOptions options;
  options.set_max_results(3);
  MP_ASSERT_OK_AND_ASSIGN(
      auto poller, BuildGraph(kQuantizedImageClassifierWithMetadata, options));
  // Build input tensors.
  std::vector<uint8> tensor(kMobileNetNumClasses, 0);
  tensor[1] = 12;
  tensor[2] = 14;
  tensor[3] = 16;
  tensor[4] = 18;

  // Send tensors and get results.
  AddTensor(tensor, Tensor::ElementType::kUInt8,
            /*quantization_parameters=*/{0.1, 10});
  MP_ASSERT_OK(Run());
  MP_ASSERT_OK_AND_ASSIGN(auto results, GetClassificationResult(poller));

  // Validate results.
  EXPECT_THAT(
      results,
      EqualsProto(
          R"pb(classifications {
                 entries {
                   categories {
                     index: 4
                     score: 0.8
                     category_name: "tiger shark"
                   }
                   categories {
                     index: 3
                     score: 0.6
                     category_name: "great white shark"
                   }
                   categories { index: 2 score: 0.4 category_name: "goldfish" }
                   timestamp_ms: 0
                 }
                 head_index: 0
                 head_name: "probability"
               })pb"));
}

TEST_F(PostprocessingTest, SucceedsWithMultipleHeads) {
  // Build graph.
  ClassifierOptions options;
  options.set_max_results(2);
  MP_ASSERT_OK_AND_ASSIGN(
      auto poller,
      BuildGraph(kFloatTwoHeadsAudioClassifierWithMetadata, options));
  // Build input tensors.
  std::vector<float> tensor_0(kTwoHeadsNumClasses[0], 0);
  tensor_0[1] = 0.2;
  tensor_0[2] = 0.4;
  tensor_0[3] = 0.6;
  std::vector<float> tensor_1(kTwoHeadsNumClasses[1], 0);
  tensor_1[1] = 0.2;
  tensor_1[2] = 0.4;
  tensor_1[3] = 0.6;

  // Send tensors and get results.
  AddTensor(tensor_0, Tensor::ElementType::kFloat32);
  AddTensor(tensor_1, Tensor::ElementType::kFloat32);
  MP_ASSERT_OK(Run());
  MP_ASSERT_OK_AND_ASSIGN(auto results, GetClassificationResult(poller));

  EXPECT_THAT(results, EqualsProto(
                           R"pb(classifications {
                                  entries {
                                    categories {
                                      index: 3
                                      score: 0.6
                                      category_name: "Narration, monologue"
                                    }
                                    categories {
                                      index: 2
                                      score: 0.4
                                      category_name: "Conversation"
                                    }
                                    timestamp_ms: 0
                                  }
                                  head_index: 0
                                  head_name: "yamnet_classification"
                                }
                                classifications {
                                  entries {
                                    categories {
                                      index: 3
                                      score: 0.6
                                      category_name: "Azara\'s Spinetail"
                                    }
                                    categories {
                                      index: 2
                                      score: 0.4
                                      category_name: "House Sparrow"
                                    }
                                    timestamp_ms: 0
                                  }
                                  head_index: 1
                                  head_name: "bird_classification"
                                })pb"));
}

TEST_F(PostprocessingTest, SucceedsWithTimestamps) {
  // Build graph.
  ClassifierOptions options;
  options.set_max_results(2);
  MP_ASSERT_OK_AND_ASSIGN(
      auto poller, BuildGraph(kQuantizedImageClassifierWithMetadata, options,
                              /*connect_timestamps=*/true));
  // Build input tensors.
  std::vector<uint8> tensor_0(kMobileNetNumClasses, 0);
  tensor_0[1] = 12;
  tensor_0[2] = 14;
  tensor_0[3] = 16;
  std::vector<uint8> tensor_1(kMobileNetNumClasses, 0);
  tensor_1[5] = 12;
  tensor_1[6] = 14;
  tensor_1[7] = 16;

  // Send tensors and get results.
  AddTensor(tensor_0, Tensor::ElementType::kUInt8,
            /*quantization_parameters=*/{0.1, 10});
  MP_ASSERT_OK(Run());
  AddTensor(tensor_1, Tensor::ElementType::kUInt8,
            /*quantization_parameters=*/{0.1, 10});
  MP_ASSERT_OK(Run(
      /*aggregation_timestamps=*/std::optional<std::vector<int>>({0, 1000}),
      /*timestamp=*/1000));

  MP_ASSERT_OK_AND_ASSIGN(auto results, GetClassificationResult(poller));

  // Validate results.
  EXPECT_THAT(
      results,
      EqualsProto(
          R"pb(classifications {
                 entries {
                   categories {
                     index: 3
                     score: 0.6
                     category_name: "great white shark"
                   }
                   categories { index: 2 score: 0.4 category_name: "goldfish" }
                   timestamp_ms: 0
                 }
                 entries {
                   categories { index: 7 score: 0.6 category_name: "stingray" }
                   categories {
                     index: 6
                     score: 0.4
                     category_name: "electric ray"
                   }
                   timestamp_ms: 1
                 }
                 head_index: 0
                 head_name: "probability"
               })pb"));
}

}  // namespace
}  // namespace tasks
}  // namespace mediapipe
