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

#include "mediapipe/tasks/cc/components/processors/embedding_postprocessing_graph.h"

#include <memory>

#include "absl/flags/flag.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/graph_runner.h"
#include "mediapipe/framework/output_stream_poller.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/tasks/cc/components/calculators/tensors_to_embeddings_calculator.pb.h"
#include "mediapipe/tasks/cc/components/containers/proto/embeddings.pb.h"
#include "mediapipe/tasks/cc/components/processors/proto/embedder_options.pb.h"
#include "mediapipe/tasks/cc/components/processors/proto/embedding_postprocessing_graph_options.pb.h"
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
using ::mediapipe::tasks::components::containers::proto::EmbeddingResult;
using ::mediapipe::tasks::core::ModelResources;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/";
constexpr char kMobileNetV3Embedder[] =
    "vision/mobilenet_v3_small_100_224_embedder.tflite";
// Abusing a few classifiers (topologically similar to embedders) in order to
// add coverage.
constexpr char kQuantizedImageClassifierWithMetadata[] =
    "vision/mobilenet_v1_0.25_224_quant.tflite";
constexpr char kQuantizedImageClassifierWithoutMetadata[] =
    "vision/mobilenet_v1_0.25_192_quantized_1_default_1.tflite";

constexpr char kTestModelResourcesTag[] = "test_model_resources";
constexpr int kMobileNetV3EmbedderEmbeddingSize = 1024;

constexpr char kTensorsTag[] = "TENSORS";
constexpr char kTensorsName[] = "tensors";
constexpr char kTimestampsTag[] = "TIMESTAMPS";
constexpr char kTimestampsName[] = "timestamps";
constexpr char kEmbeddingsTag[] = "EMBEDDINGS";
constexpr char kEmbeddingsName[] = "embeddings";
constexpr char kTimestampedEmbeddingsTag[] = "TIMESTAMPED_EMBEDDINGS";
constexpr char kTimestampedEmbeddingsName[] = "timestamped_embeddings";

// Helper function to get ModelResources.
absl::StatusOr<std::unique_ptr<ModelResources>> CreateModelResourcesForModel(
    absl::string_view model_name) {
  auto external_file = std::make_unique<core::proto::ExternalFile>();
  external_file->set_file_name(JoinPath("./", kTestDataDirectory, model_name));
  return ModelResources::Create(kTestModelResourcesTag,
                                std::move(external_file));
}

class ConfigureTest : public tflite::testing::Test {};

TEST_F(ConfigureTest, SucceedsWithQuantizedModelWithMetadata) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateModelResourcesForModel(kQuantizedImageClassifierWithMetadata));
  proto::EmbedderOptions options_in;
  options_in.set_l2_normalize(true);

  proto::EmbeddingPostprocessingGraphOptions options_out;
  MP_ASSERT_OK(ConfigureEmbeddingPostprocessingGraph(*model_resources,
                                                     options_in, &options_out));

  EXPECT_THAT(
      options_out,
      EqualsProto(
          ParseTextProtoOrDie<proto::EmbeddingPostprocessingGraphOptions>(
              R"pb(tensors_to_embeddings_options {
                     embedder_options { l2_normalize: true }
                     head_names: "probability"
                   }
                   has_quantized_outputs: true)pb")));
}

TEST_F(ConfigureTest, SucceedsWithQuantizedModelWithoutMetadata) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateModelResourcesForModel(kQuantizedImageClassifierWithoutMetadata));
  proto::EmbedderOptions options_in;
  options_in.set_quantize(true);

  proto::EmbeddingPostprocessingGraphOptions options_out;
  MP_ASSERT_OK(ConfigureEmbeddingPostprocessingGraph(*model_resources,
                                                     options_in, &options_out));

  EXPECT_THAT(
      options_out,
      EqualsProto(
          ParseTextProtoOrDie<proto::EmbeddingPostprocessingGraphOptions>(
              R"pb(tensors_to_embeddings_options {
                     embedder_options { quantize: true }
                   }
                   has_quantized_outputs: true)pb")));
}

TEST_F(ConfigureTest, SucceedsWithFloatModelWithMetadata) {
  MP_ASSERT_OK_AND_ASSIGN(auto model_resources,
                          CreateModelResourcesForModel(kMobileNetV3Embedder));
  proto::EmbedderOptions options_in;
  options_in.set_quantize(true);
  options_in.set_l2_normalize(true);

  proto::EmbeddingPostprocessingGraphOptions options_out;
  MP_ASSERT_OK(ConfigureEmbeddingPostprocessingGraph(*model_resources,
                                                     options_in, &options_out));

  EXPECT_THAT(
      options_out,
      EqualsProto(
          ParseTextProtoOrDie<proto::EmbeddingPostprocessingGraphOptions>(
              R"pb(tensors_to_embeddings_options {
                     embedder_options { quantize: true l2_normalize: true }
                     head_names: "feature"
                   }
                   has_quantized_outputs: false)pb")));
}

class PostprocessingTest : public tflite::testing::Test {
 protected:
  absl::StatusOr<OutputStreamPoller> BuildGraph(
      absl::string_view model_name, const proto::EmbedderOptions& options,
      bool connect_timestamps = false,
      const std::vector<absl::string_view>& ignored_head_names = {}) {
    MP_ASSIGN_OR_RETURN(auto model_resources,
                        CreateModelResourcesForModel(model_name));

    Graph graph;
    auto& postprocessing = graph.AddNode(
        "mediapipe.tasks.components.processors."
        "EmbeddingPostprocessingGraph");
    auto* postprocessing_options =
        &postprocessing
             .GetOptions<proto::EmbeddingPostprocessingGraphOptions>();
    for (const absl::string_view head_name : ignored_head_names) {
      postprocessing_options->mutable_tensors_to_embeddings_options()
          ->add_ignored_head_names(std::string(head_name));
    }
    MP_RETURN_IF_ERROR(ConfigureEmbeddingPostprocessingGraph(
        *model_resources, options, postprocessing_options));
    graph[Input<std::vector<Tensor>>(kTensorsTag)].SetName(kTensorsName) >>
        postprocessing.In(kTensorsTag);
    if (connect_timestamps) {
      graph[Input<std::vector<Timestamp>>(kTimestampsTag)].SetName(
          kTimestampsName) >>
          postprocessing.In(kTimestampsTag);
      postprocessing.Out(kTimestampedEmbeddingsTag)
              .SetName(kTimestampedEmbeddingsName) >>
          graph[Output<std::vector<EmbeddingResult>>(
              kTimestampedEmbeddingsTag)];
    } else {
      postprocessing.Out(kEmbeddingsTag).SetName(kEmbeddingsName) >>
          graph[Output<EmbeddingResult>(kEmbeddingsTag)];
    }

    MP_RETURN_IF_ERROR(calculator_graph_.Initialize(graph.GetConfig()));
    if (connect_timestamps) {
      MP_ASSIGN_OR_RETURN(auto poller, calculator_graph_.AddOutputStreamPoller(
                                           kTimestampedEmbeddingsName));
      MP_RETURN_IF_ERROR(calculator_graph_.StartRun(/*extra_side_packets=*/{}));
      return poller;
    }
    MP_ASSIGN_OR_RETURN(
        auto poller, calculator_graph_.AddOutputStreamPoller(kEmbeddingsName));
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

TEST_F(PostprocessingTest, SucceedsWithoutAggregation) {
  // Build graph.
  proto::EmbedderOptions options;
  MP_ASSERT_OK_AND_ASSIGN(auto poller,
                          BuildGraph(kMobileNetV3Embedder, options));
  // Build input tensor.
  std::vector<float> tensor(kMobileNetV3EmbedderEmbeddingSize, 0);
  tensor[0] = 1.0;

  // Send tensor and get results.
  AddTensor(tensor, Tensor::ElementType::kFloat32);
  MP_ASSERT_OK(Run());
  MP_ASSERT_OK_AND_ASSIGN(auto results, GetResult<EmbeddingResult>(poller));

  // Validate results.
  EXPECT_TRUE(results.has_timestamp_ms());
  EXPECT_EQ(results.timestamp_ms(), 0);
  EXPECT_EQ(results.embeddings_size(), 1);
  EXPECT_EQ(results.embeddings(0).head_index(), 0);
  EXPECT_EQ(results.embeddings(0).head_name(), "feature");
  EXPECT_EQ(results.embeddings(0).float_embedding().values_size(),
            kMobileNetV3EmbedderEmbeddingSize);
  EXPECT_FLOAT_EQ(results.embeddings(0).float_embedding().values(0), 1.0);
  for (int i = 1; i < kMobileNetV3EmbedderEmbeddingSize; ++i) {
    EXPECT_FLOAT_EQ(results.embeddings(0).float_embedding().values(i), 0.0);
  }
}

TEST_F(PostprocessingTest, SucceedsWithFilter) {
  // Build graph.
  proto::EmbedderOptions options;
  MP_ASSERT_OK_AND_ASSIGN(
      auto poller,
      BuildGraph(kMobileNetV3Embedder, options, /*connect_timestamps=*/false,
                 /*ignored_head_names=*/{"feature"}));
  // Build input tensor.
  std::vector<float> tensor(kMobileNetV3EmbedderEmbeddingSize, 0);
  tensor[0] = 1.0;

  // Send tensor and get results.
  AddTensor(tensor, Tensor::ElementType::kFloat32);
  MP_ASSERT_OK(Run());
  MP_ASSERT_OK_AND_ASSIGN(auto results, GetResult<EmbeddingResult>(poller));

  // Validate results.
  EXPECT_TRUE(results.has_timestamp_ms());
  EXPECT_EQ(results.timestamp_ms(), 0);
  EXPECT_EQ(results.embeddings_size(), 0);
}

TEST_F(PostprocessingTest, SucceedsWithAggregation) {
  // Build graph.
  proto::EmbedderOptions options;
  MP_ASSERT_OK_AND_ASSIGN(auto poller, BuildGraph(kMobileNetV3Embedder, options,
                                                  /*connect_timestamps=*/true));
  // Build input tensors.
  std::vector<float> tensor_0(kMobileNetV3EmbedderEmbeddingSize, 0);
  tensor_0[0] = 1.0;
  std::vector<float> tensor_1(kMobileNetV3EmbedderEmbeddingSize, 0);
  tensor_1[0] = 2.0;

  // Send tensors and get results.
  AddTensor(tensor_0, Tensor::ElementType::kFloat32);
  MP_ASSERT_OK(Run());
  AddTensor(tensor_1, Tensor::ElementType::kFloat32);
  MP_ASSERT_OK(Run(
      /*aggregation_timestamps=*/std::optional<std::vector<int>>({0, 1000}),
      /*timestamp=*/1000));
  MP_ASSERT_OK_AND_ASSIGN(auto results,
                          GetResult<std::vector<EmbeddingResult>>(poller));

  // Validate results.
  EXPECT_EQ(results.size(), 2);
  // First timestamp.
  EXPECT_EQ(results[0].timestamp_ms(), 0);
  EXPECT_EQ(results[0].embeddings(0).head_index(), 0);
  EXPECT_EQ(results[0].embeddings(0).head_name(), "feature");
  EXPECT_EQ(results[0].embeddings(0).float_embedding().values_size(),
            kMobileNetV3EmbedderEmbeddingSize);
  EXPECT_FLOAT_EQ(results[0].embeddings(0).float_embedding().values(0), 1.0);
  for (int i = 1; i < kMobileNetV3EmbedderEmbeddingSize; ++i) {
    EXPECT_FLOAT_EQ(results[0].embeddings(0).float_embedding().values(i), 0.0);
  }
  // Second timestamp.
  EXPECT_EQ(results[1].timestamp_ms(), 1);
  EXPECT_EQ(results[1].embeddings(0).head_index(), 0);
  EXPECT_EQ(results[1].embeddings(0).head_name(), "feature");
  EXPECT_EQ(results[1].embeddings(0).float_embedding().values_size(),
            kMobileNetV3EmbedderEmbeddingSize);
  EXPECT_FLOAT_EQ(results[1].embeddings(0).float_embedding().values(0), 2.0);
  for (int i = 1; i < kMobileNetV3EmbedderEmbeddingSize; ++i) {
    EXPECT_FLOAT_EQ(results[1].embeddings(0).float_embedding().values(i), 0.0);
  }
}

}  // namespace
}  // namespace processors
}  // namespace components
}  // namespace tasks
}  // namespace mediapipe
