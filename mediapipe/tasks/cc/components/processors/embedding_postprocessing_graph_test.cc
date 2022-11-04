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

#include "mediapipe/tasks/cc/components/processors/embedding_postprocessing_graph.h"

#include <memory>

#include "absl/flags/flag.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/components/processors/proto/embedder_options.pb.h"
#include "mediapipe/tasks/cc/components/processors/proto/embedding_postprocessing_graph_options.pb.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "tensorflow/lite/core/shims/cc/shims_test_util.h"

namespace mediapipe {
namespace tasks {
namespace components {
namespace processors {
namespace {

using ::mediapipe::file::JoinPath;
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

// Helper function to get ModelResources.
absl::StatusOr<std::unique_ptr<ModelResources>> CreateModelResourcesForModel(
    absl::string_view model_name) {
  auto external_file = std::make_unique<core::proto::ExternalFile>();
  external_file->set_file_name(JoinPath("./", kTestDataDirectory, model_name));
  return ModelResources::Create(kTestModelResourcesTag,
                                std::move(external_file));
}

class ConfigureTest : public tflite_shims::testing::Test {};

TEST_F(ConfigureTest, SucceedsWithQuantizedModelWithMetadata) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateModelResourcesForModel(kQuantizedImageClassifierWithMetadata));
  proto::EmbedderOptions options_in;
  options_in.set_l2_normalize(true);

  proto::EmbeddingPostprocessingGraphOptions options_out;
  MP_ASSERT_OK(ConfigureEmbeddingPostprocessing(*model_resources, options_in,
                                                &options_out));

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
  MP_ASSERT_OK(ConfigureEmbeddingPostprocessing(*model_resources, options_in,
                                                &options_out));

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
  MP_ASSERT_OK(ConfigureEmbeddingPostprocessing(*model_resources, options_in,
                                                &options_out));

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

// TODO: add E2E Postprocessing tests once timestamp aggregation is
// supported.

}  // namespace
}  // namespace processors
}  // namespace components
}  // namespace tasks
}  // namespace mediapipe
