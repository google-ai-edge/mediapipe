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

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/model_resources_cache.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/core/proto/model_resources_calculator.pb.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/shims/cc/shims_test_util.h"

namespace mediapipe {
namespace tasks {
namespace core {
namespace {

constexpr char kTestModelResourcesTag[] = "test_model_resources";

constexpr char kTestModelWithMetadataPath[] =
    "mediapipe/tasks/testdata/core/"
    "mobilenet_v1_0.25_224_quant.tflite";

// This file is a corrupted version of the original file. Some bytes have been
// trimmed as follow:
//
// tail -c +3 mobilenet_v1_0.25_224_1_default_1.tflite \
// > corrupted_mobilenet_v1_0.25_224_1_default_1.tflite
constexpr char kCorruptedModelPath[] =
    "mediapipe/tasks/testdata/core/"
    "corrupted_mobilenet_v1_0.25_224_1_default_1.tflite";

CalculatorGraphConfig GenerateGraphConfig(
    const std::string& model_resources_tag,
    const std::string& model_file_name) {
  std::string model_resources_tag_field = "";
  if (!model_resources_tag.empty()) {
    model_resources_tag_field =
        absl::Substitute("model_resources_tag: \"$0\"", model_resources_tag);
  }
  std::string model_file_field = "";
  if (!model_file_name.empty()) {
    model_file_field =
        absl::Substitute("model_file {file_name: \"$0\"}", model_file_name);
  }
  return ParseTextProtoOrDie<CalculatorGraphConfig>(absl::Substitute(
      R"(
    output_side_packet: "model"
    output_side_packet: "op_resolver"
    output_side_packet: "metadata_extractor"
    node {
      calculator: "ModelResourcesCalculator"
      output_side_packet: "MODEL:model"
      output_side_packet: "OP_RESOLVER:op_resolver"
      output_side_packet: "METADATA_EXTRACTOR:metadata_extractor"
      options {
        [mediapipe.tasks.core.proto.ModelResourcesCalculatorOptions.ext] {
          $0
          $1
        }
      }
    })",
      /*$0=*/model_resources_tag_field,
      /*$1=*/model_file_field));
}

void CheckOutputPackets(CalculatorGraph* graph) {
  auto status_or_model_packet = graph->GetOutputSidePacket("model");
  MP_ASSERT_OK(status_or_model_packet.status());
  Packet model_packet = status_or_model_packet.value();
  ASSERT_FALSE(model_packet.IsEmpty());
  MP_ASSERT_OK(model_packet.ValidateAsType<ModelResources::ModelPtr>());
  EXPECT_TRUE(model_packet.Get<ModelResources::ModelPtr>()->initialized());

  auto status_or_op_resolver_packet = graph->GetOutputSidePacket("op_resolver");
  MP_ASSERT_OK(status_or_op_resolver_packet.status());
  Packet op_resolver_packet = status_or_op_resolver_packet.value();
  ASSERT_FALSE(op_resolver_packet.IsEmpty());
  MP_EXPECT_OK(op_resolver_packet.ValidateAsType<tflite::OpResolver>());

  auto status_or_medata_extractor_packet =
      graph->GetOutputSidePacket("metadata_extractor");
  MP_ASSERT_OK(status_or_medata_extractor_packet.status());
  Packet metadata_extractor_packet = status_or_medata_extractor_packet.value();
  ASSERT_FALSE(metadata_extractor_packet.IsEmpty());
  MP_EXPECT_OK(metadata_extractor_packet
                   .ValidateAsType<metadata::ModelMetadataExtractor>());
}

void RunGraphWithGraphService(std::unique_ptr<ModelResources> model_resources,
                              CalculatorGraph* graph) {
  std::shared_ptr<ModelResourcesCache> model_resources_cache =
      std::make_shared<ModelResourcesCache>();
  MP_ASSERT_OK(
      model_resources_cache->AddModelResources(std::move(model_resources)));
  MP_ASSERT_OK(graph->SetServiceObject(kModelResourcesCacheService,
                                       model_resources_cache));
  MP_ASSERT_OK(
      graph->Initialize(GenerateGraphConfig(kTestModelResourcesTag, "")));
  MP_ASSERT_OK(graph->Run());
}

}  // namespace

class ModelResourcesCalculatorTest : public tflite_shims::testing::Test {};

TEST_F(ModelResourcesCalculatorTest, MissingCalculatorOptions) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(
      R"pb(
        output_side_packet: "model"
        node {
          calculator: "ModelResourcesCalculator"
          output_side_packet: "MODEL:model"
        })pb");
  CalculatorGraph graph;
  auto status = graph.Initialize(graph_config);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              testing::HasSubstr("must specify at least one of "
                                 "'model_resources_tag' or 'model_file'"));
}

TEST_F(ModelResourcesCalculatorTest, EmptyModelResourcesTag) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(
      R"pb(
        output_side_packet: "model"
        node {
          calculator: "ModelResourcesCalculator"
          output_side_packet: "MODEL:model"
          options {
            [mediapipe.tasks.core.proto.ModelResourcesCalculatorOptions.ext] {
              model_resources_tag: ""
            }
          }
        })pb");
  CalculatorGraph graph;
  auto status = graph.Initialize(graph_config);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              testing::HasSubstr("'model_resources_tag' should not be empty"));
}

TEST_F(ModelResourcesCalculatorTest, EmptyExternalFileProto) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(
      R"pb(
        output_side_packet: "model"
        node {
          calculator: "ModelResourcesCalculator"
          output_side_packet: "MODEL:model"
          options {
            [mediapipe.tasks.core.proto.ModelResourcesCalculatorOptions.ext] {
              model_file: {}
            }
          }
        })pb");
  CalculatorGraph graph;
  auto status = graph.Initialize(graph_config);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              testing::HasSubstr(
                  "'model_file' must specify at least one of "
                  "'file_content', 'file_descriptor_meta', or 'file_name'"));
}

TEST_F(ModelResourcesCalculatorTest, GraphServiceNotAvailable) {
  CalculatorGraph graph;
  MP_ASSERT_OK(
      graph.Initialize(GenerateGraphConfig(kTestModelResourcesTag, "")));
  auto status = graph.Run();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              testing::HasSubstr(
                  "Service \"mediapipe::tasks::ModelResourcesCacheService\", "
                  "required by node ModelResourcesCalculator, was not "
                  "provided and cannot be created"));
}

TEST_F(ModelResourcesCalculatorTest, CorruptedModelPath) {
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(GenerateGraphConfig("", kCorruptedModelPath)));
  auto status = graph.Run();
  EXPECT_THAT(status.message(),
              testing::HasSubstr("The model is not a valid Flatbuffer"));
}

TEST_F(ModelResourcesCalculatorTest, UseModelResourcesGraphService) {
  auto model_file = std::make_unique<proto::ExternalFile>();
  model_file->set_file_name(kTestModelWithMetadataPath);
  auto status_or_model_resources =
      ModelResources::Create(kTestModelResourcesTag, std::move(model_file));
  ASSERT_TRUE(status_or_model_resources.ok());

  CalculatorGraph graph;
  RunGraphWithGraphService(std::move(status_or_model_resources.value()),
                           &graph);
  CheckOutputPackets(&graph);
}

TEST_F(ModelResourcesCalculatorTest, CreateLocalModelResources) {
  CalculatorGraph graph;
  MP_ASSERT_OK(
      graph.Initialize(GenerateGraphConfig("", kTestModelWithMetadataPath)));
  MP_ASSERT_OK(graph.Run());
  CheckOutputPackets(&graph);
}

TEST_F(ModelResourcesCalculatorTest, ModelResourcesIsUnavailable) {
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.SetServiceObject(kModelResourcesCacheService,
                                      std::make_shared<ModelResourcesCache>()));
  MP_ASSERT_OK(
      graph.Initialize(GenerateGraphConfig(kTestModelResourcesTag, "")));
  auto status = graph.Run();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              testing::HasSubstr("no 'model_file' field to create a local "
                                 "ModelResources."));
}

TEST_F(ModelResourcesCalculatorTest, FallbackToCreateLocalModelResources) {
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.SetServiceObject(kModelResourcesCacheService,
                                      std::make_shared<ModelResourcesCache>()));
  MP_ASSERT_OK(graph.Initialize(
      GenerateGraphConfig(kTestModelResourcesTag, kTestModelWithMetadataPath)));
  MP_ASSERT_OK(graph.Run());
  CheckOutputPackets(&graph);
}

}  // namespace core
}  // namespace tasks
}  // namespace mediapipe
