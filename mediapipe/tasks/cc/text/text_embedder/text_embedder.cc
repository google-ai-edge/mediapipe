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

#include "mediapipe/tasks/cc/text/text_embedder/text_embedder.h"

#include <memory>

#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "mediapipe/tasks/cc/components/containers/proto/embeddings.pb.h"
#include "mediapipe/tasks/cc/components/processors/embedder_options.h"
#include "mediapipe/tasks/cc/components/processors/proto/embedder_options.pb.h"
#include "mediapipe/tasks/cc/components/utils/cosine_similarity.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "mediapipe/tasks/cc/core/task_api_factory.h"
#include "mediapipe/tasks/cc/text/text_embedder/proto/text_embedder_graph_options.pb.h"

namespace mediapipe::tasks::text::text_embedder {
namespace {

constexpr char kTextTag[] = "TEXT";
constexpr char kEmbeddingsTag[] = "EMBEDDINGS";
constexpr char kTextInStreamName[] = "text_in";
constexpr char kEmbeddingsStreamName[] = "embeddings_out";
constexpr char kGraphTypeName[] =
    "mediapipe.tasks.text.text_embedder.TextEmbedderGraph";

using ::mediapipe::tasks::components::containers::ConvertToEmbeddingResult;
using ::mediapipe::tasks::components::containers::proto::EmbeddingResult;

// Creates a MediaPipe graph config that contains a single node of type
// "mediapipe.tasks.text.text_embedder.TextEmbedderGraph".
CalculatorGraphConfig CreateGraphConfig(
    std::unique_ptr<proto::TextEmbedderGraphOptions> options_proto) {
  api2::builder::Graph graph;
  auto& task_graph = graph.AddNode(kGraphTypeName);
  task_graph.GetOptions<proto::TextEmbedderGraphOptions>().Swap(
      options_proto.get());
  graph.In(kTextTag).SetName(kTextInStreamName) >> task_graph.In(kTextTag);
  task_graph.Out(kEmbeddingsTag).SetName(kEmbeddingsStreamName) >>
      graph.Out(kEmbeddingsTag);
  return graph.GetConfig();
}

// Converts the user-facing TextEmbedderOptions struct to the internal
// TextEmbedderGraphOptions proto.
std::unique_ptr<proto::TextEmbedderGraphOptions>
ConvertTextEmbedderOptionsToProto(TextEmbedderOptions* options) {
  auto options_proto = std::make_unique<proto::TextEmbedderGraphOptions>();
  auto base_options_proto = std::make_unique<tasks::core::proto::BaseOptions>(
      tasks::core::ConvertBaseOptionsToProto(&(options->base_options)));
  options_proto->mutable_base_options()->Swap(base_options_proto.get());
  auto embedder_options_proto =
      std::make_unique<components::processors::proto::EmbedderOptions>(
          components::processors::ConvertEmbedderOptionsToProto(
              &(options->embedder_options)));
  options_proto->mutable_embedder_options()->Swap(embedder_options_proto.get());
  return options_proto;
}

}  // namespace

absl::StatusOr<std::unique_ptr<TextEmbedder>> TextEmbedder::Create(
    std::unique_ptr<TextEmbedderOptions> options) {
  std::unique_ptr<proto::TextEmbedderGraphOptions> options_proto =
      ConvertTextEmbedderOptionsToProto(options.get());
  return core::TaskApiFactory::Create<TextEmbedder,
                                      proto::TextEmbedderGraphOptions>(
      CreateGraphConfig(std::move(options_proto)),
      std::move(options->base_options.op_resolver));
}

absl::StatusOr<TextEmbedderResult> TextEmbedder::Embed(absl::string_view text) {
  MP_ASSIGN_OR_RETURN(
      auto output_packets,
      runner_->Process(
          {{kTextInStreamName, MakePacket<std::string>(std::string(text))}}));
  return ConvertToEmbeddingResult(
      output_packets[kEmbeddingsStreamName].Get<EmbeddingResult>());
}

absl::StatusOr<double> TextEmbedder::CosineSimilarity(
    const components::containers::Embedding& u,
    const components::containers::Embedding& v) {
  return components::utils::CosineSimilarity(u, v);
}

}  // namespace mediapipe::tasks::text::text_embedder
