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
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "mediapipe/tasks/cc/components/containers/proto/embeddings.pb.h"
#include "mediapipe/tasks/cc/components/processors/embedder_options.h"
#include "mediapipe/tasks/cc/components/processors/proto/embedder_options.pb.h"
#include "mediapipe/tasks/cc/components/utils/cosine_similarity.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "mediapipe/tasks/cc/core/running_mode.h"
#include "mediapipe/tasks/cc/core/task_api_factory.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/text/text_embedder/proto/text_embedder_graph_options.pb.h"

namespace mediapipe::tasks::text::text_embedder {
namespace {

constexpr char kTaskName[] = "TextEmbedder";
constexpr char kTextTag[] = "TEXT";
constexpr char kEmbeddingsTag[] = "EMBEDDINGS";
constexpr char kTextInStreamName[] = "text_in";
constexpr char kEmbeddingsStreamName[] = "embeddings_out";
constexpr char kGraphTypeName[] =
    "mediapipe.tasks.text.text_embedder.TextEmbedderGraph";

using ::mediapipe::tasks::components::containers::ConvertToEmbeddingResult;
using ::mediapipe::tasks::components::containers::proto::EmbeddingResult;

constexpr absl::string_view kQueryTemplate = "task: $0 | query: $1";
constexpr absl::string_view kDocumentTemplate = "title: $0 | text: $1";

std::string GetTaskString(const EmbeddingType& task_type) {
  switch (task_type) {
    case EmbeddingType::RETRIEVAL_QUERY:
      return "search result";
    case EmbeddingType::SEMANTIC_SIMILARITY:
      return "sentence similarity";
    case EmbeddingType::CLASSIFICATION:
      return "classification";
    case EmbeddingType::CLUSTERING:
      return "clustering";
    case EmbeddingType::QUESTION_ANSWERING:
      return "question answering";
    case EmbeddingType::FACT_CHECKING:
      return "fact checking";
    case EmbeddingType::CODE_RETRIEVAL:
      return "code retrieval";
    default:
      return "search result";
  }
}

std::string GetFormattedEmbeddingText(absl::string_view text,
                                      const TextFormatContext& format_context) {
  EmbeddingType task_type = format_context.task_type;
  bool is_query = format_context.role != TextRole::kDocument;
  const std::string title =
      format_context.title.has_value() && !format_context.title->empty()
          ? *format_context.title
          : "none";
  switch (task_type) {
    case EmbeddingType::RETRIEVAL_DOCUMENT:
      return absl::Substitute(kDocumentTemplate, title, text);
    case EmbeddingType::RETRIEVAL_QUERY:
      return absl::Substitute(kQueryTemplate, GetTaskString(task_type), text);
    case EmbeddingType::QUESTION_ANSWERING:
    case EmbeddingType::FACT_CHECKING:
    case EmbeddingType::CODE_RETRIEVAL:
      return is_query ? absl::Substitute(kQueryTemplate,
                                         GetTaskString(task_type), text)
                      : absl::Substitute(kDocumentTemplate, title, text);
    default:
      return absl::Substitute(kQueryTemplate, GetTaskString(task_type), text);
  }
}

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
      core::TaskRunnerOptions{
          .config = CreateGraphConfig(std::move(options_proto)),
          .task_name = kTaskName,
          .task_running_mode = core::RunningMode::kUnspecified,
          .op_resolver = std::move(options->base_options.op_resolver),
          .host_environment = options->base_options.host_environment,
          .host_system = options->base_options.host_system,
          .host_version = options->base_options.host_version,
          .ca_bundle_path = options->base_options.ca_bundle_path});
}

absl::StatusOr<TextEmbedderResult> TextEmbedder::Embed(absl::string_view text) {
  MP_ASSIGN_OR_RETURN(
      auto output_packets,
      runner_->Process(
          {{kTextInStreamName, MakePacket<std::string>(std::string(text))}}));
  return ConvertToEmbeddingResult(
      output_packets[kEmbeddingsStreamName].Get<EmbeddingResult>());
}

absl::StatusOr<TextEmbedderResult> TextEmbedder::Embed(
    absl::string_view text, const TextFormatContext& format_context) {
  std::string processed_text = GetFormattedEmbeddingText(text, format_context);
  MP_ASSIGN_OR_RETURN(
      auto output_packets,
      runner_->Process(
          {{kTextInStreamName, MakePacket<std::string>(processed_text)}}));
  return ConvertToEmbeddingResult(
      output_packets[kEmbeddingsStreamName].Get<EmbeddingResult>());
}

absl::StatusOr<double> TextEmbedder::CosineSimilarity(
    const components::containers::Embedding& u,
    const components::containers::Embedding& v) {
  return components::utils::CosineSimilarity(u, v);
}

}  // namespace mediapipe::tasks::text::text_embedder
