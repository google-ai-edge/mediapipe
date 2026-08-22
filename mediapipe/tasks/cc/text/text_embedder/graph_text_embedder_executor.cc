/* Copyright 2026 The MediaPipe Authors. All Rights Reserved.

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

#include "mediapipe/tasks/cc/text/text_embedder/graph_text_embedder_executor.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "mediapipe/tasks/cc/components/containers/proto/embeddings.pb.h"
#include "mediapipe/tasks/cc/components/processors/embedder_options.h"
#include "mediapipe/tasks/cc/components/processors/proto/embedder_options.pb.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "mediapipe/tasks/cc/core/running_mode.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/text/text_embedder/proto/text_embedder_graph_options.pb.h"
#include "mediapipe/tasks/cc/text/text_embedder/text_embedder_executor.h"

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

}  // namespace

GraphTextEmbedderExecutor::GraphTextEmbedderExecutor(
    std::unique_ptr<tasks::core::TaskRunner> runner)
    : runner_(std::move(runner)) {}

absl::StatusOr<TextEmbedderResult> GraphTextEmbedderExecutor::Embed(
    absl::string_view text) {
  ABSL_ASSIGN_OR_RETURN(
      auto output_packets,
      runner_->Process(
          {{kTextInStreamName, MakePacket<std::string>(std::string(text))}}));
  return ConvertToEmbeddingResult(
      output_packets[kEmbeddingsStreamName].Get<EmbeddingResult>());
}

absl::Status GraphTextEmbedderExecutor::Close() { return runner_->Close(); }

absl::StatusOr<std::unique_ptr<GraphTextEmbedderExecutor>>
GraphTextEmbedderExecutor::Create(
    tasks::core::TaskRunnerOptions task_runner_options) {
  ABSL_ASSIGN_OR_RETURN(
      auto runner, core::TaskRunner::Create(std::move(task_runner_options)));
  return std::make_unique<GraphTextEmbedderExecutor>(std::move(runner));
}

}  // namespace mediapipe::tasks::text::text_embedder
