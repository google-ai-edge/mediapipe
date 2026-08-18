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

#include "mediapipe/tasks/cc/text/text_embedder/litert_lm_text_embedder_executor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "mediapipe/tasks/cc/components/processors/embedder_options.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/core/logging/factory/logging_factory.h"
#include "mediapipe/tasks/cc/core/logging/tasks_logger.h"
#include "mediapipe/tasks/cc/core/running_mode.h"
#include "mediapipe/tasks/cc/text/text_embedder/text_embedder_executor.h"
#include "odml/litert_lm/runtime/core/embedding_engine_impl.h"  // from @odml
#include "odml/litert_lm/runtime/engine/embedding_engine.h"     // from @odml
#include "odml/litert_lm/runtime/engine/embedding_engine_settings.h"  // from @odml
#include "odml/litert_lm/runtime/engine/io_types.h"  // from @odml
#include "odml/litert_lm/runtime/executor/executor_settings_base.h"  // from @odml
#include "odml/litert_lm/runtime/util/memory_mapped_file.h"  // from @odml

namespace mediapipe::tasks::text::text_embedder {
namespace {

using ::litert::lm::EmbeddingEngineImpl;
using ::litert::lm::EmbeddingEngineSettings;
using ::litert::lm::EmbeddingOptions;
using ::litert::lm::InputData;
using ::litert::lm::InputText;
using ::litert::lm::MemoryMappedFile;
using ::litert::lm::ModelAssets;
using ::mediapipe::tasks::components::processors::EmbedderOptions;
using ::mediapipe::tasks::core::BaseOptions;

}  // namespace

LiteRtLmTextEmbedderExecutor::LiteRtLmTextEmbedderExecutor(
    std::shared_ptr<MemoryMappedFile> shared_mmap,
    std::unique_ptr<::litert::lm::EmbeddingEngine> engine, bool l2_normalize,
    bool quantize,
    std::unique_ptr<tasks::core::logging::TasksLogger> tasks_logger)
    : shared_mmap_(std::move(shared_mmap)),
      engine_(std::move(engine)),
      l2_normalize_(l2_normalize),
      quantize_(quantize),
      tasks_logger_(std::move(tasks_logger)) {}

LiteRtLmTextEmbedderExecutor::~LiteRtLmTextEmbedderExecutor() {
  Close().IgnoreError();
}

absl::StatusOr<TextEmbedderResult> LiteRtLmTextEmbedderExecutor::Embed(
    absl::string_view text) {
  mediapipe::Timestamp task_logger_ts(tasks_logger_timestamp_++);
  tasks_logger_->RecordCpuInputArrival(task_logger_ts);
  absl::Cleanup log_invocation_end = [this, task_logger_ts] {
    tasks_logger_->RecordInvocationEnd(task_logger_ts);
  };

  std::vector<InputData> contents;
  contents.push_back(InputText(std::string(text)));

  EmbeddingOptions options;
  options.normalize = l2_normalize_;
  options.insert_special_tokens = true;

  ABSL_ASSIGN_OR_RETURN(auto response,
                        engine_->ComputeEmbedding(contents, options),
                        _ << "Failed to compute embeddings.");

  TextEmbedderResult result;
  components::containers::Embedding embedding;

  if (quantize_) {
    std::string quantized;
    quantized.resize(response.embedding.size());
    for (size_t i = 0; i < response.embedding.size(); ++i) {
      float val = response.embedding[i];
      int unclamped_value = static_cast<int>(roundf(val * 128));
      quantized[i] =
          static_cast<char>(std::max(-128, std::min(unclamped_value, 127)));
    }
    embedding.quantized_embedding = std::move(quantized);
  } else {
    embedding.float_embedding = std::move(response.embedding);
  }

  result.embeddings.push_back(std::move(embedding));
  return result;
}

absl::Status LiteRtLmTextEmbedderExecutor::Close() {
  if (tasks_logger_) {
    tasks_logger_->LogSessionEnd();
    tasks_logger_.reset();
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<LiteRtLmTextEmbedderExecutor>>
LiteRtLmTextEmbedderExecutor::Create(const BaseOptions& base_options,
                                     const EmbedderOptions& embedder_options) {
  if (base_options.model_asset_path.empty()) {
    return absl::FailedPreconditionError("No model asset path specified.");
  }

  ::litert::lm::Backend backend = ::litert::lm::Backend::CPU;
  if (base_options.delegate == BaseOptions::GPU) {
    backend = ::litert::lm::Backend::GPU;
  }

  // Load directly as a raw `.litertlm` model file.
  ABSL_ASSIGN_OR_RETURN(
      auto mmap_file, MemoryMappedFile::Create(base_options.model_asset_path));
  auto shared_mmap = std::shared_ptr<MemoryMappedFile>(std::move(mmap_file));

  ABSL_ASSIGN_OR_RETURN(
      auto model_assets,
      ModelAssets::Create(shared_mmap, base_options.model_asset_path));
  ABSL_ASSIGN_OR_RETURN(auto settings,
                        EmbeddingEngineSettings::CreateDefault(
                            std::move(model_assets), backend, backend));

  ABSL_ASSIGN_OR_RETURN(auto engine,
                        EmbeddingEngineImpl::Create(std::move(settings)));

  auto tasks_logger = tasks::core::logging::CreateTasksLogger(
      {.task_name = "TextEmbedder",
       .task_running_mode = tasks::core::RunningMode::kUnspecified,
       .host_environment = base_options.host_environment,
       .host_system = base_options.host_system,
       .host_version = base_options.host_version,
       .app_id = base_options.app_id,
       .app_version = base_options.app_version,
       .ca_bundle_path = base_options.ca_bundle_path});
  tasks_logger->LogSessionStart();

  return std::make_unique<LiteRtLmTextEmbedderExecutor>(
      std::move(shared_mmap), std::move(engine), embedder_options.l2_normalize,
      embedder_options.quantize, std::move(tasks_logger));
}

}  // namespace mediapipe::tasks::text::text_embedder
