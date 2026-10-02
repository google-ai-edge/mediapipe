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

#include "mediapipe/tasks/cc/retrieval/universal_embedder/universal_embedder.h"

#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "mediapipe/tasks/cc/components/utils/cosine_similarity.h"
#include "mediapipe/tasks/cc/components/utils/tga_utils.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/core/embedding_provider.h"
#include "mediapipe/tasks/cc/core/logging/factory/logging_factory.h"
#include "mediapipe/tasks/cc/core/logging/tasks_logger.h"
#include "mediapipe/tasks/cc/core/running_mode.h"
#include "runtime/core/embedding_engine_impl.h"        // from @litert_lm
#include "runtime/engine/embedding_engine.h"           // from @litert_lm
#include "runtime/engine/embedding_engine_settings.h"  // from @litert_lm
#include "runtime/engine/io_types.h"                   // from @litert_lm
#include "runtime/executor/executor_settings_base.h"   // from @litert_lm
#ifdef __EMSCRIPTEN__
#include "runtime/util/file_data_stream.h"  // from @litert_lm
#endif
#include "runtime/util/memory_mapped_file.h"  // from @litert_lm

namespace mediapipe::tasks::retrieval::universal_embedder {
namespace {

using ::litert::lm::EmbeddingEngineImpl;
using ::litert::lm::EmbeddingEngineSettings;
using ::litert::lm::EmbeddingOptions;
using ::litert::lm::InputAudio;
using ::litert::lm::InputData;
using ::litert::lm::InputImage;
using ::litert::lm::InputText;
using ::litert::lm::MemoryMappedFile;
using ::litert::lm::ModelAssets;
using ::mediapipe::tasks::components::utils::EncodeToTga;

::litert::lm::Backend ToLiteRtLmBackend(
    tasks::core::BaseOptions::Delegate delegate) {
  switch (delegate) {
    case tasks::core::BaseOptions::GPU:
      return ::litert::lm::Backend::GPU;
    case tasks::core::BaseOptions::NPU:
      return ::litert::lm::Backend::NPU;
    default:
      return ::litert::lm::Backend::CPU;
  }
}

}  // namespace

UniversalEmbedder::UniversalEmbedder(
    std::shared_ptr<MemoryMappedFile> shared_mmap,
    std::unique_ptr<::litert::lm::EmbeddingEngine> engine, bool l2_normalize,
    std::unique_ptr<tasks::core::logging::TasksLogger> tasks_logger)
    : shared_mmap_(std::move(shared_mmap)),
      engine_(std::move(engine)),
      l2_normalize_(l2_normalize),
      tasks_logger_(std::move(tasks_logger)) {}

absl::StatusOr<std::unique_ptr<UniversalEmbedder>> UniversalEmbedder::Create(
    std::unique_ptr<UniversalEmbedderOptions> options) {
  auto tasks_logger = tasks::core::logging::CreateTasksLogger(
      {.task_name = "UniversalEmbedder",
       .task_running_mode = tasks::core::RunningMode::kUnspecified,
       .host_environment = options->base_options.host_environment,
       .host_system = options->base_options.host_system,
       .host_version = options->base_options.host_version,
       .app_id = options->base_options.app_id,
       .app_version = options->base_options.app_version,
       .ca_bundle_path = options->base_options.ca_bundle_path});

  if (options->base_options.model_asset_path.empty()) {
    return absl::FailedPreconditionError(
        "No model asset path specified in BaseOptions.");
  }

  std::shared_ptr<MemoryMappedFile> shared_mmap = nullptr;
#ifdef __EMSCRIPTEN__
  ABSL_ASSIGN_OR_RETURN(auto file_stream,
                        ::litert::lm::FileDataStream::Create(
                            options->base_options.model_asset_path));
  ABSL_ASSIGN_OR_RETURN(auto model_assets, ModelAssets::Create(file_stream));
#else
  ABSL_ASSIGN_OR_RETURN(
      auto mmap_file,
      MemoryMappedFile::Create(options->base_options.model_asset_path));
  shared_mmap = std::shared_ptr<MemoryMappedFile>(std::move(mmap_file));
  ABSL_ASSIGN_OR_RETURN(
      auto model_assets,
      ModelAssets::Create(shared_mmap, options->base_options.model_asset_path));
#endif

  const tasks::core::BaseOptions::Delegate base_delegate =
      options->base_options.delegate;
  ABSL_ASSIGN_OR_RETURN(
      auto settings,
      EmbeddingEngineSettings::CreateDefault(
          std::move(model_assets),
          ToLiteRtLmBackend(options->text_delegate.value_or(base_delegate)),
          ToLiteRtLmBackend(options->vision_delegate.value_or(base_delegate)),
          ToLiteRtLmBackend(options->audio_delegate.value_or(base_delegate))));

  if (options->max_input_length.has_value()) {
    settings.SetMaxInputLength(options->max_input_length);
  }
  if (options->vision_tokens_per_image.has_value()) {
    settings.SetVisionTokensPerImage(options->vision_tokens_per_image);
  }

  if (options->activation_data_type.has_value()) {
    ::litert::lm::ActivationDataType litert_activation_type =
        ::litert::lm::ActivationDataType::FLOAT32;
    switch (*options->activation_data_type) {
      case ActivationDataType::FLOAT32:
        litert_activation_type = ::litert::lm::ActivationDataType::FLOAT32;
        break;
      case ActivationDataType::FLOAT16:
        litert_activation_type = ::litert::lm::ActivationDataType::FLOAT16;
        break;
      case ActivationDataType::INT16:
        litert_activation_type = ::litert::lm::ActivationDataType::INT16;
        break;
      case ActivationDataType::INT8:
        litert_activation_type = ::litert::lm::ActivationDataType::INT8;
        break;
    }
    settings.GetMutableMainExecutorSettings().SetActivationDataType(
        litert_activation_type);
    if (settings.GetMutableVisionExecutorSettings().has_value()) {
      settings.GetMutableVisionExecutorSettings()->SetActivationDataType(
          litert_activation_type);
    }
    if (settings.GetMutableAudioExecutorSettings().has_value()) {
      settings.GetMutableAudioExecutorSettings()->SetActivationDataType(
          litert_activation_type);
    }
  }

  if (options->cache_dir.has_value() && !options->cache_dir->empty()) {
    settings.GetMutableMainExecutorSettings().SetCacheDir(*options->cache_dir);
    if (settings.GetMutableVisionExecutorSettings().has_value()) {
      settings.GetMutableVisionExecutorSettings()->SetCacheDir(
          *options->cache_dir);
    }
    if (settings.GetMutableAudioExecutorSettings().has_value()) {
      settings.GetMutableAudioExecutorSettings()->SetCacheDir(
          *options->cache_dir);
    }
  }

#ifdef __EMSCRIPTEN__
  // In WebAssembly / Emscripten, virtual filesystem does not support XNNPack
  // mmap weight caching.
  settings.GetMutableMainExecutorSettings().SetDisableWeightCache(true);
  settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  if (settings.GetMutableVisionExecutorSettings().has_value()) {
    settings.GetMutableVisionExecutorSettings()->SetDisableWeightCache(true);
    settings.GetMutableVisionExecutorSettings()->SetCacheDir(":nocache");
  }
  if (settings.GetMutableAudioExecutorSettings().has_value()) {
    settings.GetMutableAudioExecutorSettings()->SetDisableWeightCache(true);
    settings.GetMutableAudioExecutorSettings()->SetCacheDir(":nocache");
  }
#endif

  ABSL_ASSIGN_OR_RETURN(auto engine,
                        EmbeddingEngineImpl::Create(std::move(settings)));

  tasks_logger->LogSessionStart();

  return absl::WrapUnique(
      new UniversalEmbedder(std::move(shared_mmap), std::move(engine),
                            options->l2_normalize, std::move(tasks_logger)));
}

absl::StatusOr<components::containers::EmbeddingResult>
UniversalEmbedder::ExecuteInference(const std::vector<InputData>& contents,
                                    absl::string_view error_message) {
  mediapipe::Timestamp task_logger_ts(tasks_logger_timestamp_++);
  tasks_logger_->RecordCpuInputArrival(task_logger_ts);
  absl::Cleanup log_invocation_end = [this, task_logger_ts] {
    tasks_logger_->RecordInvocationEnd(task_logger_ts);
  };

  EmbeddingOptions options;
  options.normalize = l2_normalize_;
  options.insert_special_tokens = true;

  ABSL_ASSIGN_OR_RETURN(auto response,
                        engine_->ComputeEmbedding(contents, options),
                        _ << error_message);

  components::containers::EmbeddingResult result;
  components::containers::Embedding embedding;
  embedding.float_embedding = std::move(response.embedding);
  embedding.head_index = 0;
  result.embeddings.push_back(std::move(embedding));

  return result;
}

absl::StatusOr<components::containers::EmbeddingResult>
UniversalEmbedder::EmbedText(absl::string_view text) {
  std::vector<InputData> contents;
  contents.push_back(InputText(std::string(text)));
  return ExecuteInference(contents, "Failed to compute text embedding.");
}

absl::StatusOr<components::containers::EmbeddingResult>
UniversalEmbedder::EmbedImage(absl::string_view image_bytes) {
  std::vector<InputData> contents;
  contents.push_back(InputImage(std::string(image_bytes)));
  return ExecuteInference(contents, "Failed to compute image embedding.");
}

absl::StatusOr<components::containers::EmbeddingResult>
UniversalEmbedder::EmbedImage(const mediapipe::Image& image) {
  ABSL_ASSIGN_OR_RETURN(std::string tga_bytes, EncodeToTga(image));
  return EmbedImage(tga_bytes);
}

absl::StatusOr<components::containers::EmbeddingResult>
UniversalEmbedder::EmbedAudio(const std::vector<float>& audio_data) {
  std::vector<InputData> contents;
  contents.push_back(InputAudio(audio_data));
  return ExecuteInference(contents, "Failed to compute audio embedding.");
}

absl::StatusOr<components::containers::EmbeddingResult>
UniversalEmbedder::EmbedContent(const std::vector<Part>& content) {
  std::vector<InputData> contents;
  for (const auto& part : content) {
    auto status = std::visit(
        [&contents](const auto& arg) -> absl::Status {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, TextPart>) {
            contents.push_back(InputText(arg.text));
          } else if constexpr (std::is_same_v<T, ImagePart>) {
            contents.push_back(InputImage(arg.image_bytes));
          } else if constexpr (std::is_same_v<T, AudioPart>) {
            contents.push_back(InputAudio(arg.audio_data));
          }
          return absl::OkStatus();
        },
        part);
    if (!status.ok()) {
      return status;
    }
  }

  return ExecuteInference(contents, "Failed to compute content embedding.");
}

absl::StatusOr<double> UniversalEmbedder::CosineSimilarity(
    const components::containers::Embedding& u,
    const components::containers::Embedding& v) {
  return components::utils::CosineSimilarity(u, v);
}

std::unique_ptr<core::EmbeddingProvider> UniversalEmbedder::GetProvider() {
  class UniversalEmbedderProvider : public core::EmbeddingProvider {
   public:
    explicit UniversalEmbedderProvider(UniversalEmbedder* embedder)
        : embedder_(embedder) {}
    absl::StatusOr<std::optional<std::vector<float>>> EmbedContent(
        const std::vector<mediapipe::tasks::core::TaskPart>& content) override {
      std::vector<UniversalEmbedder::Part> cpp_parts;
      cpp_parts.reserve(content.size());
      for (const auto& part : content) {
        if (std::holds_alternative<mediapipe::tasks::core::TextPart>(part)) {
          cpp_parts.push_back(UniversalEmbedder::TextPart{
              .text = std::get<mediapipe::tasks::core::TextPart>(part).text});
        } else if (std::holds_alternative<mediapipe::tasks::core::ImagePart>(
                       part)) {
          cpp_parts.push_back(UniversalEmbedder::ImagePart{
              .image_bytes = std::get<mediapipe::tasks::core::ImagePart>(part)
                                 .image_bytes});
        } else if (std::holds_alternative<mediapipe::tasks::core::AudioPart>(
                       part)) {
          cpp_parts.push_back(UniversalEmbedder::AudioPart{
              .audio_data = std::get<mediapipe::tasks::core::AudioPart>(part)
                                .audio_data});
        }
      }
      auto result_status = embedder_->EmbedContent(cpp_parts);
      if (!result_status.ok()) {
        return result_status.status();
      }
      if (result_status->embeddings.empty()) {
        return std::nullopt;
      }
      return result_status->embeddings[0].float_embedding;
    }

   private:
    UniversalEmbedder* embedder_;
  };
  return std::make_unique<UniversalEmbedderProvider>(this);
}

UniversalEmbedder::~UniversalEmbedder() { Close().IgnoreError(); }

absl::Status UniversalEmbedder::Close() {
  if (tasks_logger_) {
    tasks_logger_->LogSessionEnd();
    tasks_logger_.reset();
  }
  return absl::OkStatus();
}

}  // namespace mediapipe::tasks::retrieval::universal_embedder
