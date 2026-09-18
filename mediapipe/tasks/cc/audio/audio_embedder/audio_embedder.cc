// Copyright 2026 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/tasks/cc/audio/audio_embedder/audio_embedder.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "mediapipe/tasks/cc/components/processors/embedder_options.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/core/logging/factory/logging_factory.h"
#include "mediapipe/tasks/cc/core/running_mode.h"
#include "runtime/core/embedding_engine_impl.h"        // from @litert_lm
#include "runtime/engine/embedding_engine.h"           // from @litert_lm
#include "runtime/engine/embedding_engine_settings.h"  // from @litert_lm
#include "runtime/engine/io_types.h"                   // from @litert_lm
#include "runtime/executor/embedding/embedding_executor_base.h"  // from @litert_lm
#include "runtime/executor/executor_settings_base.h"  // from @litert_lm
#include "support/util/io_types.h"                    // from @litert_lm
#include "support/util/memory_mapped_file.h"          // from @litert_lm

#ifdef _WIN32
#include <io.h>
#endif

namespace mediapipe::tasks::audio::audio_embedder {
namespace {

using ::litert::lm::EmbeddingEngineImpl;
using ::litert::lm::EmbeddingEngineSettings;
using ::litert::lm::EmbeddingOptions;
using ::litert::lm::InputAudio;
using ::litert::lm::InputData;
using ::litert::lm::ModelAssets;
using ::litert::support::InMemoryFile;
using ::litert::support::MemoryMappedFile;
using ::mediapipe::tasks::components::processors::EmbedderOptions;
using ::mediapipe::tasks::core::BaseOptions;

}  // namespace

absl::StatusOr<::litert::support::InputAudio> PreprocessAudio(
    const Matrix& audio_clip, double audio_sample_rate) {
  int num_channels = audio_clip.rows();
  int num_samples = audio_clip.cols();
  std::vector<float> audio_data;
  audio_data.reserve(num_channels * num_samples);
  for (int j = 0; j < num_samples; ++j) {
    for (int i = 0; i < num_channels; ++i) {
      audio_data.push_back(audio_clip(i, j));
    }
  }
  return ::litert::support::InputAudio(std::move(audio_data));
}

AudioEmbedder::AudioEmbedder() = default;
AudioEmbedder::~AudioEmbedder() = default;

absl::StatusOr<std::unique_ptr<AudioEmbedder>> AudioEmbedder::Create(
    std::unique_ptr<AudioEmbedderOptions> options) {
  const BaseOptions& base_options = options->base_options;
  const EmbedderOptions& embedder_options = options->embedder_options;
  std::shared_ptr<MemoryMappedFile> shared_mmap;
  if (!base_options.model_asset_path.empty()) {
    ABSL_ASSIGN_OR_RETURN(auto mmap_file, MemoryMappedFile::Create(
                                              base_options.model_asset_path));
    shared_mmap = std::shared_ptr<MemoryMappedFile>(std::move(mmap_file));
  } else if (base_options.model_asset_buffer != nullptr) {
    ABSL_ASSIGN_OR_RETURN(
        auto in_mem_file,
        InMemoryFile::Create(*base_options.model_asset_buffer));
    shared_mmap = std::shared_ptr<MemoryMappedFile>(std::move(in_mem_file));
  } else if (base_options.model_asset_descriptor_meta.fd != -1) {
    uint64_t offset = base_options.model_asset_descriptor_meta.offset >= 0
                          ? base_options.model_asset_descriptor_meta.offset
                          : 0u;
    uint64_t length = base_options.model_asset_descriptor_meta.length >= 0
                          ? base_options.model_asset_descriptor_meta.length
                          : 0u;
#ifdef _WIN32
    HANDLE platform_file = reinterpret_cast<HANDLE>(
        _get_osfhandle(base_options.model_asset_descriptor_meta.fd));
#else
    int platform_file = base_options.model_asset_descriptor_meta.fd;
#endif
    ABSL_ASSIGN_OR_RETURN(auto mmap_file, MemoryMappedFile::Create(
                                              platform_file, offset, length));
    shared_mmap = std::shared_ptr<MemoryMappedFile>(std::move(mmap_file));
  } else {
    return absl::FailedPreconditionError("No model asset specified.");
  }

  ::litert::lm::Backend backend = ::litert::lm::Backend::CPU;
  if (base_options.delegate == BaseOptions::GPU) {
    backend = ::litert::lm::Backend::GPU;
  }

  ABSL_ASSIGN_OR_RETURN(
      auto model_assets,
      ModelAssets::Create(shared_mmap, base_options.model_asset_path));
  ABSL_ASSIGN_OR_RETURN(
      auto settings, EmbeddingEngineSettings::CreateDefault(
                         std::move(model_assets), backend, backend, backend));

  ABSL_ASSIGN_OR_RETURN(auto engine,
                        EmbeddingEngineImpl::Create(std::move(settings)));

  auto tasks_logger = tasks::core::logging::CreateTasksLogger(
      {.task_name = "AudioEmbedder",
       .task_running_mode = tasks::core::RunningMode::kUnspecified,
       .host_environment = base_options.host_environment,
       .host_system = base_options.host_system,
       .host_version = base_options.host_version,
       .app_id = base_options.app_id,
       .app_version = base_options.app_version,
       .ca_bundle_path = base_options.ca_bundle_path});
  tasks_logger->LogSessionStart();

  auto embedder = std::make_unique<AudioEmbedder>();
  embedder->shared_mmap_ = std::move(shared_mmap);
  embedder->engine_ = std::move(engine);
  embedder->l2_normalize_ = embedder_options.l2_normalize;
  embedder->quantize_ = embedder_options.quantize;
  embedder->tasks_logger_ = std::move(tasks_logger);
  return embedder;
}

absl::StatusOr<std::vector<AudioEmbedderResult>> AudioEmbedder::Embed(
    Matrix audio_clip, double audio_sample_rate) {
  mediapipe::Timestamp task_logger_ts(tasks_logger_timestamp_++);
  tasks_logger_->RecordCpuInputArrival(task_logger_ts);

  std::vector<InputData> contents;
  ABSL_ASSIGN_OR_RETURN(auto input_audio,
                        PreprocessAudio(audio_clip, audio_sample_rate));
  contents.push_back(std::move(input_audio));

  EmbeddingOptions options;
  options.normalize = l2_normalize_;
  options.input_overflow_strategy =
      ::litert::lm::InputOverflowStrategy::kTruncate;

  ABSL_ASSIGN_OR_RETURN(auto response,
                        engine_->ComputeEmbedding(contents, options),
                        _ << "Failed to compute embeddings.");

  AudioEmbedderResult result;
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

  tasks_logger_->RecordInvocationEnd(task_logger_ts);
  return std::vector<AudioEmbedderResult>{result};
}

absl::Status AudioEmbedder::Close() {
  if (tasks_logger_) {
    tasks_logger_->LogSessionEnd();
    tasks_logger_.reset();
  }
  return absl::OkStatus();
}

}  // namespace mediapipe::tasks::audio::audio_embedder
