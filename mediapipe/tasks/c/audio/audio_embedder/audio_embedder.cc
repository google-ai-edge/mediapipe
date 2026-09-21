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

#include "mediapipe/tasks/c/audio/audio_embedder/audio_embedder.h"

#include <memory>
#include <utility>

#include "Eigen/Core"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/c/audio/core/common.h"
#include "mediapipe/tasks/c/components/containers/embedding_result.h"
#include "mediapipe/tasks/c/components/containers/embedding_result_converter.h"
#include "mediapipe/tasks/c/components/processors/embedder_options_converter.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"
#include "mediapipe/tasks/cc/audio/audio_embedder/audio_embedder.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "mediapipe/tasks/cc/components/utils/cosine_similarity.h"

using ::mediapipe::tasks::audio::audio_embedder::AudioEmbedder;

// C API wrapper for the MediaPipe AudioEmbedder.
struct MpAudioEmbedderInternal {
  std::unique_ptr<AudioEmbedder> instance;
};

namespace mediapipe::tasks::c::audio::audio_embedder {

namespace {

using ::mediapipe::tasks::audio::audio_embedder::AudioEmbedder;
using ::mediapipe::tasks::audio::audio_embedder::AudioEmbedderOptions;
using ::mediapipe::tasks::c::components::containers::CppCloseEmbeddingResult;
using ::mediapipe::tasks::c::components::containers::CppConvertToCppEmbedding;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToEmbeddingResult;
using ::mediapipe::tasks::c::components::processors::
    CppConvertToEmbedderOptions;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;

typedef ::mediapipe::tasks::components::containers::Embedding CppEmbedding;

AudioEmbedder* GetCppEmbedder(MpAudioEmbedderPtr wrapper) {
  ABSL_CHECK(wrapper != nullptr) << "AudioEmbedder is null.";
  return wrapper->instance.get();
}

mediapipe::Matrix CppConvertToMatrix(const MpAudioData* audio_data) {
  int num_channels = audio_data->num_channels;
  int num_samples_per_channel = audio_data->audio_data_size / num_channels;
  // Convert the buffer to a row-major matrix where rows represent samples and
  // columns represent channels and then transpose to a mediapipe::Matrix with
  // channels as rows and samples as columns.
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>>
      interleaved_map(audio_data->audio_data, num_samples_per_channel,
                      num_channels);
  return interleaved_map.transpose();
}

absl::StatusOr<std::unique_ptr<AudioEmbedder>> CppCreateAudioEmbedder(
    const MpAudioEmbedderOptions& options) {
  auto cpp_options = std::make_unique<AudioEmbedderOptions>();
  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  CppConvertToEmbedderOptions(options.embedder_options,
                              &cpp_options->embedder_options);

  return AudioEmbedder::Create(std::move(cpp_options));
}

}  // namespace

absl::Status MpAudioEmbedderCreate(struct MpAudioEmbedderOptions* options,
                                   MpAudioEmbedderPtr* embedder_out) {
  ABSL_ASSIGN_OR_RETURN(auto embedder, CppCreateAudioEmbedder(*options));
  *embedder_out = new MpAudioEmbedderInternal{std::move(embedder)};
  return absl::OkStatus();
}

absl::Status MpAudioEmbedderEmbed(MpAudioEmbedderPtr embedder,
                                  const MpAudioData* audio_data,
                                  MpAudioEmbedderResult* result_out) {
  auto* cpp_embedder = GetCppEmbedder(embedder);
  mediapipe::Matrix audio_matrix = CppConvertToMatrix(audio_data);
  ABSL_ASSIGN_OR_RETURN(
      auto cpp_result,
      cpp_embedder->Embed(audio_matrix, audio_data->sample_rate));

  if (cpp_result.empty()) {
    result_out->results = nullptr;
    result_out->results_count = 0;
    return absl::OkStatus();
  }

  auto c_embeddings = std::make_unique<MpEmbeddingResult[]>(cpp_result.size());
  result_out->results_count = cpp_result.size();
  for (int i = 0; i < result_out->results_count; ++i) {
    CppConvertToEmbeddingResult(cpp_result.at(i), &(c_embeddings.get()[i]));
  }
  result_out->results = c_embeddings.release();
  return absl::OkStatus();
}

void MpAudioEmbedderCloseResult(MpAudioEmbedderResult* result) {
  if (result->results) {
    for (int i = 0; i < result->results_count; ++i) {
      CppCloseEmbeddingResult(&result->results[i]);
    }
    delete[] result->results;
  }
  result->results = nullptr;
  result->results_count = 0;
}

absl::Status MpAudioEmbedderClose(MpAudioEmbedderPtr embedder) {
  auto* cpp_embedder = GetCppEmbedder(embedder);
  ABSL_RETURN_IF_ERROR(cpp_embedder->Close());
  delete embedder;
  return absl::OkStatus();
}

absl::Status MpAudioEmbedderCosineSimilarity(const MpEmbedding* u,
                                             const MpEmbedding* v,
                                             double* similarity) {
  CppEmbedding cpp_u;
  CppConvertToCppEmbedding(*u, &cpp_u);
  CppEmbedding cpp_v;
  CppConvertToCppEmbedding(*v, &cpp_v);
  ABSL_ASSIGN_OR_RETURN(
      *similarity,
      mediapipe::tasks::components::utils::CosineSimilarity(cpp_u, cpp_v));
  return absl::OkStatus();
}

}  // namespace mediapipe::tasks::c::audio::audio_embedder

extern "C" {

MP_EXPORT MpStatus MpAudioEmbedderCreate(struct MpAudioEmbedderOptions* options,
                                         MpAudioEmbedderPtr* embedder_out,
                                         char** error_msg) {
  absl::Status status =
      mediapipe::tasks::c::audio::audio_embedder::MpAudioEmbedderCreate(
          options, embedder_out);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpAudioEmbedderEmbed(MpAudioEmbedderPtr embedder,
                                        const MpAudioData* audio_data,
                                        MpAudioEmbedderResult* result_out,
                                        char** error_msg) {
  absl::Status status =
      mediapipe::tasks::c::audio::audio_embedder::MpAudioEmbedderEmbed(
          embedder, audio_data, result_out);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT void MpAudioEmbedderCloseResult(MpAudioEmbedderResult* result) {
  mediapipe::tasks::c::audio::audio_embedder::MpAudioEmbedderCloseResult(
      result);
}

MP_EXPORT MpStatus MpAudioEmbedderClose(MpAudioEmbedderPtr embedder,
                                        char** error_msg) {
  absl::Status status =
      mediapipe::tasks::c::audio::audio_embedder::MpAudioEmbedderClose(
          embedder);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpAudioEmbedderCosineSimilarity(const struct MpEmbedding* u,
                                                   const struct MpEmbedding* v,
                                                   double* similarity,
                                                   char** error_msg) {
  absl::Status status = mediapipe::tasks::c::audio::audio_embedder::
      MpAudioEmbedderCosineSimilarity(u, v, similarity);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

}  // extern "C"
