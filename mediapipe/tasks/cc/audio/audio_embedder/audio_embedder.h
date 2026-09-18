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

#ifndef MEDIAPIPE_TASKS_CC_AUDIO_AUDIO_EMBEDDER_AUDIO_EMBEDDER_H_
#define MEDIAPIPE_TASKS_CC_AUDIO_AUDIO_EMBEDDER_AUDIO_EMBEDDER_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "mediapipe/tasks/cc/components/processors/embedder_options.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/core/logging/tasks_logger.h"
#include "runtime/engine/embedding_engine.h"          // from @litert_lm
#include "runtime/executor/executor_settings_base.h"  // from @litert_lm

namespace mediapipe::tasks::audio::audio_embedder {

// Alias the shared EmbeddingResult struct as result type.
using AudioEmbedderResult =
    ::mediapipe::tasks::components::containers::EmbeddingResult;

struct AudioEmbedderOptions {
  // Base options for configuring the underlying Task library, such as
  // specifying the LiteRT-LM model bundle and accelerator bindings.
  tasks::core::BaseOptions base_options;

  // Options for configuring the embedder behavior, such as L2 normalization
  // and scalar quantization.
  components::processors::EmbedderOptions embedder_options;
};

// Performs audio embedding extraction on audio clips or audio stream.
class AudioEmbedder {
 public:
  AudioEmbedder();
  ~AudioEmbedder();

  // Creates an AudioEmbedder from the provided options.
  static absl::StatusOr<std::unique_ptr<AudioEmbedder>> Create(
      std::unique_ptr<AudioEmbedderOptions> options);

  // Performs embedding extraction on the provided audio clip.
  //
  // The audio clip is represented as a MediaPipe Matrix that has the number of
  // channels rows and the number of samples per channel columns. The method
  // accepts audio clips with various length and audio sample rate. It's
  // required to provide the corresponding audio sample rate along with the
  // input audio clips.
  //
  // The input audio clip may be longer than what the model is able to process
  // in a single inference. When this occurs, the input audio clip is split into
  // multiple chunks starting at different timestamps. For this reason, this
  // function returns a vector of EmbeddingResult objects, each associated
  // with a timestamp corresponding to the start (in milliseconds) of the chunk
  // data that was extracted.
  absl::StatusOr<std::vector<AudioEmbedderResult>> Embed(
      Matrix audio_clip, double audio_sample_rate);

  // Shuts down the AudioEmbedder when all works are done.
  absl::Status Close();

 private:
  std::shared_ptr<::litert::lm::MemoryMappedFile> shared_mmap_;
  std::unique_ptr<::litert::lm::EmbeddingEngine> engine_;
  bool l2_normalize_ = false;
  bool quantize_ = false;

  std::unique_ptr<tasks::core::logging::TasksLogger> tasks_logger_;
  int64_t tasks_logger_timestamp_ = 0;
};

}  // namespace mediapipe::tasks::audio::audio_embedder

#endif  // MEDIAPIPE_TASKS_CC_AUDIO_AUDIO_EMBEDDER_AUDIO_EMBEDDER_H_
