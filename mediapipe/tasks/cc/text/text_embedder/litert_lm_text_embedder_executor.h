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

#ifndef MEDIAPIPE_TASKS_CC_TEXT_TEXT_EMBEDDER_LITERT_LM_TEXT_EMBEDDER_EXECUTOR_H_
#define MEDIAPIPE_TASKS_CC_TEXT_TEXT_EMBEDDER_LITERT_LM_TEXT_EMBEDDER_EXECUTOR_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/tasks/cc/components/processors/embedder_options.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/text/text_embedder/text_embedder_executor.h"
#include "odml/litert_lm/runtime/engine/embedding_engine.h"  // from @odml
#include "odml/litert_lm/runtime/util/memory_mapped_file.h"  // from @odml

namespace mediapipe::tasks::text::text_embedder {

// A TextEmbedderExecutor that uses the LiteRT LM embedding engine to perform
// embedding.
class LiteRtLmTextEmbedderExecutor : public TextEmbedderExecutor {
 public:
  static absl::StatusOr<std::unique_ptr<LiteRtLmTextEmbedderExecutor>> Create(
      const tasks::core::BaseOptions& base_options,
      const components::processors::EmbedderOptions& embedder_options);

  LiteRtLmTextEmbedderExecutor(
      std::shared_ptr<::litert::lm::MemoryMappedFile> shared_mmap,
      std::unique_ptr<::litert::lm::EmbeddingEngine> engine, bool l2_normalize,
      bool quantize);

  absl::StatusOr<TextEmbedderResult> Embed(absl::string_view text) override;
  absl::Status Close() override;

 private:
  std::shared_ptr<::litert::lm::MemoryMappedFile> shared_mmap_;
  std::unique_ptr<::litert::lm::EmbeddingEngine> engine_;
  bool l2_normalize_;
  bool quantize_;
};

}  // namespace mediapipe::tasks::text::text_embedder

#endif  // MEDIAPIPE_TASKS_CC_TEXT_TEXT_EMBEDDER_LITERT_LM_TEXT_EMBEDDER_EXECUTOR_H_
