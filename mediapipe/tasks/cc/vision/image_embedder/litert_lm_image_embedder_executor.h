/* Copyright 2026 The MediaPipe Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
you may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MEDIAPIPE_TASKS_CC_VISION_IMAGE_EMBEDDER_LITERT_LM_IMAGE_EMBEDDER_EXECUTOR_H_
#define MEDIAPIPE_TASKS_CC_VISION_IMAGE_EMBEDDER_LITERT_LM_IMAGE_EMBEDDER_EXECUTOR_H_

#include <functional>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "mediapipe/tasks/cc/components/processors/embedder_options.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/vision/image_embedder/image_embedder_executor.h"
#include "odml/litert_lm/runtime/engine/embedding_engine.h"  // from @odml
#include "odml/litert_lm/runtime/util/memory_mapped_file.h"  // from @odml
#include "odml/litert_lm/support/util/io_types.h"            // from @odml

namespace mediapipe::tasks::vision::image_embedder {

using ::mediapipe::Image;
using ::mediapipe::tasks::components::processors::EmbedderOptions;
using ::mediapipe::tasks::core::BaseOptions;

class LiteRtLmImageEmbedderExecutor : public ImageEmbedderExecutor {
 public:
  using EmbeddingResult =
      ::mediapipe::tasks::components::containers::EmbeddingResult;

  static absl::StatusOr<std::unique_ptr<LiteRtLmImageEmbedderExecutor>> Create(
      const BaseOptions& base_options, const EmbedderOptions& embedder_options,
      std::function<void(absl::StatusOr<EmbeddingResult>, const Image&,
                         int64_t)>
          result_callback = nullptr);

  LiteRtLmImageEmbedderExecutor(
      std::shared_ptr<::litert::lm::MemoryMappedFile> shared_mmap,
      std::unique_ptr<::litert::lm::EmbeddingEngine> engine, bool l2_normalize,
      bool quantize,
      std::function<void(absl::StatusOr<EmbeddingResult>, const Image&,
                         int64_t)>
          result_callback = nullptr);

  ~LiteRtLmImageEmbedderExecutor() override;

  absl::StatusOr<ImageEmbedderResult> Embed(
      Image image, const NormalizedRect& norm_rect) override;

  absl::StatusOr<ImageEmbedderResult> EmbedForVideo(
      Image image, const NormalizedRect& norm_rect,
      int64_t timestamp_ms) override;

  absl::Status EmbedAsync(Image image, const NormalizedRect& norm_rect,
                          int64_t timestamp_ms) override;

  absl::Status Close() override;

 private:
  absl::StatusOr<::litert::support::InputImage> PreprocessImage(
      const Image& image);

  std::shared_ptr<::litert::lm::MemoryMappedFile> shared_mmap_;
  std::unique_ptr<::litert::lm::EmbeddingEngine> engine_;
  bool l2_normalize_;
  bool quantize_;
  std::function<void(absl::StatusOr<EmbeddingResult>, const Image&, int64_t)>
      result_callback_;
};

}  // namespace mediapipe::tasks::vision::image_embedder

#endif  // MEDIAPIPE_TASKS_CC_VISION_IMAGE_EMBEDDER_LITERT_LM_IMAGE_EMBEDDER_EXECUTOR_H_
