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

#ifndef MEDIAPIPE_TASKS_CC_VISION_IMAGE_EMBEDDER_IMAGE_EMBEDDER_EXECUTOR_H_
#define MEDIAPIPE_TASKS_CC_VISION_IMAGE_EMBEDDER_IMAGE_EMBEDDER_EXECUTOR_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"

namespace mediapipe::tasks::vision::image_embedder {

// Alias the shared EmbeddingResult struct as result type.
using ImageEmbedderResult =
    ::mediapipe::tasks::components::containers::EmbeddingResult;

class ImageEmbedderExecutor {
 public:
  virtual ~ImageEmbedderExecutor() = default;

  virtual absl::StatusOr<ImageEmbedderResult> Embed(
      mediapipe::Image image, const NormalizedRect& norm_rect) = 0;

  virtual absl::StatusOr<ImageEmbedderResult> EmbedForVideo(
      mediapipe::Image image, const NormalizedRect& norm_rect,
      int64_t timestamp_ms) = 0;

  virtual absl::Status EmbedAsync(mediapipe::Image image,
                                  const NormalizedRect& norm_rect,
                                  int64_t timestamp_ms) = 0;

  virtual absl::Status Close() = 0;
};

}  // namespace mediapipe::tasks::vision::image_embedder

#endif  // MEDIAPIPE_TASKS_CC_VISION_IMAGE_EMBEDDER_IMAGE_EMBEDDER_EXECUTOR_H_
