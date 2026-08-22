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

#ifndef MEDIAPIPE_TASKS_CC_VISION_IMAGE_EMBEDDER_GRAPH_IMAGE_EMBEDDER_EXECUTOR_H_
#define MEDIAPIPE_TASKS_CC_VISION_IMAGE_EMBEDDER_GRAPH_IMAGE_EMBEDDER_EXECUTOR_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/vision/image_embedder/image_embedder_executor.h"

namespace mediapipe::tasks::vision::image_embedder {

class GraphImageEmbedderExecutor : public ImageEmbedderExecutor {
 public:
  static absl::StatusOr<std::unique_ptr<GraphImageEmbedderExecutor>> Create(
      std::unique_ptr<tasks::core::TaskRunner> runner);

  explicit GraphImageEmbedderExecutor(
      std::unique_ptr<tasks::core::TaskRunner> runner);

  absl::StatusOr<ImageEmbedderResult> Embed(
      mediapipe::Image image, const NormalizedRect& norm_rect) override;

  absl::StatusOr<ImageEmbedderResult> EmbedForVideo(
      mediapipe::Image image, const NormalizedRect& norm_rect,
      int64_t timestamp_ms) override;

  absl::Status EmbedAsync(mediapipe::Image image,
                          const NormalizedRect& norm_rect,
                          int64_t timestamp_ms) override;

  absl::Status Close() override;

 private:
  std::unique_ptr<tasks::core::TaskRunner> runner_;
};

}  // namespace mediapipe::tasks::vision::image_embedder

#endif  // MEDIAPIPE_TASKS_CC_VISION_IMAGE_EMBEDDER_GRAPH_IMAGE_EMBEDDER_EXECUTOR_H_
