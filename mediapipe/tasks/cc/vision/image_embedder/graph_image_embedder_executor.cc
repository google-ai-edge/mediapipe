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

#include "mediapipe/tasks/cc/vision/image_embedder/graph_image_embedder_executor.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "mediapipe/tasks/cc/components/containers/proto/embeddings.pb.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/vision/image_embedder/image_embedder_executor.h"

namespace mediapipe::tasks::vision::image_embedder {
namespace {

constexpr char kEmbeddingsStreamName[] = "embeddings_out";
constexpr char kImageInStreamName[] = "image_in";
constexpr char kNormRectStreamName[] = "norm_rect_in";
constexpr int kMicroSecondsPerMilliSecond = 1000;

using ::mediapipe::NormalizedRect;
using ::mediapipe::tasks::components::containers::ConvertToEmbeddingResult;
using ::mediapipe::tasks::components::containers::proto::EmbeddingResult;

}  // namespace

GraphImageEmbedderExecutor::GraphImageEmbedderExecutor(
    std::unique_ptr<tasks::core::TaskRunner> runner)
    : runner_(std::move(runner)) {}

absl::StatusOr<ImageEmbedderResult> GraphImageEmbedderExecutor::Embed(
    mediapipe::Image image, const NormalizedRect& norm_rect) {
  ABSL_ASSIGN_OR_RETURN(
      auto output_packets,
      runner_->Process(
          {{kImageInStreamName, MakePacket<Image>(std::move(image))},
           {kNormRectStreamName, MakePacket<NormalizedRect>(norm_rect)}}));
  return ConvertToEmbeddingResult(
      output_packets[kEmbeddingsStreamName].Get<EmbeddingResult>());
}

absl::StatusOr<ImageEmbedderResult> GraphImageEmbedderExecutor::EmbedForVideo(
    mediapipe::Image image, const NormalizedRect& norm_rect,
    int64_t timestamp_ms) {
  ABSL_ASSIGN_OR_RETURN(
      auto output_packets,
      runner_->Process(
          {{kImageInStreamName,
            MakePacket<Image>(std::move(image))
                .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))},
           {kNormRectStreamName,
            MakePacket<NormalizedRect>(norm_rect).At(
                Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))}}));
  return ConvertToEmbeddingResult(
      output_packets[kEmbeddingsStreamName].Get<EmbeddingResult>());
}

absl::Status GraphImageEmbedderExecutor::EmbedAsync(
    mediapipe::Image image, const NormalizedRect& norm_rect,
    int64_t timestamp_ms) {
  return runner_->Send(
      {{kImageInStreamName,
        MakePacket<Image>(std::move(image))
            .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))},
       {kNormRectStreamName,
        MakePacket<NormalizedRect>(norm_rect).At(
            Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))}});
}

absl::Status GraphImageEmbedderExecutor::Close() { return runner_->Close(); }

absl::StatusOr<std::unique_ptr<GraphImageEmbedderExecutor>>
GraphImageEmbedderExecutor::Create(
    std::unique_ptr<tasks::core::TaskRunner> runner) {
  return std::make_unique<GraphImageEmbedderExecutor>(std::move(runner));
}

}  // namespace mediapipe::tasks::vision::image_embedder
