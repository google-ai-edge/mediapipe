/* Copyright 2022 The MediaPipe Authors.

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

#ifndef MEDIAPIPE_TASKS_CC_VISION_IMAGE_EMBEDDER_IMAGE_EMBEDDER_H_
#define MEDIAPIPE_TASKS_CC_VISION_IMAGE_EMBEDDER_IMAGE_EMBEDDER_H_

#include <functional>
#include <memory>
#include <optional>

#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "mediapipe/tasks/cc/components/processors/embedder_options.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/vision/core/base_vision_task_api.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/image_embedder/image_embedder_executor.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace image_embedder {

// Alias the shared EmbeddingResult struct as result type.
using ImageEmbedderResult =
    ::mediapipe::tasks::components::containers::EmbeddingResult;

// The options for configuring a MediaPipe image embedder task.
struct ImageEmbedderOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  tasks::core::BaseOptions base_options;

  // The running mode of the task. Default to the image mode.
  // Image embedder has three running modes:
  // 1) The image mode for embedding image on single image inputs.
  // 2) The video mode for embedding image on the decoded frames of a video.
  // 3) The live stream mode for embedding image on the live stream of input
  // data, such as from camera. In this mode, the "result_callback" below must
  // be specified to receive the embedding results asynchronously.
  core::RunningMode running_mode = core::RunningMode::IMAGE;

  // Options for configuring the embedder behavior, such as L2-normalization or
  // scalar-quantization.
  components::processors::EmbedderOptions embedder_options;

  // The user-defined result callback for processing live stream data.
  // The result callback should only be specified when the running mode is set
  // to RunningMode::LIVE_STREAM.
  std::function<void(absl::StatusOr<ImageEmbedderResult>, const Image&,
                     int64_t)>
      result_callback = nullptr;
};

// Performs embedding extraction on images.
class ImageEmbedder : core::BaseVisionTaskApi {
 public:
  using BaseVisionTaskApi::BaseVisionTaskApi;

  // Creates an ImageEmbedder from the provided options. A non-default
  // OpResolver can be specified in the BaseOptions in order to support custom
  // Ops or specify a subset of built-in Ops.
  static absl::StatusOr<std::unique_ptr<ImageEmbedder>> Create(
      std::unique_ptr<ImageEmbedderOptions> options);

  ~ImageEmbedder() override;

  // Performs embedding extraction on the provided single image.
  absl::StatusOr<ImageEmbedderResult> Embed(
      mediapipe::Image image,
      std::optional<core::ImageProcessingOptions> image_processing_options =
          std::nullopt);

  // Performs embedding extraction on the provided video frame.
  absl::StatusOr<ImageEmbedderResult> EmbedForVideo(
      mediapipe::Image image, int64_t timestamp_ms,
      std::optional<core::ImageProcessingOptions> image_processing_options =
          std::nullopt);

  // Sends live image data to embedder, and the results will be available via
  // the "result_callback" provided in the ImageEmbedderOptions.
  absl::Status EmbedAsync(mediapipe::Image image, int64_t timestamp_ms,
                          std::optional<core::ImageProcessingOptions>
                              image_processing_options = std::nullopt);

  // Shuts down the ImageEmbedder when all works are done.
  absl::Status Close();

  // Utility function to compute cosine similarity [1] between two embeddings.
  static absl::StatusOr<double> CosineSimilarity(
      const components::containers::Embedding& u,
      const components::containers::Embedding& v);

 private:
  std::unique_ptr<ImageEmbedderExecutor> executor_;
  core::RunningMode running_mode_;
};

}  // namespace image_embedder
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_IMAGE_EMBEDDER_IMAGE_EMBEDDER_H_
