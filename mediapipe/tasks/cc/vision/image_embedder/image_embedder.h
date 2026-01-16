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

#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "mediapipe/tasks/cc/components/processors/embedder_options.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/vision/core/base_vision_task_api.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace image_embedder {

// Alias the shared EmbeddingResult struct as result typo.
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
//
// The API expects a TFLite model with optional, but strongly recommended,
// TFLite Model Metadata.
//
// Input tensor:
//   (kTfLiteUInt8/kTfLiteFloat32)
//    - image input of size `[batch x height x width x channels]`.
//    - batch inference is not supported (`batch` is required to be 1).
//    - only RGB inputs are supported (`channels` is required to be 3).
//    - if type is kTfLiteFloat32, NormalizationOptions are required to be
//      attached to the metadata for input normalization.
// At least one output tensor with:
//   (kTfLiteUInt8/kTfLiteFloat32)
//    - `N` components corresponding to the `N` dimensions of the returned
//      feature vector for this output layer.
//    - Either 2 or 4 dimensions, i.e. `[1 x N]` or `[1 x 1 x 1 x N]`.
class ImageEmbedder : core::BaseVisionTaskApi {
 public:
  using BaseVisionTaskApi::BaseVisionTaskApi;

  // Creates an ImageEmbedder from the provided options. A non-default
  // OpResolver can be specified in the BaseOptions in order to support custom
  // Ops or specify a subset of built-in Ops.
  static absl::StatusOr<std::unique_ptr<ImageEmbedder>> Create(
      std::unique_ptr<ImageEmbedderOptions> options);

  // Performs embedding extraction on the provided single image.
  //
  // The optional 'image_processing_options' parameter can be used to specify:
  // - the rotation to apply to the image before performing embedding
  //   extraction, by setting its 'rotation_degrees' field.
  // and/or
  // - the region-of-interest on which to perform embedding extraction, by
  //   setting its 'region_of_interest' field. If not specified, the full image
  //   is used.
  // If both are specified, the crop around the region-of-interest is extracted
  // first, then the specified rotation is applied to the crop.
  //
  // Only use this method when the ImageEmbedder is created with the image
  // running mode.
  //
  // The image can be of any size with format RGB or RGBA.
  absl::StatusOr<ImageEmbedderResult> Embed(
      mediapipe::Image image,
      std::optional<core::ImageProcessingOptions> image_processing_options =
          std::nullopt);

  // Performs embedding extraction on the provided video frame.
  //
  // The optional 'image_processing_options' parameter can be used to specify:
  // - the rotation to apply to the image before performing embedding
  //   extraction, by setting its 'rotation_degrees' field.
  // and/or
  // - the region-of-interest on which to perform embedding extraction, by
  //   setting its 'region_of_interest' field. If not specified, the full image
  //   is used.
  // If both are specified, the crop around the region-of-interest is extracted
  // first, then the specified rotation is applied to the crop.
  //
  // Only use this method when the ImageEmbedder is created with the video
  // running mode.
  //
  // The image can be of any size with format RGB or RGBA. It's required to
  // provide the video frame's timestamp (in milliseconds). The input timestamps
  // must be monotonically increasing.
  absl::StatusOr<ImageEmbedderResult> EmbedForVideo(
      mediapipe::Image image, int64_t timestamp_ms,
      std::optional<core::ImageProcessingOptions> image_processing_options =
          std::nullopt);

  // Sends live image data to embedder, and the results will be available via
  // the "result_callback" provided in the ImageEmbedderOptions.
  //
  // The optional 'image_processing_options' parameter can be used to specify:
  // - the rotation to apply to the image before performing embedding
  //   extraction, by setting its 'rotation_degrees' field.
  // and/or
  // - the region-of-interest on which to perform embedding extraction, by
  //   setting its 'region_of_interest' field. If not specified, the full image
  //   is used.
  // If both are specified, the crop around the region-of-interest is extracted
  // first, then the specified rotation is applied to the crop.
  //
  // Only use this method when the ImageEmbedder is created with the live
  // stream running mode.
  //
  // The image can be of any size with format RGB or RGBA. It's required to
  // provide a timestamp (in milliseconds) to indicate when the input image is
  // sent to the object detector. The input timestamps must be monotonically
  // increasing.
  //
  // The "result_callback" prvoides
  //   - The embedding results as a
  //     components::containers::proto::EmbeddingResult object.
  //   - The const reference to the corresponding input image that the image
  //     embedder runs on. Note that the const reference to the image will no
  //     longer be valid when the callback returns. To access the image data
  //     outside of the callback, callers need to make a copy of the image.
  //   - The input timestamp in milliseconds.
  absl::Status EmbedAsync(mediapipe::Image image, int64_t timestamp_ms,
                          std::optional<core::ImageProcessingOptions>
                              image_processing_options = std::nullopt);

  // Shuts down the ImageEmbedder when all works are done.
  absl::Status Close() { return runner_->Close(); }

  // Utility function to compute cosine similarity [1] between two embeddings.
  // May return an InvalidArgumentError if e.g. the embeddings are of different
  // types (quantized vs. float), have different sizes, or have a an L2-norm of
  // 0.
  //
  // [1]: https://en.wikipedia.org/wiki/Cosine_similarity
  static absl::StatusOr<double> CosineSimilarity(
      const components::containers::Embedding& u,
      const components::containers::Embedding& v);
};

}  // namespace image_embedder
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_IMAGE_EMBEDDER_IMAGE_EMBEDDER_H_
