/* Copyright 2023 The MediaPipe Authors.

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

#ifndef MEDIAPIPE_TASKS_CC_VISION_FACE_STYLIZER_FACE_STYLIZER_H_
#define MEDIAPIPE_TASKS_CC_VISION_FACE_STYLIZER_FACE_STYLIZER_H_

#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/vision/core/base_vision_task_api.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "tensorflow/lite/kernels/register.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace face_stylizer {

// The options for configuring a mediapipe face stylizer task.
struct FaceStylizerOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  tasks::core::BaseOptions base_options;

  // The user-defined result callback for processing live stream data.
  // The result callback should only be specified when the running mode is set
  // to RunningMode::LIVE_STREAM.
  std::function<void(absl::StatusOr<std::optional<mediapipe::Image>>,
                     const Image&, int64_t)>
      result_callback = nullptr;
};

// Performs face stylization on images.
class FaceStylizer : tasks::vision::core::BaseVisionTaskApi {
 public:
  using BaseVisionTaskApi::BaseVisionTaskApi;

  // Creates a FaceStylizer from the provided options.
  static absl::StatusOr<std::unique_ptr<FaceStylizer>> Create(
      std::unique_ptr<FaceStylizerOptions> options);

  // Performs face stylization on the provided single image.
  //
  // The optional 'image_processing_options' parameter can be used to specify:
  //   - the rotation to apply to the image before performing stylization, by
  //     setting its 'rotation_degrees' field.
  //   and/or
  //   - the region-of-interest on which to perform stylization, by setting its
  //   'region_of_interest' field. If not specified, the full image is used.
  // If both are specified, the crop around the region-of-interest is extracted
  // first, then the specified rotation is applied to the crop.
  //
  // Only use this method when the FaceStylizer is created with the image
  // running mode.
  //
  // The input image can be of any size with format RGB or RGBA.
  // When no face is detected on the input image, the method returns a
  // std::nullopt. Otherwise, returns the stylized image of the most visible
  // face. The stylized output image size is the same as the model output size.
  absl::StatusOr<std::optional<mediapipe::Image>> Stylize(
      mediapipe::Image image,
      std::optional<core::ImageProcessingOptions> image_processing_options =
          std::nullopt);

  // Shuts down the FaceStylizer when all works are done.
  absl::Status Close() { return runner_->Close(); }
};

}  // namespace face_stylizer
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_FACE_STYLIZER_FACE_STYLIZER_H_
