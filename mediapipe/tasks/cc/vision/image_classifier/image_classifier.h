/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

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

#ifndef MEDIAPIPE_TASKS_CC_VISION_IMAGE_CLASSIFIER_IMAGE_CLASSIFIER_H_
#define MEDIAPIPE_TASKS_CC_VISION_IMAGE_CLASSIFIER_IMAGE_CLASSIFIER_H_

#include <functional>
#include <memory>
#include <optional>

#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/components/containers/proto/classifications.pb.h"
#include "mediapipe/tasks/cc/components/processors/classifier_options.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/vision/core/base_vision_task_api.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace image_classifier {

// The options for configuring a Mediapipe image classifier task.
struct ImageClassifierOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  tasks::core::BaseOptions base_options;

  // The running mode of the task. Default to the image mode.
  // Image classifier has three running modes:
  // 1) The image mode for classifying image on single image inputs.
  // 2) The video mode for classifying image on the decoded frames of a video.
  // 3) The live stream mode for classifying image on the live stream of input
  // data, such as from camera. In this mode, the "result_callback" below must
  // be specified to receive the segmentation results asynchronously.
  core::RunningMode running_mode = core::RunningMode::IMAGE;

  // Options for configuring the classifier behavior, such as score threshold,
  // number of results, etc.
  components::processors::ClassifierOptions classifier_options;

  // The user-defined result callback for processing live stream data.
  // The result callback should only be specified when the running mode is set
  // to RunningMode::LIVE_STREAM.
  std::function<void(
      absl::StatusOr<components::containers::proto::ClassificationResult>,
      const Image&, int64)>
      result_callback = nullptr;
};

// Performs classification on images.
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
//    -  `N `classes and either 2 or 4 dimensions, i.e. `[1 x N]` or
//       `[1 x 1 x 1 x N]`
//    - optional (but recommended) label map(s) as AssociatedFile-s with type
//      TENSOR_AXIS_LABELS, containing one label per line. The first such
//      AssociatedFile (if any) is used to fill the `class_name` field of the
//      results. The `display_name` field is filled from the AssociatedFile (if
//      any) whose locale matches the `display_names_locale` field of the
//      `ImageClassifierOptions` used at creation time ("en" by default, i.e.
//      English). If none of these are available, only the `index` field of the
//      results will be filled.
//    - optional score calibration can be attached using ScoreCalibrationOptions
//      and an AssociatedFile with type TENSOR_AXIS_SCORE_CALIBRATION. See
//      metadata_schema.fbs [1] for more details.
//
// An example of such model can be found at:
// https://tfhub.dev/bohemian-visual-recognition-alliance/lite-model/models/mushroom-identification_v1/1
//
// [1]:
// https://github.com/google/mediapipe/blob/6cdc6443b6a7ed662744e2a2ce2d58d9c83e6d6f/mediapipe/tasks/metadata/metadata_schema.fbs#L456
class ImageClassifier : tasks::vision::core::BaseVisionTaskApi {
 public:
  using BaseVisionTaskApi::BaseVisionTaskApi;

  // Creates an ImageClassifier from the provided options. A non-default
  // OpResolver can be specified in the BaseOptions in order to support custom
  // Ops or specify a subset of built-in Ops.
  static absl::StatusOr<std::unique_ptr<ImageClassifier>> Create(
      std::unique_ptr<ImageClassifierOptions> options);

  // Performs image classification on the provided single image. Classification
  // is performed on the region of interest specified by the `roi` argument if
  // provided, or on the entire image otherwise.
  //
  // Only use this method when the ImageClassifier is created with the image
  // running mode.
  //
  // The image can be of any size with format RGB or RGBA.
  // TODO: describe exact preprocessing steps once
  // YUVToImageCalculator is integrated.
  absl::StatusOr<components::containers::proto::ClassificationResult> Classify(
      mediapipe::Image image,
      std::optional<mediapipe::NormalizedRect> roi = std::nullopt);

  // Performs image classification on the provided video frame. Classification
  // is performed on the region of interested specified by the `roi` argument if
  // provided, or on the entire image otherwise.
  //
  // Only use this method when the ImageClassifier is created with the video
  // running mode.
  //
  // The image can be of any size with format RGB or RGBA. It's required to
  // provide the video frame's timestamp (in milliseconds). The input timestamps
  // must be monotonically increasing.
  absl::StatusOr<components::containers::proto::ClassificationResult>
  ClassifyForVideo(mediapipe::Image image, int64 timestamp_ms,
                   std::optional<mediapipe::NormalizedRect> roi = std::nullopt);

  // Sends live image data to image classification, and the results will be
  // available via the "result_callback" provided in the ImageClassifierOptions.
  // Classification is performed on the region of interested specified by the
  // `roi` argument if provided, or on the entire image otherwise.
  //
  // Only use this method when the ImageClassifier is created with the live
  // stream running mode.
  //
  // The image can be of any size with format RGB or RGBA. It's required to
  // provide a timestamp (in milliseconds) to indicate when the input image is
  // sent to the object detector. The input timestamps must be monotonically
  // increasing.
  //
  // The "result_callback" prvoides
  //   - The classification results as a ClassificationResult object.
  //   - The const reference to the corresponding input image that the image
  //     classifier runs on. Note that the const reference to the image will no
  //     longer be valid when the callback returns. To access the image data
  //     outside of the callback, callers need to make a copy of the image.
  //   - The input timestamp in milliseconds.
  absl::Status ClassifyAsync(
      mediapipe::Image image, int64 timestamp_ms,
      std::optional<mediapipe::NormalizedRect> roi = std::nullopt);

  // TODO: add Classify() variants taking a region of interest as
  // additional argument.

  // Shuts down the ImageClassifier when all works are done.
  absl::Status Close() { return runner_->Close(); }
};

}  // namespace image_classifier
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_IMAGE_CLASSIFIER_IMAGE_CLASSIFIER_H_
