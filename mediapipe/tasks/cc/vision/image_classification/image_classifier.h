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

#ifndef MEDIAPIPE_TASKS_CC_VISION_IMAGE_CLASSIFICATION_IMAGE_CLASSIFIER_H_
#define MEDIAPIPE_TASKS_CC_VISION_IMAGE_CLASSIFICATION_IMAGE_CLASSIFIER_H_

#include <memory>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/cc/components/containers/classifications.pb.h"
#include "mediapipe/tasks/cc/core/base_task_api.h"
#include "mediapipe/tasks/cc/vision/image_classification/image_classifier_options.pb.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/kernels/register.h"

namespace mediapipe {
namespace tasks {
namespace vision {

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
//
// An example of such model can be found at:
// https://tfhub.dev/bohemian-visual-recognition-alliance/lite-model/models/mushroom-identification_v1/1
class ImageClassifier : core::BaseTaskApi {
 public:
  using BaseTaskApi::BaseTaskApi;

  // Creates an ImageClassifier from the provided options. A non-default
  // OpResolver can be specified in order to support custom Ops or specify a
  // subset of built-in Ops.
  static absl::StatusOr<std::unique_ptr<ImageClassifier>> Create(
      std::unique_ptr<ImageClassifierOptions> options,
      std::unique_ptr<tflite::OpResolver> resolver =
          absl::make_unique<tflite::ops::builtin::BuiltinOpResolver>());

  // Performs actual classification on the provided Image.
  //
  // TODO: describe exact preprocessing steps once
  // YUVToImageCalculator is integrated.
  absl::StatusOr<ClassificationResult> Classify(mediapipe::Image image);

  // TODO: add Classify() variant taking a region of interest as
  // additional argument.

  // TODO: add ClassifyAsync() method for the streaming use case.
};

}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_IMAGE_CLASSIFICATION_IMAGE_CLASSIFIER_H_
