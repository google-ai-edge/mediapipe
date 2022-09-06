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

#ifndef MEDIAPIPE_TASKS_CC_VISION_SEGMENTATION_IMAGE_SEGMENTER_H_
#define MEDIAPIPE_TASKS_CC_VISION_SEGMENTATION_IMAGE_SEGMENTER_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/cc/core/base_task_api.h"
#include "mediapipe/tasks/cc/vision/segmentation/image_segmenter_options.pb.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/kernels/register.h"

namespace mediapipe {
namespace tasks {
namespace vision {

// Performs segmentation on images.
//
// The API expects a TFLite model with mandatory TFLite Model Metadata.
//
// Input tensor:
//   (kTfLiteUInt8/kTfLiteFloat32)
//    - image input of size `[batch x height x width x channels]`.
//    - batch inference is not supported (`batch` is required to be 1).
//    - RGB and greyscale inputs are supported (`channels` is required to be
//      1 or 3).
//    - if type is kTfLiteFloat32, NormalizationOptions are required to be
//      attached to the metadata for input normalization.
// Output tensors:
//  (kTfLiteUInt8/kTfLiteFloat32)
//   - list of segmented masks.
//   - if `output_type` is CATEGORY_MASK, uint8 Image, Image vector of size 1.
//   - if `output_type` is CONFIDENCE_MASK, float32 Image list of size
//     `cahnnels`.
//   - batch is always 1
// An example of such model can be found at:
// https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/metadata/2
class ImageSegmenter : core::BaseTaskApi {
 public:
  using BaseTaskApi::BaseTaskApi;

  // Creates a Segmenter from the provided options. A non-default
  // OpResolver can be specified in order to support custom Ops or specify a
  // subset of built-in Ops.
  static absl::StatusOr<std::unique_ptr<ImageSegmenter>> Create(
      std::unique_ptr<ImageSegmenterOptions> options,
      std::unique_ptr<tflite::OpResolver> resolver =
          absl::make_unique<tflite::ops::builtin::BuiltinOpResolver>());

  // Runs the actual segmentation task.
  absl::StatusOr<std::vector<mediapipe::Image>> Segment(mediapipe::Image image);
};

}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_SEGMENTATION_IMAGE_SEGMENTER_H_
